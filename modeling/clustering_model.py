"""K-Means playstyle clustering + k-selection (elbow/silhouette) + PCA visualization +
descriptive cluster labeling.

Note: with only 9 features (6 attacking, 3 defensive - see feature_engineering.py's
module docstring for why passing/dribbling groups were dropped), clusters mostly
separate players by attacking output level and defensive workrate, not the richer
passing/dribbling-style distinctions the original spec envisioned. Documented in the
dashboard's About/Methodology tab, not hidden.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from feature_engineering import FEATURE_COLUMNS, build_feature_matrix

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

K_RANGE = range(4, 13)

FEATURE_LABELS = {
    "Gls/90": "goal-scoring",
    "Ast/90": "assist-creating",
    "Sh/90": "shot volume",
    "SoT%": "shot accuracy",
    "G/Sh": "finishing efficiency",
    "G/SoT": "clinical finishing",
    "TklW/90": "tackling",
    "Int/90": "interceptions",
    "TklWInt/90": "defensive workrate",
}


def sweep_k(X_scaled: np.ndarray) -> dict:
    results = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        results[k] = {"inertia": float(km.inertia_), "silhouette": float(sil)}
    return results


def choose_k(sweep_results: dict) -> tuple[int, str]:
    best_k = max(sweep_results, key=lambda k: sweep_results[k]["silhouette"])
    best_sil = sweep_results[best_k]["silhouette"]
    reasoning = (
        f"k={best_k} chosen: highest silhouette score ({best_sil:.3f}) across the "
        f"k=4..12 sweep. Inertia decreases monotonically (expected) without a sharp "
        f"elbow given the small (9-dimension) feature space, so silhouette - which "
        f"directly measures cluster separation - is the more reliable signal here."
    )
    return best_k, reasoning


def label_cluster(centroid: pd.Series, top_n: int = 4) -> tuple[str, pd.DataFrame]:
    ranked = centroid.abs().sort_values(ascending=False).head(top_n)
    top_features = pd.DataFrame(
        {"feature": ranked.index, "mean_zscore": [centroid[f] for f in ranked.index]}
    )
    phrases = []
    for _, row in top_features.iterrows():
        direction = "high" if row["mean_zscore"] > 0 else "low"
        phrases.append(f"{direction} {FEATURE_LABELS[row['feature']]}")
    label = ", ".join(phrases[:3]).capitalize()
    return label, top_features


def main():
    outfield = pd.read_csv(PROCESSED_DIR / "players_outfield.csv")
    X_scaled, scaler, _ = build_feature_matrix(outfield)

    print("Sweeping k=4..12...")
    sweep_results = sweep_k(X_scaled.values)
    for k, r in sweep_results.items():
        print(f"  k={k}: inertia={r['inertia']:.1f}, silhouette={r['silhouette']:.3f}")

    best_k, reasoning = choose_k(sweep_results)
    print(f"\n{reasoning}")

    kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42).fit(X_scaled.values)
    outfield = outfield.set_index("stint_id")
    outfield["cluster"] = kmeans.labels_
    outfield = outfield.reset_index()

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled.values)
    outfield["pca_x"] = pca_coords[:, 0]
    outfield["pca_y"] = pca_coords[:, 1]

    cluster_info = {}
    for cluster_id in sorted(outfield["cluster"].unique()):
        members = outfield[outfield["cluster"] == cluster_id]
        member_ids = members["stint_id"]
        centroid = X_scaled.loc[X_scaled.index.isin(member_ids)].mean()
        label, top_features = label_cluster(centroid)
        cluster_info[int(cluster_id)] = {
            "label": label,
            "member_count": int(len(members)),
            "top_features": top_features.to_dict(orient="records"),
        }
        print(f"\nCluster {cluster_id} ({len(members)} players): {label}")
        print(top_features.to_string(index=False))
        print("Sample players:", ", ".join(members["Player"].head(5).tolist()))

    outfield.to_csv(PROCESSED_DIR / "players_clustered.csv", index=False)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(kmeans, ARTIFACTS_DIR / "kmeans_model.pkl")
    joblib.dump(pca, ARTIFACTS_DIR / "pca_model.pkl")

    with open(ARTIFACTS_DIR / "k_selection.json", "w") as f:
        json.dump(
            {"sweep": sweep_results, "chosen_k": best_k, "reasoning": reasoning},
            f,
            indent=2,
        )
    with open(ARTIFACTS_DIR / "cluster_labels.json", "w") as f:
        json.dump(cluster_info, f, indent=2)

    assert outfield["cluster"].isnull().sum() == 0
    assert outfield.groupby("stint_id").size().max() == 1

    print(f"\nWrote {PROCESSED_DIR / 'players_clustered.csv'}")


if __name__ == "__main__":
    main()
