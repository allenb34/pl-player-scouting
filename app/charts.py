"""Matplotlib chart helpers: radar chart (Player Scout) and PCA scatter (Clusters)."""
import numpy as np
import matplotlib.pyplot as plt

RADAR_FEATURES = ["Gls/90", "Ast/90", "Sh/90", "SoT%", "TklW/90", "Int/90"]


def radar_chart(query_row, match_rows, feature_cols=RADAR_FEATURES):
    """query_row: Series. match_rows: list of Series (top matches)."""
    angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    def normalize(row):
        # Min-max scale each feature 0-1 across query + matches for readable overlay
        return [row[c] for c in feature_cols]

    all_rows = [query_row] + match_rows
    maxes = {c: max(r[c] for r in all_rows) or 1 for c in feature_cols}
    mins = {c: min(r[c] for r in all_rows) for c in feature_cols}

    def scaled(row):
        vals = []
        for c in feature_cols:
            span = maxes[c] - mins[c]
            vals.append((row[c] - mins[c]) / span if span > 0 else 0.5)
        vals += vals[:1]
        return vals

    ax.plot(angles, scaled(query_row), linewidth=2, label=f"{query_row['Player']} (query)", color="#1f77b4")
    ax.fill(angles, scaled(query_row), alpha=0.15, color="#1f77b4")

    colors = ["#ff7f0e", "#2ca02c", "#d62728"]
    for i, row in enumerate(match_rows[:3]):
        ax.plot(angles, scaled(row), linewidth=1.5, label=row["Player"], color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, fontsize=9)
    ax.set_yticks([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title("Per-90 profile: query vs. top matches", fontsize=11)
    fig.tight_layout()
    return fig


def pca_scatter(df, cluster_labels):
    fig, ax = plt.subplots(figsize=(8, 6))
    clusters = sorted(df["cluster"].unique())
    cmap = plt.get_cmap("tab10")

    for cluster_id in clusters:
        subset = df[df["cluster"] == cluster_id]
        label = cluster_labels.get(str(cluster_id), {}).get("label", f"Cluster {cluster_id}")
        ax.scatter(
            subset["pca_x"], subset["pca_y"],
            label=f"{cluster_id}: {label}", alpha=0.7, s=30,
            color=cmap(cluster_id % 10),
        )

    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("Playstyle clusters (PCA projection)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    return fig
