"""NearestNeighbors similarity search over the standardized outfield feature matrix.

Fits one model per broad position group (FW/MF/DF) for the default same-position
search, plus one over all outfield players for the position-agnostic toggle.
"""
import difflib
from pathlib import Path

import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from feature_engineering import FEATURE_COLUMNS, build_feature_matrix

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

POSITION_GROUPS = ["FW", "MF", "DF"]


class PlayerNotFound(Exception):
    def __init__(self, message: str, suggestions: list[str] | None = None):
        super().__init__(message)
        self.suggestions = suggestions or []


class AmbiguousPlayer(Exception):
    def __init__(self, message: str, candidates: pd.DataFrame):
        super().__init__(message)
        self.candidates = candidates


def fit_models(outfield: pd.DataFrame, metric: str = "cosine"):
    """Returns dict: {'agnostic': model, 'FW': model, 'MF': model, 'DF': model}."""
    X_scaled, scaler, _ = build_feature_matrix(outfield)

    models = {}
    models["agnostic"] = NearestNeighbors(metric=metric).fit(X_scaled.values)
    models["_agnostic_index"] = X_scaled.index.tolist()

    for group in POSITION_GROUPS:
        group_ids = outfield.loc[outfield["primary_pos"] == group, "stint_id"]
        X_group = X_scaled.loc[X_scaled.index.isin(group_ids)]
        models[group] = NearestNeighbors(metric=metric).fit(X_group.values)
        models[f"_{group}_index"] = X_group.index.tolist()

    return models, scaler, X_scaled


def resolve_player(outfield: pd.DataFrame, player_name: str, team: str | None = None) -> str:
    """Returns the resolved stint_id, or raises PlayerNotFound / AmbiguousPlayer."""
    name_norm = player_name.strip().lower()
    matches = outfield[outfield["Player"].str.strip().str.lower() == name_norm]

    if matches.empty:
        all_names = outfield["Player"].unique().tolist()
        suggestions = difflib.get_close_matches(player_name, all_names, n=5, cutoff=0.6)
        raise PlayerNotFound(f"No player found matching '{player_name}'.", suggestions)

    if len(matches) > 1:
        if team:
            team_norm = team.strip().lower()
            matches = matches[matches["Squad"].str.strip().str.lower() == team_norm]
            if len(matches) == 1:
                return matches.iloc[0]["stint_id"]
        raise AmbiguousPlayer(
            f"Multiple players match '{player_name}' - specify team.",
            matches[["Player", "Squad", "Pos", "Min"]],
        )

    return matches.iloc[0]["stint_id"]


def find_similar_players(
    outfield: pd.DataFrame,
    models: dict,
    X_scaled: pd.DataFrame,
    player_name: str,
    team: str | None = None,
    n: int = 10,
    same_position_only: bool = True,
) -> pd.DataFrame:
    stint_id = resolve_player(outfield, player_name, team)
    query_row = outfield[outfield["stint_id"] == stint_id].iloc[0]

    if same_position_only:
        group = query_row["primary_pos"]
        model = models[group]
        index_list = models[f"_{group}_index"]
    else:
        model = models["agnostic"]
        index_list = models["_agnostic_index"]

    query_vec = X_scaled.loc[stint_id].values.reshape(1, -1)
    n_query = min(n + 1, len(index_list))  # +1 to drop self after
    distances, indices = model.kneighbors(query_vec, n_neighbors=n_query)

    result_ids = [index_list[i] for i in indices[0]]
    result_distances = distances[0]

    rows = []
    for rid, dist in zip(result_ids, result_distances):
        if rid == stint_id:
            continue
        row = outfield[outfield["stint_id"] == rid].iloc[0]
        rows.append(
            {
                "Player": row["Player"],
                "Squad": row["Squad"],
                "Pos": row["Pos"],
                "similarity_score": 1 - dist,
                "raw_distance": dist,
                **{col: row[col] for col in FEATURE_COLUMNS},
            }
        )

    result_df = pd.DataFrame(rows).sort_values("similarity_score", ascending=False).head(n)
    return result_df.reset_index(drop=True)


def main():
    outfield = pd.read_csv(PROCESSED_DIR / "players_outfield.csv")
    models, scaler, X_scaled = fit_models(outfield)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    for key in ["agnostic"] + POSITION_GROUPS:
        joblib.dump(models[key], ARTIFACTS_DIR / f"nn_model_{key}.pkl")

    result = find_similar_players(outfield, models, X_scaled, "Bukayo Saka", n=10)
    print("Top matches for Bukayo Saka (same position only):")
    print(result[["Player", "Squad", "Pos", "similarity_score", "raw_distance", "Gls/90", "Ast/90"]])


if __name__ == "__main__":
    main()
