"""Per-90 feature engineering, standardization, position grouping, and GK split.

Feature set is deliberately narrower than the original spec. Confirmed during data
collection (2026-09-16): FBref's free "Copy for Excel" export does not include any
usable Passing or Possession/Dribbling stats (headers exist, all values blank except
Passing's Ast, which just duplicates Standard's Ast), and Defense only populates
TklW and Int (not raw Tkl, thirds splits, challenges, blocks, or clearances). xG/npxG/
xAG are absent from Standard and Shooting entirely. See README's Key decisions section.

Usable feature groups, built only from confirmed-populated columns:
  Attacking : Gls/90, Ast/90, Sh/90, SoT%, G/Sh, G/SoT
  Defensive : TklW/90, Int/90, TklWInt/90 (TklW+Int combined)

Per-90 stats are always computed here from raw counts / 90s - never trusted from a
source's own precomputed per-90 column - so behavior is identical regardless of which
of the 3 ingestion paths produced the input.
"""
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

ATTACKING_RAW = ["Gls", "Ast"]
ATTACKING_RATE_COLS = ["SoT%", "G/Sh", "G/SoT"]  # already rates, not per-90'd further
DEFENSIVE_RAW = ["TklW", "Int"]
SHOOTING_RAW = ["Sh", "SoT"]

FEATURE_COLUMNS = [
    "Gls/90", "Ast/90", "Sh/90", "SoT%", "G/Sh", "G/SoT",
    "TklW/90", "Int/90", "TklWInt/90",
]


def compute_per90(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    assert (df["90s"] > 0).all(), "0-minute rows should have been dropped by the 600-min filter"

    df["Gls/90"] = df["Gls"] / df["90s"]
    df["Ast/90"] = df["Ast"] / df["90s"]
    df["Sh/90"] = df["Sh"] / df["90s"]
    df["TklW/90"] = df["TklW"] / df["90s"]
    df["Int/90"] = df["Int"] / df["90s"]
    df["TklWInt/90"] = (df["TklW"] + df["Int"]) / df["90s"]

    # Rate stats (SoT%, G/Sh, G/SoT) are undefined (NaN) for 0-shot players - fill with 0
    # rather than drop, since a player with 0 shots genuinely has 0 shooting efficiency
    # to compare, not missing data.
    for col in ["SoT%", "G/Sh", "G/SoT"]:
        df[col] = df[col].fillna(0.0)

    return df


def split_goalkeepers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    is_gk = df["Pos"].str.contains("GK")
    return df[~is_gk].copy(), df[is_gk].copy()


def add_position_groups(df: pd.DataFrame) -> pd.DataFrame:
    """FBref codes multi-position players as concatenated 2-letter tokens (e.g.
    'MFFW', 'DFMF'), not comma-separated - confirmed from real data, not the
    comma-separated format the original spec assumed. primary_pos = first token.
    """
    df = df.copy()
    df["primary_pos"] = df["Pos"].str[:2]
    df["secondary_pos"] = df["Pos"].str[2:4].replace("", None)
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, pd.DataFrame]:
    """Returns (scaled_features_df indexed by stint_id, fitted scaler, raw per-90 df)."""
    X_raw = df.set_index("stint_id")[FEATURE_COLUMNS]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw), index=X_raw.index, columns=FEATURE_COLUMNS
    )
    return X_scaled, scaler, X_raw


def main():
    df = pd.read_csv(PROCESSED_DIR / "players_filtered.csv")
    df = compute_per90(df)
    outfield, goalkeepers = split_goalkeepers(df)
    outfield = add_position_groups(outfield)

    goalkeepers.to_csv(PROCESSED_DIR / "goalkeepers.csv", index=False)
    outfield.to_csv(PROCESSED_DIR / "players_outfield.csv", index=False)

    print(f"Outfield: {len(outfield)} rows, Goalkeepers: {len(goalkeepers)} rows")
    print("Primary position counts:")
    print(outfield["primary_pos"].value_counts())

    X_scaled, scaler, X_raw = build_feature_matrix(outfield)
    print(f"\nFeature matrix: {X_scaled.shape}")
    print("Per-feature mean (should be ~0):", X_scaled.mean().round(3).to_dict())
    print("Per-feature std (should be ~1):", X_scaled.std().round(3).to_dict())

    X_scaled.to_csv(PROCESSED_DIR / "feature_matrix_scaled.csv")
    X_raw.to_csv(PROCESSED_DIR / "feature_matrix_raw.csv")

    import joblib
    ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler_outfield.pkl")


if __name__ == "__main__":
    main()
