"""Joins the 5 FBref stat tables into one modeling dataset.

Reads data/raw/fbref/{standard,shooting,passing,possession,defense}.csv (already
normalized to a flat, canonical schema by whichever fetch path produced them - see
normalize_columns.py). Writes:
  data/processed/players_merged.csv            - all 5 tables joined, before minutes filter
  data/processed/duplicate_players_flagged.csv - players appearing under >1 Squad
  data/processed/players_filtered.csv          - after the 600-minute cutoff
  data/processed/excluded_players.csv          - dropped rows, with reason
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "fbref"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

MIN_MINUTES = 600

STAT_TYPES = ["standard", "shooting", "passing", "possession", "defense"]

# Columns present in every table that would otherwise collide on merge (kept only
# from the "standard" table, the left side of every join).
SHARED_INFO_COLS = {"Player", "Nation", "Pos", "Squad", "Age", "Born"}


def load_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for stat_type in STAT_TYPES:
        path = RAW_DIR / f"{stat_type}.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found - run fetch_player_stats.py first.")
        df = pd.read_csv(path)
        if "Comp" in df.columns:
            df = df[df["Comp"] == "Premier League"]
        tables[stat_type] = df
    return tables


def merge_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = tables["standard"].copy()

    for stat_type in ["shooting", "passing", "possession", "defense"]:
        other = tables[stat_type]
        # Drop shared info columns from the right side so the join only adds new stats
        drop_cols = [c for c in other.columns if c in SHARED_INFO_COLS and c not in ("Player", "Squad")]
        other = other.drop(columns=drop_cols)
        merged = merged.merge(other, on=["Player", "Squad"], how="inner", suffixes=("", f"_{stat_type}_dup"))

    return merged


def flag_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    dupe_players = df["Player"][df["Player"].duplicated(keep=False)]
    duplicates = df[df["Player"].isin(dupe_players)].copy()
    duplicates = duplicates.sort_values("Player")
    return duplicates


def add_stint_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stint_id"] = df["Player"] + "__" + df["Squad"]
    return df


def apply_minutes_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    below_cutoff = df["Min"] < MIN_MINUTES
    excluded = df[below_cutoff].copy()
    excluded["reason"] = "below_600_min_threshold"
    filtered = df[~below_cutoff].copy()
    return filtered, excluded


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    tables = load_tables()
    print("Input row counts:", {k: len(v) for k, v in tables.items()})

    merged = merge_tables(tables)
    merged = add_stint_id(merged)
    merged.to_csv(PROCESSED_DIR / "players_merged.csv", index=False)
    print(f"Merged: {len(merged)} rows (smallest input table: {min(len(v) for v in tables.values())} rows)")

    duplicates = flag_duplicates(merged)
    duplicates.to_csv(PROCESSED_DIR / "duplicate_players_flagged.csv", index=False)
    print(f"Duplicate-name players flagged: {duplicates['Player'].nunique()} names, {len(duplicates)} rows")

    print("Distinct Pos codes observed:", sorted(merged["Pos"].dropna().unique().tolist()))

    filtered, excluded = apply_minutes_filter(merged)
    filtered.to_csv(PROCESSED_DIR / "players_filtered.csv", index=False)
    excluded.to_csv(PROCESSED_DIR / "excluded_players.csv", index=False)
    print(f"After {MIN_MINUTES}-minute filter: {len(filtered)} kept, {len(excluded)} excluded")

    assert len(filtered) + len(excluded) == len(merged), "filtered + excluded should equal merged total"


if __name__ == "__main__":
    main()
