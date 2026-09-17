"""Diagnostics: row counts, null checks, a known-player spot-check, and a position-code
audit. Run after merge_dataset.py.
"""
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

SPOT_CHECK_PLAYER = "Bukayo Saka"
SPOT_CHECK_SQUAD = "Arsenal"


def main():
    merged = pd.read_csv(PROCESSED_DIR / "players_merged.csv")
    filtered = pd.read_csv(PROCESSED_DIR / "players_filtered.csv")
    excluded = pd.read_csv(PROCESSED_DIR / "excluded_players.csv")
    duplicates = pd.read_csv(PROCESSED_DIR / "duplicate_players_flagged.csv")

    print(f"players_merged.csv:   {len(merged)} rows")
    print(f"players_filtered.csv: {len(filtered)} rows")
    print(f"excluded_players.csv: {len(excluded)} rows")
    assert len(filtered) + len(excluded) == len(merged), "row-count arithmetic mismatch"
    print("Row-count arithmetic OK.")

    null_frac = merged.isnull().mean().sort_values(ascending=False)
    high_null = null_frac[null_frac > 0.05]
    if len(high_null):
        print("\nColumns with >5% nulls (investigate before trusting downstream features):")
        print(high_null)
    else:
        print("\nNo columns with >5% nulls.")

    print(f"\nSpot-check: {SPOT_CHECK_PLAYER} ({SPOT_CHECK_SQUAD})")
    row = filtered[(filtered["Player"] == SPOT_CHECK_PLAYER) & (filtered["Squad"] == SPOT_CHECK_SQUAD)]
    if row.empty:
        row = merged[(merged["Player"] == SPOT_CHECK_PLAYER) & (merged["Squad"] == SPOT_CHECK_SQUAD)]
        if row.empty:
            print(f"  NOT FOUND in merged data at all - check the join or the player/squad name.")
        else:
            print(f"  Found in merged but NOT in filtered - excluded for minutes, or a join issue. Row:")
            print(row.T)
    else:
        print("  Found in filtered data:")
        print(row[[c for c in ["Player", "Squad", "Pos", "Min", "90s", "Gls", "Ast"] if c in row.columns]].T)

    print(f"\nDuplicate-Squad players (mid-season transfers): {duplicates['Player'].nunique() if len(duplicates) else 0} names")
    if len(duplicates):
        print(duplicates[["Player", "Squad"]].drop_duplicates().head(10))

    print("\nDistinct Pos codes:", sorted(merged["Pos"].dropna().unique().tolist()))


if __name__ == "__main__":
    main()
