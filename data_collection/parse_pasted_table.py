"""Parses FBref's "Copy for Excel" paste format (used when a user pastes table text
directly rather than saving a browser CSV export).

Real structure observed from the Standard Stats table (2026-09-16):
  line 0: group-label row ("Playing Time", "Performance", "Per 90 Minutes", ...),
          mostly empty cells otherwise - not real headers.
  line 1: the actual column header row (Rk, Player, Nation, Pos, Squad, ...).
  line 2+: data rows.

Trailing columns beyond the real stats: a "Matches" column that is always the
literal text "Matches" (a link placeholder, not data) and a final column of
FBref's internal per-player hex ID (kept as fbref_id - a more robust dedup/join
aid than name+squad alone).
"""
from pathlib import Path

import pandas as pd

from normalize_columns import normalize_fbref_table


def _find_header_row(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        tokens = [t.strip() for t in line.split(",")]
        if "Player" in tokens and "Squad" in tokens:
            return i
    raise ValueError("Could not find a header row containing both 'Player' and 'Squad'.")


def parse_pasted_table(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    header_idx = _find_header_row(lines)

    df = pd.read_csv(path, skiprows=header_idx, header=0)
    df = df.dropna(how="all")

    df = normalize_fbref_table(df)

    # Drop the literal-text "Matches" link-placeholder column.
    if "Matches" in df.columns:
        df = df.drop(columns=["Matches"])

    # The final column (however it's labeled - often garbled in a raw paste,
    # e.g. "-9999") holds FBref's internal per-player hex ID. Rename it.
    last_col = df.columns[-1]
    if last_col not in ("Player", "Squad") and df[last_col].astype(str).str.match(r"^[0-9a-f]{6,10}$").mean() > 0.9:
        df = df.rename(columns={last_col: "fbref_id"})

    return df.reset_index(drop=True)


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../data/raw/fbref/standard_pasted.csv")
    result = parse_pasted_table(p)
    print(result.shape)
    print(result.columns.tolist())
    print(result.head(3))
