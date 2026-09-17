"""Attempt 3 (guaranteed fallback): load manually-exported FBref CSVs.

See data/raw/fbref/MANUAL_EXPORT_README.md for the exact steps the user follows to
produce these files (FBref's "Share & Export -> Get table as CSV" button, no login
required).
"""
from pathlib import Path

import pandas as pd

from http_utils import log_error
from normalize_columns import normalize_fbref_table

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "fbref"

# Confirmed during development (2026-09-16): FBref's free "Copy for Excel" export only
# populates Standard and Shooting in full. Passing/Possession are structurally present
# (headers exist) but every stat column is blank except Passing's Ast (a duplicate of
# Standard). Defense only populates TklW and Int - not raw Tkl, thirds, challenges,
# blocks, clearances, or errors. Required columns below reflect what's ACTUALLY
# available, not the original spec's full wishlist - see README's Key decisions section.
REQUIRED_COLUMNS = {
    "standard": {"Player", "Nation", "Pos", "Squad", "Age", "MP", "Starts", "Min", "90s", "Gls", "Ast"},
    "shooting": {"Player", "Squad", "Sh", "SoT", "SoT%", "Sh/90", "SoT/90"},
    "passing": {"Player", "Squad", "Ast"},
    "possession": {"Player", "Squad"},
    "defense": {"Player", "Squad", "TklW", "Int"},
}


def load_all() -> dict[str, pd.DataFrame] | None:
    """Load and validate all 5 manually-exported CSVs. Returns None (with a clear,
    actionable log entry) if any file is missing or malformed - never guesses.
    """
    results = {}
    for stat_type, required_cols in REQUIRED_COLUMNS.items():
        path = RAW_DIR / f"{stat_type}.csv"
        if not path.exists():
            log_error(
                "manual_csv_loader",
                f"{stat_type}: expected file not found at {path}. "
                f"See data/raw/fbref/MANUAL_EXPORT_README.md for export steps.",
            )
            return None

        df = pd.read_csv(path)
        df = normalize_fbref_table(df)

        missing = required_cols - set(df.columns)
        if missing:
            log_error(
                "manual_csv_loader",
                f"{stat_type}.csv is missing expected column(s): {sorted(missing)}. "
                f"Found columns: {sorted(df.columns)}. Re-export from FBref and check "
                f"the file wasn't truncated or edited.",
            )
            return None

        results[stat_type] = df

    return results


if __name__ == "__main__":
    data = load_all()
    if data is None:
        print("Manual CSV load failed (see data/raw/collection_errors.log).")
    else:
        for stat_type, df in data.items():
            print(f"{stat_type}: {df.shape}")
