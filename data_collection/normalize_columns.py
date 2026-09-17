"""Column normalization shared by all three FBref ingestion paths (direct scrape,
soccerdata, manual CSV export), so merge_dataset.py only ever sees one canonical
schema regardless of which path produced it.
"""
import pandas as pd


def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten FBref's two-row MultiIndex header (e.g. ('Performance', 'Gls') -> 'Gls').

    FBref groups columns under headers like "Playing Time", "Performance", "Expected",
    "Per 90 Minutes". The top-level label is redundant once flattened (a table only has
    one 'Gls' column) except for the "Per 90 Minutes" group, which duplicates raw-stat
    names (also 'Gls'). Since this project always computes per-90 itself from raw counts
    (see feature_engineering.py), any second occurrence of a duplicate name is dropped
    here in favor of the first (raw) occurrence.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    flat_cols = []
    for top, bottom in df.columns:
        name = bottom if bottom and not str(bottom).startswith("Unnamed") else top
        flat_cols.append(name)
    df = df.copy()
    df.columns = flat_cols
    # Keep first occurrence of any duplicate-named column (raw stat over its Per-90 twin)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def strip_repeated_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop mid-table repeated header rows (FBref repeats 'Player' as a literal row
    value every ~25 rows in some export/scrape paths)."""
    if "Player" in df.columns:
        df = df[df["Player"] != "Player"]
    if "Rk" in df.columns:
        df = df.drop(columns=["Rk"])
    return df.reset_index(drop=True)


def normalize_fbref_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full normalization chain: flatten headers, strip repeated-header
    artifact rows and the Rk column. Safe to call on output from any of the 3
    ingestion paths - a no-op on already-flat, already-clean data.
    """
    df = flatten_multiindex_columns(df)
    df = strip_repeated_header_rows(df)
    return df
