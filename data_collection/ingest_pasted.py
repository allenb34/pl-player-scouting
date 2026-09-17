"""One-off driver: parse a raw FBref paste file and write the canonical CSV.

Usage: python ingest_pasted.py <stat_type> <raw_pasted_filename>
e.g.:  python ingest_pasted.py standard standard_pasted.csv
"""
import sys
from pathlib import Path

from parse_pasted_table import parse_pasted_table

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "fbref"

if __name__ == "__main__":
    stat_type, raw_filename = sys.argv[1], sys.argv[2]
    df = parse_pasted_table(RAW_DIR / raw_filename)
    out_path = RAW_DIR / f"{stat_type}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {df.shape}")
    print(df.columns.tolist())
