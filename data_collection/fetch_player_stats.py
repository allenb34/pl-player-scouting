"""Orchestrator: tries the FBref data-source fallback chain in order and writes
the canonical CSVs to data/raw/fbref/.

Order: direct scrape -> manual CSV export. `soccerdata` is deliberately NOT
attempted automatically - see fetch_fbref_soccerdata.py's module docstring for
why (confirmed hang risk + missing stat_type support in the installed version).
Run it manually if you want to retry it explicitly.

Writes data/raw/source_used.json recording which path produced the data actually
used, so the README and dashboard can state this honestly.
"""
import json
from pathlib import Path

import fetch_fbref_direct
import manual_csv_loader

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FBREF_DIR = RAW_DIR / "fbref"
SOURCE_USED_PATH = RAW_DIR / "source_used.json"

STAT_TYPES = ["standard", "shooting", "passing", "possession", "defense"]


def main():
    FBREF_DIR.mkdir(parents=True, exist_ok=True)

    print("Attempt 1: direct FBref scrape...")
    data = fetch_fbref_direct.fetch_all()
    source = "direct"

    if data is None:
        print("Attempt 1 failed. Attempt 2 (soccerdata) skipped by default - see module docstring.")
        print("Attempt 3: manual CSV export...")
        data = manual_csv_loader.load_all()
        source = "manual"

    if data is None:
        raise SystemExit(
            "All available data-source attempts failed. Follow "
            "data/raw/fbref/MANUAL_EXPORT_README.md to provide the 5 CSVs by hand, "
            "then re-run this script."
        )

    for stat_type in STAT_TYPES:
        if stat_type not in data:
            raise SystemExit(f"Internal error: '{stat_type}' missing from a supposedly successful fetch.")
        out_path = FBREF_DIR / f"{stat_type}.csv"
        data[stat_type].to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({data[stat_type].shape[0]} rows)")

    SOURCE_USED_PATH.write_text(json.dumps({"source": source, "stat_types": STAT_TYPES}, indent=2))
    print(f"Data source used: {source}")


if __name__ == "__main__":
    main()
