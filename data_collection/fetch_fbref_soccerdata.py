"""Attempt 2: the `soccerdata` package.

Confirmed during development (2026-09-16): this path is NOT reliable in this
environment and is not called automatically by fetch_player_stats.py's default
order. Two real problems, found by actually running it, not assumed:

1. The installed soccerdata version's `read_player_season_stats()` only supports
   stat_type in {"standard", "keeper", "shooting", "playing_time", "misc"} - it does
   NOT support "passing", "possession", or "defense", so it can never produce all 5
   tables this project needs through its documented API alone.
2. Its actual Cloudflare-bypass mechanism launches a real (non-headless by default)
   Chrome browser via `seleniumbase` undetected-chromedriver, which needs to download
   a chromedriver binary on first use. In testing this hung for minutes with no
   output and no clear completion signal - impractical to depend on for a repeatable
   pipeline run. (A plain requests call AND this project's own built-in browser tool
   were also tried directly against FBref and both still hit the Cloudflare "Just a
   moment..." challenge page - see fetch_fbref_direct.py and the README's Key
   decisions section.)

This module is kept, correctly implemented, for the 2 stat types it does support
(standard, shooting) in case a future soccerdata release adds passing/possession/
defense support or the Cloudflare situation changes - but fetch_player_stats.py
does not call it by default. Run manually with `python fetch_fbref_soccerdata.py`
to retry it explicitly.
"""
import truststore

truststore.inject_into_ssl()

from http_utils import log_error
from normalize_columns import normalize_fbref_table

SEASON = "2025-2026"

# Stat types this installed soccerdata version's official API actually supports
# for read_player_season_stats(). passing/possession/defense are NOT in this list.
SUPPORTED_STAT_TYPES = {"standard", "shooting"}


def fetch_all():
    """Attempt standard + shooting only (the only 2 of 5 required tables this
    package version's official API supports). Always returns None for the
    remaining 3 - the orchestrator will fall through to the manual CSV path
    for those regardless of whether this call succeeds.
    """
    try:
        import soccerdata as sd
    except ImportError as e:
        log_error("fetch_fbref_soccerdata", f"soccerdata not importable: {e}")
        return None

    try:
        fb = sd.FBref(leagues="ENG-Premier League", seasons=SEASON)
    except Exception as e:
        log_error("fetch_fbref_soccerdata", f"failed to init FBref client: {e}")
        return None

    results = {}
    for stat_type in SUPPORTED_STAT_TYPES:
        try:
            df = fb.read_player_season_stats(stat_type=stat_type)
            results[stat_type] = normalize_fbref_table(df.reset_index())
        except Exception as e:
            log_error("fetch_fbref_soccerdata", f"{stat_type}: {type(e).__name__}: {e}")
            return None

    # passing/possession/defense are not obtainable via this package version's
    # official API - report this explicitly rather than silently returning a
    # partial (and therefore unusable) result set.
    log_error(
        "fetch_fbref_soccerdata",
        "standard+shooting succeeded but passing/possession/defense are unsupported "
        "by this soccerdata version - treating as an incomplete attempt.",
    )
    return None


if __name__ == "__main__":
    data = fetch_all()
    if data is None:
        print("soccerdata attempt did not produce all 5 tables (see data/raw/collection_errors.log).")
    else:
        for stat_type, df in data.items():
            print(f"{stat_type}: {df.shape}")
