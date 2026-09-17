"""Attempt 1: direct FBref scrape via requests + pandas.read_html.

Confirmed during development (2026-09-16, from this machine, not a cloud sandbox):
plain requests with a realistic browser User-Agent still gets Cloudflare's "Just a
moment..." JS-challenge page (HTTP 403) on every request. This module is kept as the
first, cheapest attempt in the fallback chain in case Cloudflare's posture changes
later, but it is expected to fail through to fetch_fbref_soccerdata.py / the manual
CSV path in practice.
"""
import re
from io import StringIO

import pandas as pd
import requests

from http_utils import RateLimiter, get_html, log_error
from normalize_columns import normalize_fbref_table

SEASON = "2025-2026"
BASE = f"https://fbref.com/en/comps/9/{SEASON}"

STAT_URLS = {
    "standard": f"{BASE}/stats/{SEASON}-Premier-League-Stats",
    "shooting": f"{BASE}/shooting/{SEASON}-Premier-League-Stats",
    "passing": f"{BASE}/passing/{SEASON}-Premier-League-Stats",
    "possession": f"{BASE}/possession/{SEASON}-Premier-League-Stats",
    "defense": f"{BASE}/defense/{SEASON}-Premier-League-Stats",
}

# Table element id FBref uses for each stat category's player table
TABLE_IDS = {
    "standard": "stats_standard",
    "shooting": "stats_shooting",
    "passing": "stats_passing",
    "possession": "stats_possession",
    "defense": "stats_defense",
}


def _extract_table(html_text: str, table_id: str) -> pd.DataFrame | None:
    """Find a table by id, either directly in the HTML or inside an HTML comment
    (FBref wraps some tables in comments to defeat naive scrapers)."""
    try:
        tables = pd.read_html(StringIO(html_text), attrs={"id": table_id})
        if tables:
            return tables[0]
    except ValueError:
        pass

    # Fallback: search HTML comments for the table
    for match in re.finditer(r"<!--(.*?)-->", html_text, re.DOTALL):
        comment_body = match.group(1)
        if table_id not in comment_body:
            continue
        try:
            tables = pd.read_html(StringIO(comment_body), attrs={"id": table_id})
            if tables:
                return tables[0]
        except ValueError:
            continue
    return None


def fetch_all() -> dict[str, pd.DataFrame] | None:
    """Attempt all 5 stat tables via direct scrape. Returns None (partial results
    discarded) unless all 5 succeed - the orchestrator only accepts a full set.
    """
    session = requests.Session()
    rate_limiter = RateLimiter(min_interval=4.0)
    results = {}

    for stat_type, url in STAT_URLS.items():
        html_text = get_html(session, url, rate_limiter, context=f"fbref_direct:{stat_type}")
        if html_text is None:
            log_error("fetch_fbref_direct", f"{stat_type}: no HTML returned, aborting direct-scrape attempt")
            return None

        table = _extract_table(html_text, TABLE_IDS[stat_type])
        if table is None:
            log_error("fetch_fbref_direct", f"{stat_type}: table id '{TABLE_IDS[stat_type]}' not found in response")
            return None

        results[stat_type] = normalize_fbref_table(table)

    return results


if __name__ == "__main__":
    data = fetch_all()
    if data is None:
        print("Direct FBref scrape failed (see data/raw/collection_errors.log).")
    else:
        for stat_type, df in data.items():
            print(f"{stat_type}: {df.shape}")
