# PL Player Scouting & Similarity System

Given a Premier League player, finds statistically similar players and groups all
outfield players into data-driven "playstyle" clusters via K-Means. A Streamlit
dashboard ties both together: search a player, see their closest matches and which
cluster they belong to.

This README documents what was actually built and why, including a real data-source
limitation discovered mid-build that reshaped the feature set. It is not a sales pitch
for the tool's scouting depth.

**The headline finding, up front: FBref's free data export does not include passing or
dribbling/carrying stats, and defense is reduced to two columns.** This tool measures
**attacking output and shot profile, plus a thin defensive-activity signal** (tackles
won, interceptions) - not the passing-style and dribbling/carrying dimensions the
original spec called for. See "Key decisions and why" below for how this was found and
confirmed, and the dashboard's About/Methodology tab for the full disclosure.

**Tech stack**: `pandas` (data wrangling), `scikit-learn` (`NearestNeighbors`, `KMeans`,
`PCA`, `StandardScaler`), `matplotlib` (radar chart, PCA scatter), `streamlit`
(dashboard). Also `requests`, `soccerdata`, `truststore` (data collection attempts -
see below), `joblib` (model persistence).

Data provided by [FBref.com](https://fbref.com/), courtesy of
[Sports Reference](https://www.sports-reference.com/sharing.html).

## Architecture

```
data_collection/
  http_utils.py             -> shared rate limiter, retry/backoff, error logging (truststore-patched)
  fetch_fbref_direct.py     -> Attempt 1: direct requests scrape (confirmed non-viable, see below)
  fetch_fbref_soccerdata.py -> Attempt 2: soccerdata package (confirmed non-viable, see below)
  manual_csv_loader.py      -> Attempt 3: validated loader for user-provided FBref exports
  fetch_player_stats.py     -> orchestrator: tries 1 -> 3 (2 skipped by default), writes source_used.json
  normalize_columns.py      -> shared column normalization (MultiIndex flatten, dedup, repeated-header strip)
  parse_pasted_table.py     -> parses FBref's "Copy for Excel" paste format specifically
  ingest_pasted.py          -> one-off driver used to save each pasted table to its canonical CSV
  merge_dataset.py          -> joins the 5 tables on (Player, Squad), flags duplicates, 600-min filter
  sanity_check.py           -> row counts, null audit, Saka spot-check, position-code audit

modeling/
  feature_engineering.py    -> per-90 conversion, StandardScaler, position grouping, GK split
  similarity_model.py       -> NearestNeighbors (per position group + agnostic) + find_similar_players()
  clustering_model.py       -> KMeans + k-selection sweep (silhouette) + PCA + cluster labeling
  artifacts/                -> pickled models, k_selection.json, cluster_labels.json

app/
  app.py                    -> Streamlit entry, routes to the 3 tabs
  data_loader.py             -> @st.cache_data / @st.cache_resource loaders
  scout_tab.py               -> Player Scout page
  clusters_tab.py            -> Playstyle Clusters page
  about_tab.py                -> About/Methodology page (full limitation disclosure)
  charts.py                  -> radar chart + PCA scatter (matplotlib)

data/
  raw/fbref/                -> canonical per-category CSVs (standard/shooting/passing/possession/defense)
                                + MANUAL_EXPORT_README.md + collection_errors.log
  raw/source_used.json      -> which ingestion path actually produced the data
  processed/                -> merged, filtered, feature, and clustered datasets
```

Pipeline: **collect the 5 FBref stat tables** (fallback chain, see below) → **merge on
(Player, Squad)** with a 600-minute filter → **engineer per-90 features** (only from
confirmed-populated columns) → **fit similarity models** (per position + agnostic) →
**fit K-Means + PCA** → **Streamlit dashboard**.

## Key decisions and why

**FBref could not be scraped automatically, confirmed via three independent methods
from this local machine, not just a cloud sandbox.** Plain `requests` with a realistic
browser User-Agent got Cloudflare's "Just a moment..." JS-challenge page (HTTP 403) on
every attempt. The `soccerdata` package's Cloudflare-bypass mechanism (a real,
non-headless Chrome browser via Selenium/undetected-chromedriver) hung for minutes with
no output during testing and was abandoned as impractical; separately, its official API
for this version doesn't even support `passing`/`possession`/`defense` stat types (only
`standard`/`shooting`/`playing_time`/`keeper`/`misc`), confirmed by reading its source.
A real Chromium browser (this project's own built-in browser tool) also got stuck on the
same Cloudflare challenge after 14+ seconds. All three findings are recorded in
`fetch_fbref_direct.py` and `fetch_fbref_soccerdata.py`'s module docstrings.

**The only viable path was the user manually copying FBref's "Share & Export -> Copy for
Excel" output and pasting it in** - a legitimate, no-login, Sports-Reference-sanctioned
export mechanism. `fetch_player_stats.py`'s orchestrator still tries the direct scrape
first and only falls through to the manual path (via `manual_csv_loader.py`) if that
fails, so an automated re-run stays possible if Cloudflare's posture ever changes.

**The manually-exported data itself turned out to have a large, real gap: Passing and
Possession tables are structurally present (correct headers) but every stat column is
blank, and Defense only populates `TklW` and `Int` - not raw `Tkl`, thirds splits,
challenges, blocks, or clearances.** This was discovered by actually inspecting the
pasted data, not assumed - `passing_pasted.csv` and `possession_pasted.csv` in
`data/raw/fbref/` show every row's stat columns empty except Passing's `Ast` (a
duplicate of Standard's `Ast`). xG/npxG/xAG are similarly absent from both Standard and
Shooting entirely. The likely explanation: FBref has gated these specific
"advanced"/analytical columns behind a paid Stathead tier, leaving only the basic
counting-stat tables (Standard, Shooting) and a couple of Defense columns free.

**Given this, the original spec's Passing and Dribbling/carrying feature groups were
dropped entirely rather than built on fabricated or misleading proxies.** This was a
deliberate scope decision, confirmed with the user mid-build: build honestly with what's
actually available (Attacking + a thin Defensive signal) and document the gap
prominently, rather than pretend to a "well-rounded playstyle" comparison the data can't
support. `feature_engineering.py`'s module docstring and the dashboard's About tab both
state this plainly.

**Goalkeepers are excluded from similarity and clustering entirely, not modeled
separately.** The 5 tables pulled don't include FBref's Goalkeeping/Advanced
Goalkeeping tables (save%, clean sheets, etc.), so there's no GK-specific data to
compare on - forcing 26 goalkeepers into an outfield feature space that doesn't
describe them would produce meaningless "similarity."

**FBref codes multi-position players as concatenated tokens (`FWMF`, `DFMF`), not
comma-separated (`FW,MF`) as the build plan assumed** - confirmed from the real merged
data (`Distinct Pos codes: ['DF', 'DFMF', 'FW', 'FWMF', 'GK', 'MF', 'MFDF', 'MFFW']`).
`add_position_groups()` in `feature_engineering.py` takes the first 2 characters as
`primary_pos` accordingly.

**k=4 was chosen for K-Means by silhouette score, not the elbow method**, because
inertia decreases smoothly without a sharp elbow given the small (9-dimension) feature
space - silhouette directly measures cluster separation and gave a clear peak at k=4
(0.217, vs. 0.17-0.20 for k=5-12). Full sweep in `modeling/artifacts/k_selection.json`.

## Known limitations

- **No passing or dribbling/carrying data at all** - the single biggest gap versus the
  original spec. A player's passing range, progressive carrying, or take-on ability
  plays no role in these results.
- **No expected-goals (xG/xAG) data** - shot volume and accuracy stand in for shot
  quality, which is a real difference (a striker taking low-quality shots looks similar
  to one taking high-quality ones if their volume/accuracy happen to match).
- **Defense is reduced to tackles-won + interceptions** - no blocks, clearances, or
  tackle-success-rate signal.
- **Small clusters are possible** even with only k=4 over ~356 outfield players -
  member counts are shown prominently in the dashboard so a cluster's reliability can be
  judged directly (smallest cluster here: 43 players).
- **Mid-season transfers appear as separate rows**, not merged - 14 players (28 rows)
  are flagged in `data/processed/duplicate_players_flagged.csv`, and each stint
  independently passes the 600-minute cutoff, so a transferred player can appear twice
  with different (possibly noisier) per-90 profiles.
- **The manual data-collection path is not automatable** for a weekly refresh the way
  the sibling PL projects' GitHub Actions pipelines are - it depends on a human copying
  FBref's export output, since the automated paths are confirmed blocked.

## How to run it

**Setup:**
```bash
cd pl-player-scouting
pip install -r requirements.txt
```

**Data collection (already done for the 2025-26 season - see `data/raw/fbref/`):**
```bash
cd data_collection
python fetch_player_stats.py   # tries direct scrape, falls back to manual_csv_loader.py
```
If it falls through to the manual path, follow `data/raw/fbref/MANUAL_EXPORT_README.md`
(5 FBref URLs, "Share & Export -> Copy for Excel", paste into the corresponding
`*_pasted.csv` file, then re-run). Then:
```bash
python merge_dataset.py    # -> data/processed/players_merged.csv / players_filtered.csv
python sanity_check.py     # optional: row counts, null audit, Saka spot-check
```

**Modeling:**
```bash
cd ../modeling
python feature_engineering.py  # -> data/processed/players_outfield.csv + feature matrices
python similarity_model.py     # fits + pickles NearestNeighbors models, spot-checks Saka
python clustering_model.py     # fits + pickles KMeans/PCA, writes players_clustered.csv
```

**Dashboard:**
```bash
cd ../app
python -m streamlit run app.py --server.headless true --server.port 8509
```
(On Windows, `python -m streamlit`, not bare `streamlit`, avoids a TLS/PATH resolution
issue behind this machine's proxy - the same reason `truststore.inject_into_ssl()` is
wired into `data_collection/http_utils.py`.)

## Future improvements

- **Retry the automated scrape periodically** - `fetch_fbref_direct.py` and
  `fetch_fbref_soccerdata.py` are kept in place specifically so a future run can pick up
  automatically if Cloudflare's blocking or FBref's paid-tier gating changes.
- **A Kaggle-sourced dataset** ("Football Players Stats") was identified as a
  potential source for the missing passing/dribbling columns but requires a Kaggle API
  token the user doesn't currently have - worth revisiting if one becomes available.
- **Re-run once xG/passing data becomes accessible** to rebuild the full spec's feature
  set (Attacking + Passing + Dribbling + Defensive) and get genuinely richer playstyle
  clusters instead of the current attacking-output-dominated ones.
