"""About/Methodology tab: data source, season, cutoffs, feature list, k-selection
reasoning, and limitations - stated plainly, not undersold."""
import streamlit as st

from data_loader import load_k_selection, load_source_used, load_goalkeepers


def render():
    st.header("About / Methodology")

    st.subheader("Data source")
    source_info = load_source_used()
    st.write(
        f"**Season**: 2025-26 Premier League (most recently completed season at build "
        f"time). **Source**: FBref.com (Sports Reference). **Ingestion path actually "
        f"used**: `{source_info['source']}` - {source_info['note']}"
    )
    st.markdown(
        "Data provided by [FBref.com](https://fbref.com/), courtesy of "
        "[Sports Reference](https://www.sports-reference.com/sharing.html)."
    )

    st.subheader("Minutes cutoff")
    st.write(
        "Players with fewer than **600 minutes** played this season are excluded from "
        "similarity and clustering - below that, per-90 rates are dominated by "
        "small-sample noise (e.g. a substitute who scored once in 40 minutes would show "
        "an absurd Gls/90). A low-minutes player's absence from search results is "
        "expected, not a bug - see `data/processed/excluded_players.csv`."
    )

    st.subheader("Feature set - and a real limitation")
    st.warning(
        "**This is narrower than a typical scouting tool.** FBref's free 'Copy for "
        "Excel' export (the only viable data path after direct scraping and the "
        "`soccerdata` package both failed against Cloudflare - see README) does not "
        "include usable Passing or Possession/Dribbling stats, and xG/xAG are absent "
        "entirely. Defense only provides tackles-won and interceptions - not raw "
        "tackles, thirds breakdown, blocks, or clearances. As a result, this tool "
        "measures **attacking output and shot profile, plus a thin defensive-activity "
        "signal** - not passing style or dribbling/carrying ability, which the original "
        "spec called for but no accessible data source currently supports."
    )
    st.write("**Features actually used** (all computed as raw count ÷ minutes-played/90, not trusted from any source's own per-90 column):")
    st.markdown(
        "- **Attacking**: Gls/90, Ast/90, Sh/90, SoT%, G/Sh, G/SoT\n"
        "- **Defensive**: TklW/90, Int/90, TklWInt/90 (combined)"
    )

    st.subheader("Goalkeepers")
    gks = load_goalkeepers()
    st.write(
        f"**{len(gks)} goalkeepers** (600+ minutes) are excluded from similarity and "
        f"clustering entirely. The 5 stat tables pulled don't include FBref's "
        f"Goalkeeping/Advanced Goalkeeping tables (save%, clean sheets, etc.), so there's "
        f"no meaningful GK-specific data to compare them on - rather than force them into "
        f"an outfield feature space that doesn't describe them, they're cleanly excluded."
    )

    st.subheader("Position grouping")
    st.write(
        "Default similarity search restricts candidates to the same broad position "
        "group (FW/MF/DF), using the **first** position code FBref lists for "
        "multi-position players (e.g. 'FWMF' → FW). This is a deliberate simplification "
        "for players who play multiple roles. Toggle 'Same position only' off to compare "
        "across all outfield positions."
    )

    st.subheader("K-Means: choosing k")
    k_info = load_k_selection()
    st.write(k_info["reasoning"])
    sweep_rows = [
        {"k": k, "inertia": round(v["inertia"], 1), "silhouette": round(v["silhouette"], 3)}
        for k, v in k_info["sweep"].items()
    ]
    st.table(sweep_rows)

    st.subheader("Known limitations")
    st.markdown(
        "- **No passing or dribbling/carrying data** (see feature set warning above) - "
        "the single biggest gap versus the original spec.\n"
        "- **Small clusters are possible** even with only 4 clusters over ~350 players - "
        "member counts are shown prominently in the Clusters tab so a small cluster's "
        "reliability can be judged directly.\n"
        "- **Mid-season transfers appear as separate rows** (not merged) for the same "
        "player at two clubs - both rows independently pass the 600-minute cutoff, so a "
        "transferred player can show up twice with different, possibly noisier profiles.\n"
        "- **No expected-goals (xG/xAG) data** - FBref's free export didn't include it "
        "for this season at build time, so shot volume/accuracy stand in for shot quality."
    )
