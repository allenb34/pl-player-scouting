"""Player Scout tab: search a player, see top-N similar players and a radar chart."""
import streamlit as st

from data_loader import load_clustered_players, load_similarity_models
from charts import radar_chart, RADAR_FEATURES

from similarity_model import find_similar_players, PlayerNotFound, AmbiguousPlayer


def render():
    st.header("Player Scout")
    st.caption(
        "Search a Premier League player (600+ minutes this season) to find "
        "statistically similar players based on attacking output and defensive workrate."
    )

    outfield = load_clustered_players()
    models, X_scaled = load_similarity_models()

    display_names = (outfield["Player"] + " (" + outfield["Squad"] + ")").tolist()
    name_to_stint = dict(zip(display_names, outfield["stint_id"]))

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_display = st.selectbox("Player", sorted(display_names))
    with col2:
        same_pos_only = st.toggle("Same position only", value=True)
    with col3:
        n = st.slider("Number of matches", 5, 20, 10)

    stint_id = name_to_stint[selected_display]
    query_row = outfield[outfield["stint_id"] == stint_id].iloc[0]

    st.subheader(f"{query_row['Player']} - {query_row['Squad']} ({query_row['Pos']})")
    stat_cols = st.columns(6)
    stat_cols[0].metric("Gls/90", f"{query_row['Gls/90']:.2f}")
    stat_cols[1].metric("Ast/90", f"{query_row['Ast/90']:.2f}")
    stat_cols[2].metric("Sh/90", f"{query_row['Sh/90']:.2f}")
    stat_cols[3].metric("SoT%", f"{query_row['SoT%']:.1f}")
    stat_cols[4].metric("TklW/90", f"{query_row['TklW/90']:.2f}")
    stat_cols[5].metric("Int/90", f"{query_row['Int/90']:.2f}")

    try:
        results = find_similar_players(
            outfield, models, X_scaled, query_row["Player"], team=query_row["Squad"],
            n=n, same_position_only=same_pos_only,
        )
    except (PlayerNotFound, AmbiguousPlayer) as e:
        st.error(str(e))
        return

    st.subheader(f"Top {n} similar players")
    display_cols = ["Player", "Squad", "Pos", "similarity_score", "raw_distance"] + RADAR_FEATURES
    display_df = results[display_cols].copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "Rank"
    st.dataframe(
        display_df.style.format(
            {c: "{:.3f}" for c in ["similarity_score", "raw_distance"] + RADAR_FEATURES}
        ),
        use_container_width=True,
    )

    st.subheader("Radar comparison (top 3)")
    top3_rows = [results.iloc[i] for i in range(min(3, len(results)))]
    fig = radar_chart(query_row, top3_rows)
    st.pyplot(fig)
