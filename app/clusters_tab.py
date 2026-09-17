"""Playstyle Clusters tab: PCA scatter, per-cluster summary, drill-down player list."""
import streamlit as st

from data_loader import load_clustered_players, load_cluster_labels
from charts import pca_scatter


def render():
    st.header("Playstyle Clusters")
    st.caption(
        "K-Means clusters over standardized per-90 features. With the confirmed "
        "feature set (attacking output + defensive workrate only - see About tab), "
        "clusters mostly separate players by scoring volume and defensive activity, "
        "not passing or dribbling style."
    )

    outfield = load_clustered_players()
    cluster_labels = load_cluster_labels()

    fig = pca_scatter(outfield, cluster_labels)
    st.pyplot(fig)

    st.subheader("Cluster summary")
    summary_rows = []
    for cid, info in sorted(cluster_labels.items(), key=lambda x: int(x[0])):
        summary_rows.append(
            {"Cluster": cid, "Label": info["label"], "Members": info["member_count"]}
        )
    st.table(summary_rows)

    st.subheader("Drill into a cluster")
    cluster_options = sorted(cluster_labels.keys(), key=int)
    selected = st.selectbox(
        "Cluster",
        cluster_options,
        format_func=lambda cid: f"{cid}: {cluster_labels[cid]['label']} ({cluster_labels[cid]['member_count']} players)",
    )

    info = cluster_labels[selected]
    st.markdown("**Top features (mean z-score within cluster):**")
    st.table(info["top_features"])

    members = outfield[outfield["cluster"] == int(selected)]
    display_cols = ["Player", "Squad", "Pos", "Gls/90", "Ast/90", "Sh/90", "TklW/90", "Int/90"]
    st.dataframe(
        members[display_cols].sort_values("Gls/90", ascending=False).style.format(
            {c: "{:.2f}" for c in ["Gls/90", "Ast/90", "Sh/90", "TklW/90", "Int/90"]}
        ),
        use_container_width=True,
    )
