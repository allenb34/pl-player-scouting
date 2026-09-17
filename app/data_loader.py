"""Cached data/model loaders shared across all dashboard tabs."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODELING_DIR = Path(__file__).resolve().parent.parent / "modeling"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ARTIFACTS_DIR = MODELING_DIR / "artifacts"

if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))


@st.cache_data
def load_clustered_players() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "players_clustered.csv")


@st.cache_data
def load_goalkeepers() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "goalkeepers.csv")


@st.cache_data
def load_cluster_labels() -> dict:
    with open(ARTIFACTS_DIR / "cluster_labels.json") as f:
        return json.load(f)


@st.cache_data
def load_k_selection() -> dict:
    with open(ARTIFACTS_DIR / "k_selection.json") as f:
        return json.load(f)


@st.cache_data
def load_source_used() -> dict:
    path = Path(__file__).resolve().parent.parent / "data" / "raw" / "source_used.json"
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_similarity_models():
    from feature_engineering import build_feature_matrix

    outfield = load_clustered_players()
    models = {}
    for key in ["agnostic", "FW", "MF", "DF"]:
        models[key] = joblib.load(ARTIFACTS_DIR / f"nn_model_{key}.pkl")

    X_scaled, _, _ = build_feature_matrix(outfield)

    models["_agnostic_index"] = X_scaled.index.tolist()
    for group in ["FW", "MF", "DF"]:
        group_ids = outfield.loc[outfield["primary_pos"] == group, "stint_id"]
        models[f"_{group}_index"] = X_scaled.loc[X_scaled.index.isin(group_ids)].index.tolist()

    return models, X_scaled
