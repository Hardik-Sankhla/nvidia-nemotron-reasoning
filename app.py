import os
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Nemotron Reasoning Dashboard", layout="wide")
st.title("Nemotron Reasoning Dashboard")

leaderboard_file = "reports/leaderboard.csv"

if os.path.exists(leaderboard_file):
    df = pd.read_csv(leaderboard_file)

    for col in ["experiment", "accuracy", "notes", "technique", "model", "lora_rank", "timestamp"]:
        if col not in df.columns:
            df[col] = "" if col != "accuracy" else 0.0

    # Sidebar filters
    st.sidebar.header("Filters")
    techniques = sorted([x for x in df["technique"].dropna().astype(str).unique() if x])
    models = sorted([x for x in df["model"].dropna().astype(str).unique() if x])
    ranks = sorted([int(x) for x in df["lora_rank"].dropna().astype(str).str.extract(r"(\d+)")[0].dropna().unique()])

    selected_techniques = st.sidebar.multiselect("Technique", techniques, default=techniques)
    selected_models = st.sidebar.multiselect("Model", models, default=models)
    selected_ranks = st.sidebar.multiselect("LoRA rank", ranks, default=ranks)

    filtered = df.copy()
    if selected_techniques:
        filtered = filtered[filtered["technique"].astype(str).isin(selected_techniques)]
    if selected_models:
        filtered = filtered[filtered["model"].astype(str).isin(selected_models)]
    if selected_ranks:
        filtered = filtered[filtered["lora_rank"].astype(str).str.extract(r"(\d+)")[0].fillna("0").astype(int).isin(selected_ranks)]

    filtered = filtered.sort_values(by="accuracy", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Experiments", len(filtered))
    c2.metric("Best Accuracy", f"{filtered['accuracy'].max():.4f}" if len(filtered) else "n/a")
    c3.metric("Avg Accuracy", f"{filtered['accuracy'].mean():.4f}" if len(filtered) else "n/a")

    st.subheader("Leaderboard")
    st.dataframe(filtered, use_container_width=True)

    if len(filtered):
        st.subheader("Accuracy by Experiment")
        chart_df = filtered[["experiment", "accuracy"]].set_index("experiment")
        st.line_chart(chart_df)

        if filtered["technique"].astype(str).str.len().sum() > 0:
            st.subheader("Technique Breakdown")
            grouped = filtered.groupby("technique", dropna=False)["accuracy"].mean().sort_values(ascending=False)
            st.bar_chart(grouped)
else:
    st.warning("No experiments yet. Run scripts/full_run.py first.")

st.subheader("Insights")
st.markdown(
    """
- Self-consistency usually improves stability and accuracy.
- Synthetic data helps only when quality is controlled by cleaning and filtering.
- Prompt templates should be benchmarked continuously using the leaderboard loop.
"""
)
