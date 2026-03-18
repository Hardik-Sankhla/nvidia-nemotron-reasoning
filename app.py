import os
import pandas as pd
import streamlit as st


st.title("Nemotron Reasoning Dashboard")

leaderboard_file = "reports/leaderboard.csv"

if os.path.exists(leaderboard_file):
    df = pd.read_csv(leaderboard_file)

    st.subheader("Leaderboard")
    st.dataframe(df)

    if "experiment" in df.columns and "accuracy" in df.columns:
        chart_df = df[["experiment", "accuracy"]].set_index("experiment")
        st.line_chart(chart_df)
else:
    st.warning("No experiments yet.")

st.subheader("Insights")
st.markdown(
    """
- Self-consistency improves stability.
- Synthetic data can boost performance.
- Prompt engineering quality is critical.
"""
)
