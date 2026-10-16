
import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="CancerMultiAgent | Research Dashboard",
    page_icon="🧬",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background-color: #101820;
    color: #F1F5F9;
}
[data-testid="stMetric"] {
    background: #1B2B36;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #304451;
}
h1, h2, h3 {
    color: #5EEAD4;
}
</style>
""", unsafe_allow_html=True)

st.title("CancerMultiAgent")
st.caption("Multi-Agent Simulation · Markov Models · Reinforcement Learning")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Transition Matrices",
        "Policy Comparison",
        "State Explorer",
    ],
)

if page == "Overview":
    st.header("Research Overview")
    a, b, c = st.columns(3)
    a.metric("State space", "32 states")
    b.metric("Available actions", "7")
    c.metric("Policies", "2")

    st.info(
        "Dashboard initialized. The next step is to connect "
        "your trained policies and transition matrices."
    )

elif page == "Transition Matrices":
    st.header("Transition Matrix Explorer")
    st.write(
        "Explore transition probabilities for each simulated action."
    )
    st.info("Your transition_matrices.csv will be connected here.")

elif page == "Policy Comparison":
    st.header("Q-Learning vs Double Q-Learning")
    st.info(
        "We'll load your existing policy CSVs and compare "
        "their selected actions and simulation results."
    )

elif page == "State Explorer":
    st.header("Simulated State Explorer")
    st.warning(
        "This is a research interface, not a clinical treatment tool."
    )

    state_id = st.number_input(
        "State ID",
        min_value=0,
        max_value=31,
        value=0,
        step=1,
    )

    d = state_id // 16
    p = (state_id % 16) // 8
    u = (state_id % 8) // 2
    b = state_id % 2

    st.subheader("Encoded state")
    x, y, z, w = st.columns(4)
    x.metric("D", d)
    y.metric("P", p)
    z.metric("U", u)
    w.metric("B", b)

    st.caption(
        "State encoding: S = D × 16 + P × 8 + U × 2 + B"
    )
    st.info(
        "Policy output will appear here after the existing "
        "Q-Learning and Double Q-Learning CSV files are connected."
    )
