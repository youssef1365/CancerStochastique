
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="CancerMultiAgent | Research Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_training" / "netlogo"
DATASET_CANDIDATES = [
    ROOT / "data" / "cancer_treatment_dataset.csv",
    ROOT / "output" / "cancer_treatment_dataset.csv",
    ROOT / "data" / "processed" / "cancer_treatment_dataset.csv",
    ROOT / "cancer_treatment_dataset.csv",
]

Q_FILE = RESULTS / "q_learning_policy.csv"
DOUBLE_Q_FILE = RESULTS / "double_q_learning_policy.csv"
MATRIX_FILE = RESULTS / "transition_matrices.csv"

ACTIONS = {
    0: "Observation",
    1: "Surgery",
    2: "Chemotherapy",
    3: "Radiation",
    4: "Immunotherapy",
    5: "Combination",
    6: "Palliative",
}

TERMINAL_STATES = {6, 27}

# ---------------------------------------------------------
# STYLE
# ---------------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: #101820;
    color: #edf4f5;
}
[data-testid="stSidebar"] {
    background: #15232c;
}
[data-testid="stMetric"] {
    background: #1b2b36;
    border: 1px solid #304451;
    padding: 16px;
    border-radius: 12px;
}
h1, h2, h3 {
    color: #5eead4;
}
div[data-testid="stAlert"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# DATA LOADING AND COLUMN DETECTION
# ---------------------------------------------------------

def normalize_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(df, candidates, fallback=None):
    lookup = {normalize_name(c): c for c in df.columns}

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in lookup:
            return lookup[key]

    for col in df.columns:
        key = normalize_name(col)
        if any(normalize_name(candidate) in key for candidate in candidates):
            return col

    return fallback


def read_csv_safe(path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return None


def locate_dataset():
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path

    # Search common project locations, excluding virtual environments
    for folder in [ROOT / "data", ROOT / "output", ROOT]:
        if folder.exists():
            matches = list(folder.glob("**/cancer_treatment_dataset.csv"))
            matches = [
                p for p in matches
                if ".venv" not in p.parts and "site-packages" not in p.parts
            ]
            if matches:
                return matches[0]

    return None


@st.cache_data(show_spinner=False)
def load_csv(path_string):
    return pd.read_csv(path_string)


def load_table(path):
    if not path.exists():
        return None
    try:
        return load_csv(str(path))
    except Exception as exc:
        st.error(f"Error loading {path.name}: {exc}")
        return None


def numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def prepare_policy(df, label):
    if df is None or df.empty:
        return None

    state_col = find_column(
        df,
        ["state_id", "state", "state_index", "state_from", "stateid"],
        fallback=df.columns[0],
    )

    action_col = find_column(
        df,
        [
            "selected_action",
            "policy_action",
            "action_id",
            "action_index",
            "best_action",
            "action",
        ],
    )

    # The existing NetLogo loader reads policy columns 0 and 5.
    # Use that layout as a fallback if no named action column exists.
    if action_col is None:
        if len(df.columns) >= 6:
            action_col = df.columns[5]
        elif len(df.columns) >= 2:
            action_col = df.columns[-1]

    if action_col is None:
        return None

    result = pd.DataFrame({
        "state_id": numeric_series(df[state_col]),
        "action_id": numeric_series(df[action_col]),
    }).dropna()

    result["state_id"] = result["state_id"].astype(int)
    result["action_id"] = result["action_id"].astype(int)
    result = result[
        result["state_id"].between(0, 31)
        & result["action_id"].between(0, 6)
    ]
    result = result.drop_duplicates("state_id", keep="last")
    result["action_name"] = result["action_id"].map(ACTIONS)
    result["policy"] = label
    return result


def prepare_transitions(df):
    if df is None or df.empty:
        return None

    state_col = find_column(
        df,
        ["state_from", "current_state", "source_state", "state_id", "state"],
        fallback=df.columns[0],
    )
    action_col = find_column(
        df,
        ["action_id", "action", "selected_action"],
        fallback=df.columns[1] if len(df.columns) > 1 else None,
    )
    next_col = find_column(
        df,
        ["state_to", "next_state", "destination_state", "next_state_id"],
        fallback=df.columns[2] if len(df.columns) > 2 else None,
    )

    if action_col is None or next_col is None:
        return None

    result = pd.DataFrame({
        "state_from": numeric_series(df[state_col]),
        "action_id": numeric_series(df[action_col]),
        "state_to": numeric_series(df[next_col]),
    }).dropna()

    result[["state_from", "action_id", "state_to"]] = (
        result[["state_from", "action_id", "state_to"]].astype(int)
    )

    result = result[
        result["state_from"].between(0, 31)
        & result["state_to"].between(0, 31)
        & result["action_id"].between(0, 6)
    ]

    probability_col = find_column(
        df,
        ["probability", "transition_probability", "prob", "probability_value"],
    )
    count_col = find_column(
        df,
        ["count", "frequency", "observations", "transition_count", "n"],
    )

    if probability_col is not None:
        result["weight"] = numeric_series(
            df.loc[result.index, probability_col]
        ).fillna(0).clip(lower=0)
    elif count_col is not None:
        result["weight"] = numeric_series(
            df.loc[result.index, count_col]
        ).fillna(0).clip(lower=0)
    else:
        # If neither probabilities nor counts are present,
        # each record is treated as one observed transition.
        result["weight"] = 1.0

    # Normalize within each state/action pair. This supports both
    # empirical counts and probability tables.
    totals = result.groupby(["state_from", "action_id"])["weight"].transform("sum")
    result["probability"] = np.where(
        totals > 0, result["weight"] / totals, 0.0
    )

    return result.reset_index(drop=True)


def prepare_dataset(df):
    if df is None or df.empty:
        return None

    aliases = {
        "state_from": ["state_from", "current_state", "state_id"],
        "state_to": ["state_to", "next_state", "next_state_id"],
        "action_id": ["action", "action_id", "selected_action"],
        "reward": ["reward", "immediate_reward"],
        "patient_id": ["patient_id", "patient", "id_patient"],
        "timestep": ["timestep", "time_step", "step"],
        "is_terminal": ["is_terminal", "terminal", "done"],
    }

    result = df.copy()
    for standard, candidates in aliases.items():
        source = find_column(df, candidates)
        if source is not None:
            result[standard] = df[source]

    for col in ["state_from", "state_to", "action_id", "reward", "timestep"]:
        if col in result.columns:
            result[col] = numeric_series(result[col])

    return result


# ---------------------------------------------------------
# LOAD EXISTING FILES
# ---------------------------------------------------------

dataset_path = locate_dataset()
raw_dataset = load_table(dataset_path) if dataset_path else None
dataset = prepare_dataset(raw_dataset)

raw_q = load_table(Q_FILE)
raw_double_q = load_table(DOUBLE_Q_FILE)
raw_matrix = load_table(MATRIX_FILE)

q_policy = prepare_policy(raw_q, "Q-Learning")
double_policy = prepare_policy(raw_double_q, "Double Q-Learning")
transitions = prepare_transitions(raw_matrix)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("🧬 CancerMultiAgent")
st.sidebar.caption("Multi-agent research simulation")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Transition Matrices",
        "Policy Comparison",
        "State Explorer",
        "Simulation Lab",
        "Data & Diagnostics",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Model configuration")
st.sidebar.write("State space: **32 states**")
st.sidebar.write("Action space: **7 actions**")
st.sidebar.write("Terminal IDs: **6, 27**")
st.sidebar.caption("Experimental outputs only; not for clinical decisions.")

# ---------------------------------------------------------
# COMMON HELPERS
# ---------------------------------------------------------

def decode_state(state_id):
    state_id = int(state_id)
    return {
        "D": state_id // 16,
        "P": (state_id % 16) // 8,
        "U": (state_id % 8) // 2,
        "B": state_id % 2,
    }


def policy_action(policy, state_id):
    if policy is None:
        return None
    match = policy[policy["state_id"] == int(state_id)]
    if match.empty:
        return None
    return int(match.iloc[0]["action_id"])


def transitions_for(state_id, action_id):
    if transitions is None:
        return pd.DataFrame()
    return transitions[
        (transitions["state_from"] == int(state_id))
        & (transitions["action_id"] == int(action_id))
    ].sort_values("probability", ascending=False)


def show_missing_files():
    missing = []
    if raw_q is None:
        missing.append(str(Q_FILE.relative_to(ROOT)))
    if raw_double_q is None:
        missing.append(str(DOUBLE_Q_FILE.relative_to(ROOT)))
    if raw_matrix is None:
        missing.append(str(MATRIX_FILE.relative_to(ROOT)))
    if dataset is None:
        missing.append("data/cancer_treatment_dataset.csv (or another supported location)")

    if missing:
        st.warning(
            "Some files are missing. The corresponding features will be "
            "unavailable until the files are added at these paths:\n\n"
            + "\n".join(f"- `{name}`" for name in missing)
        )


def state_status(state_id):
    if state_id == 6:
        return "Terminal state (ID 6)"
    if state_id == 27:
        return "Terminal state (ID 27)"
    return "Non-terminal state"


# ---------------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------------

if page == "Overview":
    st.title("Research Overview")
    st.write(
        "Monitor the existing dataset, transition model, and learned policies."
    )

    show_missing_files()

    total_rows = len(dataset) if dataset is not None else 0

    if dataset is not None and "patient_id" in dataset.columns:
        total_patients = dataset["patient_id"].nunique()
    else:
        total_patients = "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset transitions", f"{total_rows:,}")
    col2.metric(
        "Unique patients",
        f"{total_patients:,}" if isinstance(total_patients, int) else total_patients,
    )
    col3.metric(
        "Q policy states",
        q_policy["state_id"].nunique() if q_policy is not None else "Missing",
    )
    col4.metric(
        "Double Q states",
        double_policy["state_id"].nunique() if double_policy is not None else "Missing",
    )

    if dataset is not None:
        st.subheader("Dataset preview")
        st.dataframe(dataset.head(12), use_container_width=True)

        if "state_from" in dataset.columns:
            counts = (
                dataset["state_from"]
                .dropna()
                .astype(int)
                .value_counts()
                .reindex(range(32), fill_value=0)
                .rename_axis("state_id")
                .reset_index(name="transitions")
            )
            fig = px.bar(
                counts,
                x="state_id",
                y="transitions",
                title="Observed transitions by starting state",
                color="transitions",
                color_continuous_scale="Teal",
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        if "reward" in dataset.columns:
            rewards = dataset["reward"].dropna()
            if not rewards.empty:
                a, b, c = st.columns(3)
                a.metric("Mean recorded reward", f"{rewards.mean():.3f}")
                b.metric("Median reward", f"{rewards.median():.3f}")
                c.metric("Reward observations", f"{len(rewards):,}")
    else:
        st.info("The dataset was not found. Policy and matrix pages may still work.")

# ---------------------------------------------------------
# PAGE: TRANSITION MATRICES
# ---------------------------------------------------------

elif page == "Transition Matrices":
    st.title("Transition Matrix Explorer")
    st.write(
        "Select a state and action to inspect the estimated next-state probabilities."
    )

    if transitions is None or transitions.empty:
        st.error(
            "Transition data could not be loaded. Check the file path and column "
            "names on the Data & Diagnostics page."
        )
    else:
        action_id = st.selectbox(
            "Action",
            options=list(ACTIONS),
            format_func=lambda x: f"{x} — {ACTIONS[x]}",
        )

        view = st.radio(
            "Matrix view",
            ["All 32 starting states", "One starting state"],
            horizontal=True,
        )

        if view == "One starting state":
            selected_state = st.selectbox("Starting state", range(32))
            subset = transitions[
                (transitions["state_from"] == selected_state)
                & (transitions["action_id"] == action_id)
            ]
            matrix = np.zeros((1, 32))
            if not subset.empty:
                for _, row in subset.iterrows():
                    matrix[0, int(row["state_to"])] = row["probability"]
            y_labels = [f"State {selected_state}"]
        else:
            subset = transitions[transitions["action_id"] == action_id]
            matrix = np.zeros((32, 32))
            for _, row in subset.iterrows():
                matrix[int(row["state_from"]), int(row["state_to"])] += row["probability"]
            y_labels = [str(i) for i in range(32)]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=list(range(32)),
                y=y_labels,
                colorscale="Teal",
                zmin=0,
                zmax=1,
                colorbar={"title": "Probability"},
                hovertemplate=(
                    "Starting state: %{y}<br>"
                    "Next state: %{x}<br>"
                    "Probability: %{z:.4f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            title=f"Transition probabilities — {ACTIONS[action_id]}",
            xaxis_title="Next state",
            yaxis_title="Starting state",
            height=700,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Transition details")
        details = subset.copy()
        if not details.empty:
            details["action_name"] = details["action_id"].map(ACTIONS)
            details = details.rename(columns={
                "state_from": "Starting state",
                "state_to": "Next state",
                "action_id": "Action ID",
                "action_name": "Action",
                "probability": "Probability",
            })
            display_cols = [
                c for c in [
                    "Starting state", "Action ID", "Action", "Next state", "Probability"
                ] if c in details.columns
            ]
            st.dataframe(
                details[display_cols].sort_values("Probability", ascending=False),
                use_container_width=True,
            )
        else:
            st.info("No transitions were recorded for this action in the selected view.")

# ---------------------------------------------------------
# PAGE: POLICY COMPARISON
# ---------------------------------------------------------

elif page == "Policy Comparison":
    st.title("Q-Learning vs Double Q-Learning")
    st.write("Compare the learned action selected for every encoded state.")

    if q_policy is None or double_policy is None:
        st.error("One or both policy CSV files could not be interpreted.")
    else:
        comparison = pd.DataFrame({"state_id": range(32)})
        q_map = q_policy.set_index("state_id")["action_id"]
        d_map = double_policy.set_index("state_id")["action_id"]

        comparison["q_action_id"] = comparison["state_id"].map(q_map)
        comparison["double_q_action_id"] = comparison["state_id"].map(d_map)
        comparison["q_action"] = comparison["q_action_id"].map(ACTIONS)
        comparison["double_q_action"] = comparison["double_q_action_id"].map(ACTIONS)
        comparison["same_action"] = (
            comparison["q_action_id"] == comparison["double_q_action_id"]
        )

        valid = comparison.dropna(
            subset=["q_action_id", "double_q_action_id"]
        )
        same_count = int(valid["same_action"].sum())
        total_compared = len(valid)

        a, b, c = st.columns(3)
        a.metric("States compared", f"{total_compared}/32")
        b.metric("Same selected action", same_count)
        c.metric(
            "Agreement",
            f"{100 * same_count / total_compared:.1f}%"
            if total_compared else "N/A",
        )

        st.subheader("Action agreement by state")
        plot_data = comparison.copy()
        plot_data["Q-Learning"] = plot_data["q_action_id"]
        plot_data["Double Q-Learning"] = plot_data["double_q_action_id"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_data["state_id"],
            y=plot_data["Q-Learning"],
            mode="markers+lines",
            name="Q-Learning",
        ))
        fig.add_trace(go.Scatter(
            x=plot_data["state_id"],
            y=plot_data["Double Q-Learning"],
            mode="markers+lines",
            name="Double Q-Learning",
        ))
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="State ID",
            yaxis_title="Selected action ID",
            yaxis=dict(tickmode="array", tickvals=list(ACTIONS), ticktext=list(ACTIONS.values())),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full comparison table")
        st.dataframe(
            comparison.style.map(
                lambda value: "background-color: #174e42; color: white"
                if value is True else "",
                subset=["same_action"],
            ),
            use_container_width=True,
        )

        st.subheader("Selected action distribution")
        distribution = pd.concat([
            q_policy[["action_name"]].assign(policy="Q-Learning"),
            double_policy[["action_name"]].assign(policy="Double Q-Learning"),
        ])
        counts = (
            distribution.groupby(["policy", "action_name"])
            .size().reset_index(name="states")
        )
        fig = px.bar(
            counts,
            x="action_name",
            y="states",
            color="policy",
            barmode="group",
            title="Number of states assigned to each action",
        )
        fig.update_layout(template="plotly_dark", xaxis_title="Action", yaxis_title="States")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE: STATE EXPLORER
# ---------------------------------------------------------

elif page == "State Explorer":
    st.title("State Explorer")
    st.write(
        "Explore the model's learned policy for a state in its 32-state space."
    )
    st.warning(
        "This is a research simulation. It does not determine a real patient's "
        "medical state or recommend clinical treatment."
    )

    input_mode = st.radio(
        "Choose how to identify a state",
        ["State ID", "Encoded variables D, P, U, B"],
        horizontal=True,
    )

    if input_mode == "State ID":
        state_id = st.number_input(
            "State ID", min_value=0, max_value=31, value=0, step=1
        )
        state_id = int(state_id)
    else:
        c1, c2, c3, c4 = st.columns(4)
        d = c1.selectbox("D", [0, 1])
        p = c2.selectbox("P", [0, 1])
        u = c3.selectbox("U", [0, 1])
        b = c4.selectbox("B", [0, 1])
        state_id = int(d * 16 + p * 8 + u * 2 + b)

    decoded = decode_state(state_id)
    st.subheader(f"State {state_id}")
    st.caption(state_status(state_id))

    cols = st.columns(4)
    for col, (name, value) in zip(cols, decoded.items()):
        col.metric(name, value)

    st.caption("Encoding: S = D × 16 + P × 8 + U × 2 + B")

    q_action = policy_action(q_policy, state_id)
    double_action = policy_action(double_policy, state_id)

    left, right = st.columns(2)
    with left:
        st.subheader("Q-Learning")
        if q_action is None:
            st.warning("No policy entry found for this state.")
        else:
            st.metric("Selected action", ACTIONS[q_action])
            st.caption(f"Action ID: {q_action}")

    with right:
        st.subheader("Double Q-Learning")
        if double_action is None:
            st.warning("No policy entry found for this state.")
        else:
            st.metric("Selected action", ACTIONS[double_action])
            st.caption(f"Action ID: {double_action}")

    selected_policy_name = st.radio(
        "Inspect transitions for",
        ["Q-Learning", "Double Q-Learning"],
        horizontal=True,
    )
    selected_action = q_action if selected_policy_name == "Q-Learning" else double_action

    if selected_action is not None:
        details = transitions_for(state_id, selected_action)
        st.subheader(f"Predicted transition distribution — {ACTIONS[selected_action]}")

        if details.empty:
            st.info("No matching transition records for this state/action pair.")
        else:
            chart = details.sort_values("probability", ascending=False)
            chart = chart.rename(columns={
                "state_to": "Next state",
                "probability": "Probability",
            })
            fig = px.bar(
                chart,
                x="Next state",
                y="Probability",
                hover_data=["state_from", "action_id"],
                title="Estimated probability of each next state",
                color="Probability",
                color_continuous_scale="Teal",
            )
            fig.update_layout(template="plotly_dark", yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                chart[["state_to", "probability"]].rename(columns={
                    "state_to": "Next state",
                    "probability": "Probability",
                }),
                use_container_width=True,
            )

# ---------------------------------------------------------
# PAGE: SIMULATION LAB
# ---------------------------------------------------------

elif page == "Simulation Lab":
    st.title("Simulation Lab")
    st.write(
        "Run synthetic trajectories using a learned policy and the transition model."
    )
    st.warning(
        "These are simulated trajectories, not patient forecasts. The results "
        "depend on the quality and assumptions of the supplied synthetic data."
    )

    if transitions is None or q_policy is None or double_policy is None:
        st.error("The transition matrix and both policies must be available.")
    else:
        selected_policy_name = st.selectbox(
            "Policy",
            ["Q-Learning", "Double Q-Learning"],
        )
        selected_policy = q_policy if selected_policy_name == "Q-Learning" else double_policy

        start_state = st.number_input(
            "Starting state ID",
            min_value=0,
            max_value=31,
            value=0,
            step=1,
        )
        episodes = st.slider("Number of simulated trajectories", 10, 500, 100, 10)
        max_steps = st.slider("Maximum steps per trajectory", 1, 100, 30)
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)

        if st.button("Run simulation", type="primary"):
            rng = np.random.default_rng(int(seed))
            policy_map = selected_policy.set_index("state_id")["action_id"].to_dict()
            results = []
            trajectory_rows = []

            progress = st.progress(0)

            for episode in range(int(episodes)):
                current_state = int(start_state)
                total_reward = 0.0
                steps = 0
                reached_terminal = current_state in TERMINAL_STATES

                trajectory_rows.append({
                    "episode": episode + 1,
                    "step": 0,
                    "state_id": current_state,
                })

                while steps < int(max_steps) and not reached_terminal:
                    action = policy_map.get(current_state)
                    if action is None:
                        break

                    options = transitions_for(current_state, int(action))
                    if options.empty or options["probability"].sum() <= 0:
                        break

                    probs = options["probability"].to_numpy(dtype=float)
                    probs = probs / probs.sum()
                    chosen_index = int(rng.choice(len(options), p=probs))
                    chosen = options.iloc[chosen_index]
                    next_state = int(chosen["state_to"])

                    # Rewards are optional in the transition CSV.
                    # If the source dataset contains rewards, use its mean
                    # reward for the corresponding state/action where possible.
                    step_reward = 0.0
                    if dataset is not None and "reward" in dataset.columns:
                        reward_df = dataset
                        if "state_from" in reward_df.columns and "action_id" in reward_df.columns:
                            matched = reward_df[
                                (reward_df["state_from"] == current_state)
                                & (reward_df["action_id"] == int(action))
                            ]
                            if not matched.empty:
                                step_reward = float(matched["reward"].mean())

                    total_reward += step_reward
                    steps += 1
                    current_state = next_state
                    reached_terminal = current_state in TERMINAL_STATES

                    trajectory_rows.append({
                        "episode": episode + 1,
                        "step": steps,
                        "state_id": current_state,
                    })

                results.append({
                    "episode": episode + 1,
                    "steps": steps,
                    "final_state": current_state,
                    "terminal_reached": reached_terminal,
                    "total_reward": total_reward,
                })
                progress.progress((episode + 1) / int(episodes))

            result_df = pd.DataFrame(results)
            trajectory_df = pd.DataFrame(trajectory_rows)

            st.session_state["simulation_results"] = result_df
            st.session_state["simulation_trajectories"] = trajectory_df

        if "simulation_results" in st.session_state:
            result_df = st.session_state["simulation_results"]
            trajectory_df = st.session_state["simulation_trajectories"]

            a, b, c, d = st.columns(4)
            a.metric("Trajectories", len(result_df))
            b.metric("Mean steps", f"{result_df['steps'].mean():.2f}")
            c.metric(
                "Reached terminal state",
                f"{100 * result_df['terminal_reached'].mean():.1f}%",
            )
            d.metric("Mean accumulated reward", f"{result_df['total_reward'].mean():.3f}")

            fig = px.histogram(
                result_df,
                x="final_state",
                nbins=32,
                title="Distribution of final simulated states",
                color_discrete_sequence=["#14b8a6"],
            )
            fig.update_layout(template="plotly_dark", xaxis_title="Final state", yaxis_title="Trajectories")
            st.plotly_chart(fig, use_container_width=True)

            state_counts = trajectory_df.groupby("step")["state_id"].nunique().reset_index(
                name="distinct_states"
            )
            fig = px.line(
                state_counts,
                x="step",
                y="distinct_states",
                markers=True,
                title="Number of distinct states visited at each step",
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Trajectory results")
            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                "Download simulation results (CSV)",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="simulation_results.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------
# PAGE: DATA & DIAGNOSTICS
# ---------------------------------------------------------

elif page == "Data & Diagnostics":
    st.title("Data & Diagnostics")
    st.write("Check which local project files the dashboard has loaded.")

    files = [
        ("Dataset", dataset_path, dataset),
        ("Q-Learning policy", Q_FILE, q_policy),
        ("Double Q-Learning policy", DOUBLE_Q_FILE, double_policy),
        ("Transition matrix", MATRIX_FILE, transitions),
    ]

    for name, path, df in files:
        with st.expander(name, expanded=(df is None)):
            st.write(f"Expected path: `{path if path else 'Not found'}`")
            if df is None:
                st.error("Not loaded")
            else:
                st.success(f"Loaded successfully: {len(df):,} rows")
                st.write("Columns:", list(df.columns))
                st.dataframe(df.head(5), use_container_width=True)

    st.subheader("Policy coverage")
    for name, policy in [
        ("Q-Learning", q_policy),
        ("Double Q-Learning", double_policy),
    ]:
        if policy is None:
            st.write(f"{name}: unavailable")
        else:
            missing_states = sorted(set(range(32)) - set(policy["state_id"]))
            st.write(
                f"**{name}:** {policy['state_id'].nunique()}/32 states. "
                f"Missing states: {missing_states or 'none'}"
            )

    if transitions is not None:
        pairs = transitions.groupby(["state_from", "action_id"]).size()
        st.write(f"Available state/action pairs: {len(pairs)}")
        st.write(f"Transition records: {len(transitions):,}")
        st.write(
            "State/action pairs with no recorded transitions are not simulated; "
            "the app does not invent fallback transitions."
        )
