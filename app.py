
from pathlib import Path
import re

import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CancerStochastique",
    page_icon="🧬",
    layout="wide",
)


ACTIONS = {
    0: "Observation",
    1: "Surgery",
    2: "Chemotherapy",
    3: "Radiation",
    4: "Immunotherapy",
    5: "Combination therapy",
    6: "Palliative",
}

COMBINATION_COMPONENTS = {
    5: {
        "name": "Combination therapy",
        "components": [
            "Chemotherapy",
            "Radiation therapy",
        ],

    }
}

STAGES = ["I", "II", "III", "IV"]

# Locate the project root whether app.py is in the root
# or in a child folder such as CancerResearchApp.
APP_DIR = Path(__file__).resolve().parent
ROOT_CANDIDATES = [APP_DIR, APP_DIR.parent]

def find_project_root():
    for folder in ROOT_CANDIDATES:
        if (
            (folder / "results_training" / "netlogo").exists()
            or (folder / "results_training").exists()
        ):
            return folder
    return APP_DIR

ROOT = find_project_root()
RESULTS_DIRS = [
    ROOT / "results_training" / "netlogo",
    ROOT / "results_training",
    ROOT / "output",
    ROOT,
]

def locate_file(filename):
    for folder in RESULTS_DIRS:
        path = folder / filename
        if path.exists():
            return path

    # Search the project, avoiding virtual environments.
    for folder in [ROOT / "results_training", ROOT / "output", ROOT]:
        if not folder.exists():
            continue
        try:
            for path in folder.rglob(filename):
                if not any(
                    part in {".venv", "venv", "site-packages", ".git"}
                    for part in path.parts
                ):
                    return path
        except OSError:
            pass
    return None


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(df, names):
    lookup = {normalize(col): col for col in df.columns}

    for name in names:
        key = normalize(name)
        if key in lookup:
            return lookup[key]

    for col in df.columns:
        key = normalize(col)
        for name in names:
            if normalize(name) in key:
                return col

    return None


def load_policy(filename, label):
    path = locate_file(filename)

    if path is None:
        return None, f"File not found: {filename}"

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return None, f"Could not read {path}: {exc}"

    if df.empty:
        return None, f"{filename} is empty."

    state_col = find_column(
        df, ["state_id", "state_index", "state", "state_from"]
    )
    action_col = find_column(
        df,
        [
            "action_id",
            "selected_action",
            "policy_action",
            "best_action",
            "action_index",
            "action",
        ],
    )

    # Compatibility with the project's existing policy CSV layout.
    if state_col is None:
        state_col = df.columns[0]

    if action_col is None:
        if len(df.columns) >= 6:
            action_col = df.columns[5]
        elif len(df.columns) >= 2:
            action_col = df.columns[-1]
        else:
            return None, f"No action column found in {filename}."

    # Accept numeric action IDs or action names.
    action_lookup = {
        normalize(name): action_id for action_id, name in ACTIONS.items()
    }

    state_values = pd.to_numeric(df[state_col], errors="coerce")
    action_values = df[action_col].map(
        lambda value: (
            action_lookup.get(normalize(value))
            if not pd.isna(value) and not str(value).strip().replace(".", "", 1).isdigit()
            else value
        )
    )
    action_values = pd.to_numeric(action_values, errors="coerce")

    policy = pd.DataFrame({
        "state_id": state_values,
        "action_id": action_values,
    }).dropna()

    policy["state_id"] = policy["state_id"].astype(int)
    policy["action_id"] = policy["action_id"].astype(int)

    policy = policy[
        policy["state_id"].between(0, 31)
        & policy["action_id"].between(0, 6)
    ].drop_duplicates("state_id", keep="last")

    policy["action"] = policy["action_id"].map(ACTIONS)
    policy["model"] = label

    if policy.empty:
        return None, f"No valid state/action rows found in {filename}."

    return policy, str(path)


@st.cache_data(show_spinner=False)
def load_policy_cached(path_string, filename, label):
    # Cache policy loading by file path and filename.
    df = pd.read_csv(path_string)
    return df


def get_action(policy, state_id):
    if policy is None:
        return None

    rows = policy.loc[policy["state_id"] == int(state_id)]
    if rows.empty:
        return None

    return int(rows.iloc[0]["action_id"])


# ============================================================
# PROJECT STATE MAPPING
# ============================================================

def map_patient_to_state(
    stage,
    status,
    metastatic_site,
    previous_treatments,
):
    """
    Uses the project's existing encoding:
        state_id = D*16 + P*8 + U*2 + B

    D: disease category
    P: progression category
    U: treatment utility category
    B: biological response category
    """

    stage_index = STAGES.index(stage)
    has_metastasis = metastatic_site not in ["None", "Unknown"]
    deceased = status == "Deceased"

    # Existing project's D mapping.
    D = int(stage_index >= 2 or has_metastasis)

    # Existing project's P mapping.
    P = int(
        deceased
        or (
            stage == "IV"
            and metastatic_site in ["Brain", "Multiple"]
        )
    )

    # Existing project's U mapping.
    if previous_treatments == 0:
        U = 0
    elif deceased or stage == "IV":
        U = 1
    elif stage == "III":
        U = 2
    else:
        U = 3

    # Existing project's B mapping.
    B = int(
        metastatic_site == "Multiple"
        or (
            previous_treatments >= 3
            and has_metastasis
        )
    )

    state_id = D * 16 + P * 8 + U * 2 + B

    return state_id, {"D": D, "P": P, "U": U, "B": B}


# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
.stApp { background: #f4f7f4; }
[data-testid="stSidebar"] {
    background: #123d32;
}
[data-testid="stSidebar"] * {
    color: #f4f7f4;
}
h1, h2, h3 { color: #123d32; }
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce7df;
    border-radius: 12px;
    padding: 14px;
}
.stButton > button, [data-testid="stFormSubmitButton"] button {
    background: #08785d;
    color: white;
    border: 0;
    border-radius: 9px;
    font-weight: 650;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD THE EXISTING TRAINED POLICIES
# ============================================================

q_policy, q_source = load_policy(
    "q_learning_policy.csv", "Q-Learning"
)
double_policy, double_source = load_policy(
    "double_q_learning_policy.csv", "Double Q-Learning"
)

# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.title("🧬 CancerStochastique")
st.sidebar.caption("Multi-Agent System · Q-Learning")
page = st.sidebar.radio(
    "Navigation",
    ["Patient Assessment", "Policy Comparison", "State Explorer"],
)

st.sidebar.divider()
st.sidebar.write("**State space:** 32")
st.sidebar.write("**Actions:** 7")
st.sidebar.write("**Training:** Existing policies")

with st.sidebar.expander("Loaded model files"):
    st.write("Q-Learning")
    st.code(q_source)
    st.write("Double Q-Learning")
    st.code(double_source)

# ============================================================
# PATIENT ASSESSMENT
# ============================================================

if page == "Patient Assessment":
    st.title("Patient Case Explorer")
    st.write(
        "Enter the case information below. The application will "
        "construct the internal state and retrieve the actions "
        "selected by your trained policies."
    )

    with st.form("patient_form"):
        st.subheader("Case information")

        col1, col2 = st.columns(2)

        with col1:
            case_name = st.text_input(
                "Case label",
                placeholder="Example: Test Case 01",
            )
            cancer_type = st.selectbox(
                "Cancer type",
                ["Lung", "Breast", "Colorectal", "Prostate"],
            )
            stage = st.selectbox(
                "Cancer stage",
                STAGES,
                format_func=lambda value: f"Stage {value}",
            )

        with col2:
            locations = {
                "Lung": [
                    "Right lung", "Left lung", "Both lungs",
                    "Not specified",
                ],
                "Breast": [
                    "Left breast", "Right breast", "Both breasts",
                    "Not specified",
                ],
                "Colorectal": [
                    "Colon", "Rectum", "Colon and rectum",
                    "Not specified",
                ],
                "Prostate": ["Prostate gland", "Not specified"],
            }
            primary_location = st.selectbox(
                "Primary cancer location",
                locations[cancer_type],
            )
            status = st.selectbox(
                "Current status",
                ["Alive", "Deceased", "Unknown"],
            )
            metastatic_site = st.selectbox(
                "Spread to another site",
                [
                    "None", "Bone", "Liver", "Lung", "Brain",
                    "Multiple", "Unknown",
                ],
                format_func=lambda value: {
                    "None": "No known spread",
                    "Bone": "Bone",
                    "Liver": "Liver",
                    "Lung": "Lung",
                    "Brain": "Brain",
                    "Multiple": "Multiple sites",
                    "Unknown": "Unknown / not recorded",
                }[value],
            )

        previous_treatments = st.selectbox(
            "Number of previous treatments",
            [0, 1, 2, 3, 4],
            format_func=lambda value: {
                0: "None",
                1: "1 treatment",
                2: "2 treatments",
                3: "3 treatments",
                4: "4 or more treatments",
            }[value],
        )

        model_choice = st.radio(
            "Model to test",
            [
                "Compare both models",
                "Q-Learning",
                "Double Q-Learning",
            ],
            horizontal=True,
        )

        submitted = st.form_submit_button(
            "Run assessment",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        mapped_site = (
            "None" if metastatic_site == "Unknown"
            else metastatic_site
        )

        state_id, encoded = map_patient_to_state(
            stage=stage,
            status=status,
            metastatic_site=mapped_site,
            previous_treatments=previous_treatments,
        )

        st.session_state["case_result"] = {
            "case_name": case_name.strip() or "Unnamed case",
            "cancer_type": cancer_type,
            "primary_location": primary_location,
            "stage": stage,
            "status": status,
            "metastatic_site": metastatic_site,
            "previous_treatments": previous_treatments,
            "model_choice": model_choice,
            "state_id": state_id,
            "encoded": encoded,
        }

    case = st.session_state.get("case_result")

    if case:
        st.divider()
        st.subheader(f"Results: {case['case_name']}")

        st.write(
            f"**Cancer:** {case['cancer_type']}  \n"
            f"**Primary location:** {case['primary_location']}  \n"
            f"**Stage:** {case['stage']}  \n"
            f"**Spread:** {case['metastatic_site']}  \n"
            f"**Previous treatments:** {case['previous_treatments']}"
        )

        # Internal state is generated by the application.
        st.markdown("### Automatically constructed state")
        st.metric("Encoded state", f"S{case['state_id']}")

        encoded = case["encoded"]
        state_cols = st.columns(4)
        for col, variable in zip(state_cols, ["D", "P", "U", "B"]):
            col.metric(variable, encoded[variable])

        st.caption(
            f"Encoding: 16×{encoded['D']} + 8×{encoded['P']} "
            f"+ 2×{encoded['U']} + {encoded['B']} "
            f"= {case['state_id']}"
        )

        st.markdown("### Trained policy output")

        models = []
        if case["model_choice"] == "Compare both models":
            models = [
                ("Q-Learning", q_policy),
                ("Double Q-Learning", double_policy),
            ]
        elif case["model_choice"] == "Q-Learning":
            models = [("Q-Learning", q_policy)]
        else:
            models = [("Double Q-Learning", double_policy)]

        result_cols = st.columns(len(models))
        outputs = {}

        for col, (model_name, policy) in zip(result_cols, models):
            with col:
                st.markdown(f"#### {model_name}")

                if policy is None:
                    st.error(
                        f"{model_name} policy did not load. "
                        "Check the file path shown in the sidebar."
                    )
                    continue

                action_id = get_action(policy, case["state_id"])

                if action_id is None:
                    st.warning(
                        "This state has no action in the loaded policy."
                    )
                    continue

                outputs[model_name] = action_id

                st.success(ACTIONS[action_id])

                if action_id == 5:
                    combination = COMBINATION_COMPONENTS[5]
                    st.markdown("**Combination components**")
                    for component in combination["components"]:
                        st.write(f"- {component}")

                st.metric("Selected action ID", action_id)

                with st.expander("Policy row"):
                    st.dataframe(
                        policy[policy["state_id"] == case["state_id"]],
                        hide_index=True,
                        use_container_width=True,
                    )

        if len(outputs) == 2:
            st.markdown("### Comparison")
            q_action = outputs["Q-Learning"]
            dq_action = outputs["Double Q-Learning"]

            if q_action == dq_action:
                st.success(
                    f"Both policies select **{ACTIONS[q_action]}**."
                )
            else:
                a, b = st.columns(2)
                a.metric("Q-Learning", ACTIONS[q_action])
                b.metric("Double Q-Learning", ACTIONS[dq_action])

        with st.expander("Case data"):
            st.json({
                key: value
                for key, value in case.items()
                if key != "encoded"
            })

# ============================================================
# POLICY COMPARISON
# ============================================================

elif page == "Policy Comparison":
    st.title("Q-Learning vs Double Q-Learning")

    if q_policy is None or double_policy is None:
        st.error(
            "Both policy files must be available to compare them. "
            "Check the loaded model paths in the sidebar."
        )
    else:
        comparison = pd.DataFrame({"state_id": range(32)})

        q_map = q_policy.set_index("state_id")["action_id"]
        dq_map = double_policy.set_index("state_id")["action_id"]

        comparison["q_action_id"] = comparison["state_id"].map(q_map)
        comparison["double_q_action_id"] = comparison["state_id"].map(dq_map)
        comparison["q_action"] = comparison["q_action_id"].map(ACTIONS)
        comparison["double_q_action"] = comparison["double_q_action_id"].map(ACTIONS)
        comparison["same_action"] = (
            comparison["q_action_id"] == comparison["double_q_action_id"]
        )

        valid = comparison.dropna(
            subset=["q_action_id", "double_q_action_id"]
        )
        matches = int(valid["same_action"].sum())
        compared = len(valid)

        a, b, c = st.columns(3)
        a.metric("States compared", f"{compared}/32")
        b.metric("Matching actions", matches)
        c.metric(
            "Agreement",
            f"{matches / compared * 100:.1f}%" if compared else "N/A",
        )

        st.subheader("Full policy comparison")
        st.dataframe(comparison, hide_index=True, use_container_width=True)

        st.subheader("Action distribution")
        distribution = pd.DataFrame({
            "Q-Learning": q_policy["action"].value_counts(),
            "Double Q-Learning": double_policy["action"].value_counts(),
        }).fillna(0).astype(int)
        st.bar_chart(distribution)

        st.download_button(
            "Download comparison CSV",
            comparison.to_csv(index=False).encode("utf-8"),
            file_name="policy_comparison.csv",
            mime="text/csv",
        )

# ============================================================
# STATE EXPLORER
# ============================================================

elif page == "State Explorer":
    st.title("State Explorer")
    st.write(
        "Inspect the actions stored in the trained policies for "
        "each encoded state."
    )

    selected_state = st.selectbox(
        "Select a state",
        list(range(32)),
        format_func=lambda value: f"State {value}",
    )

    D = selected_state // 16
    P = (selected_state % 16) // 8
    U = (selected_state % 8) // 2
    B = selected_state % 2

    a, b, c, d = st.columns(4)
    a.metric("D", D)
    b.metric("P", P)
    c.metric("U", U)
    d.metric("B", B)

    rows = []
    for name, policy in [
        ("Q-Learning", q_policy),
        ("Double Q-Learning", double_policy),
    ]:
        action_id = get_action(policy, selected_state)
        rows.append({
            "Model": name,
            "State": selected_state,
            "Action ID": action_id,
            "Action": ACTIONS.get(action_id, "No stored action"),
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
