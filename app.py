
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    D_VALUES, P_VALUES, U_VALUES, B_VALUES,
    D_LABELS, P_LABELS, U_LABELS, B_LABELS,
    CANCER_TYPES, STAGES, OUTCOMES, METASTATIC_SITES,
    REWARD_IMPROVEMENT, REWARD_STABLE, REWARD_DETERIORATION,
    REWARD_DEATH, REWARD_RECOVERY, REWARD_TREATMENT_COST,
    GAMMA, N_ACTIONS, N_STATES,
)
from markov_states import (
    encode_state, decode_state, is_terminal, compute_reward,
    get_state_quality, state_to_label, build_state_space
)

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
    5: "Combination",
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


@st.cache_data(show_spinner=False)
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


@st.cache_resource(show_spinner=False)
def load_multi_agent_system(algorithm="double_q_learning"):
    try:
        from agents.multi_agent import MultiAgentSystem
        model_folder = locate_file("valid_actions.npy")
        model_dir = model_folder.parent if model_folder else None
        return MultiAgentSystem.from_models(model_dir=model_dir, algorithm=algorithm)
    except Exception:
        return None


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
    if previous_treatments == 0 or deceased:
        U = 0
    elif stage == "IV":
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
    ["Patient Assessment", "Policy Comparison", "State Explorer", "Model Index & Theory"],
)

st.sidebar.divider()
st.sidebar.write("**State space:** 32")
st.sidebar.write("**Actions:** 7")
st.sidebar.write("**Terminal states:** S06 (Recovery), S25 (Death)")
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

        # Terminal state clinical notifications
        if case["state_id"] == 25:
            st.error(
                "⚠️ **Absorbing Terminal State S25 (Terminal Death):** "
                "Patient condition represents maximum disease severity, poor prognosis, and biological resistance. "
                "The disease trajectory has reached terminus; active disease-modifying therapies cannot be administered."
            )
        elif case["state_id"] == 6:
            st.info(
                "🌟 **Absorbing Terminal State S06 (Full Curative Remission):** "
                "Patient has achieved complete curative recovery with high utility and responsive tumor biology."
            )

        # Multi-Agent Tumor Board Consensus
        st.markdown("### 🤖 Multi-Agent Tumor Board Consensus")
        algo_choice = "q_learning" if case["model_choice"] == "Q-Learning" else "double_q_learning"
        mas = load_multi_agent_system(algorithm=algo_choice)

        if mas is not None:
            mas_decision = mas.decide({
                "cancer_type": case["cancer_type"],
                "stage": case["stage"],
                "outcome": case["status"],
                "metastatic_site": case["metastatic_site"],
                "previous_treatments": case["previous_treatments"],
                "state_tuple": (encoded["D"], encoded["P"], encoded["U"], encoded["B"]),
                "state_index": case["state_id"],
            })

            if mas_decision.get("vetoed"):
                st.warning(
                    f"🚨 **Clinical Specialist Guardrail Override:** {mas_decision.get('veto_reason')}\n\n"
                    f"The reinforcement learning agent proposed **{mas_decision['proposed_action_name']}**, "
                    f"but the Tumor Board clinical specialist overruled it in favor of **{mas_decision['final_action_name']}**."
                )
            else:
                st.success(
                    f"✅ **Tumor Board Consensus ({mas_decision['consensus']:.0%}):** "
                    f"The clinical Tumor Board confirmed **{mas_decision['final_action_name']}**."
                )

            assessments = mas_decision.get("assessments", {})
            votes = mas_decision.get("votes", {})
            b_c1, b_c2, b_c3, b_c4 = st.columns(4)

            with b_c1:
                st.markdown("**Diagnostic Agent**")
                diag_eval = assessments.get("diagnostic", {})
                st.metric("Voted Action", ACTIONS.get(votes.get("DiagnosticAgent", 0), "None"))
                st.caption(diag_eval.get("assessment", ""))

            with b_c2:
                st.markdown("**Prognostic Agent**")
                prog_eval = assessments.get("prognostic", {})
                st.metric("Voted Action", ACTIONS.get(votes.get("PrognosticAgent", 0), "None"))
                st.caption(prog_eval.get("assessment", ""))

            with b_c3:
                st.markdown("**Biological Agent**")
                bio_eval = assessments.get("biological", {})
                st.metric("Voted Action", ACTIONS.get(votes.get("BiologicalAgent", 0), "None"))
                st.caption(bio_eval.get("assessment", ""))

            with b_c4:
                st.markdown("**Treatment Agent (RL)**")
                treat_eval = assessments.get("TreatmentAgent", {})
                st.metric("Proposed Action", mas_decision["proposed_action_name"])
                st.caption(f"Evaluated via {mas_decision.get('agent_type', 'RL')}")

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

    if selected_state == 6:
        st.info("🌟 **Terminal State S06 (Recovery)**: Low severity, Good prognosis, High utility, Responsive.")
    elif selected_state == 25:
        st.error("⚠️ **Terminal State S25 (Death)**: High severity, Poor prognosis, No utility, Resistant.")

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


# ============================================================
# MODEL INDEX & THEORY: VARIABLES, STATES, MARKOV & REWARDS
# ============================================================

elif page == "Model Index & Theory":
    st.title("📚 Model Index: Variables, States, Markov Dynamics & Rewards")
    st.markdown(
        "A rigorous, interactive reference documenting the clinical observation space $\\mathcal{X}(t)$, "
        "the 32-state discrete Markov model $\\mathcal{S}$, the action space $\\mathcal{A}$, the memoryless "
        "Markov property in oncology, and the differential health quality reward function $R(s, a, s')$."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Clinical Variables & Actions",
        "🗺️ 32 States & Radix Encoding",
        "🔄 Markov Process & Transitions",
        "🎯 Quality Scoring & Rewards",
    ])

    # ---------------------------------------------------------
    # TAB 1: CLINICAL VARIABLES & ACTIONS
    # ---------------------------------------------------------
    with tab1:
        st.subheader("1. Patient Observation Space: $\\mathcal{X}_i(t)$")
        st.write(
            "Every patient trajectory is tracked longitudinally across clinical visits $t = 0, 1, 2, \\dots, T$ "
            "using five primary clinical features:"
        )

        obs_data = [
            {
                "Symbol": "C_i",
                "Clinical Variable": "Primary Cancer Type",
                "Allowed Domain": "Lung, Breast, Colorectal, Prostate",
                "Role in Modeling": "Determines organ-specific progression hazard rates and metastatic site affinity.",
            },
            {
                "Symbol": "G_i",
                "Clinical Variable": "Cancer Stage",
                "Allowed Domain": "Stage I, II, III, IV",
                "Role in Modeling": "Primary determinant of baseline mortality risk and severity index D.",
            },
            {
                "Symbol": "O_i",
                "Clinical Variable": "Clinical Outcome Status",
                "Allowed Domain": "Alive, Deceased",
                "Role in Modeling": "Drives terminal death detection and prognosis indicator P.",
            },
            {
                "Symbol": "M_i",
                "Clinical Variable": "Metastatic Spread / Site",
                "Allowed Domain": "None, Bone, Liver, Lung, Brain, Multiple",
                "Role in Modeling": "Drives disease severity D, high-risk prognosis P, and biological resistance B.",
            },
            {
                "Symbol": "T_i(t)",
                "Clinical Variable": "Cumulative Treatments",
                "Allowed Domain": "Sequence of prior actions (0 to 4+)",
                "Role in Modeling": "Tracks prior therapy burden, drug exhaustion, and resistance emergence B.",
            },
        ]
        st.dataframe(pd.DataFrame(obs_data), hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("2. The 4 Abstract Markov State Dimensions: $S_i = \\{D, P, U, B\\}$")
        st.write(
            "To make reinforcement learning computationally tractable, continuous and multi-categorical observations "
            "are projected via deterministic mapping functions $f_d, f_p, f_u, f_b$ into four ordinal/binary dimensions:"
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(r"""
            **$D \in \{0, 1\}$ : Disease Severity ($f_d$)**
            * **0 (Low):** Early-stage disease (Stage I or II) with **no distant metastases**.
            * **1 (High):** Advanced locoregional disease (Stage $\ge$ III) **or presence of distant metastasis**.
            * *Clinical Rule:*
              $$D = \mathbf{1}_{\{\text{Stage } \ge \text{III} \;\lor\; \text{Metastatic Site} \ne \text{'None'}\}}$$
            """)

            st.markdown("""
            **$U \\in \\{0, 1, 2, 3\\}$ : Treatment Utility ($f_u$)**
            * **0 (None / Failure):** Patient has received no treatment yet, or treatment failed (patient deceased).
            * **1 (Low):** End-stage disease (Stage IV) alive under salvage therapy.
            * **2 (Medium):** Mid-stage disease (Stage III) alive with partial response.
            * **3 (High):** Early-stage disease (Stage I or II) alive with curative potential.
            """)

        with c2:
            st.markdown("""
            **$P \\in \\{0, 1\\}$ : Survival Prognosis ($f_p$)**
            * **0 (Good):** Expected intermediate/long-term progression-free survival.
            * **1 (Poor):** High short-term mortality risk.
            * *Clinical Rule:*
              $$P = \\mathbf{1}_{\\{\\text{Status} = \\text{'Deceased'} \\;\\lor\\; (\\text{Stage IV} \\;\\land\\; \\text{Met} \\in \\{\\text{'Brain'}, \\text{'Multiple'}\\})\\}}$$
            """)

            st.markdown("""
            **$B \\in \\{0, 1\\}$ : Biological Response / Resistance ($f_b$)**
            * **0 (Responsive):** Tumor cells remain sensitive to frontline standard therapies.
            * **1 (Resistant):** Acquired tumor resistance or polyclonal dissemination.
            * *Clinical Rule:*
              $$B = \\mathbf{1}_{\\{\\text{Met} = \\text{'Multiple'} \\;\\lor\\; (|T| \\ge 3 \\;\\land\\; \\text{Met} \\ne \\text{'None'})\\}}$$
            """)

        st.divider()
        st.subheader("3. Action Space: $\\mathcal{A}$ ($|\\mathcal{A}| = 7$ Clinical Actions)")
        act_rows = [
            {"Action ID": 0, "Action Name": "Observation", "Cost Penalty": "0.0", "Clinical Rationale": "Active surveillance / watchful waiting for indolent early tumors.", "Multi-Agent Guardrail": "Vetoed by DiagnosticAgent if D=1."},
            {"Action ID": 1, "Action Name": "Surgery", "Cost Penalty": "-1.0", "Clinical Rationale": "Complete surgical resection with curative intent.", "Multi-Agent Guardrail": "Recommended for localized resectable tumors."},
            {"Action ID": 2, "Action Name": "Chemotherapy", "Cost Penalty": "-1.0", "Clinical Rationale": "Systemic cytotoxic agents targeting rapidly dividing malignant cells.", "Multi-Agent Guardrail": "Redirected to Immunotherapy if B=1 (resistant)."},
            {"Action ID": 3, "Action Name": "Radiation", "Cost Penalty": "-1.0", "Clinical Rationale": "Targeted ionizing radiation for locoregional control.", "Multi-Agent Guardrail": "Commonly deployed in localized breast/lung cohorts."},
            {"Action ID": 4, "Action Name": "Immunotherapy", "Cost Penalty": "-1.0", "Clinical Rationale": "Immune checkpoint inhibitors restoring T-cell anti-tumor response.", "Multi-Agent Guardrail": "Primary fallback when tumor is chemo-resistant (B=1)."},
            {"Action ID": 5, "Action Name": "Combination", "Cost Penalty": "-1.0", "Clinical Rationale": "Concurrent chemo-radiotherapy or chemo-immunotherapy regimens.", "Multi-Agent Guardrail": "Vetoed if D=0 (overtreatment prevention)."},
            {"Action ID": 6, "Action Name": "Palliative", "Cost Penalty": "-1.0", "Clinical Rationale": "Symptom alleviation, pain control, and end-of-life care.", "Multi-Agent Guardrail": "Mandated by PrognosticAgent if D=1, P=1, B=1."},
        ]
        st.dataframe(pd.DataFrame(act_rows), hide_index=True, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: 32 STATES & RADIX ENCODING
    # ---------------------------------------------------------
    with tab2:
        st.subheader("1. Mixed-Radix Bit-Shift State Encoding Formula")
        st.markdown(
            "The state space is a discrete Cartesian product of dimensions "
            "$|D| \\times |P| \\times |U| \\times |B| = 2 \\times 2 \\times 4 \\times 2 = 32$ states. "
            "Any 4-tuple $(D, P, U, B)$ is mapped bijectively to an integer index $S \\in [0, 31]$ using:"
        )

        st.latex(r"S = 16 \times D + 8 \times P + 2 \times U + 1 \times B")

        st.caption(
            "Multipliers are derived from positional radix sizing: "
            "$16 = 2 \\times 4 \\times 2$, $8 = 4 \\times 2$, $2 = 2$, $1 = 1$."
        )

        st.divider()
        st.subheader("2. Interactive State Encoder & Decoder")
        col_enc1, col_enc2 = st.columns(2)
        with col_enc1:
            test_d = st.selectbox("Disease Severity (D)", [0, 1], format_func=lambda x: f"{x} - {D_LABELS[x]}")
            test_p = st.selectbox("Survival Prognosis (P)", [0, 1], format_func=lambda x: f"{x} - {P_LABELS[x]}")
        with col_enc2:
            test_u = st.selectbox("Treatment Utility (U)", [0, 1, 2, 3], format_func=lambda x: f"{x} - {U_LABELS[x]}")
            test_b = st.selectbox("Biological Response (B)", [0, 1], format_func=lambda x: f"{x} - {B_LABELS[x]}")

        calc_index = test_d * 16 + test_p * 8 + test_u * 2 + test_b
        calc_quality = get_state_quality(calc_index)
        calc_terminal = is_terminal(calc_index)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calculated State ID", f"S{calc_index:02d}")
        m2.metric("Encoding Math", f"16({test_d}) + 8({test_p}) + 2({test_u}) + 1({test_b})")
        m3.metric("Quality Score Q(s)", f"{calc_quality:.1f}")
        m4.metric("Terminal Status", "Terminal" if calc_terminal else "Non-terminal")

        st.divider()
        st.subheader("3. Absorbing Terminal States")
        term_col1, term_col2 = st.columns(2)
        with term_col1:
            st.success("""
            🌟 **State 06: Full Curative Recovery**
            * **Tuple:** $(D=0, P=0, U=3, B=0)$
            * **Condition:** Low severity, good prognosis, high utility, responsive.
            * **Terminal Reward:** $+30.0$ bonus.
            * **Quality Score:** $Q(S06) = +3.0$ (highest possible health level).
            * **Absorption:** 0 outgoing transitions; patient has completed curative therapy.
            """)
        with term_col2:
            st.error("""
            ⚠️ **State 25: Terminal Disease / Death**
            * **Tuple:** $(D=1, P=1, U=0, B=1)$
            * **Condition:** High severity, poor prognosis, no utility, resistant.
            * **Terminal Penalty:** $-50.0$ penalty.
            * **Quality Score:** $Q(S25) = -6.0$ (lowest possible health level).
            * **Absorption:** 0 outgoing transitions; trajectory has permanently ended.
            """)

        st.divider()
        st.subheader("4. Complete Master Reference Table: All 32 States")
        filter_opt = st.radio(
            "Filter State Space View",
            ["All 32 States", "Terminal States Only", "High Severity Only (D=1)", "Treatment Resistant Only (B=1)"],
            horizontal=True
        )

        all_states_data = []
        for s in range(32):
            d, p, u, b = decode_state(s)
            is_term = is_terminal(s)
            q = get_state_quality(s)
            all_states_data.append({
                "State": f"S{s:02d}",
                "State ID": s,
                "Tuple (D,P,U,B)": f"({d}, {p}, {u}, {b})",
                "Severity (D)": D_LABELS[d],
                "Prognosis (P)": P_LABELS[p],
                "Utility (U)": U_LABELS[u],
                "Response (B)": B_LABELS[b],
                "Health Score Q(s)": q,
                "Terminal Status": "Recovery (S06)" if s == 6 else ("Death (S25)" if s == 25 else "Active"),
            })

        df_all_states = pd.DataFrame(all_states_data)
        if filter_opt == "Terminal States Only":
            df_all_states = df_all_states[df_all_states["Terminal Status"] != "Active"]
        elif filter_opt == "High Severity Only (D=1)":
            df_all_states = df_all_states[df_all_states["Severity (D)"] == "High"]
        elif filter_opt == "Treatment Resistant Only (B=1)":
            df_all_states = df_all_states[df_all_states["Response (B)"] == "Resistant"]

        st.dataframe(df_all_states, hide_index=True, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: MARKOV PROCESS & TRANSITIONS
    # ---------------------------------------------------------
    with tab3:
        st.subheader("1. The Markov Property in Cancer Disease Progression")
        st.markdown(
            "A stochastic process possesses the **Markov Property** if the conditional probability "
            "distribution of future states depends solely upon the present state and action, and not on the past sequence:"
        )

        st.latex(r"P(S_{t+1} = s' \mid S_t = s, A_t = a, S_{t-1}, A_{t-1}, \dots, S_0, A_0) = P(S_{t+1} = s' \mid S_t = s, A_t = a)")

        st.info("""
        **Why does the Markov property hold in CancerStochastique?**
        Although biological cancer progression is historically conditioned, our state space design captures the essential history:
        * Prior cumulative therapy lines and drug exhaustion are internalized inside **$U(t)$**.
        * Accumulated clonal selection and mutational escape are internalized inside **$B$**.
        Therefore, conditioning on the current state $S_t = (D, P, U, B)$ and treatment action $A_t = a$ renders future progression $S_{t+1}$ conditionally independent of previous time steps.
        """)

        st.divider()
        st.subheader("2. Transition Probability Tensor: $\\mathcal{P}(s' \\mid s, a)$")
        st.write(
            "The stochastic environment is characterized by a 3D transition probability tensor "
            "of shape **$(7 \\text{ actions} \\times 32 \\text{ from-states} \\times 32 \\text{ to-states})$**, "
            "estimated from 249,258 clinical transitions."
        )

        tensor_col1, tensor_col2 = st.columns(2)
        with tensor_col1:
            st.markdown("""
            * **Total state-action pairs:** $7 \\times 32 = 224$
            * **Observed state-action pairs:** 144 / 224
            * **Row stochasticity condition:**
              $$\\sum_{s'=0}^{31} \\mathcal{P}(s' \\mid s, a) = 1.0 \\quad \\forall (s, a) \\text{ where } N(s, a) > 0$$
            """)
        with tensor_col2:
            st.markdown("""
            * **Absorbing boundary conditions:**
              $$\\mathcal{P}(s' \\mid S06, a) = 0 \\quad \\text{and} \\quad \\mathcal{P}(s' \\mid S25, a) = 0$$
              No patient can transition away from recovery or death.
            """)

        st.divider()
        st.subheader("3. Interactive Transition Matrix Heatmap by Action")

        # Load transition matrix from results_training
        npz_file = locate_file("transition_matrices.npz")
        if npz_file and npz_file.exists():
            matrix_data = np.load(npz_file)
            transition_tensor = matrix_data["matrices"]  # Shape (7, 32, 32)
            counts_tensor = matrix_data["counts"]

            selected_action_id = st.selectbox(
                "Select treatment action to inspect Markov transition probabilities",
                list(range(7)),
                format_func=lambda i: f"Action {i}: {ACTIONS[i]}"
            )

            act_matrix = transition_tensor[selected_action_id]
            act_counts = counts_tensor[selected_action_id]

            st.caption(
                f"Showing empirical $32 \\times 32$ Markov transition probability matrix for **{ACTIONS[selected_action_id]}**. "
                f"Total observed transitions for this action: {int(act_counts.sum()):,}."
            )

            state_labels = [f"S{i:02d}" for i in range(32)]
            fig = px.imshow(
                act_matrix,
                labels=dict(x="Destination State (s')", y="Origin State (s)", color="Probability"),
                x=state_labels,
                y=state_labels,
                color_continuous_scale="Teal",
                zmin=0.0,
                zmax=1.0,
                aspect="auto",
            )
            fig.update_layout(
                height=650,
                xaxis_title="Destination State s'",
                yaxis_title="Origin State s",
                margin=dict(l=40, r=40, t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Transition matrix file (transition_matrices.npz) not found in results_training.")

    # ---------------------------------------------------------
    # TAB 4: QUALITY SCORING & REWARDS
    # ---------------------------------------------------------
    with tab4:
        st.subheader("1. Clinical Health Quality Scoring Metric: $Q(s)$")
        st.write(
            "Rather than using heuristic rewards, CancerStochastique defines an objective scalar "
            "**health quality score** $Q(s)$ for every state $s \\in \\mathcal{S}$:"
        )

        st.latex(r"Q(s) = -\Big(2D + 2P - U + 2B\Big)")

        st.markdown("""
        **Clinical Weighting Rationale:**
        * **Disease Severity ($D=1$):** $-2.0$ penalty (high tumor burden, metastases).
        * **Poor Prognosis ($P=1$):** $-2.0$ penalty (acute mortality risk).
        * **Biological Resistance ($B=1$):** $-2.0$ penalty (treatment refractory).
        * **Treatment Utility ($U \\in \\{0, 1, 2, 3\\}$):** $+1.0 \\times U$ health benefit.
        """)

        st.divider()
        st.subheader("2. Piecewise Differential Reward Formulation: $R(s, a, s')$")
        st.write(
            "The reward received when taking action $a$ in state $s$ and transitioning to state $s'$ "
            "is governed by the health differential $\\Delta = Q(s') - Q(s)$ plus an intervention cost penalty:"
        )

        st.latex(r"""
        R(s, a, s') = \begin{cases}
        +30.0 & \text{if } s' = 6 \text{ (Full Recovery)} \\
        -50.0 & \text{if } s' = 25 \text{ (Terminal Death)} \\
        +10.0 \times \Delta & \text{if } \Delta > 0 \text{ (Health Improvement)} \\
        +1.0 & \text{if } \Delta = 0 \text{ (Disease Stability)} \\
        -5.0 \times |\Delta| & \text{if } \Delta < 0 \text{ (Clinical Deterioration)}
        \end{cases} \quad - \quad \mathbf{1}_{\{a > 0\}}
        """)

        st.caption(
            "The $-1.0$ penalty on active interventions ($a > 0$) prevents unnecessary toxicity and overtreatment, "
            "encouraging active surveillance (Observation, $a=0$) when safe."
        )

        st.divider()
        st.subheader("3. Interactive Step-by-Step Reward Calculator")
        st.write("Pick an origin state, an action, and a destination state to compute the exact transition reward:")

        r_c1, r_c2, r_c3 = st.columns(3)
        with r_c1:
            calc_from = st.selectbox("Origin State (s_from)", list(range(32)), format_func=lambda i: f"S{i:02d} - {state_to_label(i)[:25]}...", index=10)
        with r_c2:
            calc_act = st.selectbox("Action Taken (a)", list(range(7)), format_func=lambda i: f"Action {i}: {ACTIONS[i]}", index=2)
        with r_c3:
            calc_to = st.selectbox("Destination State (s_to)", list(range(32)), format_func=lambda i: f"S{i:02d} - {state_to_label(i)[:25]}...", index=6)

        q_from = get_state_quality(calc_from)
        q_to = get_state_quality(calc_to)
        delta_q = q_to - q_from
        cost_penalty = -1.0 if calc_act > 0 else 0.0
        computed_r = compute_reward(calc_from, calc_act, calc_to)

        st.markdown(f"""
        * **Origin Health Score $Q(s)$: ** `{q_from:.1f}`
        * **Destination Health Score $Q(s')$: ** `{q_to:.1f}`
        * **Net Health Differential $\\Delta$: ** `{delta_q:+.1f}`
        * **Action Toxicity Penalty: ** `{cost_penalty:.1f}`
        * **Calculated Total Reward $R(s, a, s')$: ** **`{computed_r:+.1f}`**
        """)

        cost_str = " - 1.0" if calc_act > 0 else " + 0.0"
        if calc_to == 6:
            st.success("🌟 Awarded Full Recovery Bonus (+30.0).")
        elif calc_to == 25:
            st.error("⚠️ Penalized with Absorbing Death Penalty (-50.0).")
        elif delta_q > 0:
            st.info(f"📈 Patient improved by {delta_q:+.1f} points: reward = (+10.0 × {delta_q:.1f}){cost_str} = {computed_r:.1f}.")
        elif delta_q == 0:
            st.info(f"⚖️ Disease stabilized: reward = (+1.0){cost_str} = {computed_r:.1f}.")
        else:
            st.warning(f"📉 Disease deteriorated by {delta_q:.1f} points: reward = (-5.0 × {abs(delta_q):.1f}){cost_str} = {computed_r:.1f}.")

        st.divider()
        st.subheader("4. Connection to the Bellman Equation")
        st.markdown(
            "Tabular Q-Learning and Double Q-Learning optimize expected discounted returns by solving the **Bellman Optimality Equation**:"
        )

        st.latex(r"Q^*(s, a) = \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ R(s, a, s') + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a') \right]")

        st.caption(
            "With discount factor $\\gamma = 0.95$, the policy balances short-term therapy burden against long-term disease remission and survival."
        )

