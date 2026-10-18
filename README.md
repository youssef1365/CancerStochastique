# 🧬 CancerStochastique

> **Stochastic Modeling, Multi-Agent Reinforcement Learning & Simulation for Adaptive Cancer Treatment Decision Support**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![NetLogo](https://img.shields.io/badge/Simulation-NetLogo_7.0-green.svg)](https://ccl.northwestern.edu/netlogo/)

---

## 🔬 Overview

**CancerStochastique** is a computational oncology research framework that models dynamic cancer treatment decision-making as a **Markov Decision Process (MDP)**. Rather than relying solely on static clinical protocols or unconstrained "black-box" reinforcement learning, the framework combines:

1. **Stochastic State Representation**: A compact 32-state discrete Markov model capturing disease burden, prognosis, treatment utility, and biological resistance.
2. **Tabular Reinforcement Learning**: Standard **Q-Learning** and **Double Q-Learning** (to mitigate maximization bias under stochastic transitions).
3. **Multi-Agent "Tumor Board" Guardrails**: A consensus system where specialized clinical agents (Diagnostic, Prognostic, Biological) possess veto powers to prevent dangerous algorithmic treatment recommendations.
4. **Dual-Engine Simulation & Visualization**:
   * **Streamlit Research Dashboard**: Interactive patient case assessment, policy comparisons, and state transition analysis.
   * **NetLogo 7.0 Agent-Based Model**: Micro-simulation of patient cohorts navigating disease states over time.

---

## 📐 Mathematical Formulation

### 1. Patient Observation Space $\mathcal{X}(t)$
Each patient is observed through clinical variables:
$$\mathcal{X}_i(t) = \{C_i, G_i, O_i, M_i, T_i(t)\}$$
* $C_i \in \{\text{Lung, Breast, Colorectal, Prostate}\}$: Primary cancer type.
* $G_i \in \{\text{I, II, III, IV}\}$: Stage at entry.
* $O_i \in \{\text{Alive, Deceased}\}$: Vital status.
* $M_i \in \{\text{None, Bone, Liver, Lung, Brain, Multiple}\}$: Metastatic site.
* $T_i(t)$: Cumulative treatment history.

### 2. Markov State Space $\mathcal{S}$ ($|\mathcal{S}| = 32$)
The observation space is deterministically projected into 4 binary/ordinal clinical components:
$$\mathcal{S}_i(t) = \{D_i, P_i, U_i(t), B_i\}$$

$$\text{State Index } S = 16 \times D + 8 \times P + 2 \times U + B \quad \in [0, 31]$$

| Component | Values | Clinical Meaning | Projection Function |
| :--- | :--- | :--- | :--- |
| **$D$ (Disease Severity)** | $\{0, 1\}$ | $0=\text{Low}, 1=\text{High}$ | $f_d$: Stage $\ge \text{III}$ or metastasis $\to 1$ |
| **$P$ (Prognosis)** | $\{0, 1\}$ | $0=\text{Good}, 1=\text{Poor}$ | $f_p$: Deceased or Stage IV with Brain/Multiple mets $\to 1$ |
| **$U$ (Treatment Utility)**| $\{0, 1, 2, 3\}$| $0=\text{None}, 1=\text{Low}, 2=\text{Med}, 3=\text{High}$ | $f_u$: Based on prior therapies, stage, and vital status |
| **$B$ (Biological Response)**| $\{0, 1\}$ | $0=\text{Responsive}, 1=\text{Resistant}$ | $f_b$: Multiple metastases or $\ge 3$ lines with metastasis $\to 1$ |

#### Absorbing Terminal States:
* **State 6**: $(D=0, P=0, U=3, B=0) \implies$ **Full Recovery** ($+30.0$ reward).
* **State 25**: $(D=1, P=1, U=0, B=1) \implies$ **Terminal Death** ($-50.0$ penalty).

### 3. Action Space $\mathcal{A}$ ($|\mathcal{A}| = 7$)
$$\mathcal{A} = \{\text{Observation (0), Surgery (1), Chemotherapy (2), Radiation (3), Immunotherapy (4), Combination (5), Palliative (6)}\}$$

### 4. Reward Formulation $R(s, a, s')$
Rewards are governed by a clinical health quality scoring metric:
$$Q(s) = -(2D + 2P - U + 2B)$$
$$\Delta = Q(s') - Q(s)$$

$$R(s, a, s') = \begin{cases}
+30.0 & \text{if } s' = 6 \text{ (Recovery)} \\
-50.0 & \text{if } s' = 25 \text{ (Death)} \\
+10.0 \times \Delta & \text{if } \Delta > 0 \text{ (Health improvement)} \\
+1.0 & \text{if } \Delta = 0 \text{ (Stable condition)} \\
-5.0 \times |\Delta| & \text{if } \Delta < 0 \text{ (Clinical deterioration)}
\end{cases} \quad - \quad \mathbf{1}_{\{a > 0\}}$$

*(The $-1.0$ penalty on non-observation actions discourages overtreatment and unnecessary drug toxicity).*

---

## 🤖 Multi-Agent Tumor Board Architecture

```
   ┌─────────────────────────────────────────────────────────────┐
   │                    Patient Case Data                        │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
 ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
 │DiagnosticAgent│        │PrognosticAgent│        │BiologicalAgent│
 │ Evaluates D   │        │ Evaluates P   │        │ Evaluates B   │
 └───────┬───────┘        └───────┬───────┘        └───────┬───────┘
         │                        │                        │
         │                ┌───────▼───────┐                │
         │                │TreatmentAgent │                │
         │                │ Learned Q(s,a)│                │
         │                └───────┬───────┘                │
         │                        │                        │
         └───────────────┐        ▼        ┌───────────────┘
                         ▼        ▼        ▼
                      ┌──────────────────────┐
                      │  MultiAgentSystem    │
                      │  Voting & Consensus  │
                      └──────────┬───────────┘
                                 ▼
                         Selected Action
```

* **DiagnosticAgent**: Vetoes Observation if $D=1$ (advocates Chemo); vetoes Combination if $D=0$.
* **PrognosticAgent**: Recommends Palliative care if disease is high severity, poor prognosis, and resistant ($D=1, P=1, B=1$).
* **BiologicalAgent**: Redirects Chemotherapy to Immunotherapy when tumor resistance ($B=1$) is detected.
* **TreatmentAgent**: Proposes $\arg\max_a Q(s, a)$ from the learned offline RL policy table.
* **Consensus Engine**: Gathers votes, tallies majority consensus, and breaks ties via the Treatment Agent.

---

## 📁 Repository Structure

```
CancerStochastique/
├── config.py                 # Core constants, reward weights, spaces
├── markov_states.py          # State encoders/decoders, mappers fd, fp, fu, fb, rewards
├── matrix.py                 # Diagnostic script inspecting empirical dataset transitions
├── pyproject.toml            # Project metadata and dependencies
├── requirements.txt          # Python dependencies for pip
├── app.py                    # Streamlit interactive web dashboard
├── agents/
│   ├── base_agent.py         # Abstract agent interface
│   ├── diagnostic_agent.py   # Diagnostician agent (evaluates D)
│   ├── prognostic_agent.py   # Prognostician agent (evaluates P)
│   ├── biological_agent.py   # Oncobiologist agent (evaluates B)
│   ├── treatment_agent.py    # Policy executor agent (queries Q-values)
│   ├── multi_agent.py        # MAS orchestrator (voting, consensus)
│   └── train.py              # Main training pipeline & transition tensor builder
├── q_learnings/
│   ├── q_learning.py         # Tabular Q-Learning agent with action masking
│   └── double_q_learning.py  # Double Q-Learning agent (QA, QB)
├── data/
│   ├── data_generator.py     # 25,000-patient synthetic cohort generator
│   └── cancer_treatment_dataset.csv # Recorded transition records
├── simulation/
│   └── cancer_simulation.nlogo.nlogox # NetLogo 7.0 visual agent-based simulation
├── results_training/
│   ├── models/               # Serialized Q-tables (.npy) & 3D matrices (.npz)
│   ├── netlogo/              # Exported CSVs consumed by NetLogo and Streamlit
│   └── tables/               # Checkpoint metrics, episodes, policy evaluations
└── tests/
    ├── test_markov_states.py # Unit tests for state encoding & rewards
    ├── test_agents.py        # Unit tests for multi-agent veto rules
    └── test_q_learning.py    # Unit tests for Q and Double Q Bellman updates
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install dependencies using standard `pip` or modern `uv`:

```bash
git clone https://github.com/youssef1365/CancerStochastique.git
cd CancerStochastique

# Using pip:
pip install -r requirements.txt

# Or using uv (recommended):
uv sync
```

### 2. Run Automated Unit Tests

Ensure all mathematical formulas, state encodings, and agent vetoes pass validation:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### 3. Launch the Interactive Streamlit Web App

Start the research dashboard:

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to:
* Assess synthetic patient cases and observe state encodings.
* Compare Q-Learning vs. Double Q-Learning treatment choices.
* Explore the 32 discrete states, terminal status, and associated policies.

### 4. Train the Reinforcement Learning Agents

Train Double Q-Learning or standard Q-Learning on the patient dataset:

```bash
# Train Double Q-Learning (10,000 episodes, 50 checkpoints)
python3 agents/train.py --data data/cancer_treatment_dataset.csv --algorithm double_q_learning

# Train standard Q-Learning
python3 agents/train.py --data data/cancer_treatment_dataset.csv --algorithm q_learning
```

Outputs are automatically saved into `results_training/`, including `.npy` arrays, `.csv` policy tables, and NetLogo input files.

### 5. Run NetLogo Visual Agent-Based Simulation

1. Open **NetLogo 7.0+**.
2. File $\to$ Open $\to$ `simulation/cancer_simulation.nlogo.nlogox`.
3. Click **Setup** to initialize the virtual patient cohort.
4. Click **Go** to watch the cohort evolve through the disease Markov states according to learned treatment policies.

---

## ⚠️ Research Disclaimer

> **IMPORTANT**: This software is an educational research prototype designed to investigate stochastic modeling and reinforcement learning dynamics in simulated oncology environments. **It is not certified for clinical use and does not provide real medical advice or clinical treatment recommendations.** Do not enter real or identifiable patient data.
