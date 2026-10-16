"""
Mathematical Foundation:
    Observed patient:  Xi(t) = {Ci, Gi, Oi, Mi, Ti(t)}
    Markov state:      Si    = {Di, Pi, Ui(t), Bi}  ∈  S

    Mapping functions:
        fd: X → {0,1}       Disease severity (Low=0, High=1)
        fp: X → {0,1}       Prognosis (Good=0, Poor=1)
        fu: X → {0,1,2,3}   Treatment utility (None=0, Low=1, Medium=2, High=3)
        fb: X → {0,1}       Biological response (Responsive=0, Resistant=1)

    State space:  |S| = 2 × 2 × 4 × 2 = 32 states
    Action space: |A| = 7 actions
"""

RANDOM_SEED = 42

# Dataset parameters

N_PATIENTS = 25000
MAX_TIMESTEPS = 20  # Max trajectory length per patient
MIN_TIMESTEPS = 3  # Min trajectory length per patient


# Observed patient features:  Xi(t) = {Ci, Gi, Oi, Mi, Ti(t)}


CANCER_TYPES = ["Lung", "Breast", "Colorectal", "Prostate"]  # Ci
STAGES = ["I", "II", "III", "IV"]  # Gi
OUTCOMES = ["Alive", "Deceased"]  # Oi
METASTATIC_SITES = ["None", "Bone", "Liver", "Lung", "Brain", "Multiple"]  # Mi



# Markov state components:  Si = {Di, Pi, Ui(t), Bi}

# fd: Disease severity
D_VALUES = [0, 1]  # 0=Low, 1=High
D_LABELS = {0: "Low", 1: "High"}

# fp: Prognosis
P_VALUES = [0, 1]  # 0=Good, 1=Poor
P_LABELS = {0: "Good", 1: "Poor"}

# fu: Treatment utility
U_VALUES = [0, 1, 2, 3]  # 0=None, 1=Low, 2=Medium, 3=High
U_LABELS = {0: "None", 1: "Low", 2: "Medium", 3: "High"}

# fb: Biological response
B_VALUES = [0, 1]  # 0=Responsive, 1=Resistant
B_LABELS = {0: "Responsive", 1: "Resistant"}

# Total state space
N_STATES = len(D_VALUES) * len(P_VALUES) * len(U_VALUES) * len(B_VALUES)  # = 32
assert N_STATES == 32, f"State space should be 32, got {N_STATES}"


# Action space:  A = set of possible treatments

ACTIONS = [
    "Observation",  # a=0 : watchful waiting
    "Surgery",  # a=1 : surgical intervention
    "Chemotherapy",  # a=2 : chemotherapy
    "Radiation",  # a=3 : radiation therapy
    "Immunotherapy",  # a=4 : immunotherapy
    "Combination",  # a=5 : combination therapy (chemo + radiation / chemo + immuno)
    "Palliative",  # a=6 : palliative care
]
N_ACTIONS = len(ACTIONS)  # = 7


# Q-Learning hyperparameters

ALPHA = 0.1  # Learning rate
GAMMA = 0.95  # Discount factor
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Exploration decay per episode
N_EPISODES = 10000  # Training episodes


# Reward function parameters

# R(s, a, s') — reward for transitioning from state s to s' via action a
REWARD_IMPROVEMENT = +10.0  # State improves (lower severity, better prognosis)
REWARD_STABLE = +1.0  # State remains the same
REWARD_DETERIORATION = -5.0  # State worsens
REWARD_DEATH = -50.0  # Patient reaches terminal/death state
REWARD_RECOVERY = +30.0  # Patient reaches best possible state
REWARD_TREATMENT_COST = -1.0  # Small penalty per treatment step (encourages efficiency)


# Terminal states (absorbing states)

# States where D=1, P=1, U=0, B=1 → worst outcome (death-like)
# States where D=0, P=0, U=3, B=0 → best outcome (recovery)
TERMINAL_REWARD_MAP = {
    # (D, P, U, B) → is_terminal
    (1, 1, 0, 1): True,  # Worst: high severity, poor prognosis, no utility, resistant
    (0, 0, 3, 0): True,  # Best: low severity, good prognosis, high utility, responsive
}


# Multi-Agent System configuration

AGENT_TYPES = [
    "DiagnosticAgent",  # Evaluates disease severity (D)
    "PrognosticAgent",  # Evaluates prognosis (P)
    "TreatmentAgent",  # Manages treatment utility (U) — Q-learning agent
    "BiologicalAgent",  # Evaluates biological response (B)
]


