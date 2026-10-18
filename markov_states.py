"""
State Space Definition & Mapping Functions

Maps between:
    Observation space X:  Xi(t) = {Ci, Gi, Oi, Mi, Ti(t)}
    Markov state space S: Si    = {Di, Pi, Ui(t), Bi}

Mapping functions:
    fd(Xi) → Di ∈ {0,1}       Disease severity
    fp(Xi) → Pi ∈ {0,1}       Prognosis
    fu(Xi) → Ui ∈ {0,1,2,3}   Treatment utility
    fb(Xi) → Bi ∈ {0,1}       Biological response

State encoding:
    state_index = D * 16 + P * 8 + U * 2 + B
    giving indices 0..31 for the 32 possible states
"""

import numpy as np
from config import (
    D_VALUES, P_VALUES, U_VALUES, B_VALUES,
    N_STATES, STAGES, METASTATIC_SITES, CANCER_TYPES, OUTCOMES
)



# State encoding / decoding

def encode_state(D: int, P: int, U: int, B: int) -> int:

    assert D in D_VALUES and P in P_VALUES and U in U_VALUES and B in B_VALUES
    return D * 16 + P * 8 + U * 2 + B


def decode_state(index: int) -> tuple:
    assert 0 <= index < N_STATES
    B = index % 2
    U = (index // 2) % 4
    P = (index // 8) % 2
    D = (index // 16) % 2
    return (D, P, U, B)


def state_to_label(index: int) -> str:
    D, P, U, B = decode_state(index)
    from config import D_LABELS, P_LABELS, U_LABELS, B_LABELS
    return f"S{index:02d}(D={D_LABELS[D]}, P={P_LABELS[P]}, U={U_LABELS[U]}, B={B_LABELS[B]})"


def build_state_space():

    states = []
    state_index_map = {}

    for D in D_VALUES:
        for P in P_VALUES:
            for U in U_VALUES:
                for B in B_VALUES:
                    idx = encode_state(D, P, U, B)
                    states.append((D, P, U, B))
                    state_index_map[(D, P, U, B)] = idx

    assert len(states) == N_STATES
    return states, state_index_map



def fd(cancer_type: str, stage: str, metastatic_site: str) -> int:

    stage_idx = STAGES.index(stage)
    has_metastasis = metastatic_site != "None"

    if stage_idx >= 2 or has_metastasis:
        return 1  # High severity
    return 0  # Low severity


def fp(cancer_type: str, stage: str, outcome: str, metastatic_site: str) -> int:

    if outcome == "Deceased":
        return 1  # Poor

    stage_idx = STAGES.index(stage)
    if stage_idx == 3 and metastatic_site in ["Brain", "Multiple"]:
        return 1  # Poor

    return 0  # Good


def fu(treatments_so_far: list, outcome: str, stage: str) -> int:

    n_treatments = len(treatments_so_far)

    if n_treatments == 0:
        return 0  # No treatment yet

    stage_idx = STAGES.index(stage)

    if outcome == "Deceased":
        return 0  # None — treatment failure (maps deceased patients to terminal state S25)

    if stage_idx >= 3:
        return 1  # Low — late stage
    elif stage_idx >= 2:
        return 2  # Medium — mid stage, alive
    else:
        return 3  # High — early stage, alive


def fb(cancer_type: str, treatments_so_far: list, metastatic_site: str) -> int:

    if metastatic_site == "Multiple":
        return 1  # Resistant

    if len(treatments_so_far) >= 3 and metastatic_site != "None":
        return 1  # Resistant

    return 0  # Responsive


def observation_to_state(cancer_type, stage, outcome, metastatic_site, treatments):

    D = fd(cancer_type, stage, metastatic_site)
    P = fp(cancer_type, stage, outcome, metastatic_site)
    U = fu(treatments, outcome, stage)
    B = fb(cancer_type, treatments, metastatic_site)

    state_idx = encode_state(D, P, U, B)
    return state_idx, (D, P, U, B)



# Reward function:  R(s, a, s')


def compute_reward(state_from: int, action: int, state_to: int) -> float:

    from config import (
        REWARD_IMPROVEMENT, REWARD_STABLE, REWARD_DETERIORATION,
        REWARD_DEATH, REWARD_RECOVERY, REWARD_TREATMENT_COST,
        TERMINAL_REWARD_MAP
    )

    D_from, P_from, U_from, B_from = decode_state(state_from)
    D_to, P_to, U_to, B_to = decode_state(state_to)

    # Quality scoring
    quality_from = -(2 * D_from + 2 * P_from - U_from + 2 * B_from)
    quality_to = -(2 * D_to + 2 * P_to - U_to + 2 * B_to)

    # Check terminal states
    state_to_tuple = (D_to, P_to, U_to, B_to)
    if state_to_tuple == (1, 1, 0, 1):  # Worst state
        return REWARD_DEATH
    if state_to_tuple == (0, 0, 3, 0):  # Best state
        return REWARD_RECOVERY

    # Compute differential reward
    delta = quality_to - quality_from

    if delta > 0:
        reward = REWARD_IMPROVEMENT * delta
    elif delta == 0:
        reward = REWARD_STABLE
    else:
        reward = REWARD_DETERIORATION * abs(delta)

    # Treatment cost penalty (except for observation)
    if action > 0:
        reward += REWARD_TREATMENT_COST

    return reward



# Terminal state checking


def is_terminal(state_index: int) -> bool:
    state_tuple = decode_state(state_index)
    return state_tuple in [(1, 1, 0, 1), (0, 0, 3, 0)]


def get_state_quality(state_index: int) -> float:
    D, P, U, B = decode_state(state_index)
    return -(2 * D + 2 * P - U + 2 * B)


if __name__ == "__main__":
    states, state_map = build_state_space()
    print(f"State space |S| = {len(states)}")
    print(f"{'Index':<6} {'(D,P,U,B)':<12} {'Label':<50} {'Quality':<8} {'Terminal'}")
    print("─" * 90)
    for idx, (D, P, U, B) in enumerate(states):
        label = state_to_label(idx)
        quality = get_state_quality(idx)
        terminal = "YES" if is_terminal(idx) else ""
        print(f"{idx:<6} ({D},{P},{U},{B}){'':>4} {label:<50} {quality:<8.1f} {terminal}")
