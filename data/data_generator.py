import argparse
import json
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config as cfg
from markov_states import (
    observation_to_state, compute_reward, is_terminal, decode_state,
)

GENERATOR_VERSION = "v1"

# ----------------------------------------------------------------------------
# Generation assumptions (illustrative, invented - NOT clinical estimates)
# ----------------------------------------------------------------------------
STAGE_START_P = [0.30, 0.30, 0.20, 0.20]  # stages I..IV at entry

# Behaviour policy over actions [Obs, Surg, Chemo, Rad, Immuno, Comb, Pall]
BEHAVIOUR_POLICY = {
    0: [0.40, 0.25, 0.10, 0.10, 0.05, 0.07, 0.03],  # stage I
    1: [0.30, 0.25, 0.15, 0.10, 0.07, 0.10, 0.03],  # stage II
    2: [0.12, 0.12, 0.26, 0.15, 0.12, 0.17, 0.06],  # stage III
    3: [0.08, 0.04, 0.24, 0.10, 0.15, 0.22, 0.17],  # stage IV
}

# per-step stage-progression probability (stages I..III), by stage
PROG_BASE = [0.12, 0.14, 0.12]
TYPE_PROG = {"Lung": 1.3, "Breast": 0.8, "Colorectal": 1.0, "Prostate": 0.7}
#                Obs   Surg  Chemo Rad   Immu  Comb  Pall
ACT_PROG = [1.00, 0.45, 0.60, 0.60, 0.65, 0.40, 0.90]
ACT_META = [1.00, 0.60, 0.70, 0.70, 0.70, 0.50, 0.90]
ACT_HAZ = [1.00, 0.70, 0.80, 0.85, 0.80, 0.70, 0.90]

META_NEW = [0.02, 0.05, 0.10, 0.15]  # None -> single site, by stage
META_MULTI = [0.03, 0.06, 0.12, 0.20]  # single -> Multiple, by stage
DEATH_HAZ = [0.002, 0.010, 0.040, 0.100]  # per step, by stage
SITE_HAZ_MULT = {"None": 1.0, "Bone": 1.5, "Liver": 1.8, "Lung": 1.6,
                 "Brain": 2.0, "Multiple": 2.5}

SITE_WEIGHTS = {  # which single site appears first, by cancer type
    "Lung": {"Brain": .35, "Bone": .30, "Liver": .25, "Lung": .10},
    "Breast": {"Bone": .45, "Liver": .25, "Lung": .15, "Brain": .15},
    "Colorectal": {"Liver": .60, "Lung": .30, "Bone": .07, "Brain": .03},
    "Prostate": {"Bone": .85, "Liver": .05, "Lung": .05, "Brain": .05},
}

CENSOR_P = 0.03  # per-step loss-to-follow-up probability once t >= MIN_TIMESTEPS
INTERVAL_DAYS = (60, 120)
DATE_RANGE_START = date(2015, 1, 1)
DATE_RANGE_DAYS = 2190

RARE_STATE_THRESHOLD = 30  # outgoing transitions from a state
RARE_PAIR_THRESHOLD = 5  # transitions for a (state, action) pair

TRAJ_COLUMNS = ["patient_id", "timestep", "start_date", "end_date",
                "state_from", "action", "reward", "state_to",
                "is_terminal", "data_source"]
NETLOGO_COLUMNS = ["action", "state_from", "state_to", "count",
                   "probability", "observed"]

END_TERMINAL = "terminal_state"
END_DEATH = "death_nonterminal_state"
END_CENSOR = "loss_to_follow_up_censoring"
END_MAXT = "administrative_censoring_max_timesteps"


# ----------------------------------------------------------------------------
# Latent patient simulation
# ----------------------------------------------------------------------------
def _pick_site(rng, cancer):
    w = SITE_WEIGHTS[cancer]
    names = list(w)
    p = np.array([w[n] for n in names], dtype=float)
    return str(rng.choice(names, p=p / p.sum()))


def _state_of(p):
    idx, _ = observation_to_state(
        p["cancer"], cfg.STAGES[p["stage"]], p["outcome"], p["site"],
        p["treatments"])
    return idx


def _init_patient(rng):
    """Sample an alive, untreated patient whose state is not terminal."""
    while True:
        cancer = str(rng.choice(cfg.CANCER_TYPES))
        stage = int(rng.choice(4, p=STAGE_START_P))
        site = "None"
        r = rng.random()
        if stage == 1 and r < 0.05:
            site = _pick_site(rng, cancer)
        elif stage == 2 and r < 0.15:
            site = _pick_site(rng, cancer)
        elif stage == 3 and r < 0.60:
            site = _pick_site(rng, cancer)
            if rng.random() < 0.25:
                site = "Multiple"
        p = {"cancer": cancer, "stage": stage, "outcome": "Alive",
             "site": site, "treatments": []}
        if not is_terminal(_state_of(p)):
            return p


def _advance(p, a, rng):
    """Apply one action step to the latent patient. Returns (new_p, died)."""
    tx = list(p["treatments"])
    if a > 0:
        tx.append(cfg.ACTIONS[a])
    stage, site, cancer = p["stage"], p["site"], p["cancer"]

    if stage < 3:
        pr = min(0.95, PROG_BASE[stage] * TYPE_PROG[cancer] * ACT_PROG[a])
        if rng.random() < pr:
            stage += 1

    if site == "None":
        if rng.random() < min(0.95, META_NEW[stage] * ACT_META[a]):
            site = _pick_site(rng, cancer)
    elif site != "Multiple":
        if rng.random() < min(0.95, META_MULTI[stage] * ACT_META[a]):
            site = "Multiple"

    haz = min(0.95, DEATH_HAZ[stage] * SITE_HAZ_MULT[site] * ACT_HAZ[a])
    died = bool(rng.random() < haz)

    new_p = {"cancer": cancer, "stage": stage,
             "outcome": "Deceased" if died else "Alive",
             "site": site, "treatments": tx}
    return new_p, died


def generate_trajectory(pid, rng, data_source):
    p = _init_patient(rng)
    s_from = _state_of(p)
    start_state = s_from
    d = DATE_RANGE_START + timedelta(days=int(rng.integers(0, DATE_RANGE_DAYS)))
    diag_date = d
    rows, t, end_reason = [], 0, END_MAXT

    while t < cfg.MAX_TIMESTEPS:
        a = int(rng.choice(cfg.N_ACTIONS, p=BEHAVIOUR_POLICY[p["stage"]]))
        p_new, died = _advance(p, a, rng)
        s_to = _state_of(p_new)
        term = bool(is_terminal(s_to))
        reward = float(compute_reward(s_from, a, s_to))
        d_end = d + timedelta(days=int(rng.integers(INTERVAL_DAYS[0],
                                                    INTERVAL_DAYS[1] + 1)))
        rows.append({
            "patient_id": pid, "timestep": t,
            "start_date": d.isoformat(), "end_date": d_end.isoformat(),
            "state_from": s_from, "action": a, "reward": reward,
            "state_to": s_to, "is_terminal": term,
            "data_source": data_source,
        })
        p, s_from, d, t = p_new, s_to, d_end, t + 1

        if term:
            end_reason = END_TERMINAL
            break
        if died:
            end_reason = END_DEATH
            break
        if t >= cfg.MAX_TIMESTEPS:
            end_reason = END_MAXT
            break
        if t >= cfg.MIN_TIMESTEPS and rng.random() < CENSOR_P:
            end_reason = END_CENSOR
            break

    summary = {
        "patient_id": pid, "start_state": start_state,
        "n_transitions": len(rows), "end_reason": end_reason,
        "final_state": rows[-1]["state_to"],
        "followup_days": (d - diag_date).days,
        "cancer_type_latent": None,  # filled below (latent, informational)
    }
    return rows, summary


def build_dataset(n_patients, seed):
    rng = np.random.default_rng(seed)
    data_source = f"SYNTHETIC_GENERATOR_{GENERATOR_VERSION}_seed{seed}"
    all_rows, summaries = [], []
    for i in range(n_patients):
        pid = f"P{i:06d}"
        rows, summ = generate_trajectory(pid, rng, data_source)
        all_rows.extend(rows)
        summaries.append(summ)
    df = pd.DataFrame(all_rows, columns=TRAJ_COLUMNS)
    ps = pd.DataFrame(summaries).drop(columns=["cancer_type_latent"])
    return df, ps, data_source


# ----------------------------------------------------------------------------
# Splitting (by patient_id only)
# ----------------------------------------------------------------------------
def split_patients(patient_ids, seed, frac=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed + 1)
    ids = np.array(patient_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(round(frac[0] * n))
    n_va = int(round(frac[1] * n))
    split = {}
    for i, pid in enumerate(ids):
        split[pid] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return split


# ----------------------------------------------------------------------------
# Counts, matrices, support
# ----------------------------------------------------------------------------
def count_array(df):
    C = np.zeros((cfg.N_ACTIONS, cfg.N_STATES, cfg.N_STATES), dtype=np.int64)
    g = df.groupby(["action", "state_from", "state_to"]).size()
    for (a, s, s2), n in g.items():
        C[a, s, s2] = n
    return C


def probability_array(C):
    """P[a,s,:] = C[a,s,:]/sum; rows with zero support stay NaN (unsupported)."""
    P = np.full(C.shape, np.nan, dtype=float)
    tot = C.sum(axis=2)
    for a in range(C.shape[0]):
        for s in range(C.shape[1]):
            if tot[a, s] > 0:
                P[a, s, :] = C[a, s, :] / tot[a, s]
    return P, tot


def state_label(i):
    D, Pp, U, B = decode_state(i)
    return (f"D={cfg.D_LABELS[D]}|P={cfg.P_LABELS[Pp]}|"
            f"U={cfg.U_LABELS[U]}|B={cfg.B_LABELS[B]}")


def support_table(tot):
    """Per (state, action): count + status. Statuses:
       observed                       - >=1 observed transition
       possible_unobserved            - state has outgoing data for other
                                        actions; generator permits this action
                                        but it was never observed
       no_information                 - no outgoing transitions at all for the
                                        state (terminal or unreachable); no
                                        transition info exists.
    """
    state_total = tot.sum(axis=0)
    rows = []
    for s in range(cfg.N_STATES):
        for a in range(cfg.N_ACTIONS):
            n = int(tot[a, s])
            if n > 0:
                status = "observed"
            elif state_total[s] > 0 and not is_terminal(s):
                status = "possible_unobserved"
            else:
                status = "no_information"
            rows.append({"state_from": s, "action": a,
                         "action_name": cfg.ACTIONS[a], "count": n,
                         "status": status,
                         "rare_pair": 0 < n < RARE_PAIR_THRESHOLD})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate(df, ps, split_map, P, tot):
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    add("Exact required column set/order", list(df.columns) == TRAJ_COLUMNS)
    add("No nulls in required non-date columns",
        df[["patient_id", "timestep", "state_from", "action", "reward",
            "state_to", "is_terminal", "data_source"]].notna().all().all())
    add("state_from/state_to within [0,31]",
        df.state_from.between(0, 31).all() and df.state_to.between(0, 31).all())
    add("action within [0,6]", df.action.between(0, 6).all())

    # state/component agreement (round trip through the project's codec)
    from markov_states import encode_state
    rt = all(encode_state(*decode_state(int(s))) == s
             for s in pd.unique(pd.concat([df.state_from, df.state_to])))
    add("encode(decode(state)) == state for all states used", rt)

    # reward recomputation
    rew = np.array([compute_reward(int(f), int(a), int(t)) for f, a, t in
                    zip(df.state_from, df.action, df.state_to)])
    add("Rewards equal compute_reward() recomputation",
        np.allclose(rew, df.reward.values))

    # terminal flag
    tf = np.array([is_terminal(int(s)) for s in df.state_to])
    add("is_terminal matches is_terminal(state_to)",
        (tf == df.is_terminal.values).all())

    d = df.sort_values(["patient_id", "timestep"]).reset_index(drop=True)
    g = d.groupby("patient_id")
    first = g.timestep.transform("min")
    add("Timesteps start at 0", (first == 0).all())
    seq_ok = (g.timestep.diff().dropna() == 1).all()
    add("Timesteps sequential (+1) within patient", seq_ok)

    nxt = g.state_from.shift(-1)
    mask = nxt.notna()
    add("state_to(t) == state_from(t+1) within patient",
        (d.state_to[mask].values == nxt[mask].values).all())

    # no rows after a terminal row
    term_prev = g.is_terminal.shift(1).fillna(False).astype(bool)
    add("No rows after a genuine terminal state", not term_prev.any())

    # no rows after death (death rows are final rows)
    last = d.groupby("patient_id").tail(1).set_index("patient_id")
    ps_i = ps.set_index("patient_id")
    add("n_transitions in summary equals row counts",
        (d.groupby("patient_id").size() == ps_i.n_transitions).all())
    term_final = last.is_terminal
    add("is_terminal only ever True on last row of a patient",
        d.is_terminal.sum() == term_final.sum())
    add("end_reason=terminal_state <=> last row is_terminal",
        ((ps_i.end_reason == END_TERMINAL) == term_final.reindex(ps_i.index)).all())
    add("Non-terminal endings are not flagged terminal "
        "(death/censoring not mislabeled)",
        (~term_final.reindex(ps_i.index)[ps_i.end_reason != END_TERMINAL]).all())

    # dates
    sd = pd.to_datetime(d.start_date)
    ed = pd.to_datetime(d.end_date)
    add("end_date >= start_date", (ed >= sd).all())
    nsd = sd.groupby(d.patient_id).shift(-1)
    m2 = nsd.notna()
    add("Consecutive intervals contiguous (next start == prev end)",
        (nsd[m2].values == ed[m2].values).all())

    # splits
    sp = d.patient_id.map(split_map)
    pat_sets = {k: set(d.patient_id[sp == k]) for k in ("train", "val", "test")}
    disjoint = (not (pat_sets["train"] & pat_sets["val"]) and
                not (pat_sets["train"] & pat_sets["test"]) and
                not (pat_sets["val"] & pat_sets["test"]))
    add("Splits disjoint by patient_id", disjoint,
        f"train={len(pat_sets['train'])}, val={len(pat_sets['val'])}, "
        f"test={len(pat_sets['test'])}")

    # matrices
    rowsum = np.nansum(P, axis=2)
    supported = tot > 0
    add("Supported rows sum to 1 (tol 1e-9)",
        np.allclose(rowsum[supported], 1.0, atol=1e-9))
    add("Unsupported rows remain NaN (not uniform/zero-filled)",
        np.isnan(P[~supported]).all())
    add("All probabilities nonnegative", (np.nan_to_num(P) >= 0).all())
    return checks


# ----------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------
DATA_DICTIONARY = """# Data dictionary

**All data is SYNTHETIC** (see `data_source`). It is not real clinical data.

## trajectories*.csv
| Column | Type | Description |
|---|---|---|
| patient_id | string | Unique synthetic patient id (`P000000`...). |
| timestep | int | 0-based sequential transition index within the patient. |
| start_date | date (synthetic) | Synthetic start of the interval (ISO 8601). |
| end_date | date (synthetic) | Synthetic end; equals next row's start_date. |
| state_from | int 0-31 | Current state, `D*16+P*8+U*2+B`. |
| action | int 0-6 | 0 Observation, 1 Surgery, 2 Chemotherapy, 3 Radiation, 4 Immunotherapy, 5 Combination, 6 Palliative. |
| reward | float | `markov_states.compute_reward(state_from, action, state_to)`. |
| state_to | int 0-31 | Next state. |
| is_terminal | bool | True only if `state_to` is terminal per `markov_states.is_terminal()` (states 25 and 6). Death/censoring never set this. |
| data_source | string | `SYNTHETIC_GENERATOR_<version>_seed<seed>`. |

## patients_summary.csv
| Column | Description |
|---|---|
| patient_id | As above. |
| start_state | state_from of timestep 0. |
| n_transitions | Number of rows for the patient. |
| end_reason | `terminal_state`, `death_nonterminal_state`, `loss_to_follow_up_censoring`, or `administrative_censoring_max_timesteps`. |
| final_state | state_to of the last row. |
| followup_days | Synthetic follow-up duration (days). |
| split | train / val / test (assigned by patient). |

## transitions_netlogo.csv
Columns: `action,state_from,state_to,count,probability,observed`.
One row per observed (action, state_from, state_to) triple (observed=1);
`probability = count / count(state_from, action)`. Unsupported state-action
pairs have no rows (nothing is invented).

## state_action_support.csv
`status`: `observed` / `possible_unobserved` / `no_information`.
`rare_pair` is True when 0 < count < %d.

## matrices/P_action_<a>.csv
32x32 row-stochastic matrix for action a. Rows for unsupported state-action
pairs are left EMPTY (NaN). `transition_matrices.npz` holds the same arrays
(`P`, shape 7x32x32, NaN for unsupported; `counts`).
""" % RARE_PAIR_THRESHOLD


def write_all(out_dir, df, ps, split_map, data_source, matrix_source, seed, args):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "matrices"), exist_ok=True)

    df = df.sort_values(["patient_id", "timestep"]).reset_index(drop=True)
    ps = ps.copy()
    ps["split"] = ps.patient_id.map(split_map)
    sp = df.patient_id.map(split_map)

    df.to_csv(os.path.join(out_dir, "trajectories.csv"), index=False)
    for k in ("train", "val", "test"):
        df[sp == k].to_csv(os.path.join(out_dir, f"trajectories_{k}.csv"), index=False)
    ps.to_csv(os.path.join(out_dir, "patients_summary.csv"), index=False)

    src_df = df[sp == "train"] if matrix_source == "train" else df
    C = count_array(src_df)
    P, tot = probability_array(C)

    # distributions (all data)
    len_dist = ps.n_transitions.value_counts().sort_index().rename_axis(
        "trajectory_length").reset_index(name="n_patients")
    len_dist.to_csv(os.path.join(out_dir, "trajectory_length_distribution.csv"), index=False)

    n_start = ps.start_state.value_counts()
    n_from = df.state_from.value_counts()
    n_to = df.state_to.value_counts()
    n_term = df[df.is_terminal].state_to.value_counts()
    srows = []
    for s in range(cfg.N_STATES):
        D, Pp, U, B = decode_state(s)
        srows.append({
            "state": s, "D": D, "P": Pp, "U": U, "B": B,
            "label": state_label(s), "terminal": bool(is_terminal(s)),
            "n_as_start": int(n_start.get(s, 0)),
            "n_as_state_from": int(n_from.get(s, 0)),
            "n_as_state_to": int(n_to.get(s, 0)),
            "n_terminal_transitions_into": int(n_term.get(s, 0)),
        })
    sf = pd.DataFrame(srows)
    sf.to_csv(os.path.join(out_dir, "state_frequency.csv"), index=False)

    af = df.action.value_counts().sort_index().rename_axis("action").reset_index(name="count")
    af["action_name"] = af.action.map(lambda a: cfg.ACTIONS[a])
    af.to_csv(os.path.join(out_dir, "action_frequency.csv"), index=False)

    sup = support_table(tot)
    sup.to_csv(os.path.join(out_dir, "state_action_support.csv"), index=False)

    trip = []
    for a, s, s2 in zip(*np.nonzero(C)):
        trip.append({"action": int(a), "state_from": int(s), "state_to": int(s2),
                     "count": int(C[a, s, s2]), "probability": float(P[a, s, s2]),
                     "observed": 1})
    trip_df = pd.DataFrame(trip, columns=NETLOGO_COLUMNS)
    trip_df.to_csv(os.path.join(out_dir, "transitions_netlogo.csv"), index=False)
    trip_df.to_csv(os.path.join(out_dir, "transition_counts.csv"), index=False)

    for a in range(cfg.N_ACTIONS):
        m = pd.DataFrame(P[a], index=range(cfg.N_STATES), columns=range(cfg.N_STATES))
        m.index.name = "state_from"
        m.to_csv(os.path.join(out_dir, "matrices", f"P_action_{a}.csv"),
                 float_format="%.8f")
    np.savez(os.path.join(out_dir, "transition_matrices.npz"), P=P, counts=C)

    # unsupported / rare report
    out_tot = tot.sum(axis=0)
    rare_states = [int(s) for s in range(cfg.N_STATES)
                   if 0 < out_tot[s] < RARE_STATE_THRESHOLD]
    unobserved_states = [int(s) for s in range(cfg.N_STATES)
                         if out_tot[s] == 0]
    never_visited = [int(s) for s in range(cfg.N_STATES)
                     if n_from.get(s, 0) == 0 and n_to.get(s, 0) == 0
                     and n_start.get(s, 0) == 0]
    status_counts = sup.status.value_counts().to_dict()
    unsup = sup[sup.status != "observed"]
    unsup.to_csv(os.path.join(out_dir, "unsupported_state_action_pairs.csv"), index=False)
    pd.DataFrame({"state": rare_states,
                  "outgoing_transitions": [int(out_tot[s]) for s in rare_states],
                  "label": [state_label(s) for s in rare_states]}).to_csv(
        os.path.join(out_dir, "rare_states.csv"), index=False)

    # validation
    checks = validate(df, ps, split_map, P, tot)
    all_ok = all(ok for _, ok, _ in checks)

    # report
    n_pat, n_rows = ps.shape[0], len(df)
    L = ps.n_transitions
    end_counts = ps.end_reason.value_counts()
    lines = []
    lines.append("# Validation & generation report\n")
    lines.append("> **SYNTHETIC DATA.** Not clinical evidence. Rates in the "
                 "generator are invented for simulation; do not interpret "
                 "frequencies as real-world outcomes.\n")
    lines.append(f"- Generator: `{data_source}`  | seed: {seed}")
    lines.append(f"- Patients: **{n_pat}**  | transitions: **{n_rows}**")
    lines.append(f"- Trajectory length: min {L.min()}, median {L.median():.0f}, "
                 f"mean {L.mean():.2f}, max {L.max()}")
    lines.append(f"- Patients with length >= MIN_TIMESTEPS ({cfg.MIN_TIMESTEPS}): "
                 f"{(L >= cfg.MIN_TIMESTEPS).mean():.1%} "
                 "(shorter ones ended in a terminal state or death, which "
                 "override the minimum)")
    lines.append(f"- Split by patient: " + ", ".join(
        f"{k}={int((ps.split == k).sum())}" for k in ("train", "val", "test")))
    lines.append(f"- Matrix/support source: **{matrix_source}** split\n")
    lines.append("## Trajectory endings")
    for k, v in end_counts.items():
        lines.append(f"- {k}: {v} ({v / n_pat:.1%})")
    lines.append("\n## Terminal states (from `is_terminal`)")
    lines.append("- Terminal state indices: " +
                 str([s for s in range(cfg.N_STATES) if is_terminal(s)]))
    lines.append(f"- Terminal transitions: {int(df.is_terminal.sum())}")
    lines.append("- Death is NOT absorbing in the project's state model: "
                 "deaths that map to a non-terminal state end the trajectory "
                 "with `end_reason=death_nonterminal_state` and "
                 "`is_terminal=False`.\n")
    lines.append("## Coverage")
    lines.append(f"- State-action pairs: {status_counts}")
    lines.append(f"- Observed pairs: {status_counts.get('observed', 0)} / "
                 f"{cfg.N_STATES * cfg.N_ACTIONS}")
    lines.append(f"- States never visited at all (start/from/to): {never_visited}")
    lines.append(f"- States with zero outgoing transitions "
                 f"(terminal or unreachable): {unobserved_states}")
    lines.append(f"- Rare states (< {RARE_STATE_THRESHOLD} outgoing): {rare_states}")
    lines.append(f"- Rare pairs (< {RARE_PAIR_THRESHOLD}): "
                 f"{int(sup.rare_pair.sum())}\n")
    lines.append("## Validation checks")
    for name, ok, detail in checks:
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name}"
                     + (f" ({detail})" if detail else ""))
    lines.append(f"\n**Overall: {'ALL CHECKS PASSED' if all_ok else 'FAILURES PRESENT'}**\n")
    lines.append("## Assumptions")
    lines.append("- Treatment bookkeeping: every action > 0 (incl. Palliative) "
                 "appends to treatments_so_far; Observation does not. Verify "
                 "against your training scripts.")
    lines.append("- Stage progression, metastasis and death rates are invented "
                 "illustrative constants (see top of generate_dataset.py).")
    lines.append("- Any action is permitted from any non-terminal state in the "
                 "generator (behaviour policy is stage-dependent but nonzero "
                 "for all actions).")
    lines.append("- Dates are synthetic (2015-2020 diagnosis, 60-120 day intervals).")
    lines.append("- Probabilities are empirical frequencies from the "
                 f"`{matrix_source}` split only; unsupported pairs have no row "
                 "in transitions_netlogo.csv and NaN rows in the matrices.")
    lines.append("- The NetLogo CSV path/format could not be checked against "
                 "your loader; change --out-dir or rename files if needed.")
    with open(os.path.join(out_dir, "validation_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "data_dictionary.md"), "w") as f:
        f.write(DATA_DICTIONARY)
    with open(os.path.join(out_dir, "generation_metadata.json"), "w") as f:
        json.dump({"generator_version": GENERATOR_VERSION, "seed": seed,
                   "n_patients": n_pat, "n_transitions": n_rows,
                   "matrix_source": matrix_source, "synthetic": True,
                   "args": vars(args)}, f, indent=2)
    return checks, all_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-patients", type=int, default=max(cfg.N_PATIENTS, 25000))
    ap.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    ap.add_argument("--out-dir", default="dataset_out")
    ap.add_argument("--matrix-source", choices=["train", "all"], default="train")
    ap.add_argument("--allow-small", action="store_true",
                    help="permit fewer than 25000 patients (testing only)")
    args = ap.parse_args()

    if args.n_patients < 25000 and not args.allow_small:
        sys.exit("n-patients must be >= 25000 (use --allow-small to override).")

    df, ps, data_source = build_dataset(args.n_patients, args.seed)
    split_map = split_patients(ps.patient_id.tolist(), args.seed)
    checks, ok = write_all(args.out_dir, df, ps, split_map, data_source,
                           args.matrix_source, args.seed, args)

    print(f"Patients: {len(ps)}   Transitions: {len(df)}")
    for name, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\nOutputs written to: {os.path.abspath(args.out_dir)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
