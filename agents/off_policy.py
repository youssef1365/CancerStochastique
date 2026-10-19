"""
Patient-level splitting and off-policy evaluation (OPE) for tabular policies.

Everything that is *fitted* (behavior policy, reward/transition model) uses the
TRAIN patients only. Everything that is *scored* uses the TEST patients only.

Estimators (all use discounted return, gamma given by the caller):
    logged  : mean return of the clinicians' actual treatment (on-policy baseline)
    IS      : trajectory-wise importance sampling (unbiased, high variance)
    WIS     : weighted (self-normalised) importance sampling
    PDWIS   : per-decision weighted importance sampling
    DR      : per-decision doubly robust, using a tabular model fitted on train

The learned greedy policy is deterministic, which makes plain importance
sampling degenerate (almost every trajectory gets weight 0). It is therefore
evaluated as an epsilon-soft version of the greedy policy.

This module does not import config, so it is usable with any state/action size.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Patient-level split
# ----------------------------------------------------------------------------

def split_by_patient(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split rows so that every patient lands entirely in train or in test."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1).")

    ids = np.sort(df["patient_id"].unique())
    if len(ids) < 2:
        raise ValueError("At least two patients are needed to split.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    n_test = int(np.clip(round(len(ids) * test_size), 1, len(ids) - 1))

    is_test = df["patient_id"].isin(shuffled[:n_test])
    return (
        df[~is_test].reset_index(drop=True),
        df[is_test].reset_index(drop=True),
    )


# ----------------------------------------------------------------------------
# Trajectories
# ----------------------------------------------------------------------------

@dataclass
class Trajectories:
    """Padded patient trajectories, shape (n_patients, horizon)."""
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminals: np.ndarray
    mask: np.ndarray

    @property
    def n(self) -> int:
        return self.states.shape[0]

    @property
    def horizon(self) -> int:
        return self.states.shape[1]


def build_trajectories(df: pd.DataFrame) -> Trajectories:
    if df.empty:
        raise ValueError("Cannot build trajectories from an empty dataset.")

    d = df.sort_values(
        ["patient_id", "timestep"], kind="stable"
    ).reset_index(drop=True)

    grouped = d.groupby("patient_id", sort=False)
    traj_idx = grouped.ngroup().to_numpy()
    step_idx = grouped.cumcount().to_numpy()

    n = int(traj_idx.max()) + 1
    horizon = int(step_idx.max()) + 1

    def pad(column, dtype):
        arr = np.zeros((n, horizon), dtype=dtype)
        arr[traj_idx, step_idx] = d[column].to_numpy(dtype=dtype)
        return arr

    mask = np.zeros((n, horizon), dtype=bool)
    mask[traj_idx, step_idx] = True

    return Trajectories(
        states=pad("state_from", int),
        actions=pad("action", int),
        rewards=pad("reward", float),
        next_states=pad("state_to", int),
        terminals=pad("is_terminal", bool),
        mask=mask,
    )
# ----------------------------------------------------------------------------
# Behavior policy, target policy, tabular model
# ----------------------------------------------------------------------------

def state_action_counts(df: pd.DataFrame, n_states: int, n_actions: int):
    counts = np.zeros((n_states, n_actions), dtype=float)
    np.add.at(
        counts,
        (df["state_from"].to_numpy(int), df["action"].to_numpy(int)),
        1,
    )
    return counts


def behavior_policy(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Laplace-smoothed estimate of pi_b(a|s). alpha > 0 keeps it positive."""
    if alpha <= 0:
        raise ValueError("alpha must be positive so pi_b(a|s) > 0.")
    n_actions = counts.shape[1]
    return (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * n_actions)


def greedy_policy(Q: np.ndarray, counts: np.ndarray, min_count: int = 1):
    """
    Greedy action per state, restricted to actions observed >= min_count times
    in the training data. States with no supported action fall back to the raw
    argmax and are flagged as unsupported.

    Returns (policy, supported_state_mask).
    """
    support = counts >= min_count
    supported_state = support.any(axis=1)
    masked = np.where(support, Q, -np.inf)
    policy = np.where(supported_state, masked.argmax(axis=1), Q.argmax(axis=1))
    return policy.astype(int), supported_state


def soft_policy(policy, supported_state, n_actions: int, epsilon: float):
    """Epsilon-soft version of a deterministic policy (uniform if unsupported)."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1].")
    n_states = len(policy)
    probs = np.full((n_states, n_actions), epsilon / n_actions)
    probs[np.arange(n_states), policy] += 1.0 - epsilon
    probs[~supported_state] = 1.0 / n_actions
    return probs


def fit_tabular_model(df: pd.DataFrame, n_states: int, n_actions: int):
    """
    Empirical reward R[s,a] and NON-terminal transition mass P[s,a,s'].
    Rows of P sum to P(not terminal | s,a); terminal transitions carry no
    future value. Unobserved (s,a) fall back to pooled-over-actions statistics
    for that state, and to the global mean reward (with no future) otherwise.
    """
    s = df["state_from"].to_numpy(int)
    a = df["action"].to_numpy(int)
    s2 = df["state_to"].to_numpy(int)
    r = df["reward"].to_numpy(float)
    nonterminal = ~df["is_terminal"].astype(bool).to_numpy()

    n = np.zeros((n_states, n_actions))
    r_sum = np.zeros((n_states, n_actions))
    trans = np.zeros((n_states, n_actions, n_states))

    np.add.at(n, (s, a), 1)
    np.add.at(r_sum, (s, a), r)
    np.add.at(trans, (s[nonterminal], a[nonterminal], s2[nonterminal]), 1)

    n_s = n.sum(axis=1)
    r_s = r_sum.sum(axis=1)
    trans_s = trans.sum(axis=1)
    global_r = r_sum.sum() / n.sum()

    observed = n > 0
    state_seen = n_s > 0

    R = np.where(
        observed,
        r_sum / np.maximum(n, 1),
        np.where(state_seen[:, None], (r_s / np.maximum(n_s, 1))[:, None], global_r),
    )
    P = np.where(
        observed[..., None],
        trans / np.maximum(n, 1)[..., None],
        np.where(
            state_seen[:, None, None],
            (trans_s / np.maximum(n_s, 1)[:, None])[:, None, :],
            0.0,
        ),
    )
    return R, P


def evaluate_policy_model(R, P, pi, gamma: float, max_iter: int = 5000, tol=1e-10):
    """Iterative policy evaluation on the tabular model. Returns (Q, V)."""
    V = np.zeros(R.shape[0])
    for _ in range(max_iter):
        Q = R + gamma * (P @ V)
        V_new = (pi * Q).sum(axis=1)
        converged = np.abs(V_new - V).max() < tol
        V = V_new
        if converged:
            break
    return R + gamma * (P @ V), V


# ----------------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------------

def _estimates(cum, disc_r, dr_term, mask=None):
    """All point estimates from cumulative ratios and discounted rewards."""
    w = cum[:, -1]
    G = disc_r.sum(axis=1)
    w_sum = w.sum()

    num_t = (cum * disc_r).sum(axis=0)
    if mask is not None:
        den_t = (cum * mask).sum(axis=0)
    else:
        den_t = cum.sum(axis=0)
    per_step = np.divide(num_t, den_t, out=np.zeros_like(num_t), where=den_t > 0)

    return {
        "logged": G.mean(),
        "IS": (w * G).mean(),
        "WIS": (w * G).sum() / w_sum if w_sum > 0 else np.nan,
        "PDWIS": per_step.sum(),
        "DR": dr_term.sum(axis=1).mean(),
    }


def evaluate_target(
    name: str,
    traj: Trajectories,
    pi_e: np.ndarray,
    pi_b: np.ndarray,
    Q_hat: np.ndarray,
    V_hat: np.ndarray,
    gamma: float,
    n_bootstrap: int = 500,
    seed: int = 0,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Evaluate a target policy on held-out patient trajectories."""
    s = traj.states
    a = traj.actions
    r = traj.rewards
    s2 = traj.next_states
    terminals = traj.terminals
    mask = traj.mask

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")

    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1).")

    if n_bootstrap < 0:
        raise ValueError("n_bootstrap cannot be negative.")

    # Importance ratios for observed actions only.
    rho = np.ones_like(r, dtype=float)
    rows, cols = np.nonzero(mask)

    rho[rows, cols] = (
        pi_e[s[rows, cols], a[rows, cols]]
        / pi_b[s[rows, cols], a[rows, cols]]
    )

    cum = np.cumprod(rho, axis=1)
    cum_prev = np.concatenate(
        [np.ones((traj.n, 1)), cum[:, :-1]], axis=1
    )

    discount = gamma ** np.arange(traj.horizon)
    disc_r = r * discount * mask

    # Sequential doubly robust estimator:
    # V(s_0) + sum_t gamma^t * rho_0:t *
    # [r_t + gamma * V(s_{t+1}) - Q(s_t, a_t)].
    continuation = (~terminals).astype(float)

    td_error = (
        r
        + gamma * continuation * V_hat[s2]
        - Q_hat[s, a]
    )

    corrections = (
        discount[None, :]
        * cum
        * td_error
        * mask
    )

    dr_term = corrections
    dr_term[:, 0] += V_hat[s[:, 0]]

    point = _estimates(cum, disc_r, dr_term, mask)

    rng = np.random.default_rng(seed)
    boots = {k: np.full(n_bootstrap, np.nan) for k in point}

    for b in range(n_bootstrap):
        idx = rng.integers(0, traj.n, traj.n)
        estimates = _estimates(
            cum[idx],
            disc_r[idx],
            dr_term[idx],
            mask[idx],
        )
        for k, value in estimates.items():
            boots[k][b] = value

    w = cum[:, -1]
    w_sum = w.sum()
    ess = (
        float(w_sum ** 2 / np.sum(w ** 2))
        if w_sum > 0 else np.nan
    )
    max_share = (
        float(w.max() / w_sum)
        if w_sum > 0 else np.nan
    )

    lo_q = (1 - ci) / 2 * 100
    hi_q = (1 + ci) / 2 * 100
    results = []

    def add(label, estimate, samples):
        if n_bootstrap > 0 and np.isfinite(samples).any():
            lo, hi = np.nanpercentile(samples, [lo_q, hi_q])
        else:
            lo = hi = np.nan

        results.append({
            "policy": name,
            "estimator": label,
            "estimate": float(estimate),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "ess": ess,
            "max_weight_share": max_share,
            "n_patients": traj.n,
        })

    for key, value in point.items():
        add(key, value, boots[key])

    for key in ("IS", "WIS", "PDWIS", "DR"):
        add(
            f"{key}_minus_logged",
            point[key] - point["logged"],
            boots[key] - boots["logged"],
        )

    return pd.DataFrame(results)

# ----------------------------------------------------------------------------
# One-call driver
# ----------------------------------------------------------------------------

def run_ope(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    Q: np.ndarray,
    n_states: int,
    n_actions: int,
    gamma: float,
    epsilon: float = 0.1,
    alpha: float = 1.0,
    min_count: int = 1,
    n_bootstrap: int = 500,
    seed: int = 42,
):
    """
    Evaluate the greedy policy implied by Q on held-out patients.

    Returns (ope_table, diagnostics). Compared policies:
        learned         epsilon-soft greedy policy from Q (support-masked)
        behavior_clone  estimated clinician policy (sanity check)
        uniform_random  reference lower bound
    """
    counts = state_action_counts(train_df, n_states, n_actions)
    pi_b = behavior_policy(counts, alpha)

    policy, supported = greedy_policy(Q, counts, min_count)
    targets = {
        "learned": soft_policy(policy, supported, n_actions, epsilon),
        "behavior_clone": pi_b,
        "uniform_random": np.full((n_states, n_actions), 1.0 / n_actions),
    }

    R, P = fit_tabular_model(train_df, n_states, n_actions)
    traj = build_trajectories(test_df)

    frames = []
    for name, pi_e in targets.items():
        Q_hat, V_hat = evaluate_policy_model(R, P, pi_e, gamma)
        frames.append(evaluate_target(
            name, traj, pi_e, pi_b, Q_hat, V_hat, gamma,
            n_bootstrap=n_bootstrap, seed=seed,
        ))

    ope = pd.concat(frames, ignore_index=True)
    logged = ope[ope["estimator"] == "logged"].iloc[[0]].assign(policy="logged")
    ope = pd.concat([logged, ope[ope["estimator"] != "logged"]],
                    ignore_index=True)

    ts = test_df["state_from"].to_numpy(int)
    ta = test_df["action"].to_numpy(int)
    diagnostics = {
        "n_test_patients": int(traj.n),
        "n_test_transitions": int(len(test_df)),
        "gamma": float(gamma),
        "epsilon": float(epsilon),
        "min_count": int(min_count),
        "test_share_in_supported_state": float(supported[ts].mean()),
        "test_share_pairs_seen_in_train": float((counts[ts, ta] >= min_count).mean()),
        "test_agreement_greedy_vs_logged": float((policy[ts] == ta).mean()),
    }
    return ope, diagnostics