import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import N_STATES, N_ACTIONS, ACTIONS, N_EPISODES
from markov_states import decode_state
from q_learnings.q_learning import QLearningAgent
from q_learnings.double_q_learning import DoubleQLearningAgent
from agents.multi_agent import MultiAgentSystem
from off_policy import (
    split_by_patient,
    state_action_counts,
    greedy_policy,
    run_ope,
)


REQUIRED_COLUMNS = [
    "patient_id",
    "timestep",
    "state_from",
    "action",
    "reward",
    "state_to",
    "is_terminal",
]

DEFAULT_GAMMA = 0.95


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def load_dataset(path):
    df = pd.read_csv(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df.empty:
        raise ValueError("The dataset is empty.")

    for column in ("state_from", "state_to", "action", "timestep"):
        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="raise")

        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite values.")

        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain integers.")

        df[column] = values.astype(int)

    for column in ("state_from", "state_to"):
        if not df[column].between(0, N_STATES - 1).all():
            raise ValueError(f"Invalid {column} index in dataset.")

    if not df["action"].between(0, N_ACTIONS - 1).all():
        raise ValueError("Invalid action index in dataset.")

    df["reward"] = pd.to_numeric(df["reward"], errors="raise")
    if not np.isfinite(df["reward"]).all():
        raise ValueError("Reward contains non-finite values.")

    terminal = df["is_terminal"]

    if terminal.dtype == bool:
        df["is_terminal"] = terminal.astype(int)
    else:
        terminal = pd.to_numeric(terminal, errors="raise")
        if not terminal.isin([0, 1]).all():
            raise ValueError("is_terminal must contain only 0 or 1.")
        df["is_terminal"] = terminal.astype(int)

    if df["patient_id"].isna().any():
        raise ValueError("patient_id contains missing values.")

    df = df.sort_values(
        ["patient_id", "timestep"]
    ).reset_index(drop=True)

    return df


def build_transition_matrices(df):
    counts = np.zeros(
        (N_ACTIONS, N_STATES, N_STATES),
        dtype=np.int64
    )

    actions = df["action"].to_numpy(dtype=int)
    states_from = df["state_from"].to_numpy(dtype=int)
    states_to = df["state_to"].to_numpy(dtype=int)

    np.add.at(counts, (actions, states_from, states_to), 1)

    totals = counts.sum(axis=2, keepdims=True)
    matrices = np.zeros(
        (N_ACTIONS, N_STATES, N_STATES),
        dtype=np.float64
    )

    np.divide(
        counts,
        totals,
        out=matrices,
        where=totals > 0
    )

    observed_rows = totals[:, :, 0] > 0
    observed_pairs = int(observed_rows.sum())
    total_pairs = N_ACTIONS * N_STATES

    row_sums = matrices.sum(axis=2)
    if not np.allclose(row_sums[observed_rows], 1.0):
        raise ValueError(
            "Transition probabilities do not sum to 1 "
            "for every observed state-action pair."
        )

    if not np.isfinite(matrices).all() or np.any(matrices < 0):
        raise ValueError("Invalid transition probabilities.")

    print("\nTransition matrix summary (training patients only)")
    print(f"  Matrix shape: {matrices.shape}")
    print(f"  Observed state-action pairs: {observed_pairs}/{total_pairs}")
    print(f"  Unobserved state-action pairs: {total_pairs - observed_pairs}")
    print(f"  Recorded transitions: {int(counts.sum()):,}")

    return counts, matrices


def export_transition_matrices(counts, matrices, model_dir, out):
    model_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = model_dir / "transition_matrices.npz"
    np.savez_compressed(
        matrix_path,
        counts=counts,
        matrices=matrices
    )

    action_ids, from_ids, to_ids = np.indices(
        (N_ACTIONS, N_STATES, N_STATES)
    )

    transition_table = pd.DataFrame({
        "action": action_ids.reshape(-1),
        "state_from": from_ids.reshape(-1),
        "state_to": to_ids.reshape(-1),
        "count": counts.reshape(-1),
        "probability": matrices.reshape(-1),
        "observed": counts.reshape(-1) > 0,
    })

    netlogo_dir = out / "netlogo"
    netlogo_dir.mkdir(parents=True, exist_ok=True)

    csv_path = netlogo_dir / "transition_matrices.csv"
    transition_table.to_csv(csv_path, index=False)

    print(f"  NPZ matrices: {matrix_path.resolve()}")
    print(f"  Transition CSV: {csv_path.resolve()}")


def get_q_table(agent, algorithm):
    if algorithm == "q_learning":
        Q = np.asarray(agent.Q, dtype=float)
    else:
        Q = np.asarray(agent.get_combined_Q(), dtype=float)

    expected_shape = (N_STATES, N_ACTIONS)

    if Q.shape != expected_shape:
        raise ValueError(
            f"Expected Q-table shape {expected_shape}, got {Q.shape}."
        )

    if not np.isfinite(Q).all():
        raise ValueError("The Q-table contains non-finite values.")

    return Q


def export_policy_files(Q, algorithm, out, sa_counts, min_count=1):
    export_dir = out / "netlogo"
    export_dir.mkdir(parents=True, exist_ok=True)

    states = [decode_state(state) for state in range(N_STATES)]
    policy, supported = greedy_policy(Q, sa_counts, min_count)

    policy = np.asarray(policy, dtype=int)
    supported = np.asarray(supported, dtype=bool)

    if policy.shape != (N_STATES,):
        raise ValueError("The exported policy must contain one action per state.")

    if not np.all((policy >= 0) & (policy < N_ACTIONS)):
        raise ValueError("The policy contains an invalid action index.")

    q_table = pd.DataFrame(
        Q,
        columns=[f"Q_action_{action}" for action in range(N_ACTIONS)]
    )
    q_table.insert(0, "state", np.arange(N_STATES))
    q_table["D"] = [state[0] for state in states]
    q_table["P"] = [state[1] for state in states]
    q_table["U"] = [state[2] for state in states]
    q_table["B"] = [state[3] for state in states]

    q_columns = [f"Q_action_{action}" for action in range(N_ACTIONS)]
    q_table = q_table[["state", "D", "P", "U", "B"] + q_columns]

    q_path = export_dir / f"{algorithm}_Q_values.csv"
    q_table.to_csv(q_path, index=False)

    state_visits = sa_counts.sum(axis=1).astype(int)

    policy_table = pd.DataFrame({
        "state": np.arange(N_STATES),
        "D": [state[0] for state in states],
        "P": [state[1] for state in states],
        "U": [state[2] for state in states],
        "B": [state[3] for state in states],
        "best_action": policy,
        "action_name": [ACTIONS[action] for action in policy],
        "Q_value": Q[np.arange(N_STATES), policy],
        "n_visits": state_visits,
        "supported": supported.astype(int),
    })

    policy_path = export_dir / f"{algorithm}_policy.csv"
    policy_table.to_csv(policy_path, index=False)

    loaded_policy = pd.read_csv(policy_path)
    loaded_q = pd.read_csv(q_path)

    if len(loaded_policy) != N_STATES or len(loaded_q) != N_STATES:
        raise ValueError("An exported policy or Q-table has an incorrect row count.")

    if not np.array_equal(
        loaded_policy["best_action"].to_numpy(dtype=int),
        policy
    ):
        raise ValueError("The exported policy does not match the computed policy.")

    print(f"\nNetLogo exports: {export_dir.resolve()}")
    print(f"  Q-values: {q_path.name}")
    print(f"  Policy:   {policy_path.name}")
    print(f"  Supported states: {int(supported.sum())}/{N_STATES}")
    print(f"  Actions: {N_ACTIONS}")

    return policy_table


def get_history(agent, name):
    values = getattr(agent, name, [])
    return list(values) if values is not None else []


def train_agent(df, algorithm, n_episodes, n_checkpoints):
    if algorithm == "q_learning":
        agent = QLearningAgent()
    else:
        agent = DoubleQLearningAgent()

    n_checkpoints = min(n_checkpoints, n_episodes)

    chunks = [
        len(chunk)
        for chunk in np.array_split(
            np.arange(n_episodes),
            n_checkpoints
        )
        if len(chunk) > 0
    ]

    checkpoint_rows = []
    episodes_done = 0
    previous_Q = None
    previous_policy = None
    training_seconds = 0.0

    for chunk_size in tqdm(chunks, desc=f"Training {algorithm}"):
        start = time.perf_counter()

        with quiet():
            agent.train_from_dataset(df, chunk_size)

        training_seconds += time.perf_counter() - start
        episodes_done += chunk_size

        Q = get_q_table(agent, algorithm)
        policy = np.argmax(Q, axis=1)

        checkpoint_rows.append({
            "algorithm": algorithm,
            "episodes": episodes_done,
            "mean_Q": float(Q.mean()),
            "max_Q": float(Q.max()),
            "mean_max_Q": float(Q.max(axis=1).mean()),
            "max_abs_delta_Q": (
                None if previous_Q is None
                else float(np.abs(Q - previous_Q).max())
            ),
            "policy_changes": (
                None if previous_policy is None
                else int(np.sum(policy != previous_policy))
            ),
        })

        previous_Q = Q.copy()
        previous_policy = policy.copy()

    return agent, pd.DataFrame(checkpoint_rows), training_seconds


def logged_action_agreement(policy, df):
    states = df["state_from"].to_numpy(dtype=int)
    actions = df["action"].to_numpy(dtype=int)
    return float(np.mean(policy[states] == actions))


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [clean_for_json(value) for value in obj]

    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    return obj


def ope_report_lines(ope, diagnostics):
    def fmt(row):
        return (
            f"{row.estimate:.3f} "
            f"[{row.ci_low:.3f}, {row.ci_high:.3f}]"
        )

    def get(policy_name, estimator):
        rows = ope[
            (ope["policy"] == policy_name)
            & (ope["estimator"] == estimator)
        ]
        if rows.empty:
            raise ValueError(
                f"Missing OPE result for {policy_name}/{estimator}."
            )
        return rows.iloc[0]

    logged = get("logged", "logged")

    lines = [
        f"Held-out patients: {diagnostics['n_test_patients']} "
        f"({diagnostics['n_test_transitions']:,} transitions), "
        f"gamma = {diagnostics['gamma']}, "
        f"target epsilon = {diagnostics['epsilon']}.",
        "",
        "| Policy | WIS | PDWIS | DR | ESS |",
        "|---|---|---|---|---|",
        f"| logged | {fmt(logged)} | - | - | - |",
    ]

    for policy_name in ("learned", "behavior_clone", "uniform_random"):
        wis = get(policy_name, "WIS")
        pdwis = get(policy_name, "PDWIS")
        dr = get(policy_name, "DR")

        lines.append(
            f"| {policy_name} | {fmt(wis)} | {fmt(pdwis)} | "
            f"{fmt(dr)} | {wis.ess:.1f} |"
        )

    lines.extend(["", "Learned minus logged:"])

    for estimator in ("WIS_minus_logged", "DR_minus_logged"):
        rows = ope[
            (ope["policy"] == "learned")
            & (ope["estimator"] == estimator)
        ]
        if rows.empty:
            continue

        row = rows.iloc[0]
        verdict = (
            "CI excludes zero"
            if row.ci_low > 0 or row.ci_high < 0
            else "CI includes zero"
        )
        lines.append(f"- {estimator}: {fmt(row)} — {verdict}")

    lines.extend([
        "",
        "Held-out data coverage:",
        f"- Transitions in states with training support: "
        f"{100 * diagnostics['test_share_in_supported_state']:.1f}%",
        f"- State-action pairs observed in training: "
        f"{100 * diagnostics['test_share_pairs_seen_in_train']:.1f}%",
        f"- Greedy action matches logged action: "
        f"{100 * diagnostics['test_agreement_greedy_vs_logged']:.1f}%",
    ])

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the multi-agent system."
    )

    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--algorithm",
        choices=["q_learning", "double_q_learning"],
        default="q_learning"
    )
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--checkpoints", type=int, default=50)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--out", default="results_training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--eval-epsilon", type=float, default=0.1)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=500)

    args = parser.parse_args()

    if args.episodes < 1 or args.checkpoints < 1 or args.samples < 1:
        parser.error("--episodes, --checkpoints and --samples must be positive.")

    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1.")

    if not 0 < args.eval_epsilon <= 1:
        parser.error("--eval-epsilon must be in (0, 1].")

    if args.min_count < 1 or args.bootstrap < 0:
        parser.error("--min-count must be >= 1 and --bootstrap >= 0.")

    np.random.seed(args.seed)

    out = Path(args.out)
    model_dir = out / "models"
    table_dir = out / "tables"

    model_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and split by patient.
    df = load_dataset(args.data)
    train_df, test_df = split_by_patient(
        df,
        args.test_size,
        args.seed
    )

    if train_df.empty or test_df.empty:
        raise ValueError("The patient-level split produced an empty partition.")

    train_patients = sorted(train_df["patient_id"].unique().tolist())
    test_patients = sorted(test_df["patient_id"].unique().tolist())

    if set(train_patients) & set(test_patients):
        raise ValueError("Patient leakage detected between train and test.")

    split_info = {
        "seed": args.seed,
        "test_size": args.test_size,
        "train_patients": train_patients,
        "test_patients": test_patients,
    }

    (out / "split.json").write_text(
        json.dumps(clean_for_json(split_info), indent=2),
        encoding="utf-8"
    )

    n_train = len(train_patients)
    n_test = len(test_patients)

    print(f"Algorithm: {args.algorithm}")
    print(
        f"Loaded {len(df):,} transitions "
        f"from {df['patient_id'].nunique():,} patients."
    )
    print(f"Train: {n_train:,} patients / {len(train_df):,} transitions")
    print(f"Test:  {n_test:,} patients / {len(test_df):,} transitions")
    print(f"Training episodes: {args.episodes:,}")

    # 2. Train only on training patients.
    agent, checkpoints, train_seconds = train_agent(
        df=train_df,
        algorithm=args.algorithm,
        n_episodes=args.episodes,
        n_checkpoints=args.checkpoints
    )

    checkpoints.to_csv(table_dir / "checkpoints.csv", index=False)

    # 3. Obtain the learned Q-table.
    Q = get_q_table(agent, args.algorithm)
    np.save(model_dir / f"{args.algorithm}_q_table.npy", Q)

    if args.algorithm == "double_q_learning":
        np.save(model_dir / "q_table_QA.npy", agent.QA)
        np.save(model_dir / "q_table_QB.npy", agent.QB)

    # 4. Calculate action support using TRAIN data only.
    sa_counts = state_action_counts(
        train_df,
        N_STATES,
        N_ACTIONS
    )

    valid_actions = np.asarray(
        sa_counts >= args.min_count,
        dtype=bool
    )

    if valid_actions.shape != (N_STATES, N_ACTIONS):
        raise ValueError(
            f"Expected action support shape {(N_STATES, N_ACTIONS)}, "
            f"got {valid_actions.shape}."
        )

    policy, supported = greedy_policy(
        Q,
        sa_counts,
        args.min_count
    )

    policy = np.asarray(policy, dtype=int)
    supported = np.asarray(supported, dtype=bool)

    if not np.all((policy >= 0) & (policy < N_ACTIONS)):
        raise ValueError("The learned policy contains invalid action indices.")

    np.save(model_dir / "valid_actions.npy", valid_actions)

    print(
        f"\nAction support: {int(valid_actions.sum())}/"
        f"{N_STATES * N_ACTIONS} state-action pairs meet "
        f"min_count={args.min_count}."
    )

    # 5. Export policy and Q-values.
    policy_table = export_policy_files(
        Q=Q,
        algorithm=args.algorithm,
        out=out,
        sa_counts=sa_counts,
        min_count=args.min_count
    )

    policy_table.to_csv(
        table_dir / f"policy_{args.algorithm}.csv",
        index=False
    )

    # 6. Save training histories.
    rewards = get_history(agent, "episode_rewards")
    lengths = get_history(agent, "episode_lengths")
    q_history = get_history(agent, "q_value_history")

    if len(rewards) != args.episodes:
        print(
            f"Recorded {len(rewards):,} episodes; "
            f"{args.episodes:,} were requested."
        )

    history_length = min(len(rewards), len(lengths), len(q_history))

    pd.DataFrame({
        "episode": np.arange(1, history_length + 1),
        "trajectory_return": rewards[:history_length],
        "trajectory_length": lengths[:history_length],
        "mean_Q": q_history[:history_length],
    }).to_csv(table_dir / "episodes.csv", index=False)

    # 7. Logged-action agreement.
    match_train = logged_action_agreement(policy, train_df)
    match_test = logged_action_agreement(policy, test_df)

    # 8. Off-policy evaluation on held-out patients.
    gamma = (
        args.gamma
        if args.gamma is not None
        else float(getattr(agent, "gamma", DEFAULT_GAMMA))
    )

    print(
        f"\nOff-policy evaluation: gamma={gamma}, "
        f"epsilon={args.eval_epsilon}, bootstrap={args.bootstrap}"
    )

    ope, ope_diag = run_ope(
        train_df=train_df,
        test_df=test_df,
        Q=Q,
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        gamma=gamma,
        epsilon=args.eval_epsilon,
        min_count=args.min_count,
        n_bootstrap=args.bootstrap,
        seed=args.seed
    )

    ope.to_csv(table_dir / "ope.csv", index=False)

    # 9. MAS evaluation using the same training-derived action mask.
    mas = MultiAgentSystem(
        q_agent=agent,
        use_double=(args.algorithm == "double_q_learning"),
        valid_actions=valid_actions
    )

    mas_results = mas.evaluate_on_dataset(
        test_df,
        n_samples=min(args.samples, n_test),
        verbose=False
    )

    mas_results.to_csv(table_dir / "mas_evaluation.csv", index=False)

    if len(mas_results):
        action_match = float(
            (
                mas_results["original_action"]
                == mas_results["mas_action"]
            ).mean()
        )

        unanimous = float(
            np.isclose(mas_results["consensus"], 1.0).mean()
        )

        proposal_accepted = float(
            mas_results["q_proposed_matches_final"].mean()
        )
    else:
        action_match = None
        unanimous = None
        proposal_accepted = None

    mas_metrics = {
        "n_samples": int(len(mas_results)),
        "recorded_action_match_rate": action_match,
        "unanimous_consensus_rate": unanimous,
        "q_proposal_accepted_rate": proposal_accepted,
    }

    # 10. Transition matrices from TRAIN patients only.
    transition_counts, transition_matrices = build_transition_matrices(
        train_df
    )

    export_transition_matrices(
        counts=transition_counts,
        matrices=transition_matrices,
        model_dir=model_dir,
        out=out
    )

    # 11. Summary.
    summary = {
        "algorithm": args.algorithm,
        "episodes_requested": args.episodes,
        "episodes_recorded": len(rewards),
        "dataset": {
            "path": args.data,
            "n_transitions": int(len(df)),
            "n_patients": int(df["patient_id"].nunique()),
            "n_train_patients": n_train,
            "n_test_patients": n_test,
        },
        "training": {
            "seconds": float(train_seconds),
            "final_epsilon": float(agent.epsilon),
            "mean_Q": float(Q.mean()),
            "max_Q": float(Q.max()),
            "mean_max_Q": float(Q.max(axis=1).mean()),
            "logged_action_match_train": match_train,
            "logged_action_match_test": match_test,
            "min_count": args.min_count,
            "supported_state_action_pairs": int(valid_actions.sum()),
        },
        "off_policy_evaluation": {
            "diagnostics": ope_diag,
            "estimates": ope.to_dict("records"),
        },
        "mas_evaluation_test_patients": mas_metrics,
    }

    (out / "summary.json").write_text(
        json.dumps(clean_for_json(summary), indent=2, allow_nan=False),
        encoding="utf-8"
    )

    # 12. Markdown report.
    def pct(value):
        return f"{100 * value:.2f}%" if value is not None else "N/A"

    report_lines = [
        "# Training and Multi-Agent System Evaluation",
        "",
        "## Algorithm",
        f"- Algorithm: {args.algorithm}",
        f"- Episodes requested: {args.episodes:,}",
        f"- Episodes recorded: {len(rewards):,}",
        f"- Training time: {train_seconds:.2f} seconds",
        "",
        "## Dataset split",
        f"- Total transitions: {len(df):,}",
        f"- Training patients: {n_train:,}",
        f"- Test patients: {n_test:,}",
        f"- Random seed: {args.seed}",
        "",
        "## Learned policy",
        f"- Mean Q-value: {Q.mean():.4f}",
        f"- Maximum Q-value: {Q.max():.4f}",
        f"- Mean maximum Q-value per state: {Q.max(axis=1).mean():.4f}",
        f"- Training logged-action agreement: {100 * match_train:.2f}%",
        f"- Test logged-action agreement: {100 * match_test:.2f}%",
        f"- Minimum action support count: {args.min_count}",
        f"- Supported state-action pairs: {int(valid_actions.sum())}",
        "",
        "## Off-policy evaluation",
        "",
        *ope_report_lines(ope, ope_diag),
        "",
        "## Multi-agent evaluation",
        f"- Patients evaluated: {len(mas_results)}",
        f"- MAS action matches recorded action: {pct(action_match)}",
        f"- Unanimous agent consensus: {pct(unanimous)}",
        f"- Q-learning proposal retained: {pct(proposal_accepted)}",
        "",
        "## Output files",
        f"- models/{args.algorithm}_q_table.npy",
        "- models/valid_actions.npy",
        "- models/transition_matrices.npz",
        "- models/q_table_QA.npy and models/q_table_QB.npy (Double Q-Learning)",
        f"- netlogo/{args.algorithm}_Q_values.csv",
        f"- netlogo/{args.algorithm}_policy.csv",
        "- netlogo/transition_matrices.csv",
        f"- tables/policy_{args.algorithm}.csv",
        "- tables/checkpoints.csv",
        "- tables/episodes.csv",
        "- tables/ope.csv",
        "- tables/mas_evaluation.csv",
        "- split.json",
        "- summary.json",
        "- report.md",
        "",
    ]

    (out / "report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print("\nTraining and evaluation complete.")
    print(f"Algorithm: {args.algorithm}")
    print(
        f"Logged-action agreement: train {100 * match_train:.2f}% / "
        f"test {100 * match_test:.2f}%"
    )
    print(f"MAS action agreement: {pct(action_match)}")
    print(f"Unanimous consensus: {pct(unanimous)}")
    print(f"Results saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
