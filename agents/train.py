"""
Training and Evaluation — Multi-Agent Treatment Decision Support

Architecture:
    DiagnosticAgent  -> D
    PrognosticAgent  -> P
    BiologicalAgent  -> B
    MultiAgentSystem -> assembles S = (D, P, U, B)
    TreatmentAgent   -> proposes actions using a trained Q-learning policy

Algorithms:
    q_learning
    double_q_learning

Usage:
    python agents/train.py --data dataset.csv --algorithm q_learning
    python agents/train.py --data dataset.csv --algorithm double_q_learning
    python agents/train.py --data dataset.csv --algorithm q_learning --episodes 20000
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import N_STATES, N_ACTIONS, ACTIONS, N_EPISODES
from markov_states import decode_state
from q_learnings.q_learning import QLearningAgent
from q_learnings.double_q_learning import DoubleQLearningAgent
from agents.multi_agent import MultiAgentSystem


REQUIRED_COLUMNS = [
    "patient_id",
    "timestep",
    "state_from",
    "action",
    "reward",
    "state_to",
    "is_terminal",
]


@contextlib.contextmanager
def quiet():
    """Suppress internal training output."""
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        yield


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    for column in ["state_from", "state_to", "action"]:
        values = pd.to_numeric(df[column], errors="raise")

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains non-finite values."
            )

        if not np.all(values == np.floor(values)):
            raise ValueError(
                f"{column} must contain integers."
            )

        df[column] = values.astype(int)

    if not df["state_from"].between(0, N_STATES - 1).all():
        raise ValueError("Invalid state_from index in dataset.")

    if not df["state_to"].between(0, N_STATES - 1).all():
        raise ValueError("Invalid state_to index in dataset.")

    if not df["action"].between(0, N_ACTIONS - 1).all():
        raise ValueError("Invalid action index in dataset.")

    if not np.isfinite(
        pd.to_numeric(df["reward"], errors="raise")
    ).all():
        raise ValueError("Reward contains non-finite values.")

    return df


def build_transition_matrices(df: pd.DataFrame):

    counts = np.zeros(
        (N_ACTIONS, N_STATES, N_STATES),
        dtype=np.int64,
    )

    actions = df["action"].to_numpy(dtype=int)
    states_from = df["state_from"].to_numpy(dtype=int)
    states_to = df["state_to"].to_numpy(dtype=int)

    np.add.at(
        counts,
        (actions, states_from, states_to),
        1,
    )

    totals = counts.sum(axis=2, keepdims=True)

    matrices = np.zeros(
        (N_ACTIONS, N_STATES, N_STATES),
        dtype=np.float64,
    )

    np.divide(
        counts,
        totals,
        out=matrices,
        where=totals > 0,
    )

    observed_pairs = int(np.count_nonzero(totals[:, :, 0]))
    total_pairs = N_ACTIONS * N_STATES
    unobserved_pairs = total_pairs - observed_pairs

    # Validate probabilities for observed state-action pairs.
    row_sums = matrices.sum(axis=2)
    observed_rows = totals[:, :, 0] > 0

    if not np.allclose(row_sums[observed_rows], 1.0):
        raise ValueError(
            "Transition probabilities do not sum to 1 "
            "for all observed state-action pairs."
        )

    if np.any(matrices < 0) or not np.isfinite(matrices).all():
        raise ValueError("Invalid transition probabilities.")

    print("\nTransition matrix summary")
    print(f"  Matrix shape: {matrices.shape}")
    print(f"  Observed state-action pairs: {observed_pairs}/{total_pairs}")
    print(f"  Unobserved state-action pairs: {unobserved_pairs}/{total_pairs}")
    print(f"  Recorded transitions counted: {int(counts.sum()):,}")

    return counts, matrices


def export_transition_matrices(
    counts: np.ndarray,
    matrices: np.ndarray,
    model_dir: Path,
    out: Path,
):

    model_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = model_dir / "transition_matrices.npz"

    np.savez_compressed(
        matrix_path,
        counts=counts,
        matrices=matrices,
    )

    # Long-format CSV: one row per action/from-state/to-state.
    action_ids, from_ids, to_ids = np.indices(
        (N_ACTIONS, N_STATES, N_STATES)
    )

    observed_counts = counts.reshape(-1)
    probabilities = matrices.reshape(-1)

    transition_table = pd.DataFrame({
        "action": action_ids.reshape(-1),
        "state_from": from_ids.reshape(-1),
        "state_to": to_ids.reshape(-1),
        "count": observed_counts,
        "probability": probabilities,
        "observed": observed_counts > 0,
    })

    netlogo_dir = out / "netlogo"
    netlogo_dir.mkdir(parents=True, exist_ok=True)

    csv_path = netlogo_dir / "transition_matrices.csv"
    transition_table.to_csv(csv_path, index=False)

    print(f"  NPZ matrices: {matrix_path.resolve()}")
    print(f"  Transition CSV: {csv_path.resolve()}")


def get_q_table(agent, algorithm: str) -> np.ndarray:
    """Return the effective action-value table used by the policy."""
    if algorithm == "q_learning":
        Q = np.asarray(agent.Q, dtype=float)
    else:
        Q = np.asarray(agent.get_combined_Q(), dtype=float)

    if Q.shape != (N_STATES, N_ACTIONS):
        raise ValueError(
            f"Expected Q-table shape {(N_STATES, N_ACTIONS)}, "
            f"got {Q.shape}."
        )

    if not np.isfinite(Q).all():
        raise ValueError(
            "The effective Q-table contains non-finite values."
        )

    return Q


def get_policy_table(agent, Q: np.ndarray) -> pd.DataFrame:
    """Return the policy table, falling back to the effective Q-table."""
    if hasattr(agent, "get_policy_table"):
        table = agent.get_policy_table()

        if isinstance(table, pd.DataFrame):
            return table

    policy = np.argmax(Q, axis=1)

    return pd.DataFrame({
        "state": np.arange(N_STATES),
        "best_action": policy,
        "action_name": [ACTIONS[a] for a in policy],
        "Q_value": Q[np.arange(N_STATES), policy],
    })


def export_policy_files(
    agent,
    Q: np.ndarray,
    algorithm: str,
    out: Path,
):
    """Export Q-values and greedy policy as CSV files for NetLogo."""
    export_dir = out / "netlogo"
    export_dir.mkdir(parents=True, exist_ok=True)

    states = [decode_state(s) for s in range(N_STATES)]

    # Export all state-action values.
    q_table = pd.DataFrame(
        Q,
        columns=[f"Q_action_{a}" for a in range(N_ACTIONS)],
    )
    q_table.insert(0, "state", np.arange(N_STATES))
    q_table["D"] = [s[0] for s in states]
    q_table["P"] = [s[1] for s in states]
    q_table["U"] = [s[2] for s in states]
    q_table["B"] = [s[3] for s in states]

    q_table = q_table[
        ["state", "D", "P", "U", "B"]
        + [f"Q_action_{a}" for a in range(N_ACTIONS)]
    ]

    q_path = export_dir / f"{algorithm}_Q_values.csv"
    q_table.to_csv(q_path, index=False)

    # Export the greedy action for every state.
    policy = np.argmax(Q, axis=1)

    policy_table = pd.DataFrame({
        "state": np.arange(N_STATES),
        "D": [s[0] for s in states],
        "P": [s[1] for s in states],
        "U": [s[2] for s in states],
        "B": [s[3] for s in states],
        "best_action": policy,
        "action_name": [ACTIONS[a] for a in policy],
        "Q_value": Q[np.arange(N_STATES), policy],
    })

    policy_path = export_dir / f"{algorithm}_policy.csv"
    policy_table.to_csv(policy_path, index=False)

    # Verify exported policy against the learned Q-table.
    loaded_policy = pd.read_csv(policy_path)
    loaded_q = pd.read_csv(q_path)

    if len(loaded_policy) != N_STATES:
        raise ValueError("Exported policy has an incorrect row count.")

    if len(loaded_q) != N_STATES:
        raise ValueError("Exported Q-table has an incorrect row count.")

    if not np.array_equal(
        loaded_policy["best_action"].to_numpy(dtype=int),
        np.argmax(Q, axis=1),
    ):
        raise ValueError("Exported policy does not match the Q-table.")

    print(f"\nNetLogo exports: {export_dir.resolve()}")
    print(f"  Q-values: {q_path.name}")
    print(f"  Policy:   {policy_path.name}")
    print(f"  States:   {N_STATES}")
    print(f"  Actions:  {N_ACTIONS}")


def get_history(agent, name: str) -> list:
    value = getattr(agent, name, [])
    return list(value) if value is not None else []


def train_agent(
    df: pd.DataFrame,
    algorithm: str,
    n_episodes: int,
    n_checkpoints: int,
):
    """Train the selected algorithm and record checkpoint metrics."""
    if algorithm == "q_learning":
        agent = QLearningAgent()
    else:
        agent = DoubleQLearningAgent()

    n_checkpoints = min(n_checkpoints, n_episodes)

    chunks = [
        len(chunk)
        for chunk in np.array_split(
            np.arange(n_episodes), n_checkpoints
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
                np.nan if previous_Q is None
                else float(np.abs(Q - previous_Q).max())
            ),
            "policy_changes": (
                np.nan if previous_policy is None
                else int(np.sum(policy != previous_policy))
            ),
        })

        previous_Q = Q.copy()
        previous_policy = policy.copy()

    return agent, pd.DataFrame(checkpoint_rows), training_seconds


def evaluate_logged_actions(Q: np.ndarray, df: pd.DataFrame) -> float:
    """Measure agreement between the greedy policy and logged actions."""
    states = df["state_from"].to_numpy(dtype=int)
    actions = df["action"].to_numpy(dtype=int)
    predicted = np.argmax(Q, axis=1)[states]

    return float(np.mean(predicted == actions))


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the multi-agent system."
    )

    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--algorithm",
        choices=["q_learning", "double_q_learning"],
        default="q_learning",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=N_EPISODES,
    )
    parser.add_argument("--checkpoints", type=int, default=50)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--out", default="results_training")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.episodes < 1 or args.checkpoints < 1 or args.samples < 1:
        parser.error("episodes, checkpoints and samples must be positive.")

    np.random.seed(args.seed)

    out = Path(args.out)
    model_dir = out / "models"
    table_dir = out / "tables"

    model_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate the dataset.
    df = load_dataset(args.data)
    n_patients = int(df["patient_id"].nunique())

    print(f"Algorithm: {args.algorithm}")
    print(f"Loaded {len(df):,} transitions.")
    print(f"Unique patients: {n_patients:,}")
    print(f"Training episodes: {args.episodes:,}")

    # 2. Train the selected learner.
    agent, checkpoints, train_seconds = train_agent(
        df=df,
        algorithm=args.algorithm,
        n_episodes=args.episodes,
        n_checkpoints=args.checkpoints,
    )

    checkpoints.to_csv(
        table_dir / "checkpoints.csv",
        index=False,
    )

    # 3. Extract and save the effective Q-table.
    Q = get_q_table(agent, args.algorithm)

    np.save(
        model_dir / f"{args.algorithm}_q_table.npy",
        Q,
    )

    if args.algorithm == "double_q_learning":
        np.save(model_dir / "q_table_QA.npy", agent.QA)
        np.save(model_dir / "q_table_QB.npy", agent.QB)

    get_policy_table(agent, Q).to_csv(
        table_dir / f"policy_{args.algorithm}.csv",
        index=False,
    )

    # 4. Export the Q-table and greedy policy for NetLogo.
    export_policy_files(agent, Q, args.algorithm, out)

    # 5. Save training histories.
    rewards = get_history(agent, "episode_rewards")
    lengths = get_history(agent, "episode_lengths")
    q_history = get_history(agent, "q_value_history")

    history_length = min(len(rewards), len(lengths), len(q_history))

    episodes_df = pd.DataFrame({
        "episode": np.arange(1, history_length + 1),
        "trajectory_return": rewards[:history_length],
        "trajectory_length": lengths[:history_length],
        "mean_Q": q_history[:history_length],
    })

    episodes_df.to_csv(
        table_dir / "episodes.csv",
        index=False,
    )

    # 6. Evaluate agreement with actions recorded in the dataset.
    logged_action_match = evaluate_logged_actions(Q, df)

    # 7. Evaluate the multi-agent system.
    mas = MultiAgentSystem(
        q_agent=agent,
        use_double=(args.algorithm == "double_q_learning"),
    )

    n_samples = min(args.samples, n_patients)

    mas_results = mas.evaluate_on_dataset(
        df,
        n_samples=n_samples,
        verbose=False,
    )

    mas_results.to_csv(
        table_dir / "mas_evaluation.csv",
        index=False,
    )

    # 8. Calculate summary metrics.
    mas_metrics = {
        "n_samples": int(len(mas_results)),
        "recorded_action_match_rate": (
            float(
                (mas_results["original_action"]
                 == mas_results["mas_action"]).mean()
            )
            if len(mas_results) else None
        ),
        "unanimous_consensus_rate": (
            float(mas_results["consensus"].mean())
            if len(mas_results) else None
        ),
        "q_proposal_accepted_rate": (
            float(mas_results["q_proposed_matches_final"].mean())
            if len(mas_results) else None
        ),
    }

    summary = {
        "algorithm": args.algorithm,
        "episodes_requested": args.episodes,
        "episodes_recorded": len(rewards),
        "dataset": {
            "path": args.data,
            "n_transitions": int(len(df)),
            "n_patients": n_patients,
        },
        "training": {
            "seconds": float(train_seconds),
            "final_epsilon": float(agent.epsilon),
            "mean_Q": float(Q.mean()),
            "max_Q": float(Q.max()),
            "mean_max_Q": float(Q.max(axis=1).mean()),
            "logged_action_match_rate": logged_action_match,
        },
        "mas_evaluation": mas_metrics,
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # 9. Generate report.
    report = f"""# Training and Multi-Agent System Evaluation

## Algorithm
- Selected algorithm: {args.algorithm}
- Episodes requested: {args.episodes:,}
- Episodes recorded: {len(rewards):,}
- Training time: {train_seconds:.2f} seconds

## Dataset
- Transitions: {len(df):,}
- Unique patients: {n_patients:,}

## Learned policy
- Mean Q-value: {Q.mean():.4f}
- Maximum Q-value: {Q.max():.4f}
- Mean maximum Q-value per state: {Q.max(axis=1).mean():.4f}
- Agreement with recorded actions: {100 * logged_action_match:.2f}%

## Multi-agent evaluation
- Patients evaluated: {len(mas_results)}
- MAS action matches recorded action: {
        f"{100 * mas_metrics['recorded_action_match_rate']:.2f}%"
        if mas_metrics["recorded_action_match_rate"] is not None
        else "N/A"
    }
- Unanimous consensus: {
        f"{100 * mas_metrics['unanimous_consensus_rate']:.2f}%"
        if mas_metrics["unanimous_consensus_rate"] is not None
        else "N/A"
    }
- Q-learning proposal accepted as final action: {
        f"{100 * mas_metrics['q_proposal_accepted_rate']:.2f}%"
        if mas_metrics["q_proposal_accepted_rate"] is not None
        else "N/A"
    }

## Output files
- models/{args.algorithm}_q_table.npy
- tables/policy_{args.algorithm}.csv
- tables/checkpoints.csv
- tables/episodes.csv
- tables/mas_evaluation.csv
- netlogo/{args.algorithm}_Q_values.csv
- netlogo/{args.algorithm}_policy.csv
- summary.json
- report.md

Action agreement and consensus measure model behavior, not clinical effectiveness.
This system is a research prototype and does not provide clinical treatment recommendations.
"""

    (out / "report.md").write_text(
        report,
        encoding="utf-8",
    )

    print("\nTraining and evaluation complete.")
    print(f"Algorithm: {args.algorithm}")
    print(f"Logged-action agreement: {100 * logged_action_match:.2f}%")
    # 9. Build and export transition matrices.
    transition_counts, transition_matrices = build_transition_matrices(df)

    export_transition_matrices(
        counts=transition_counts,
        matrices=transition_matrices,
        model_dir=model_dir,
        out=out,
    )
    print(f"Results saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
