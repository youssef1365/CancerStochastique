
import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from config import (
    N_STATES, N_ACTIONS, ACTIONS,
    ALPHA, GAMMA, EPSILON_START, EPSILON_MIN,
    EPSILON_DECAY, N_EPISODES, RANDOM_SEED
)
from markov_states import decode_state, is_terminal, get_state_quality


class DoubleQLearningAgent:

    def __init__(
        self,
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        seed=RANDOM_SEED
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.RandomState(seed + 1)

        self.QA = np.zeros((n_states, n_actions))
        self.QB = np.zeros((n_states, n_actions))

        self.valid_actions = np.zeros(
            (n_states, n_actions), dtype=bool
        )

        self.episode_rewards = []
        self.episode_lengths = []
        self.q_value_history = []

    def build_valid_actions(self, df):
        self.valid_actions.fill(False)

        pairs = df[["state_from", "action"]].drop_duplicates()

        for row in pairs.itertuples(index=False):
            state = int(row.state_from)
            action = int(row.action)

            if (
                0 <= state < self.n_states
                and 0 <= action < self.n_actions
            ):
                self.valid_actions[state, action] = True

        print(
            "Supported state-action pairs:",
            int(self.valid_actions.sum())
        )
        print(
            "States with no supported actions:",
            int((~self.valid_actions.any(axis=1)).sum())
        )

    def select_action(self, state: int, greedy: bool = False) -> int:
        valid = np.flatnonzero(self.valid_actions[state])

        if len(valid) == 0:
            return -1

        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid))

        combined = self.QA[state, valid] + self.QB[state, valid]
        return int(valid[np.argmax(combined)])

    def update(
        self, state, action, reward, next_state, done
    ):
        if not self.valid_actions[state, action]:
            return 0.0

        next_valid = np.flatnonzero(
            self.valid_actions[next_state]
        )

        if self.rng.random() < 0.5:
            # Update QA; QA selects, QB evaluates.
            if done or len(next_valid) == 0:
                target = reward
            else:
                best_action = next_valid[
                    np.argmax(self.QA[next_state, next_valid])
                ]
                target = (
                    reward
                    + self.gamma * self.QB[next_state, best_action]
                )

            td_error = target - self.QA[state, action]
            self.QA[state, action] += self.alpha * td_error

        else:
            # Update QB; QB selects, QA evaluates.
            if done or len(next_valid) == 0:
                target = reward
            else:
                best_action = next_valid[
                    np.argmax(self.QB[next_state, next_valid])
                ]
                target = (
                    reward
                    + self.gamma * self.QA[next_state, best_action]
                )

            td_error = target - self.QB[state, action]
            self.QB[state, action] += self.alpha * td_error

        return td_error

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def _prepare_trajectories(self, df: pd.DataFrame):
        """Pre-extract structured patient trajectories for fast training."""
        trajectories = []
        for _, group in df.groupby("patient_id", sort=False):
            sorted_group = group.sort_values("timestep")
            traj = []
            n_rows = len(sorted_group)
            for idx, row in enumerate(sorted_group.itertuples(index=False)):
                state = int(row.state_from)
                action = int(row.action)
                reward = float(row.reward)
                next_state = int(row.state_to)
                # True if marked terminal or final recorded transition of the patient
                done = bool(row.is_terminal) or (idx == n_rows - 1)
                traj.append((state, action, reward, next_state, done))
            if traj:
                trajectories.append(traj)
        return trajectories

    def train_from_dataset(
        self, df: pd.DataFrame, n_episodes: int = N_EPISODES, trajectories=None
    ):
        if not self.valid_actions.any():
            self.build_valid_actions(df)

        if trajectories is None:
            if not hasattr(self, "_cached_trajectories") or self._cached_trajectories is None:
                self._cached_trajectories = self._prepare_trajectories(df)
            trajectories = self._cached_trajectories

        if not trajectories:
            raise ValueError("Training dataset contains no valid patient trajectories.")

        n_traj = len(trajectories)
        for _ in range(n_episodes):
            traj = trajectories[self.rng.choice(n_traj)]
            episode_reward = 0.0
            steps = 0

            for state, action, reward, next_state, done in traj:
                self.update(state, action, reward, next_state, done)
                episode_reward += reward
                steps += 1
                if done:
                    break

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            self.q_value_history.append(
                float(np.mean(self.get_combined_Q()))
            )
            self.decay_epsilon()

    def get_combined_Q(self) -> np.ndarray:
        return (self.QA + self.QB) / 2

    def get_policy(self) -> np.ndarray:
        combined = self.get_combined_Q()
        policy = np.full(self.n_states, -1, dtype=int)

        for state in range(self.n_states):
            valid = np.flatnonzero(self.valid_actions[state])

            if len(valid) > 0:
                policy[state] = valid[
                    np.argmax(combined[state, valid])
                ]

        return policy

    def get_policy_table(self) -> pd.DataFrame:
        policy = self.get_policy()
        combined = self.get_combined_Q()
        rows = []

        for state in range(self.n_states):
            D, P, U, B = decode_state(state)
            action = int(policy[state])

            rows.append({
                "state": state,
                "D": D,
                "P": P,
                "U": U,
                "B": B,
                "best_action": action,
                "action_name": (
                    ACTIONS[action] if action >= 0
                    else "NO_SUPPORTED_ACTION"
                ),
                "Q_value": (
                    combined[state, action] if action >= 0
                    else np.nan
                ),
                "QA_value": (
                    self.QA[state, action] if action >= 0
                    else np.nan
                ),
                "QB_value": (
                    self.QB[state, action] if action >= 0
                    else np.nan
                ),
                "is_terminal": is_terminal(state),
                "state_quality": get_state_quality(state)
            })

        return pd.DataFrame(rows)
