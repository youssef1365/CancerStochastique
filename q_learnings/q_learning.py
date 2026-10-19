import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    N_STATES, N_ACTIONS, ACTIONS,
    ALPHA, GAMMA, EPSILON_START, EPSILON_MIN,
    EPSILON_DECAY, N_EPISODES, RANDOM_SEED
)
from markov_states import decode_state

from markov_states import is_terminal

TERMINAL_STATES = {
    state for state in range(N_STATES)
    if is_terminal(state)
}

REQUIRED_COLUMNS = {
    "patient_id", "timestep", "state_from",
    "action", "reward", "state_to", "is_terminal"
}


class QLearningAgent:
    def __init__(
        self,
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        seed=RANDOM_SEED,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros((n_states, n_actions), dtype=float)
        self.valid_actions = np.zeros((n_states, n_actions), dtype=bool)

        self.episode_rewards = []
        self.episode_lengths = []
        self.q_value_history = []

    def _prepare_dataset(self, df):
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing columns: {sorted(missing)}")
        if df.empty:
            raise ValueError("Cannot train on an empty dataset.")

        data = df.copy()
        numeric = ["timestep", "state_from", "action", "reward", "state_to"]
        for col in numeric:
            data[col] = pd.to_numeric(data[col], errors="raise")

        for col in ["state_from", "state_to"]:
            if not data[col].between(0, self.n_states - 1).all():
                raise ValueError(f"{col} must be between 0 and {self.n_states - 1}.")

        if not data["action"].between(0, self.n_actions - 1).all():
            raise ValueError(f"action must be between 0 and {self.n_actions - 1}.")

        data["is_terminal"] = (
            data["is_terminal"].astype(str).str.strip().str.lower()
            .map({"true": True, "1": True, "yes": True,
                  "false": False, "0": False, "no": False})
        )
        if data["is_terminal"].isna().any():
            raise ValueError("is_terminal contains invalid boolean values.")

        return data.sort_values(
            ["patient_id", "timestep"], kind="stable"
        ).reset_index(drop=True)

    def _build_valid_actions(self, df):
        mask = np.zeros_like(self.valid_actions)
        pairs = df[["state_from", "action"]].drop_duplicates()

        for state, action in pairs.itertuples(index=False, name=None):
            state, action = int(state), int(action)
            if state not in TERMINAL_STATES:
                mask[state, action] = True

        self.valid_actions = mask

    def _valid_for_state(self, state):
        return np.flatnonzero(self.valid_actions[int(state)])

    def select_action(self, state, greedy=False):
        state = int(state)
        if state in TERMINAL_STATES:
            return -1

        actions = self._valid_for_state(state)
        if len(actions) == 0:
            return -1

        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(actions))

        return int(actions[np.argmax(self.Q[state, actions])])

    def update(self, state, action, reward, next_state, done):
        state, action, next_state = int(state), int(action), int(next_state)

        if not (0 <= state < self.n_states and
                0 <= next_state < self.n_states and
                0 <= action < self.n_actions):
            raise ValueError("State or action is outside the configured range.")

        if not self.valid_actions[state, action]:
            raise ValueError(
                f"Unobserved state/action pair: state={state}, action={action}"
            )

        target = float(reward)
        next_actions = self._valid_for_state(next_state)

        if not done and next_state not in TERMINAL_STATES and len(next_actions):
            target += self.gamma * np.max(self.Q[next_state, next_actions])

        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min, self.epsilon * self.epsilon_decay
        )

    def train_from_dataset(self, df, n_episodes=N_EPISODES, trajectories=None):
        data = self._prepare_dataset(df)
        self._build_valid_actions(data)

        groups = [
            group.sort_values("timestep")
            for _, group in data.groupby("patient_id", sort=False)
        ]
        if not groups:
            raise ValueError("No patient trajectories found.")
        if n_episodes < 1:
            raise ValueError("n_episodes must be at least 1.")

        for _ in tqdm(range(n_episodes), desc="Q-Learning Training"):
            trajectory = groups[int(self.rng.integers(len(groups)))]
            total_reward = 0.0
            steps = 0

            for row in trajectory.itertuples(index=False):
                state = int(row.state_from)
                action = int(row.action)
                reward = float(row.reward)
                next_state = int(row.state_to)
                done = bool(row.is_terminal) or next_state in TERMINAL_STATES

                self.update(state, action, reward, next_state, done)
                total_reward += reward
                steps += 1

                if done:
                    break

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            self.q_value_history.append(float(np.mean(self.Q)))
            self.decay_epsilon()

        return self

    def get_policy(self):
        policy = np.full(self.n_states, -1, dtype=int)
        for state in range(self.n_states):
            actions = self._valid_for_state(state)
            if state not in TERMINAL_STATES and len(actions):
                policy[state] = int(actions[np.argmax(self.Q[state, actions])])
        return policy

    def get_policy_table(self):
        policy = self.get_policy()
        rows = []

        for state, action in enumerate(policy):
            D, P, U, B = decode_state(state)
            valid = self._valid_for_state(state)
            rows.append({
                "state": state,
                "D": D, "P": P, "U": U, "B": B,
                "best_action": int(action),
                "action_name": ACTIONS[action] if action >= 0 else "No supported action",
                "Q_value": float(self.Q[state, action]) if action >= 0 else np.nan,
                "is_terminal": state in TERMINAL_STATES,
                "state_quality": -(2 * D + 2 * P - U + 2 * B),
                "n_supported_actions": len(valid),
            })

        return pd.DataFrame(rows)