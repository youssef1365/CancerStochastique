
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NETLOGO = ROOT / "results_training" / "netlogo"
DATASET = ROOT / "data" / "cancer_treatment_dataset.csv"

sys.path.insert(0, str(ROOT))

from config import N_STATES, N_ACTIONS, ACTIONS


class TestExportedPolicies(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.errors = []
        cls.reports = []

        required = {
            "q_learning": (
                NETLOGO / "q_learning_policy.csv",
                NETLOGO / "q_learning_Q_values.csv",
            ),
            "double_q_learning": (
                NETLOGO / "double_q_learning_policy.csv",
                NETLOGO / "double_q_learning_Q_values.csv",
            ),
        }

        cls.policies = {}
        cls.q_tables = {}

        for algorithm, (policy_path, q_path) in required.items():
            if not policy_path.exists():
                cls.errors.append(f"Missing policy file: {policy_path}")
                continue
            if not q_path.exists():
                cls.errors.append(f"Missing Q-table file: {q_path}")
                continue

            cls.policies[algorithm] = pd.read_csv(policy_path)
            cls.q_tables[algorithm] = pd.read_csv(q_path)

        cls.counts = None
        if DATASET.exists():
            df = pd.read_csv(DATASET)
            if {"state_from", "action"}.issubset(df.columns):
                # This dataset contains all logged transitions, not necessarily
                # the training split. Use it as a diagnostic, not proof of
                # training-only action support.
                cls.counts = np.zeros(
                    (N_STATES, N_ACTIONS), dtype=np.int64
                )
                states = df["state_from"].to_numpy(dtype=int)
                actions = df["action"].to_numpy(dtype=int)
                valid = (
                    (states >= 0) & (states < N_STATES)
                    & (actions >= 0) & (actions < N_ACTIONS)
                )
                np.add.at(cls.counts, (states[valid], actions[valid]), 1)

    def test_policy_files_exist(self):
        self.assertFalse(
            self.errors,
            "\n".join(self.errors),
        )

    def test_each_policy_has_all_states(self):
        for name, policy in self.policies.items():
            with self.subTest(policy=name):
                self.assertEqual(len(policy), N_STATES)
                self.assertIn("state", policy.columns)
                self.assertIn("best_action", policy.columns)
                self.assertEqual(
                    sorted(policy["state"].astype(int).tolist()),
                    list(range(N_STATES)),
                    f"{name}: missing, duplicate, or invalid state IDs",
                )

    def test_actions_are_valid(self):
        for name, policy in self.policies.items():
            with self.subTest(policy=name):
                actions = policy["best_action"].to_numpy(dtype=int)
                self.assertTrue(
                    np.all((actions >= 0) & (actions < N_ACTIONS)),
                    f"{name}: contains action IDs outside valid range",
                )

                if "action_name" in policy.columns:
                    for row in policy.itertuples(index=False):
                        action = int(row.best_action)
                        self.assertEqual(
                            str(row.action_name),
                            str(ACTIONS[action]),
                            f"{name}: action name does not match action ID "
                            f"for state {row.state}",
                        )

    def test_q_tables_have_valid_shape_and_values(self):
        for name, q_table in self.q_tables.items():
            with self.subTest(policy=name):
                columns = [f"Q_action_{a}" for a in range(N_ACTIONS)]
                self.assertTrue(
                    set(columns).issubset(q_table.columns),
                    f"{name}: missing Q-value columns",
                )
                self.assertEqual(len(q_table), N_STATES)

                q = q_table[columns].to_numpy(dtype=float)
                self.assertEqual(q.shape, (N_STATES, N_ACTIONS))
                self.assertTrue(
                    np.isfinite(q).all(),
                    f"{name}: Q-table contains NaN or infinity",
                )

                self.assertEqual(
                    sorted(q_table["state"].astype(int).tolist()),
                    list(range(N_STATES)),
                )

    def test_policy_matches_q_values(self):
        """Check each exported action maximizes Q among eligible actions."""
        for name, policy in self.policies.items():
            if name not in self.q_tables:
                continue

            with self.subTest(policy=name):
                qdf = self.q_tables[name].sort_values("state")
                pdf = policy.sort_values("state")
                q = qdf[
                    [f"Q_action_{a}" for a in range(N_ACTIONS)]
                ].to_numpy(dtype=float)

                actions = pdf["best_action"].to_numpy(dtype=int)

                for state, action in enumerate(actions):
                    if "supported" in pdf.columns and int(
                        pdf.iloc[state]["supported"]
                    ) == 1 and self.counts is not None:
                        eligible = np.flatnonzero(
                            self.counts[state] >= 1
                        )
                        if len(eligible):
                            best_value = np.max(q[state, eligible])
                            self.assertAlmostEqual(
                                q[state, action],
                                best_value,
                                places=6,
                                msg=(
                                    f"{name}: state {state} action {action} "
                                    "is not greedy among dataset-supported "
                                    "actions. Check against training-split "
                                    "counts before treating this as a failure."
                                ),
                            )
                    else:
                        self.assertAlmostEqual(
                            q[state, action],
                            np.max(q[state]),
                            places=6,
                            msg=(
                                f"{name}: state {state} action {action} "
                                "does not maximize the exported Q-table"
                            ),
                        )

    def test_recorded_q_value_matches_selected_action(self):
        for name, policy in self.policies.items():
            if name not in self.q_tables or "Q_value" not in policy.columns:
                continue

            with self.subTest(policy=name):
                qdf = self.q_tables[name].sort_values("state")
                pdf = policy.sort_values("state")
                q = qdf[
                    [f"Q_action_{a}" for a in range(N_ACTIONS)]
                ].to_numpy(dtype=float)
                actions = pdf["best_action"].to_numpy(dtype=int)
                stored = pdf["Q_value"].to_numpy(dtype=float)
                expected = q[np.arange(N_STATES), actions]

                np.testing.assert_allclose(
                    stored,
                    expected,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"{name}: exported Q_value is inconsistent",
                )

    def test_report_unvisited_or_unsupported_states(self):
        """Diagnostic: unsupported states are reported, not automatically failed."""
        for name, policy in self.policies.items():
            if "supported" not in policy.columns:
                continue

            unsupported = policy.loc[
                policy["supported"].astype(int) == 0, "state"
            ].astype(int).tolist()

            print(f"\n{name}: unsupported states = {unsupported}")

            if "n_visits" in policy.columns:
                zero_visit = policy.loc[
                    policy["n_visits"].astype(int) == 0, "state"
                ].astype(int).tolist()
                print(f"{name}: zero-visit states = {zero_visit}")

    def test_compare_q_learning_and_double_q_learning(self):
        if not {
            "q_learning", "double_q_learning"
        }.issubset(self.policies):
            self.skipTest("Both policy exports are required.")

        q_policy = self.policies["q_learning"].sort_values("state")
        dq_policy = self.policies["double_q_learning"].sort_values("state")

        a = q_policy["best_action"].to_numpy(dtype=int)
        b = dq_policy["best_action"].to_numpy(dtype=int)
        agreement = float(np.mean(a == b))

        print(f"\nQ-Learning vs Double Q-Learning agreement: {agreement:.2%}")
        print(f"Different states: {np.flatnonzero(a != b).tolist()}")

        self.assertEqual(len(a), N_STATES)
        self.assertEqual(len(b), N_STATES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
