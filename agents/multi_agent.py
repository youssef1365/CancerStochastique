
import numpy as np
import pandas as pd
from collections import Counter

from config import N_STATES, N_ACTIONS, ACTIONS
from markov_states import encode_state, decode_state
from agents.diagnostic_agent import DiagnosticAgent
from agents.prognostic_agent import PrognosticAgent
from agents.treatment_agent import TreatmentAgent
from agents.biological_agent import BiologicalAgent


class MultiAgentSystem:
    def __init__(
        self,
        q_agent=None,
        double_q_agent=None,
        use_double=False,
        valid_actions=None
    ):
        self.diagnostic = DiagnosticAgent()
        self.prognostic = PrognosticAgent()
        self.biological = BiologicalAgent()

        self.valid_actions = self._validate_action_mask(valid_actions)

        self.treatment = TreatmentAgent(
            q_agent=q_agent,
            double_q_agent=double_q_agent,
            use_double=use_double,
            valid_actions=self.valid_actions
        )

        self.agents = [
            self.diagnostic,
            self.prognostic,
            self.biological,
            self.treatment
        ]

        self.decision_log = []

    @staticmethod
    def _validate_action_mask(valid_actions):
        if valid_actions is None:
            return None

        mask = np.asarray(valid_actions, dtype=bool)

        if mask.shape != (N_STATES, N_ACTIONS):
            raise ValueError(
                f"valid_actions must have shape "
                f"({N_STATES}, {N_ACTIONS}), got {mask.shape}"
            )

        return mask.copy()

    @staticmethod
    def _valid_action(action):
        try:
            action = int(action)
        except (TypeError, ValueError):
            return False

        return 0 <= action < N_ACTIONS

    def _select_valid_action(self, action, state_idx):
        if not self._valid_action(action):
            raise ValueError(
                f"TreatmentAgent returned invalid action {action} "
                f"for state {state_idx}"
            )

        if self.valid_actions is None:
            return int(action)

        supported = self.valid_actions[state_idx]

        if not supported.any():
            return int(action)

        if not supported[action]:
            raise ValueError(
                f"Action {action} ({ACTIONS[action]}) is unsupported "
                f"for state {state_idx}"
            )

        return int(action)

    def _build_assessments(self, patient_data, state_tuple):
        D, P, U, B = state_tuple

        self.diagnostic.observations = dict(patient_data)
        self.prognostic.observations = dict(patient_data)
        self.biological.observations = dict(patient_data)

        self.diagnostic.observations.update({
            "D": D,
            "state_tuple": state_tuple
        })
        self.prognostic.observations.update({
            "P": P,
            "state_tuple": state_tuple
        })
        self.biological.observations.update({
            "B": B,
            "state_tuple": state_tuple
        })

        diagnostic_result = self.diagnostic.evaluate()
        prognostic_result = self.prognostic.evaluate()
        biological_result = self.biological.evaluate()

        return {
            "diagnostic": diagnostic_result,
            "prognostic": prognostic_result,
            "biological": biological_result
        }

    def decide(self, patient_data, verbose=False):
        patient_data = dict(patient_data)

        if "state_tuple" in patient_data:
            state_tuple = tuple(
                int(value) for value in patient_data["state_tuple"]
            )
            if len(state_tuple) != 4:
                raise ValueError(
                    "state_tuple must contain (D, P, U, B)"
                )
            state_idx = encode_state(*state_tuple)
        else:
            self.diagnostic.observations = dict(patient_data)
            self.prognostic.observations = dict(patient_data)
            self.biological.observations = dict(patient_data)

            diagnostic_result = self.diagnostic.evaluate()
            prognostic_result = self.prognostic.evaluate()
            biological_result = self.biological.evaluate()

            D = int(diagnostic_result["D"])
            P = int(prognostic_result["P"])
            B = int(biological_result["B"])
            U = int(patient_data.get("U", 0))

            state_tuple = (D, P, U, B)
            state_idx = encode_state(*state_tuple)

        if not all(value in (0, 1, 2, 3) for value in state_tuple):
            raise ValueError(f"Invalid state tuple: {state_tuple}")

        if not 0 <= state_idx < N_STATES:
            raise ValueError(f"Invalid state index: {state_idx}")

        encoded_tuple = decode_state(state_idx)
        if tuple(encoded_tuple) != tuple(state_tuple):
            raise ValueError(
                f"State tuple {state_tuple} does not match "
                f"encoded state {state_idx}"
            )

        supplied_index = patient_data.get("state_index")
        if supplied_index is not None and int(supplied_index) != state_idx:
            raise ValueError(
                f"state_index {supplied_index} does not match "
                f"state_tuple {state_tuple}"
            )

        assessments = self._build_assessments(
            patient_data,
            state_tuple
        )

        self.treatment.observations = dict(patient_data)
        self.treatment.observations.update({
            "state_index": state_idx,
            "state_tuple": state_tuple,
            "D": state_tuple[0],
            "P": state_tuple[1],
            "U": state_tuple[2],
            "B": state_tuple[3]
        })

        treatment_result = self.treatment.evaluate()
        proposed_action = int(treatment_result["recommended_action"])
        proposed_action = self._select_valid_action(
            proposed_action,
            state_idx
        )

        votes = {}
        for agent in self.agents:
            vote = agent.vote(proposed_action, state_tuple)

            # Keep every agent's vote within the actions supported
            # by the training data for this state.
            if (
                    self.valid_actions is not None
                    and self.valid_actions[state_idx].any()
                    and (
                    not self._valid_action(vote)
                    or not self.valid_actions[state_idx, int(vote)]
            )
            ):
                vote = proposed_action

            vote = self._select_valid_action(vote, state_idx)
            votes[agent.name] = vote

        counts = Counter(votes.values())
        highest_vote_count = max(counts.values())
        tied_actions = {
            action for action, count in counts.items()
            if count == highest_vote_count
        }

        if proposed_action in tied_actions:
            final_action = proposed_action
        else:
            final_action = min(
                tied_actions,
                key=lambda action: (
                    -float(treatment_result["q_values"][action]),
                    action
                )
            )

        result = {
            "state_index": state_idx,
            "state_tuple": state_tuple,
            "D": state_tuple[0],
            "P": state_tuple[1],
            "U": state_tuple[2],
            "B": state_tuple[3],
            "assessments": assessments,
            "proposed_action": proposed_action,
            "proposed_action_name": ACTIONS[proposed_action],
            "final_action": final_action,
            "recommended_action": final_action,
            "action_name": ACTIONS[final_action],
            "q_values": np.asarray(
                treatment_result["q_values"], dtype=float
            ).copy(),
            "agent_type": treatment_result["agent_type"],
            "votes": votes,
            "consensus": counts[final_action] / len(self.agents),
            "q_proposed_matches_final": proposed_action == final_action,
            "assessment": treatment_result.get("assessment", "")
        }

        self.decision_log.append(result)

        if verbose:
            print(
                f"State {state_idx} {state_tuple} | "
                f"Proposed: {ACTIONS[proposed_action]} | "
                f"Final: {ACTIONS[final_action]} | "
                f"Consensus: {result['consensus']:.1%}"
            )

        return result

    def evaluate_on_dataset(
        self,
        df,
        n_samples=100,
        verbose=False
    ):
        required = {
            "patient_id",
            "timestep",
            "state_from",
            "action",
            "state_to",
            "is_terminal"
        }
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Dataset missing required columns: {sorted(missing)}"
            )

        if df.empty:
            return pd.DataFrame()

        data = df.copy()
        data["state_from"] = pd.to_numeric(
            data["state_from"], errors="raise"
        ).astype(int)
        data["action"] = pd.to_numeric(
            data["action"], errors="raise"
        ).astype(int)

        if not data["state_from"].between(0, N_STATES - 1).all():
            raise ValueError("Dataset contains invalid state_from values")

        if not data["action"].between(0, N_ACTIONS - 1).all():
            raise ValueError("Dataset contains invalid action values")

        first_rows = (
            data.sort_values(["patient_id", "timestep"])
            .groupby("patient_id", sort=False)
            .first()
            .reset_index()
        )

        n_samples = max(0, min(int(n_samples), len(first_rows)))
        if n_samples == 0:
            return pd.DataFrame()

        sampled = first_rows.sample(
            n=n_samples,
            random_state=42
        )

        results = []

        for _, row in sampled.iterrows():
            state_idx = int(row["state_from"])
            state_tuple = tuple(decode_state(state_idx))

            patient_data = {
                "patient_id": row["patient_id"],
                "state_index": state_idx,
                "state_tuple": state_tuple,
                "D": state_tuple[0],
                "P": state_tuple[1],
                "U": state_tuple[2],
                "B": state_tuple[3]
            }

            decision = self.decide(
                patient_data,
                verbose=verbose
            )

            original_action = int(row["action"])
            final_action = int(decision["final_action"])
            proposed_action = int(decision["proposed_action"])

            results.append({
                "patient_id": row["patient_id"],
                "state_index": state_idx,
                "state_tuple": state_tuple,
                "original_action": ACTIONS[original_action],
                "original_action_index": original_action,
                "mas_action": ACTIONS[final_action],
                "mas_action_index": final_action,
                "proposed_action": ACTIONS[proposed_action],
                "proposed_action_index": proposed_action,
                "consensus": decision["consensus"],
                "q_proposed_matches_final": (
                    proposed_action == final_action
                )
            })

        return pd.DataFrame(results)
