
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import ACTIONS
from markov_states import decode_state

try:
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.prognostic_agent import PrognosticAgent
    from agents.treatment_agent import TreatmentAgent
    from agents.biological_agent import BiologicalAgent
except ImportError:
    from diagnostic_agent import DiagnosticAgent
    from prognostic_agent import PrognosticAgent
    from treatment_agent import TreatmentAgent
    from biological_agent import BiologicalAgent


class MultiAgentSystem:
    """Multi-agent orchestration for synthetic state-based evaluation."""

    def __init__(self, q_agent=None, double_q_agent=None, use_double=False):
        self.diagnostic = DiagnosticAgent()
        self.prognostic = PrognosticAgent()
        self.treatment = TreatmentAgent(q_agent, double_q_agent, use_double)
        self.biological = BiologicalAgent()

        self.agents = [
            self.diagnostic,
            self.prognostic,
            self.treatment,
            self.biological,
        ]
        self.decision_log = []

    def decide(self, patient_data: dict, verbose: bool = False) -> dict:
        # Dataset states are already encoded as (D, P, U, B).
        state_tuple = patient_data.get("state_tuple")

        if state_tuple is None:
            for agent in self.agents:
                agent.observe(patient_data)

            assessments = {
                agent.name: agent.evaluate()
                for agent in self.agents
            }

            D = assessments["DiagnosticAgent"]["D"]
            P = assessments["PrognosticAgent"]["P"]
            U = patient_data.get("U", 0)
            B = assessments["BiologicalAgent"]["B"]
            state_tuple = (D, P, U, B)
        else:
            D, P, U, B = map(int, state_tuple)

            # Use the supplied encoded state instead of inventing
            # clinical fields that do not exist in the dataset.
            assessments = {
                "DiagnosticAgent": {
                    "D": D,
                    "assessment": f"Encoded severity state D={D}",
                },
                "PrognosticAgent": {
                    "P": P,
                    "assessment": f"Encoded prognosis state P={P}",
                },
                "BiologicalAgent": {
                    "B": B,
                    "assessment": f"Encoded biological state B={B}",
                },
            }

            for agent in self.agents:
                agent.observe(patient_data)

        state_idx = int(patient_data.get(
            "state_index",
            self._encode_state(state_tuple),
        ))

        self.treatment.observations["state_index"] = state_idx
        treatment_eval = self.treatment.evaluate()
        assessments["TreatmentAgent"] = treatment_eval

        proposed_action = int(treatment_eval["recommended_action"])

        votes = {
            agent.name: int(agent.vote(proposed_action, state_tuple))
            for agent in self.agents
        }

        vote_counts = Counter(votes.values())
        final_action, count = vote_counts.most_common(1)[0]

        # Treatment agent breaks a tie for the leading vote.
        if sum(v == count for v in vote_counts.values()) > 1:
            final_action = votes["TreatmentAgent"]

        result = {
            "state_index": state_idx,
            "state_tuple": state_tuple,
            "proposed_action": proposed_action,
            "proposed_action_name": ACTIONS[proposed_action],
            "votes": votes,
            "vote_names": {k: ACTIONS[v] for k, v in votes.items()},
            "final_action": final_action,
            "final_action_name": ACTIONS[final_action],
            "assessments": assessments,
            "consensus": len(set(votes.values())) == 1,
        }

        self.decision_log.append(result)

        if verbose:
            self._print_decision(result)

        return result

    @staticmethod
    def _encode_state(state_tuple):
        D, P, U, B = state_tuple
        return D * 16 + P * 8 + U * 2 + B

    def _print_decision(self, result):
        print(f"\nMAS Decision — State S{result['state_index']:02d}")
        print(f"State tuple: {result['state_tuple']}")

        for name, assessment in result["assessments"].items():
            print(f"{name}: {assessment.get('assessment', assessment)}")

        print(f"Proposed action: {result['proposed_action_name']}")
        print(f"Votes: {result['vote_names']}")
        print(f"Final action: {result['final_action_name']}")
        print(f"Consensus: {result['consensus']}")

    def evaluate_on_dataset(
        self,
        df: pd.DataFrame,
        n_samples: int = 100,
        verbose: bool = False,
    ) -> pd.DataFrame:
        print(f"\n{'=' * 70}")
        print(f"EVALUATING MULTI-AGENT SYSTEM ON {n_samples} SAMPLES")
        print(f"{'=' * 70}")

        required = {
            "patient_id", "timestep", "state_from",
            "action", "state_to", "is_terminal",
        }
        missing = required.difference(df.columns)

        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing)}"
            )

        patient_ids = df["patient_id"].drop_duplicates().to_numpy()
        sample_ids = np.random.choice(
            patient_ids,
            size=min(n_samples, len(patient_ids)),
            replace=False,
        )

        # Fast lookup: filter dataframe for sampled IDs only
        sample_df = df[df["patient_id"].isin(sample_ids)].sort_values(["patient_id", "timestep"])
        patient_first_rows = sample_df.groupby("patient_id", as_index=False).first()

        results = []

        for row in patient_first_rows.itertuples(index=False):
            pid = row.patient_id
            state_idx = int(row.state_from)
            D, P, U, B = map(int, decode_state(state_idx))

            patient_data = {
                "patient_id": pid,
                "state_index": state_idx,
                "state_tuple": (D, P, U, B),
                "D": D,
                "P": P,
                "U": U,
                "B": B,
            }

            decision = self.decide(patient_data, verbose=verbose)

            original_action = int(row.action)

            results.append({
                "patient_id": pid,
                "state_index": state_idx,
                "state_tuple": str(decision["state_tuple"]),
                "original_action": ACTIONS[original_action],
                "mas_action": decision["final_action_name"],
                "proposed_action": decision["proposed_action_name"],
                "consensus": decision["consensus"],
                "q_proposed_matches_final": (
                    decision["proposed_action"] == decision["final_action"]
                ),
            })

        results_df = pd.DataFrame(results)

        if not results_df.empty:
            match_rate = (
                results_df["original_action"] == results_df["mas_action"]
            ).mean()
            consensus_rate = results_df["consensus"].mean()
            q_match_rate = results_df["q_proposed_matches_final"].mean()

            print("\nEvaluation results:")
            print(f"  MAS matches recorded action: {100 * match_rate:.1f}%")
            print(f"  Agent consensus rate:        {100 * consensus_rate:.1f}%")
            print(f"  Q proposal accepted:         {100 * q_match_rate:.1f}%")

        return results_df
