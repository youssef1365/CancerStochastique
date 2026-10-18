
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from agents.base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent

from config import N_ACTIONS, ACTIONS
from markov_states import encode_state, decode_state


class TreatmentAgent(BaseAgent):
    """Treatment selection using standard or Double Q-learning."""

    def __init__(self, q_agent=None, double_q_agent=None, use_double=False):
        super().__init__("TreatmentAgent")
        self.q_agent = q_agent
        self.double_q_agent = double_q_agent
        self.use_double = use_double

    def _get_q_values(self, state_idx):
        if self.use_double:
            agent = self.double_q_agent or self.q_agent

            if agent is None:
                return np.zeros(N_ACTIONS), "Random"

            if hasattr(agent, "get_combined_Q"):
                return agent.get_combined_Q()[state_idx], "Double Q-Learning"

            if hasattr(agent, "QA") and hasattr(agent, "QB"):
                q_values = (agent.QA[state_idx] + agent.QB[state_idx]) / 2.0
                return q_values, "Double Q-Learning"

            raise AttributeError(
                "Double Q-learning agent must expose get_combined_Q(), "
                "or both QA and QB."
            )

        if self.q_agent is not None:
            if hasattr(self.q_agent, "Q"):
                return self.q_agent.Q[state_idx], "Q-Learning"

            if hasattr(self.q_agent, "QA") and hasattr(self.q_agent, "QB"):
                q_values = (self.q_agent.QA[state_idx] + self.q_agent.QB[state_idx]) / 2.0
                return q_values, "Double Q-Learning"

            raise AttributeError(
                "The supplied learning agent exposes neither Q nor QA/QB."
            )

        return np.zeros(N_ACTIONS), "Random"

    def evaluate(self) -> dict:
        state_idx = self.observations.get("state_index", 0)
        D, P, U, B = decode_state(state_idx)

        q_values, agent_type = self._get_q_values(state_idx)
        best_action = int(np.argmax(q_values))

        return {
            "recommended_action": best_action,
            "action_name": ACTIONS[best_action],
            "q_values": q_values,
            "agent_type": agent_type,
            "assessment": (
                f"Recommended: {ACTIONS[best_action]} "
                f"(Q={q_values[best_action]:.3f}, via {agent_type})"
            ),
        }

    def vote(self, proposed_action: int, state: tuple) -> int:
        """Vote for the action with the highest learned value."""
        state_idx = encode_state(*state)
        q_values, agent_type = self._get_q_values(state_idx)

        if agent_type == "Random":
            return proposed_action

        return int(np.argmax(q_values))
