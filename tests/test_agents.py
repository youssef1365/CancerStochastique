import unittest
from config import ACTIONS
from agents.diagnostic_agent import DiagnosticAgent
from agents.prognostic_agent import PrognosticAgent
from agents.biological_agent import BiologicalAgent
from agents.treatment_agent import TreatmentAgent
from agents.multi_agent import MultiAgentSystem


class TestAgents(unittest.TestCase):

    def test_diagnostic_agent_vetoes(self):
        diag = DiagnosticAgent()
        # High severity (D=1): observation (0) should be vetoed in favor of chemotherapy (2)
        state_high = (1, 0, 1, 0)
        self.assertEqual(diag.vote(proposed_action=0, state=state_high), 2)

        # Low severity (D=0): aggressive combination (5) should be vetoed in favor of observation (0)
        state_low = (0, 0, 1, 0)
        self.assertEqual(diag.vote(proposed_action=5, state=state_low), 0)

        # Neutral: acceptable proposal passes unchanged
        self.assertEqual(diag.vote(proposed_action=1, state=state_low), 1)

    def test_prognostic_agent_vetoes(self):
        prog = PrognosticAgent()
        # High severity, poor prognosis, resistant tumor (D=1, P=1, B=1) -> advocate palliative (6)
        terminal_risk_state = (1, 1, 0, 1)
        self.assertEqual(prog.vote(proposed_action=2, state=terminal_risk_state), 6)

        # Good prognosis passes proposed action
        self.assertEqual(prog.vote(proposed_action=2, state=(0, 0, 2, 0)), 2)

    def test_biological_agent_vetoes(self):
        bio = BiologicalAgent()
        # Resistant tumor (B=1): Chemotherapy (2) is redirected to Immunotherapy (4)
        resistant_state = (1, 0, 1, 1)
        self.assertEqual(bio.vote(proposed_action=2, state=resistant_state), 4)

        # Responsive tumor (B=0): Chemotherapy passes
        responsive_state = (1, 0, 1, 0)
        self.assertEqual(bio.vote(proposed_action=2, state=responsive_state), 2)

    def test_multi_agent_system_decision(self):
        mas = MultiAgentSystem()
        patient_data = {
            "cancer_type": "Lung",
            "stage": "IV",
            "outcome": "Alive",
            "metastatic_site": "Brain",
            "previous_treatments": 2,
        }

        result = mas.decide(patient_data)
        self.assertIn("state_index", result)
        self.assertIn("state_tuple", result)
        self.assertIn("votes", result)
        self.assertIn("final_action", result)
        self.assertTrue(0 <= result["final_action"] < 7)
        self.assertIn(result["final_action_name"], ACTIONS)


if __name__ == "__main__":
    unittest.main()
