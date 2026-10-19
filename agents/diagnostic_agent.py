import sys
from pathlib import Path

# Ensure root directory is on sys.path if invoked from within agents/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from agents.base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent

from config import D_LABELS
from markov_states import fd


class DiagnosticAgent(BaseAgent):
    """
    Evaluates disease severity D ∈ {0, 1}.

    Applies fd mapping function and provides clinical assessment
    of disease burden based on stage and metastatic status.
    """

    def __init__(self):
        super().__init__("DiagnosticAgent")

    def evaluate(self) -> dict:
        has_clinical = "stage" in self.observations or "metastatic_site" in self.observations
        if "D" in self.observations:
            D = int(self.observations["D"])
        else:
            stage = self.observations.get('stage', 'I')
            met_site = self.observations.get('metastatic_site', 'None')
            cancer_type = self.observations.get('cancer_type', 'Lung')
            D = fd(cancer_type, stage, met_site)

        if has_clinical:
            stage = self.observations.get('stage', 'I')
            met_site = self.observations.get('metastatic_site', 'None')
            confidence = 0.9 if D == 1 and met_site != "None" else 0.75
            assessment = (
                f"Disease severity: {D_LABELS[D]} "
                f"(Stage {stage}, Met: {met_site})"
            )
        else:
            confidence = 0.85
            assessment = f"Disease severity: {D_LABELS[D]} (Encoded state D={D})"

        return {
            'D': D,
            'severity': D_LABELS[D],
            'confidence': confidence,
            'assessment': assessment
        }

    def vote(self, proposed_action: int, state: tuple) -> int:
        D, P, U, B = state

        # If low severity, veto overly aggressive treatments
        if D == 0 and proposed_action == 5:  # Combination too aggressive
            return 0  # Suggest observation

        # If high severity, veto observation
        if D == 1 and proposed_action == 0:  # Observation inadequate
            return 2  # Suggest chemotherapy

        return proposed_action