from config import D_LABELS
from markov_states import fd
from base_agent import BaseAgent


class DiagnosticAgent(BaseAgent):
    """
    Evaluates disease severity D ∈ {0, 1}.

    Applies fd mapping function and provides clinical assessment
    of disease burden based on stage and metastatic status.
    """

    def __init__(self):
        super().__init__("DiagnosticAgent")

    def evaluate(self) -> dict:
        stage = self.observations.get('stage', 'I')
        met_site = self.observations.get('metastatic_site', 'None')
        cancer_type = self.observations.get('cancer_type', 'Lung')

        D = fd(cancer_type, stage, met_site)

        return {
            'D': D,
            'severity': D_LABELS[D],
            'confidence': 0.9 if D == 1 and met_site != "None" else 0.75,
            'assessment': f"Disease severity: {D_LABELS[D]} "
                          f"(Stage {stage}, Met: {met_site})"
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