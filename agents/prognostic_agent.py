from config import P_LABELS
from markov_states import fp
from base_agent import BaseAgent


class PrognosticAgent(BaseAgent):
    """
    Evaluates prognosis P ∈ {0, 1}.

    Applies fp mapping function and provides survival prognosis
    based on outcome history, stage, and metastatic patterns.
    """

    def __init__(self):
        super().__init__("PrognosticAgent")

    def evaluate(self) -> dict:
        cancer_type = self.observations.get('cancer_type', 'Lung')
        stage = self.observations.get('stage', 'I')
        outcome = self.observations.get('outcome', 'Alive')
        met_site = self.observations.get('metastatic_site', 'None')

        P = fp(cancer_type, stage, outcome, met_site)

        return {
            'P': P,
            'prognosis': P_LABELS[P],
            'confidence': 0.85,
            'assessment': f"Prognosis: {P_LABELS[P]} "
                          f"(Outcome: {outcome}, Stage {stage})"
        }

    def vote(self, proposed_action: int, state: tuple) -> int:
        D, P, U, B = state

        # Poor prognosis + palliative might be better
        if P == 1 and D == 1 and B == 1:
            return 6  # Palliative

        return proposed_action