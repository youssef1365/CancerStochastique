from config import B_LABELS
from markov_states import (fb)
from base_agent import BaseAgent


class BiologicalAgent(BaseAgent):
    """
    Evaluates biological response B ∈ {0, 1}.

    Applies fb mapping function. Assesses whether the tumor
    is responsive or resistant to treatment.
    """

    def __init__(self):
        super().__init__("BiologicalAgent")

    def evaluate(self) -> dict:
        cancer_type = self.observations.get('cancer_type', 'Lung')
        treatments = self.observations.get('treatments', [])
        met_site = self.observations.get('metastatic_site', 'None')

        B = fb(cancer_type, treatments, met_site)

        return {
            'B': B,
            'response': B_LABELS[B],
            'confidence': 0.80,
            'assessment': f"Biological response: {B_LABELS[B]} "
                          f"(Treatments: {len(treatments)}, Met: {met_site})"
        }

    def vote(self, proposed_action: int, state: tuple) -> int:
        D, P, U, B = state

        # If resistant, chemotherapy alone is less effective
        if B == 1 and proposed_action == 2:
            return 4  # Suggest immunotherapy

        return proposed_action