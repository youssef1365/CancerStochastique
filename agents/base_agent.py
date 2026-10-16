class BaseAgent:
    """Base class for all agents in the MAS."""

    def __init__(self, name: str):
        self.name = name
        self.observations = {}

    def observe(self, patient_data: dict):
        """Receive patient observation data."""
        self.observations = patient_data.copy()

    def evaluate(self) -> dict:
        """Evaluate and return assessment. Override in subclass."""
        raise NotImplementedError

    def vote(self, proposed_action: int, state: tuple) -> int:
        """
        Vote on a proposed action.
        Returns: the action this agent recommends (may differ from proposed).
        """
        return proposed_action  # Default: agree