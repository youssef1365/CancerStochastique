import unittest
import numpy as np
from q_learnings.q_learning import QLearningAgent
from q_learnings.double_q_learning import DoubleQLearningAgent


class TestQLearning(unittest.TestCase):

    def test_q_learning_agent_update(self):
        agent = QLearningAgent(n_states=32, n_actions=7)
        agent.valid_actions[0, 1] = True
        agent.valid_actions[1, 2] = True

        initial_q = agent.Q[0, 1]
        td_error = agent.update(state=0, action=1, reward=10.0, next_state=1, done=False)
        self.assertNotEqual(td_error, 0.0)
        self.assertGreater(agent.Q[0, 1], initial_q)

        # Terminal state update
        agent.update(state=0, action=1, reward=-50.0, next_state=25, done=True)
        self.assertLess(agent.Q[0, 1], 10.0)

    def test_double_q_learning_update(self):
        agent = DoubleQLearningAgent(n_states=32, n_actions=7)
        agent.valid_actions[0, 1] = True
        agent.valid_actions[1, 2] = True

        for _ in range(20):
            agent.update(state=0, action=1, reward=5.0, next_state=1, done=False)

        combined = agent.get_combined_Q()
        self.assertEqual(combined.shape, (32, 7))
        self.assertGreater(combined[0, 1], 0)


if __name__ == "__main__":
    unittest.main()
