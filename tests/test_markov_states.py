import unittest
from config import (
    D_VALUES, P_VALUES, U_VALUES, B_VALUES, N_STATES,
    REWARD_DEATH, REWARD_RECOVERY
)
from markov_states import (
    encode_state, decode_state, is_terminal, compute_reward,
    fd, fp, fu, fb
)


class TestMarkovStates(unittest.TestCase):

    def test_state_encoding_decoding_bijective(self):
        """Verify state encoding/decoding is 1:1 and covers all 32 states."""
        seen_indices = set()
        for D in D_VALUES:
            for P in P_VALUES:
                for U in U_VALUES:
                    for B in B_VALUES:
                        idx = encode_state(D, P, U, B)
                        self.assertTrue(0 <= idx < N_STATES)
                        seen_indices.add(idx)
                        decoded = decode_state(idx)
                        self.assertEqual(decoded, (D, P, U, B))

        self.assertEqual(len(seen_indices), 32)
        self.assertEqual(seen_indices, set(range(32)))

    def test_terminal_states(self):
        """Verify terminal states are exactly S06 (Recovery) and S25 (Death)."""
        self.assertTrue(is_terminal(6))
        self.assertTrue(is_terminal(25))

        # State 6: D=0, P=0, U=3, B=0
        self.assertEqual(decode_state(6), (0, 0, 3, 0))
        # State 25: D=1, P=1, U=0, B=1
        self.assertEqual(decode_state(25), (1, 1, 0, 1))

        # All other states must NOT be terminal
        for s in range(N_STATES):
            if s not in (6, 25):
                self.assertFalse(is_terminal(s), f"State {s} should not be terminal")

    def test_clinical_mappers(self):
        """Verify clinical mapping functions adhere to oncology definitions."""
        # fd: severity
        self.assertEqual(fd("Lung", "I", "None"), 0)
        self.assertEqual(fd("Lung", "III", "None"), 1)
        self.assertEqual(fd("Lung", "I", "Brain"), 1)

        # fp: prognosis
        self.assertEqual(fp("Breast", "I", "Alive", "None"), 0)
        self.assertEqual(fp("Breast", "I", "Deceased", "None"), 1)
        self.assertEqual(fp("Breast", "IV", "Alive", "Brain"), 1)

        # fu: treatment utility
        self.assertEqual(fu([], "Alive", "I"), 0)
        self.assertEqual(fu(["Surgery"], "Deceased", "II"), 0)  # Deceased -> 0
        self.assertEqual(fu(["Surgery"], "Alive", "I"), 3)       # Early stage alive -> 3
        self.assertEqual(fu(["Chemo"], "Alive", "III"), 2)       # Mid stage alive -> 2
        self.assertEqual(fu(["Chemo"], "Alive", "IV"), 1)        # Late stage alive -> 1

        # fb: biological response / resistance
        self.assertEqual(fb("Colorectal", ["Chemo"], "None"), 0)
        self.assertEqual(fb("Colorectal", ["Chemo"], "Multiple"), 1)
        self.assertEqual(fb("Colorectal", ["Chemo", "Rad", "Immuno"], "Liver"), 1)

    def test_reward_function(self):
        """Verify rewards for recovery, death, improvement, and action costs."""
        # Terminal death (transition into state 25)
        r_death = compute_reward(state_from=10, action=2, state_to=25)
        self.assertEqual(r_death, REWARD_DEATH)

        # Terminal recovery (transition into state 6)
        r_rec = compute_reward(state_from=10, action=1, state_to=6)
        self.assertEqual(r_rec, REWARD_RECOVERY)

        # Action penalty check: Observation (a=0) vs Treatment (a=1)
        r_obs = compute_reward(state_from=0, action=0, state_to=0)
        r_surg = compute_reward(state_from=0, action=1, state_to=0)
        self.assertEqual(r_obs - r_surg, 1.0)


if __name__ == "__main__":
    unittest.main()
