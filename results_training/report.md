# Training and Multi-Agent System Evaluation

## Algorithm
- Algorithm: double_q_learning
- Episodes requested: 10,000
- Episodes recorded: 10,000
- Training time: 3.57 seconds

## Dataset split
- Total transitions: 249,258
- Training patients: 20,000
- Test patients: 5,000
- Random seed: 42

## Learned policy
- Mean Q-value: 18.0175
- Maximum Q-value: 46.4628
- Mean maximum Q-value per state: 27.3559
- Training logged-action agreement: 21.65%
- Test logged-action agreement: 22.07%
- Minimum action support count: 1
- Supported state-action pairs: 144

## Off-policy evaluation

Held-out patients: 5000 (49,397 transitions), gamma = 0.95, target epsilon = 0.1.

| Policy | WIS | PDWIS | DR | ESS |
|---|---|---|---|---|
| logged | 15.989 [15.319, 16.672] | - | - | - |
| learned | -3.780 [-24.175, 14.848] | 18.770 [-11.508, 43.796] | 25.205 [18.968, 29.371] | 10.5 |
| behavior_clone | 15.989 [15.319, 16.672] | 25.706 [24.481, 27.036] | 28.029 [27.328, 28.741] | 5000.0 |
| uniform_random | 8.085 [6.594, 9.777] | 27.478 [25.165, 29.718] | 27.763 [27.440, 28.095] | 1538.2 |

Learned minus logged:
- WIS_minus_logged: -19.770 [-40.223, -1.049] — CI excludes zero
- DR_minus_logged: 9.216 [2.750, 13.350] — CI excludes zero

Held-out data coverage:
- Transitions in states with training support: 100.0%
- State-action pairs observed in training: 100.0%
- Greedy action matches logged action: 22.1%

## Multi-agent evaluation
- Patients evaluated: 100
- MAS action matches recorded action: 23.00%
- Unanimous agent consensus: 74.00%
- Q-learning proposal retained: 74.00%

## Output files
- models/double_q_learning_q_table.npy
- models/valid_actions.npy
- models/transition_matrices.npz
- models/q_table_QA.npy and models/q_table_QB.npy (Double Q-Learning)
- netlogo/double_q_learning_Q_values.csv
- netlogo/double_q_learning_policy.csv
- netlogo/transition_matrices.csv
- tables/policy_double_q_learning.csv
- tables/checkpoints.csv
- tables/episodes.csv
- tables/ope.csv
- tables/mas_evaluation.csv
- split.json
- summary.json
- report.md
