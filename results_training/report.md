# Training and Multi-Agent System Evaluation

## Algorithm
- Selected algorithm: double_q_learning
- Episodes requested: 10,000
- Episodes recorded: 10,000
- Training time: 7.02 seconds

## Dataset
- Transitions: 249,258
- Unique patients: 25,000

## Learned policy
- Mean Q-value: 9.7316
- Maximum Q-value: 32.8733
- Mean maximum Q-value per state: 15.6482
- Agreement with recorded actions: 21.38%

## Multi-agent evaluation
- Patients evaluated: 100
- MAS action matches recorded action: 25.00%
- Unanimous consensus: 59.00%
- Q-learning proposal accepted as final action: 100.00%

## Output files
- models/double_q_learning_q_table.npy
- tables/policy_double_q_learning.csv
- tables/checkpoints.csv
- tables/episodes.csv
- tables/mas_evaluation.csv
- netlogo/double_q_learning_Q_values.csv
- netlogo/double_q_learning_policy.csv
- summary.json
- report.md

Action agreement and consensus measure model behavior, not clinical effectiveness.
This system is a research prototype and does not provide clinical treatment recommendations.
