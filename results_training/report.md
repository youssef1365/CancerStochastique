# Training and Multi-Agent System Evaluation

## Algorithm
- Selected algorithm: double_q_learning
- Episodes requested: 10,000
- Episodes recorded: 10,000
- Training time: 25.50 seconds

## Dataset
- Transitions: 249,258
- Unique patients: 25,000

## Learned policy
- Mean Q-value: 17.2509
- Maximum Q-value: 45.4260
- Mean maximum Q-value per state: 26.3510
- Agreement with recorded actions: 21.82%

## Multi-agent evaluation
- Patients evaluated: 100
- MAS action matches recorded action: 29.00%
- Unanimous consensus: 68.00%
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
