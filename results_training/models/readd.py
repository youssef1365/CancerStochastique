import numpy as np

data = np.load("transition_matrices.npz")

counts = data["counts"]
matrices = data["matrices"]

print("Shapes:", counts.shape, matrices.shape)

for action in range(7):
    state_counts = counts[action, 6, :]
    probabilities = matrices[action, 6, :]

    print(f"\nAction {action}")
    print("Observed transitions:", state_counts.sum())
    print("Probability sum:", probabilities.sum())

    for next_state in np.where(probabilities > 0)[0]:
        print(
            f"  State 6 -> State {next_state}: "
            f"count={state_counts[next_state]}, "
            f"probability={probabilities[next_state]:.4f}"
        )