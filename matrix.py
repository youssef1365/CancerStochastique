import pandas as pd
from markov_states import decode_state, state_to_label, is_terminal

df = pd.read_csv("data/cancer_treatment_dataset.csv")

# Decode unique state_from occurrences
states_from = sorted(df["state_from"].unique())
from_rows = []
for s in states_from:
    D, P, U, B = decode_state(int(s))
    count = int((df["state_from"] == s).sum())
    from_rows.append({
        "state": s,
        "D": D,
        "P": P,
        "U": U,
        "B": B,
        "terminal": is_terminal(int(s)),
        "transitions_from": count,
        "label": state_to_label(int(s)),
    })

print("=== Observed 'state_from' breakdown ===")
print(pd.DataFrame(from_rows).to_string(index=False))

# Decode unique state_to occurrences
states_to = sorted(df["state_to"].unique())
to_rows = []
for s in states_to:
    D, P, U, B = decode_state(int(s))
    count = int((df["state_to"] == s).sum())
    to_rows.append({
        "state": s,
        "D": D,
        "P": P,
        "U": U,
        "B": B,
        "terminal": is_terminal(int(s)),
        "transitions_to": count,
        "label": state_to_label(int(s)),
    })

print("\n=== Observed 'state_to' breakdown ===")
print(pd.DataFrame(to_rows).to_string(index=False))