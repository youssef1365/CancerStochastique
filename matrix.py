import pandas as pd

df = pd.read_csv("data/cancer_treatment_dataset.csv")

print("state_from vs D/P/U/B:")
print(
    df[["state_from", "D_from", "P_from", "U_from", "B_from"]]
    .drop_duplicates()
    .sort_values("state_from")
    .to_string(index=False)
)

print("\nstate_to vs D/P/U/B:")
print(
    df[["state_to", "D_to", "P_to", "U_to", "B_to"]]
    .drop_duplicates()
    .sort_values("state_to")
    .to_string(index=False)
)