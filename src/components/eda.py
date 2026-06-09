import pandas as pd

df = pd.read_csv(
    "data/raw/all_tickets_processed_improved_v3.csv"
)

print("\nDataset Shape:")
print(df.shape)

print("\nTopic Distribution:")
print(df["Topic_group"].value_counts())

print("\nUnique Topics:")
print(df["Topic_group"].unique())

print("\nNumber of Classes:")
print(df["Topic_group"].nunique())