import pandas as pd
DATA_PATH = "data/raw/all_tickets_processed_improved_v3.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)

    print("=" * 50)
    print("DATASET SHAPE")
    print("=" * 50)
    print(df.shape)

    print("\n" + "=" * 50)
    print("COLUMN NAMES")
    print("=" * 50)
    print(df.columns.tolist())

    print("\n" + "=" * 50)
    print("FIRST 5 ROWS")
    print("=" * 50)
    print(df.head())

    print("\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)
    print(df.isnull().sum())

if __name__ == "__main__":
    load_data()