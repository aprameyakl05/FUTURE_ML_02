import pandas as pd

df = pd.read_csv(
    "data/raw/customer_support_tickets.csv"
)

for ticket_type in df["Ticket Type"].unique():

    print("\n" + "="*50)
    print(ticket_type)

    sample = df[
        df["Ticket Type"] == ticket_type
    ]["Ticket Subject"].head(10)

    print(sample.tolist())