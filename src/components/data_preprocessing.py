import pandas as pd
import re
import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

def preprocess_data():

    df = pd.read_csv(
        "data/raw/customer_support_tickets.csv"
    )

    df["text"] = (
        df["Ticket Subject"] +
        " " +
        df["Ticket Description"]
    )

    df["clean_text"] = df["text"].apply(
        clean_text
    )

    print(df[[
        "text",
        "clean_text"
    ]].head())

    
    print("\nOriginal Text Sample:\n")
    print(df["text"].iloc[0])

    print("\nCleaned Text Sample:\n")
    print(df["clean_text"].iloc[0])
    return df

if __name__ == "__main__":
    preprocess_data()