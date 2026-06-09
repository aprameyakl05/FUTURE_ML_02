import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load Dataset
df = pd.read_csv(
    "data/raw/all_tickets_processed_improved_v3.csv"
)

X = df["Document"]
y = df["Topic_group"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=10000
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

pipeline.fit(X_train, y_train)

joblib.dump(
    pipeline,
    "models/ticket_classifier.pkl"
)

print("Model Saved Successfully!")