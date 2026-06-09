import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# Load Data
df = pd.read_csv(
    "data/raw/all_tickets_processed_improved_v3.csv"
)

# Features and Target
X = df["Document"]

y = df["Topic_group"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Pipeline
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
        LogisticRegression(
            max_iter=1000
        )
    )
])

# Train
pipeline.fit(
    X_train,
    y_train
)

# Predict
predictions = pipeline.predict(
    X_test
)

# Evaluation
print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        predictions
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)