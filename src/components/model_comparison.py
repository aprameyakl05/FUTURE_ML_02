import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

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

models = {

    "Logistic Regression":
    LogisticRegression(max_iter=1000),

    "Naive Bayes":
    MultinomialNB(),

    "Random Forest":
    RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

for name, model in models.items():

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
            model
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"{name}: {accuracy:.4f}")