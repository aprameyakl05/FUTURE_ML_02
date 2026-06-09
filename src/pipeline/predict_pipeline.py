import joblib
print("Predict pipeline started...")
model = joblib.load(
    "models/ticket_classifier.pkl"
)

while True:

    text = input(
        "\nEnter Ticket Text (or quit): "
    )

    if text.lower() == "quit":
        break

    prediction = model.predict(
        [text]
    )[0]

    probability = max(
        model.predict_proba([text])[0]
    )

    print(
        f"\nPredicted Topic: {prediction}"
    )

    print(
        f"Confidence: {probability:.2%}"
    )