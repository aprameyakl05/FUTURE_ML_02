import streamlit as st
import joblib

# Page Config
st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="centered"
)

# Load Model
model = joblib.load(
    "models/ticket_classifier.pkl"
)

# Title
st.title("🎫 Support Ticket Classifier")

st.write(
    "Classify IT support tickets into the appropriate department."
)

# Input
ticket_text = st.text_area(
    "Enter Ticket Description"
)

# Button
if st.button("Classify Ticket"):

    if ticket_text.strip():

        prediction = model.predict(
            [ticket_text]
        )[0]

        confidence = max(
            model.predict_proba(
                [ticket_text]
            )[0]
        )

        st.success(
            f"Predicted Department: {prediction}"
        )
        st.progress(float(confidence))

        st.info(
            f"Confidence Score: {confidence:.2%}"
        )

    else:
        st.warning(
            "Please enter a ticket description."
        )