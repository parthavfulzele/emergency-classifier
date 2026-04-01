import os
import re
import json
import joblib
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "classifier.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "model", "metrics.json")

st.set_page_config(page_title="Emergency Classifier", page_icon="🚨", layout="centered")


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


# Sidebar
st.sidebar.title("Model Info")
if os.path.exists(METRICS_PATH):
    metrics = load_metrics()
    st.sidebar.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
    st.sidebar.metric("CV Accuracy", f"{metrics['cv_accuracy']:.1%}")
    st.sidebar.metric("Emergency F1", f"{metrics['emergency_f1']:.1%}")
    st.sidebar.metric("Non-Emergency F1", f"{metrics['non_emergency_f1']:.1%}")
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Dataset", metrics["total_samples"])
    st.sidebar.metric("Training Samples", metrics["train_samples"])
    st.sidebar.metric("Test Samples", metrics["test_samples"])
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Best Model:** {metrics['best_classifier']}")
st.sidebar.markdown("**Features:** Word + Char N-grams (TF-IDF)")

# Main
st.title("Emergency Text Classifier")
st.markdown("Enter a text description and the model will classify it as an **emergency** or **non-emergency**.")

text_input = st.text_area("Enter text to classify:", height=120, placeholder="e.g. There's a fire in the building...")

if st.button("Classify", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    elif not os.path.exists(MODEL_PATH):
        st.error("Model not found. Run `python model/train.py` first.")
    else:
        model = load_model()
        cleaned = preprocess(text_input)
        prediction = model.predict([cleaned])[0]
        probabilities = model.predict_proba([cleaned])[0]
        confidence = probabilities[prediction]

        if prediction == 1:
            st.error(f"**EMERGENCY** — Confidence: {confidence:.1%}")
        else:
            st.success(f"**Non-Emergency** — Confidence: {confidence:.1%}")

        col1, col2 = st.columns(2)
        col1.metric("Emergency", f"{probabilities[1]:.1%}")
        col2.metric("Non-Emergency", f"{probabilities[0]:.1%}")
