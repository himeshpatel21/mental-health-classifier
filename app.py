import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="🧠",
    layout="centered"
)

MODEL_REPO = "wizardpatel/mental-health-bert"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, probs

st.title("🧠 Mental Health Classifier")
st.caption("Detects mental health distress signals in text using a fine-tuned MentalBERT model.")
st.markdown("---")

with st.spinner("Loading model from Hugging Face Hub..."):
    tokenizer, model = load_model()

text = st.text_area(
    "Enter a tweet or short text to analyze:",
    placeholder="e.g. I've been feeling really hopeless lately and can't get out of bed...",
    height=120
)

if st.button("Analyze", type="primary"):
    if text.strip():
        pred, probs = predict(text, tokenizer, model)

        label = "Distressed" if pred == 1 else "Not distressed"
        confidence = probs[pred] * 100

        if pred == 1:
            st.error(f"**{label}** — {confidence:.1f}% confidence")
        else:
            st.success(f"**{label}** — {confidence:.1f}% confidence")

        st.markdown("**Probability breakdown:**")
        col1, col2 = st.columns(2)
        col1.metric("Not distressed", f"{probs[0]*100:.1f}%")
        col2.metric("Distressed", f"{probs[1]*100:.1f}%")

        st.markdown("---")
        st.caption(
            "Model: [wizardpatel/mental-health-bert](https://huggingface.co/wizardpatel/mental-health-bert) · "
            "Base: mental/mental-bert-base-uncased · "
            "This is a research project, not a clinical tool."
        )
    else:
        st.warning("Please enter some text to analyze.")
