import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap

st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="🧠",
    layout="centered"
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 750px; }
    .result-distress {
        background: #2d1515;
        border-left: 4px solid #e05252;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 16px 0;
    }
    .result-coping {
        background: #122d1a;
        border-left: 4px solid #4caf7d;
        border-radius: 10px;
        padding: 18px 22px;
        margin: 16px 0;
    }
    .result-label {
        font-size: 20px;
        font-weight: 700;
        margin: 0 0 4px 0;
    }
    .result-conf {
        font-size: 14px;
        opacity: 0.75;
        margin: 0;
    }
    .label-distress { color: #e05252; }
    .label-coping   { color: #4caf7d; }
    .bar-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 6px 0;
        font-size: 14px;
    }
    .bar-label { width: 140px; flex-shrink: 0; }
    .bar-track {
        flex: 1;
        background: #2a2a2a;
        border-radius: 6px;
        height: 14px;
        overflow: hidden;
    }
    .bar-fill-d { height: 100%; border-radius: 6px; background: #e05252; }
    .bar-fill-c { height: 100%; border-radius: 6px; background: #4caf7d; }
    .bar-pct { width: 44px; text-align: right; flex-shrink: 0; }
    .shap-note {
        font-size: 12px;
        color: #888;
        margin-bottom: 8px;
    }
    .disclaimer {
        font-size: 12px;
        color: #666;
        border-top: 1px solid #222;
        padding-top: 12px;
        margin-top: 32px;
    }
</style>
""", unsafe_allow_html=True)

# ── model ─────────────────────────────────────────────────────────────────────
MODEL_REPO = "wizardpatel/mental-health-bert"

@st.cache_resource(show_spinner="Loading model from Hugging Face Hub…")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ── prediction ────────────────────────────────────────────────────────────────
def predict(text):
    inputs = tokenizer(
        text, return_tensors="pt",
        max_length=128, truncation=True, padding="max_length"
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred  = int(np.argmax(probs))
    # label 1 = distressed, label 0 = not distressed (matches dataset convention)
    return pred, probs

# ── SHAP ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_explainer():
    def pipeline(texts):
        inputs = tokenizer(
            list(texts), return_tensors="pt",
            max_length=128, truncation=True, padding="max_length"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.softmax(logits, dim=1).cpu().numpy()
    return shap.Explainer(pipeline, tokenizer)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧠 Mental Health Classifier")
st.caption(
    "Fine-tuned **MentalBERT** on the Mental-Health-Twitter dataset · "
    "Detects distress signals in tweet-style text"
)
st.markdown("---")

text = st.text_area(
    "Enter a tweet or short text:",
    placeholder="e.g.  I haven't slept in days, the panic attacks are getting worse…",
    height=120
)

col_btn, col_shap_toggle = st.columns([1, 3])
run       = col_btn.button("Analyse", type="primary", use_container_width=True)
show_shap = col_shap_toggle.checkbox("Show word-level explanation (SHAP)", value=True)

if run:
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        # ── prediction result ──────────────────────────────────────────────
        with st.spinner("Running model…"):
            pred, probs = predict(text)

        is_distress  = pred == 1
        label_text   = "Distress detected"   if is_distress else "Coping / Not distressed"
        label_class  = "label-distress"       if is_distress else "label-coping"
        card_class   = "result-distress"      if is_distress else "result-coping"
        confidence   = probs[pred] * 100

        st.markdown(f"""
        <div class="{card_class}">
            <p class="result-label {label_class}">{label_text}</p>
            <p class="result-conf">Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # ── confidence bars ────────────────────────────────────────────────
        p_distress = probs[1] * 100
        p_coping   = probs[0] * 100

        st.markdown(f"""
        <div class="bar-row">
            <span class="bar-label label-distress">Distress</span>
            <div class="bar-track"><div class="bar-fill-d" style="width:{p_distress:.1f}%"></div></div>
            <span class="bar-pct">{p_distress:.1f}%</span>
        </div>
        <div class="bar-row">
            <span class="bar-label label-coping">Coping / Normal</span>
            <div class="bar-track"><div class="bar-fill-c" style="width:{p_coping:.1f}%"></div></div>
            <span class="bar-pct">{p_coping:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # ── SHAP explanation ───────────────────────────────────────────────
        if show_shap:
            st.markdown("---")
            st.markdown("#### Why did the model decide this?")
            st.markdown(
                '<p class="shap-note">'
                '🔴 Red words pushed toward <b>distress</b> · '
                '🔵 Blue words pushed toward <b>coping</b> · '
                'Intensity = strength of influence'
                '</p>',
                unsafe_allow_html=True
            )
            with st.spinner("Computing SHAP values… (takes ~10 sec)"):
                try:
                    explainer  = get_explainer()
                    shap_vals  = explainer([text])
                    shap_html  = shap.plots.text(shap_vals[0, :, pred], display=False)
                    st.components.v1.html(shap_html, height=280, scrolling=True)
                except Exception as e:
                    st.info(f"SHAP explanation unavailable on this input: {e}")

# ── footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="disclaimer">'
    'Model: <a href="https://huggingface.co/wizardpatel/mental-health-bert" target="_blank">'
    'wizardpatel/mental-health-bert</a> · '
    'Base: mental/mental-bert-base-uncased · '
    'This is a research demo — not a clinical tool.'
    '</p>',
    unsafe_allow_html=True
)
