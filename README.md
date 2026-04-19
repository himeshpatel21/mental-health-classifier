# Mental Health Classifier 🧠

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Model-wizardpatel%2Fmental--health--bert-blue)](https://huggingface.co/wizardpatel/mental-health-bert)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Spaces-Live%20Demo-green)](https://huggingface.co/spaces/wizardpatel/mental-health-classifier)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/mental-health-classifier/blob/main/notebook/mental_health_final.ipynb)

Fine-tuned **MentalBERT** classifier that detects mental health distress signals in tweets. Binary classification: `Distressed (1)` vs `Not distressed (0)`.

---

## Live Demo

> Try it directly on Hugging Face Spaces — no setup needed.

[![App Screenshot](assets/app_screenshot.png)](https://huggingface.co/spaces/wizardpatel/mental-health-classifier)

---

## Pipeline

![Pipeline Flow](assets/pipeline_flow.png)

---

## Results

Evaluated on a held-out test set never seen during training.

| Metric | Score |
|--------|-------|
| Accuracy | 0.9160 |
| F1 Score | 0.9172 |
| AUC-ROC | 0.9759 |

> Fill in your scores from the `classification_report` output in the notebook.

### Sample predictions

![Sample Predictions](assets/sample_predictions.png)

---

## Model

| Property | Value |
|----------|-------|
| Base model | `mental/mental-bert-base-uncased` |
| Dataset | Mental-Health-Twitter |
| Task | Binary sequence classification |
| Max token length | 128 |
| Training epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 32 train / 64 eval |
| Precision | fp16 |
| Best checkpoint | by F1 on validation set |

The base model (MentalBERT) was pre-trained on mental health-specific Reddit and Twitter corpora, making it a stronger starting point than general BERT for this task.

---

## Dataset

**Mental-Health-Twitter** — a labelled collection of tweets annotated for mental health distress signals.

Key observations from EDA:
- 95th percentile tweet length is under 50 words → max_length=128 covers the dataset well
- Binary labels with moderate class imbalance (explored in notebook)

---

## Repo Structure

```
mental-health-classifier/
│
├── notebook/
│   └── mental_health_main.ipynb   # Full training notebook (Colab-ready)
│
├── research/
│   └── thought_process.md         # Thinking process, experiments, learnings
│
├── assets/
│   ├── pipeline_flow.png
│   ├── sample_predictions.png
│   └── app_screenshot.png
│
├── app.py                          # Streamlit app (deployed on HF Spaces)
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/himeshpatel21/mental-health-classifier
cd mental-health-classifier
pip install -r requirements.txt
streamlit run app.py
```

The app loads the model directly from Hugging Face Hub — no local model download needed.

---

## Run the Notebook

The notebook is designed for **Google Colab with GPU** (T4 or better).

1. Open via the badge above or upload `notebook/mental_health_main.ipynb` to Colab
2. Set runtime → T4 GPU
3. Add your `HF_TOKEN` to Colab Secrets
4. Run all cells top to bottom

---

## Research & Learning

Curious about the reasoning behind model choice, tokenizer decisions, and what I tried before getting here?

→ Read [`research/thought_process.md`](research/thought_process.md)

---

## Tech Stack

`transformers` · `datasets` · `torch` · `scikit-learn` · `shap` · `streamlit` · `huggingface-hub`
