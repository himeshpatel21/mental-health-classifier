# Research Journey — Mental Health Classifier

This document captures my thinking process, decisions, dead ends, and learnings while building this project. It's meant to show the *why* behind every technical choice, not just the *what*.

---

## The Starting Question

Mental health detection from text is a real problem. Clinicians can't read millions of tweets. Can a model flag distress signals reliably enough to be useful?

I wanted to understand:
1. What kind of model is suited for this (domain knowledge matters)
2. How much the base model choice affects fine-tuning outcomes
3. What "good enough" even means for a mental health classifier — is accuracy the right metric?

---

## Phase 1 — Understanding the Problem Domain

### Why not just use BERT-base?

My first instinct was to reach for `bert-base-uncased` — it's the default for text classification tutorials. But I paused and asked: what does BERT actually "know" about mental health language?

BERT was trained on Wikipedia and BooksCorpus. People don't write about depression on Wikipedia. The vocabulary of mental health distress — informal expressions, fragmented sentences, crisis language — is almost entirely absent from BERT's pre-training distribution.

This led me to look for domain-specific pre-trained models. I found two serious candidates:

| Model | Pre-training data | Notes |
|-------|------------------|-------|
| `mental/mental-bert-base-uncased` | Mental health subreddits + Twitter | Matches our task domain closely |
| `mental/mental-roberta-base` | Same corpus, RoBERTa architecture | Slightly better on some benchmarks |

I chose **MentalBERT** over MentalRoBERTa because:
- Same architecture as BERT → familiar training loop, no surprises
- The dataset is Twitter-based → MentalBERT's pre-training included Twitter data specifically
- Lower compute cost for a first pass — I can always upgrade to RoBERTa later

**Learning:** Domain-adapted pre-training is more important than architecture when the input distribution is narrow and specialized. A weaker model pre-trained on the right data will beat a stronger model pre-trained on the wrong data.

---

## Phase 2 — EDA and the max_length Decision

### Why 128 tokens?

This was not arbitrary. I looked at the word length distribution of the dataset:

- Mean tweet: ~20 words
- 95th percentile: ~45 words

BERT tokenizes sub-word, so a 45-word tweet might produce ~60–70 tokens. Setting max_length=128 means we capture 95%+ of tweets with no truncation, while keeping memory and compute manageable.

max_length=256 would have doubled the memory per sample with almost no coverage gain on this dataset. max_length=64 would have truncated ~20% of tweets — including potentially the most important crisis-signal words that appear later in the text.

**Learning:** Always do EDA before setting tokenizer parameters. The "standard" 512 from the BERT paper is for full document tasks. Twitter is not a full document task.

### The class imbalance question

I checked the label distribution early. This matters because if 90% of tweets are "not distressed," a model that always predicts 0 gets 90% accuracy — completely useless clinically.

This is why I added **F1 score and AUC-ROC** as primary metrics alongside accuracy, and set `metric_for_best_model = "f1"` in TrainingArguments. The best checkpoint is selected by F1, not accuracy.

**Learning:** For medical/mental health tasks, the cost of a false negative (missing a distress signal) is much higher than a false positive. This should drive metric choice.

---

## Phase 3 — Training Setup Decisions

### The 80/10/10 split with stratification

I used stratified splitting to ensure both classes appear in the same ratio across train, validation, and test sets. Without stratification, random chance could put most of one class into a single split — especially dangerous with imbalanced data.

### TrainingArguments choices

```python
learning_rate = 2e-5       # Standard for BERT fine-tuning, conservative
num_train_epochs = 3       # Enough for convergence, not enough to overfit
warmup_steps = 200         # ~10% of total steps, standard practice
weight_decay = 0.01        # L2 regularization — reduces overfitting
fp16 = True                # Half precision: ~2x faster on GPU, same results
```

The `fp16=True` flag was important for Colab efficiency. Without it, training on a T4 would take ~35 minutes. With it: ~15 minutes. Same final metrics.

**Why 3 epochs specifically?** BERT-based models for text classification typically converge between 2–5 epochs on moderately-sized datasets. I started with 3 as a conservative middle ground. The validation F1 plateauing after epoch 2 confirmed this was the right range for this dataset.

### load_best_model_at_end

This is a small detail with a big effect. Without it, the Trainer saves the last checkpoint — which might be epoch 3 even if epoch 2 was better on validation F1. With it, the final model is the best checkpoint across all epochs. Always use this.

---

## Phase 4 — Evaluation Thinking

### Why confusion matrix matters here

The classification report gives macro averages. But for this task, the specific cells of the confusion matrix tell a more important story:

- **False Negatives (FN):** Distressed tweet predicted as "not distressed" — the model misses someone in crisis
- **False Positives (FP):** Normal tweet predicted as "distressed" — a false alarm

In a clinical-adjacent system, FN is the dangerous error. I logged the confusion matrix explicitly in `compute_metrics` to track this during training, not just at the end.

### SHAP for explainability

I added SHAP token attribution to understand *why* the model makes predictions, not just *what* it predicts. This was motivated by the "black box" problem — a mental health classifier that can't explain its reasoning is harder to trust.

The SHAP values show which tokens push the model toward "distressed" vs "not distressed." This is useful for:
- Catching spurious correlations (e.g., if the model learned to key on specific usernames in the dataset)
- Building intuition for what mental health language patterns the model learned

---

## Phase 5 — Deployment Decision

### Why Hugging Face Spaces over localtunnel

I experimented with localtunnel inside Colab during development. It works for live testing but dies the moment the Colab session ends. It's a dev tool, not a deployment.

Hugging Face Spaces gives:
- Persistent hosting (survives session ends)
- Model loading directly from HF Hub (no file management)
- A shareable URL that actually works for anyone

The app.py was written to load the model from `wizardpatel/mental-health-bert` at startup with `@st.cache_resource` — this means the model loads once and stays in memory for subsequent requests. Without caching, every new user request would re-download 440MB.

---

## What I Would Do Differently

1. **Try class weighting:** Pass a `class_weight` parameter to handle imbalance more explicitly instead of relying on metric selection alone
2. **Compare against MentalRoBERTa:** Run the same training loop with the RoBERTa variant and compare final test F1
3. **Add confidence thresholding:** Instead of argmax, use a probability threshold (e.g., only flag as distressed if P > 0.7) to reduce false positives
4. **More thorough SHAP analysis:** Look at population-level SHAP values across the full test set to understand global feature importance, not just per-sample

---

## Key Things I Learned

- Domain-specific pre-training > architecture choice for narrow NLP tasks
- EDA drives tokenizer decisions — never use default lengths without checking the data
- F1 is the right metric for imbalanced classification; accuracy is misleading
- `load_best_model_at_end=True` is non-optional for serious training runs
- `@st.cache_resource` is the difference between a usable and unusable deployed model
- The confusion matrix is more informative than aggregate metrics for safety-adjacent tasks

