# Advanced Machine Learning: Mid-Semester Examination
### Part A — Research Paper Selection

**Student:** Prashant Kumar
**Roll Number:** 230101
**Course:** Advanced Machine Learning (Semester 6)
**University:** NST, Rishihood University (Sonipat)

---

## Selected Research Paper

| Field | Details |
|---|---|
| **Title** | Finding Deceptive Opinion Spam by Any Stretch of the Imagination |
| **Authors** | Myle Ott, Yejin Choi, Claire Cardie, Jeffrey T. Hancock |
| **Year** | 2011 |
| **Venue** | Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011) |
| **CORE Ranking** | A* |
| **Primary Method** | Support Vector Machine (SVM) |
| **Official Page** | https://aclanthology.org/P11-1032/ |
| **Free PDF** | https://aclanthology.org/P11-1032.pdf |

---

## Paper Summary

This paper investigates whether automated classifiers can detect **deceptive opinion spam** — fake hotel reviews written by paid workers — better than human judges. The authors collect a gold-standard dataset of 400 truthful and 400 deceptive positive reviews for 20 Chicago hotels, then train and compare three SVM-based classifiers using different feature sets:

- **n-gram SVM** — bag-of-words unigram and bigram word frequency features
- **LIWC-SVM** — 80 psycholinguistic feature categories (spatial language, family words, emotional words, etc.)
- **POS-SVM** — part-of-speech tag frequency features

The key finding is that the SVM classifier achieves approximately **89% accuracy**, far outperforming human judges who perform near random chance (~60%). The paper also identifies that **deceptive reviews use more imaginative and emotional language** while truthful reviews use more **specific, factual, and spatial language** (words like "bathroom", "location", "small").

---

## Why This Paper

Safeguarding customer trust is critical for modern digital storefronts. As I develop the Animaniac platform, ensuring that the "Live Your Arc" brand experience isn't compromised by fake product reviews is a top priority. This paper provides a highly interpretable, computationally lightweight baseline for detecting deceptive opinion spam using classical machine learning techniques, making it directly applicable to real-world e-commerce integrity problems.

---

## Dataset

| Field | Details |
|---|---|
| **Name** | Deceptive Opinion Spam Corpus (op_spam_v1.4) |
| **Released by** | Myle Ott (original paper author) |
| **Direct Download** | https://myleott.com/op_spam_v1.4.zip |
| **Mirror (Kaggle)** | https://www.kaggle.com/rtatman/deceptive-opinion-spam-corpus |
| **Size** | 1,600 hotel reviews |
| **Format** | Plain text files organized in folders |
| **License** | Freely available for research use |

### Dataset Structure

```
op_spam_v1.4/
├── positive_polarity/
│   ├── deceptive_from_MTurk/     # 400 fake positive reviews (Amazon Mechanical Turk)
│   └── truthful_from_TripAdvisor/ # 400 real positive reviews (TripAdvisor)
└── negative_polarity/
    ├── deceptive_from_MTurk/     # 400 fake negative reviews
    └── truthful_from_Web/        # 400 real negative reviews (Expedia, Hotels.com, etc.)
```

The dataset covers **20 Chicago hotels**. Each folder contains reviews organized by hotel and fold (1–5) for cross-validation, exactly matching the paper's experimental setup.

---

## Reproduction Pipeline

The entire paper can be reproduced in approximately 25 lines of Python using only standard libraries:

```python
import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Step 1: Load dataset
texts, labels = [], []
for label, folder in [(1, "deceptive_from_MTurk"), (0, "truthful_from_TripAdvisor")]:
    path = f"op_spam_v1.4/positive_polarity/{folder}"
    for fold in os.listdir(path):
        for fname in os.listdir(f"{path}/{fold}"):
            with open(f"{path}/{fold}/{fname}", encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label)

# Step 2: Build pipeline (TF-IDF + Linear SVM)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), norm="l2")),
    ("svm", LinearSVC(C=1.0))
])

# Step 3: 5-fold cross-validation (matches paper's evaluation method)
scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="accuracy")
print(f"Mean Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
# Expected output: ~0.87-0.89 (matching paper's reported results)
```

**Requirements:** `scikit-learn`, `numpy` — both installable via `pip install scikit-learn numpy`
**Hardware:** Any standard CPU-only laptop. Runs in under 10 seconds.

---

## Key Concepts for Understanding This Paper

### How SVM Works Here

1. Each review is converted into a **sparse TF-IDF vector** — every unique word/bigram in the vocabulary becomes a dimension, with its TF-IDF weight as the value
2. The SVM finds a **hyperplane** in this high-dimensional space that maximally separates deceptive reviews from truthful ones
3. The **margin** — the gap between the hyperplane and the nearest data points (support vectors) — is maximized during training
4. For a new review, its TF-IDF vector is computed and the **side of the hyperplane** it falls on determines the prediction

### Why Humans Fail but SVM Succeeds

Human judges perform at ~60% accuracy — barely above random chance. They consciously look for obvious signals like excessive positivity or vague language, but miss subtle **statistical patterns** across thousands of word frequencies simultaneously. The SVM detects these patterns automatically because it operates in a space with as many dimensions as the vocabulary size.

### Key Failure Cases

| Failure Mode | Description |
|---|---|
| **Cross-domain** | Model trained on hotel reviews fails on restaurant or product reviews |
| **Class imbalance** | Paper assumes 50/50 split; real-world fake review ratio is closer to 1:100 |
| **Crowdsourced vs professional spam** | Mechanical Turk workers write differently from professional review farms |
| **Vocabulary shift** | New slang and writing styles not present in 2011 training data |
| **Negation blindness** | Bag-of-words ignores "not good" vs "good" distinction |

---

## Repository Structure

```
230101-midsem/
├── README.md                  ← This file
└── llm_usage_partA.json       ← Mandatory LLM usage disclosure (Part A)
```

---

## LLM Usage Disclosure

All LLM interactions used during the paper selection process are fully documented in [`llm_usage_partA.json`](./llm_usage_partA.json).

**Tools used:** Claude (Anthropic), Gemini (Google)
**Total interactions logged:** 7
**Purpose:** Paper selection assistance, feasibility verification, dataset availability checking, venue ranking confirmation, and method alignment evaluation

As required by the exam instructions, all LLM usage is disclosed completely and honestly. All final decisions, verifications, and understanding are my own.

---

## References

- Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). Finding deceptive opinion spam by any stretch of the imagination. In *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies* (pp. 309–319). https://aclanthology.org/P11-1032/
- Original dataset: https://myleott.com/op_spam_v1.4.zip
- ACL Anthology: https://aclanthology.org/P11-1032/
- CORE Rankings Portal: https://portal.core.edu.au/conf-ranks/
