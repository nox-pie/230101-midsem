## Why this folder contains no dataset file

The dataset used in Question 2 and Question 3 is a **149-message SMS spam subset** embedded directly as Python lists inside each notebook (`task_2_1.ipynb`, `task_2_2.ipynb`, `task_2_3.ipynb`, `task_3_1.ipynb`, `task_3_2.ipynb`).

No external file download, login, or manual step is required to reproduce any result. Running any of the above notebooks from top to bottom in a clean Python environment fully reconstructs the dataset in memory.

---

## Dataset: SMS Spam Collection (embedded subset)

| Property | Value |
|---|---|
| Total samples | 149 |
| Spam (label = 1) | 75 (50.3%) |
| Ham (label = 0) | 74 (49.7%) |
| Language | English |
| Text type | Short SMS messages |
| Average word count | ~13 words per message |

### Original source

The messages are representative of the **UCI SMS Spam Collection Dataset**:

> Almeida, T.A., Gómez Hidalgo, J.M., and Yamakami, A. (2011).  
> *Contributions to the Study of SMS Spam Filtering: New Collection and Results.*  
> Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG).  
> UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The full corpus contains 5,574 labelled SMS messages. The 149-message subset used here was hand-selected to be class-balanced and representative of both classes for the purposes of this reproduction study.

---

## Why this dataset was chosen

The SMS spam classification task is structurally parallel to the deceptive opinion spam detection task in Ott et al. (2011):

- Both are **binary text classification** problems (spam/ham ↔ deceptive/truthful)
- Both use **short natural language text** as input and a binary label as output
- Both are evaluated with **classification accuracy**, the same metric reported in Table 3 of the paper
- Both datasets are **near-balanced** (~50/50), satisfying Assumption 1 from Task 1.2

The key difference — and the honest limitation documented in Task 2.1 and Task 2.3 — is that SMS spam contains explicit high-frequency trigger words (`FREE`, `WIN`, `URGENT`, `prize`) that are far more lexically discriminative than the subtle contextual signals in hotel reviews, explaining why the reproduction achieves higher accuracy than the paper.

---

## How to access the full UCI dataset (optional)

If you wish to run experiments on the full 5,574-message corpus:

```bash
# Download from UCI ML Repository
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
unzip smsspamcollection.zip
# File: SMSSpamCollection (tab-separated: label \t message)
```

The embedded subset in the notebooks is sufficient to reproduce all reported results and requires no external download.

---

## Label mapping (parallel to Ott et al. 2011)

| This dataset | Ott et al. (2011) | Integer label |
|---|---|---|
| Spam | Deceptive review | `1` |
| Ham | Truthful review | `0` |
