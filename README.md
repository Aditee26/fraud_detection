# 🏦 Adversarially Robust Fraud Detection in Banking (ML + Blockchain)

A working prototype demonstrating fraud detection with a Keras ANN benchmarked
against classical ML models, adversarial robustness testing (FGSM +
adversarial training), SHAP explainability, and a tamper-evident
blockchain-style audit trail — built as a B.Tech final-year project.

## ✨ Features

- **ANN fraud classifier** (Dense 64→32→16→1, dropout 0.3) trained with
  class weights to handle heavy class imbalance
- **Trained on the real Kaggle Credit Card Fraud Detection dataset** —
  284,807 transactions, 492 fraud cases (0.17%). The pipeline requires the
  real dataset and will not run on synthetic/placeholder data.
- **5-model benchmark** (`model_comparison.py`) — Logistic Regression,
  Decision Tree, Random Forest, XGBoost, and the ANN, all evaluated on an
  identical held-out test split for a fair comparison
- **Adversarial robustness testing** (`adversarial_test.py`) — real FGSM
  attack, measured clean vs. adversarial accuracy/precision/recall/F1, then
  adversarial training to recover robustness
- **Explainability** (`explainability.py`) — SHAP values (global + one
  per-transaction local explanation) on the XGBoost model
- **Blockchain-style audit trail** — local tamper-evident hash chain
  (`blockchain_demo.py`) + reference Solidity contract (`FraudAudit.sol`)
- **Graph-neighbour model** (`train_gnn.py`) — a GraphSAGE-inspired model
  using nearest-neighbour feature aggregation (clearly labelled as similarity
  edges because the dataset has no real account/customer IDs)
- **Transfer learning** (`train_transfer.py`) — pretrains an encoder with an
  unsupervised autoencoder task, then fine-tunes it for fraud classification
- **Streamlit UI** — 8 tabs, including GNN and transfer-learning evaluation

## 🧱 Tech Stack

`Python` · `TensorFlow / Keras` · `scikit-learn` · `XGBoost` · `SHAP` ·
`Streamlit` · `Solidity` (reference contract only)

## 📁 Project Structure

```
fraud_detection/
├── .streamlit/
│   └── config.toml            # forces a light UI theme
├── assets/
│   └── style.css               # dashboard styling (loaded by app.py)
├── artifacts/                  # generated: models, scaler, metrics, plots
├── data/
│   └── creditcard.csv          # NOT included — download separately, see below
├── app.py                      # Streamlit UI (8 tabs)
├── train_model.py              # trains the ANN on the real dataset, saves metrics + plots
├── train_gnn.py                # GNN-inspired nearest-neighbour aggregation model
├── train_transfer.py           # autoencoder pretraining + supervised fine-tuning
├── model_comparison.py         # benchmarks 4 more models against the ANN
├── explainability.py           # SHAP explainability on the XGBoost model
├── adversarial_test.py         # FGSM attack + adversarial training
├── blockchain_demo.py          # local tamper-evident hash chain
├── FraudAudit.sol              # reference Solidity smart contract
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

```bash
git clone https://github.com/Aditee26/fraud_detection.git
cd fraud_detection

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
```

### Get the dataset (required)

This project trains only on the real **Kaggle Credit Card Fraud Detection**
dataset — there is no synthetic-data fallback. Download it and place the
CSV at `data/creditcard.csv`:

| Dataset | Where to get it | File to place in `data/` |
|---|---|---|
| Credit Card Fraud Detection | [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `creditcard.csv` |

Easiest via the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
```

Verify it landed correctly:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/creditcard.csv'); print(len(df), 'rows,', (df.Class==1).sum(), 'fraud')"
```

Expected: `284807 rows, 492 fraud`.

### Train and launch

Run in this order — each step depends on artifacts from the previous one:

```bash
python train_model.py         # trains the ANN, saves artifacts/fraud_model.keras
python train_gnn.py           # trains graph-neighbour aggregation model
python train_transfer.py      # pretrains encoder, then fine-tunes classifier
python model_comparison.py    # benchmarks LogReg/DecisionTree/RandomForest/XGBoost vs the ANN
python explainability.py      # SHAP on the XGBoost model from model_comparison.py
python adversarial_test.py    # FGSM attack + adversarial training on the ANN
python -m streamlit run app.py
```

Then open **http://localhost:8501**. The dashboard includes separate GNN and
Transfer Learning tabs, plus a combined comparison of their saved metrics.

Once everything above has been run once, you only need
`python -m streamlit run app.py` for future sessions — all results are
loaded from `artifacts/`.

## 📊 Real Results

All numbers below come from an actual run on the full real dataset
(284,807 rows, 98 fraud cases in the test split) — regenerate with the
commands above if you retrain; numbers may shift slightly run to run.

### ANN baseline (`train_model.py`)

| Metric | Value |
|---|---|
| Recall | 91.84% (90 / 98 fraud caught) |
| Precision | 6.27% |
| F1 | 0.1173 |
| ROC-AUC | 0.9732 |
| PR-AUC | 0.7028 |

*(Accuracy is 97.62% but is not a meaningful metric on this dataset — a
model predicting "legitimate" for everything would score 99.83% accuracy
while catching zero fraud.)*

### 5-model comparison (`model_comparison.py`)

| Model | Precision | Recall | PR-AUC | MCC |
|---|---|---|---|---|
| Logistic Regression | 0.0610 | 0.9184 | 0.7289 | 0.2332 |
| Decision Tree | 0.0697 | 0.8469 | 0.6491 | 0.2397 |
| Random Forest | 0.8889 | 0.8163 | 0.8480 | 0.8516 |
| **XGBoost** | 0.8723 | 0.8367 | **0.8757** | **0.8541** |
| ANN | 0.0627 | 0.9184 | 0.7028 | 0.2365 |

XGBoost is the best overall model on this dataset — tree ensembles clearly
outperform the ANN on tabular fraud data here.

### Adversarial robustness (`adversarial_test.py`, epsilon = 0.1)

| | Normal ANN | Adversarially Trained ANN |
|---|---|---|
| Clean accuracy | 97.62% | 98.72% |
| Adversarial (FGSM) accuracy | 93.75% | **96.76%** |
| Adversarial precision | 2.42% | **4.52%** |
| Adversarial recall | 89.80% | 88.78% |

Adversarial training measurably improves robustness — accuracy and
precision under attack both improve substantially, with only a marginal
recall drop.

## 🔍 Explainability

SHAP on the XGBoost model shows `V14` and `V4` (anonymized PCA features
from the source dataset) as the most influential features globally. SHAP
values show what the model used to reach a decision — they are not proof
that a transaction is actually fraudulent.

## ⚠️ Known Limitations

- No concept drift detection, calibration, or SMOTE implemented yet
- Only FGSM tested (not PGD)
- Blockchain audit trail (`blockchain_demo.py`) exists but has not been
  independently verified/tested as part of this README's result set
- No relational/account/device data exists in this dataset, so no
  GraphSAGE/graph-based component is implemented

## 🎓 Academic Context

Built for the final-year project *"Adversarially Robust Fraud Detection in
Banking Using Machine Learning and Blockchain,"* Department of Computer
Science & Engineering, Meerut Institute of Engineering & Technology.

## 📄 License

MIT — feel free to fork and build on this for your own project.

## 🕸️ GNN design note

The source Kaggle dataset contains anonymized transaction features but no customer,
account, card, or merchant identifiers. Therefore, `train_gnn.py` builds a
feature-similarity neighbourhood using the `k` nearest standardized training
transactions. It combines each transaction's own features with the mean features
of its neighbours in a neural model. This is a **GraphSAGE-inspired
neighbour-aggregation baseline**, not a graph of real banking relationships.
For a production-style relational GNN, use a dataset with real entity IDs and
time-aware edges, and avoid using test labels during graph construction.

## 🔁 Transfer-learning design note

`train_transfer.py` first trains an autoencoder to reconstruct standardized
training features without using `Class`. It transfers the encoder weights into
a fraud classifier and fine-tunes the classifier using training labels. This is
self-supervised pretraining on the training split, not transfer from an external
banking dataset. Compare it against the ANN using PR-AUC, precision, recall,
F1, false positives, and the same held-out split; do not assume transfer learning
will improve results before the experiment runs.

### New training commands

```bash
python train_gnn.py
python train_transfer.py
python -m streamlit run app.py
```

Generated files:
- `artifacts/gnn_model.keras`
- `artifacts/gnn_metrics.json`
- `artifacts/gnn_reference.npz`
- `artifacts/pretrained_encoder.keras`
- `artifacts/transfer_model.keras`
- `artifacts/transfer_scaler.npz`
- `artifacts/transfer_metrics.json`
