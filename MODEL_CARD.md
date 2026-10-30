# Model card - fraud-detection research prototype

**Status:** research / teaching prototype. Not validated for production or regulatory use.

## Models
| Model | Where | Data actually used for the reported numbers |
|---|---|---|
| ANN 64-32-16-1, FGSM-trained ANN, RF, XGBoost, LR, DT | `train_model.py`, `model_comparison.py`, `adversarial_test.py` (pre-existing) | Kaggle Credit Card (random 80/20 split) - saved artifacts only; **not re-run here** (dataset unavailable) |
| Autoencoder-"transfer" and kNN-"GNN" | `train_transfer.py`, `train_gnn.py` (pre-existing) | Kaggle; self-supervised pretraining on the same data and feature-similarity edges - **not** cross-dataset transfer or a real transaction graph |
| Transfer learning (source→target) | `transfer_learning.py` | SYNTHETIC domains |
| ANN, XGBoost, LSTM, GraphSAGE, learned late fusion | `train_multimodel.py` | SYNTHETIC temporal stream with account/device ids |
| PGD / FGSM robustness, PGD adversarial training | `adversarial_research.py` | SYNTHETIC (Kaggle automatically if `data/creditcard.csv` exists) |

## Intended use
Teaching, method comparison, reproducible experimentation. **Out of scope:** automated blocking decisions.

## Evaluation protocol (new experiments)
Chronological 60/20/20 split; preprocessing fit on training rows only; raw elapsed time not used as a feature;
early stopping and decision thresholds chosen on validation only; metrics = precision, recall, F1, PR-AUC, ROC-AUC, MCC.
Accuracy is not used as a headline metric (fraud is 0.17 % of Kaggle rows; 1.5-2 % in the synthetic data).

## Headline results (all SYNTHETIC; see `PAPER_CLAIMS_MATRIX.md`, CSVs in `artifacts/research/`)
- Transfer vs scratch (PR-AUC, 3 seeds): −0.080 at 10 % labels, −0.023 at 25 %, +0.001 at 50 %, −0.011 at 100 %. **No benefit was found.** The pretrained encoder itself works on its source domain (PR-AUC 0.75).
- Learned late fusion (PR-AUC 0.691) did **not** beat the best single branch (GraphSAGE 0.740) in one seed.
- PGD adversarial training raised PR-AUC under PGD at ε=0.1 from 0.375 to 0.428, but at ε=0.3 both models collapse (0.050 / 0.067).

## Limitations and ethical notes
Small test sets (~60 synthetic frauds) → wide uncertainty; single seed for fusion/robustness/drift. Anonymised PCA features
prevent physical interpretation of perturbations. Real fraud data shifts, has label delay and fairness issues that synthetic data does not capture.
