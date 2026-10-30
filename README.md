# ðŸ¦ Adversarially Robust, Blockchain-Audited Fraud Detection (B.Tech research project)

Multi-model fraud-detection research code: ANN / XGBoost / LSTM / GraphSAGE with learned late fusion, transfer learning,
FGSM/PGD robustness, SHAP + LIME, ADWIN/DDM drift monitoring with a safe retraining workflow, a tamper-evident audit log
(local hash chain + Solidity contract) and a Streamlit dashboard.



## Quick start (Windows 11 PowerShell - details in START_HERE.md)
```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m pytest -q
python transfer_learning.py --quick
python -m streamlit run app.py
```

## What is where
| Area | Files |
|---|---|
| Original pipeline (Kaggle data) | `train_model.py`, `model_comparison.py`, `adversarial_test.py`, `explainability.py`, `train_transfer.py`, `train_gnn.py` (unchanged) |
| Shared library | `fraudlab/` (synthetic data, chronological split, preprocessing, metrics, NN, adversarial, sequences, graph, fusion, drift, registry, explain) |
| Transfer learning | `transfer_learning.py` â†’ `artifacts/research/transfer/` |
| Multi-model + fusion | `train_multimodel.py` â†’ `artifacts/research/multimodel/` |
| Robustness (FGSM, PGD, PGD-AT) | `adversarial_research.py` â†’ `artifacts/research/adversarial/` |
| Explainability | `explain_research.py` (SHAP, LIME), original `explainability.py` |
| Drift + retraining | `drift_demo.py`, `fraudlab/drift.py`, `fraudlab/registry.py` |
| Audit | `audit_chain.py`, `blockchain_demo.py`, `contracts/FraudAudit.sol`, `contracts/test/` (Hardhat) |
| Dashboard | `app.py` (8 original tabs + 4 research tabs), `research_dashboard.py` |
| Tests | `tests/` (62 pytest tests), `contracts/test/` (9 Hardhat tests) |

## Results (SYNTHETIC data; one machine; see CSVs for all metrics)
| Experiment | Outcome |
|---|---|
| Transfer vs scratch, PR-AUC (3 seeds) | âˆ’0.080 / âˆ’0.023 / +0.001 / âˆ’0.011 at 10 / 25 / 50 / 100 % target labels - **no benefit found** |
| Multi-model, test PR-AUC | GraphSAGE 0.740, LSTM 0.693, learned fusion 0.691, simple average 0.677, ANN 0.622, XGBoost 0.593 - **fusion did not beat the best branch** |
| PGD (Îµ=0.1), PR-AUC | standard ANN 0.375 vs PGD-adversarially-trained 0.428; at Îµ=0.3 both â‰ˆ0.05-0.07 - **no general robustness** |
| Drift (synthetic stream) | ADWIN alarm 597 and DDM 445 transactions after the drift point; DDM raised 2 false alarms before it |
| Audit | 62/62 pytest and 9/9 Hardhat tests pass; no public-testnet deployment |

Pre-existing Kaggle results (`artifacts/model_comparison.csv` etc.) are internally consistent but **not reproduced here**.
The former README table (99.05 % / 84.60 %) matched no artifact and was removed.

## Design guarantees (all unit-tested)
Chronological splits; preprocessing fit on train only; thresholds chosen on validation; causal LSTM windows and graph edges
(runtime assertions); fusion trained only on out-of-sample predictions; attacks respect Îµ-budget, training-range bounds and
immutable features; no graph is fabricated for data without entity ids; candidate models are never auto-promoted.

## Data
Put Kaggle `creditcard.csv` in `data/` (git-ignored) for the original pipeline. `python generate_synthetic_data.py` writes
labelled synthetic CSVs. Perturbing anonymised PCA features is a mathematical test, not a physical attack.

## Git workflow
```powershell
git checkout feature/research-extension
git push -u origin feature/research-extension     # then open a pull request; main is untouched
```

## ðŸŽ“ Academic Context

Built for the final-year project *"Adversarially Robust Fraud Detection in
Banking Using Machine Learning and Blockchain,"* Department of Computer
Science & Engineering, Meerut Institute of Engineering & Technology.

## ðŸ“„ License

MIT â€” feel free to fork and build on this for your own project.

## ðŸ•¸ï¸ GNN design note

The source Kaggle dataset contains anonymized transaction features but no customer,
account, card, or merchant identifiers. Therefore, `train_gnn.py` builds a
feature-similarity neighbourhood using the `k` nearest standardized training
transactions. It combines each transaction's own features with the mean features
of its neighbours in a neural model. This is a **GraphSAGE-inspired
neighbour-aggregation baseline**, not a graph of real banking relationships.
For a production-style relational GNN, use a dataset with real entity IDs and
time-aware edges, and avoid using test labels during graph construction.

## ðŸ” Transfer-learning design note

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
