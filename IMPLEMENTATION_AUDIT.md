# Implementation audit

Work was done on branch `feature/research-extension`, created from `origin/akshita` (commit `70670a2`, the most recent
branch and the only one containing `model_comparison.py`, `explainability.py`, `train_transfer.py`, `train_gnn.py`).
`main` (4 commits) lacks all of those; `tanu` is an older subset. Nothing on `main`, `akshita` or `tanu` was modified.
Remote push/PR was not possible from this environment (no GitHub credentials) - see PROGRESS.md.

## 1. Initial state (what was verified, not assumed)
| Component | Finding |
|---|---|
| Models | All 5 committed `.keras` models load (TF 2.21 / Keras 3.15). |
| Saved metrics | `metrics.json` and `model_comparison.csv`: precision/recall recomputed from TP/FP/FN match the stored values. **Not reproducible here** - the Kaggle CSV was unavailable and artifacts carry no data hash/commit. |
| Hash chain | Original `artifacts/fraud_blockchain.json` still verifies with the refactored code. |
| Dashboard | Hard-stops unless `data/creditcard.csv` **and** artifacts exist (so it is unusable without the 144 MB file). 8 tabs (README said 4/6). |

## 2. Problems found
1. **README is inaccurate:** lists 4 tabs, files `data_loader.py`/`generate_demo_data.py` that do not exist, a synthetic-data fallback the code does not implement, and results (ANN 99.05 % / 84.60 % FGSM accuracy) that appear in no committed artifact (the committed JSON says 97.62 % / 93.75 % at ε=0.1; the `main` branch JSON says 97.0 % / 63.3 % at ε=0.3).
2. **Splits:** random stratified split; raw `Time` used as a feature; fixed 0.5 threshold - not chronological, thresholds not validation-selected.
3. **"Transfer learning"** (`train_transfer.py`) is an autoencoder pretrained on the same dataset, not transfer between datasets. **"GNN"** (`train_gnn.py`) is a nearest-neighbour feature-similarity model; Kaggle data has no account/device ids. (Both scripts' docstrings say so; the dashboard/README wording overstates.)
4. **`adversarial_test.py`:** FGSM only, no feature bounds/immutables; adversarial training uses `validation_split` on `vstack([clean, adv])`, so the validation tail consists of perturbed copies of training rows (leaky model selection). Original files were left unchanged.
5. **Audit:** `verify_integrity` ignored index order/genesis; module imported TensorFlow; no tests; `FraudAudit.sol` stored plain `transactionId` strings, allowed `riskScoreBps` up to 65535 and had no tests.
6. `artifacts/gnn_reference.npz` (25 MB) contains standardised features of 227,845 Kaggle rows, contradicting "dataset stays external". Left in place; excluded from the release ZIP.
7. `.agents/` and `.claude/` contain broken symlinks into a local `.venv`; `package-lock.json` was an empty stub. `requirements.txt` lacked LIME; no tests existed.

## 3. Implemented (see README for commands)
`fraudlab/` (synthetic data, chronological split, train-only preprocessing, metrics, NN blocks, adversarial, sequences, graph, fusion, drift, registry, explain), `transfer_learning.py`, `train_multimodel.py`, `adversarial_research.py`, `explain_research.py`, `drift_demo.py`, `audit_chain.py` (+ refactor of `blockchain_demo.py`), `contracts/FraudAudit.sol` (moved from root, rewritten) with Hardhat tests, `research_dashboard.py` + 4 new tabs in `app.py`, 62 pytest tests, `build_claims_report.py` → `PAPER_CLAIMS_MATRIX.md`.

## 4. Known limitations
- All new experimental numbers are **synthetic**; no real-data replication (no Kaggle file, no second real dataset). Transfer learning gave **no benefit** here; late fusion did not beat the best single branch; PGD adversarial training helps only near its training ε.
- Single seed for fusion/robustness/drift; ~60 test frauds; no significance tests (none are reported, to avoid fabricating p-values).
- LSTM sequences are per-account only where account ids exist (synthetic). On Kaggle data only a global-stream window would be possible and was not run.
- ADWIN is a simplified implementation (cross-checked against `river`); DDM false-alarms on stationary data.
- Legacy tabs were smoke-tested only with a synthetic stand-in CSV, not the real dataset. Windows commands untested on Windows. Original code emits Streamlit `use_container_width` deprecation warnings (left unchanged).
- Contract tested on a local EVM only; no testnet deployment, no external audit.
- The paper text was not provided; the claim matrix is derived from its title and your brief.
