# START HERE (Windows 11 PowerShell)

Everything below runs on CPU. The Kaggle dataset is **optional**: all new experiments run on built-in synthetic data.
> The author verified these steps on Linux (Python 3.12). The PowerShell commands are the direct equivalents but were not run on Windows.

## 1. Install (Python 3.10-3.12 recommended)
```powershell
cd path\to\fraud-detection-conference-project
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```
## 2. Verify (about 1 min)
```powershell
python -m pytest -q                    # expect: 62 passed
```
## 3. Run the experiments (quick = a few minutes in total; omit --quick for full size)
```powershell
$env:TF_CPP_MIN_LOG_LEVEL = "3"
python transfer_learning.py --quick       # transfer learning: scratch vs source-pretrained, 10/25/50/100 % labels
python adversarial_research.py --quick    # FGSM, PGD, random-noise baseline, PGD adversarial training
python train_multimodel.py --quick        # ANN + XGBoost + LSTM + GraphSAGE + late fusion
python explain_research.py --synthetic --n 12000 --n-explain 6   # SHAP + LIME
python drift_demo.py                      # ADWIN / DDM + safe candidate-retraining workflow
python build_claims_report.py             # regenerates PAPER_CLAIMS_MATRIX.md
```
Or run everything: `.\scripts\quickstart.ps1`.  Results land in `artifacts\research\`.
## 4. Dashboard
```powershell
python -m streamlit run app.py            # http://localhost:8501
```
Without `data\creditcard.csv` the dashboard shows the four research tabs (Transfer Learning (research), PGD Robustness,
Concept Drift Monitoring, Research Experiment Results). With the dataset and the original artifacts, all 12 tabs appear.
Dashboards only load saved results - they never retrain.
## 5. Optional: real Kaggle data (original pipeline)
Download `creditcard.csv` from Kaggle into `data\` (never commit it), then `python train_model.py`, `python model_comparison.py`,
`python adversarial_test.py`.
## 6. Optional: Solidity tests (needs Node 18+)
```powershell
npm install
npx hardhat test                          # 9 tests, in-process EVM, no network or keys
```
## 7. Audit log without any blockchain credentials
`python blockchain_demo.py --rows 5` uses the local hash chain. Tamper tests: `python -m pytest tests/test_audit.py -q`.
## Troubleshooting
| Symptom | Fix |
|---|---|
| `No module named tensorflow` | use Python 3.10-3.12 (64-bit), then `pip install -r requirements.txt` |
| Dashboard says results "not found" | run step 3 first |
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope Process Bypass` |
| Slow first TensorFlow import | normal (10-30 s) |
| Hardhat deletes files? | it writes only to `build\`; the ML folder `artifacts\` is configured to be left alone |
| Results differ slightly from `artifacts\research` | different TF/CPU; see REPRODUCIBILITY.md |
