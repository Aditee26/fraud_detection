# Windows 11 PowerShell quick start (about 5-10 minutes on CPU). NOT executed on Windows by the author - see START_HERE.md.
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { py -3.12 -m venv .venv }
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
$env:TF_CPP_MIN_LOG_LEVEL = "3"
python -m pytest -q
python transfer_learning.py --quick
python adversarial_research.py --quick
python train_multimodel.py --quick
python explain_research.py --synthetic --n 12000 --n-explain 6
python drift_demo.py
python build_claims_report.py
Write-Host "Done. Start the dashboard with:  python -m streamlit run app.py"
