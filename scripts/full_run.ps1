# Full-size experiments (roughly 20-30 min on a laptop CPU). Writes artifacts/research/**.
$ErrorActionPreference = "Stop"
. .\.venv\Scripts\Activate.ps1
$env:TF_CPP_MIN_LOG_LEVEL = "3"
python transfer_learning.py
python adversarial_research.py
python train_multimodel.py
python explain_research.py
python drift_demo.py
python -m pytest -q --junitxml=artifacts/research/pytest_report.xml
python build_claims_report.py
