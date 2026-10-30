# Reproducibility

| Item | Value |
|---|---|
| Verified environment | Ubuntu Linux, Python 3.12.3, 1 CPU / 3 GB RAM, tensorflow-cpu 2.21.0, Keras 3.15.1, scikit-learn 1.8.0, xgboost 3.4.1, shap 0.54.1, lime 0.2.0.1, river 0.26.1, Node 22 / Hardhat 2.x |
| Windows 11 | Commands are provided but were **not executed on Windows** by the author |
| Seeds | `--seed` (default 0); transfer uses seeds 0..2; results are close but not bit-identical across machines/TF versions |
| Data | synthetic: `fraudlab/synthetic.py` (deterministic). Real: put Kaggle `creditcard.csv` in `data/` (not in Git) |

## Regenerating every result
```
python transfer_learning.py            # artifacts/research/transfer/*   (~7 min on 1 CPU)
python adversarial_research.py         # artifacts/research/adversarial/*
python train_multimodel.py             # artifacts/research/multimodel/*
python explain_research.py             # artifacts/research/explain/*
python drift_demo.py                   # artifacts/research/drift/*
python -m pytest -q --junitxml=artifacts/research/pytest_report.xml
npx hardhat test | Tee-Object artifacts/research/hardhat_test_output.txt
python build_claims_report.py          # PAPER_CLAIMS_MATRIX.md
```
Add `--quick` to the first four for a smoke run. Every output records its data origin (SYNTHETIC / KAGGLE_REAL / USER_CSV).

## Provenance of the pre-existing README/Kaggle numbers
Not reproducible here: the Kaggle file could not be obtained, and the committed artifacts carry no data hash or git commit.
We verified only that `metrics.json` and `model_comparison.csv` are internally consistent and that the `.keras` models load.
With the real file present, `adversarial_research.py`, `explain_research.py` automatically use it (chronological split on `Time`).
Real-data transfer needs a *second* dataset: `python transfer_learning.py --source-csv A.csv --target-csv B.csv`
(columns: a time column, an amount column, `Class`, numeric features; schemas may differ).
