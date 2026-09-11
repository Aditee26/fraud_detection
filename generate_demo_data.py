"""
generate_demo_data.py
----------------------
DEMO MODE DATA GENERATOR

The real "Credit Card Fraud Detection" Kaggle dataset (284,807 transactions,
0.172% fraud) could not be downloaded here because this environment has no
internet access to Kaggle. This script generates a SYNTHETIC dataset that
mimics its shape and structure so the pipeline can be trained and demoed
end-to-end:

    - Time, Amount, V1..V28 (PCA-style anonymised features), Class
    - Same ~0.17% fraud rate
    - Fraud rows are generated from a shifted/rotated distribution so the
      task is learnable but not trivial (same spirit as the real dataset,
      where fraud sits in a different region of PCA space).

If you have the real creditcard.csv (e.g. from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), drop it in
data/creditcard.csv and train_model.py will automatically use it instead
of this synthetic data. Every plot/metric produced downstream is computed
live from whichever dataset is actually loaded -- nothing is hardcoded.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_LEGIT = 28477      # scaled-down version of the real dataset (1/10th)
FRAUD_RATE = 0.00172  # matches the real dataset's 0.172% fraud rate
N_FRAUD = max(50, int(N_LEGIT * FRAUD_RATE / (1 - FRAUD_RATE)))
N_FEATURES = 28


def generate(path="data/creditcard.csv", seed=RNG_SEED):
    rng = np.random.default_rng(seed)

    # Legit transactions: standard-normal PCA-style features (mean 0, std 1),
    # like the real dataset's V1-V28 columns.
    legit_features = rng.normal(loc=0.0, scale=1.0, size=(N_LEGIT, N_FEATURES))
    legit_amount = np.round(np.abs(rng.gamma(shape=1.5, scale=45, size=N_LEGIT)), 2)
    legit_time = np.sort(rng.integers(0, 172792, size=N_LEGIT))  # ~2 days, like real data
    legit_class = np.zeros(N_LEGIT, dtype=int)

    # Fraud transactions: shifted mean + higher variance on a subset of
    # components, and generally smaller/odder amounts -- mirrors the real
    # dataset's fraud cluster sitting off-distribution in PCA space.
    shift = rng.normal(loc=2.4, scale=0.6, size=N_FEATURES)
    shift[10:] *= 0.3  # only some components carry a strong fraud signal
    fraud_features = rng.normal(loc=shift, scale=1.4, size=(N_FRAUD, N_FEATURES))
    fraud_amount = np.round(np.abs(rng.gamma(shape=1.0, scale=90, size=N_FRAUD)), 2)
    fraud_time = np.sort(rng.integers(0, 172792, size=N_FRAUD))
    fraud_class = np.ones(N_FRAUD, dtype=int)

    features = np.vstack([legit_features, fraud_features])
    amount = np.concatenate([legit_amount, fraud_amount])
    time = np.concatenate([legit_time, fraud_time])
    cls = np.concatenate([legit_class, fraud_class])

    cols = [f"V{i}" for i in range(1, N_FEATURES + 1)]
    df = pd.DataFrame(features, columns=cols)
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = cls

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"[demo data] wrote {len(df)} rows -> {path}")
    print(f"[demo data] fraud rate: {df['Class'].mean() * 100:.4f}% "
          f"({df['Class'].sum()} fraud / {len(df)} total)")
    return df


if __name__ == "__main__":
    generate()
