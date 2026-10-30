"""Synthetic transaction generator (clearly SYNTHETIC; never real bank data).

Purpose: give every research experiment a reproducible, label-isolated
environment with properties the Kaggle Credit Card dataset lacks:
  * account and device identifiers (so a real transaction graph exists),
  * per-account chronological sequences (so an LSTM has valid sequences),
  * two *domains* with different feature schemas sharing the same latent
    fraud signal (so cross-schema transfer learning is meaningful),
  * an optional concept-drift point.

Results computed on this data MUST be labelled "synthetic" and say nothing
about performance on real banking data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

LATENT_DIM = 6


def _latent_means(seed: int, drifted: bool = False):
    rng = np.random.default_rng(10_000 + seed)  # shared by both domains
    mu = rng.normal(0, 1.4, LATENT_DIM)
    if drifted:  # new fraud tactic: signal moves to different latent axes
        mu = np.roll(mu, 2) * 1.1 - 0.5
    return mu


def generate_transactions(
    n: int = 20_000,
    domain: str = "target",
    fraud_rate: float = 0.02,
    n_accounts: int = 800,
    n_devices: int = 1_200,
    n_features: int | None = None,
    seed: int = 0,
    latent_seed: int = 0,
    drift_at: float | None = None,
    days: float = 30.0,
) -> pd.DataFrame:
    """Return a time-sorted DataFrame.

    domain="target": Kaggle-like schema  Time, V1..V28, Amount, Class
    domain="source": different schema    ts, f0..f{k-1}, amt, Class
    Both add account_id, device_id, is_synthetic=True.
    `latent_seed` is the shared "fraud physics"; `seed` is sampling noise.
    `drift_at` (0-1) moves the fraud signal for transactions after that
    fraction of the timeline.
    """
    if domain not in ("target", "source"):
        raise ValueError("domain must be 'target' or 'source'")
    rng = np.random.default_rng(seed + (0 if domain == "target" else 7919))
    k = n_features or (28 if domain == "target" else 20)
    horizon = days * 86400.0
    t = np.sort(rng.uniform(0, horizon, n))

    acct = np.minimum(rng.zipf(1.6, n) - 1, n_accounts - 1).astype(int)
    rng.shuffle(acct)
    own_dev = (acct * 7 + rng.integers(0, 2, n)) % n_devices
    ring_devices = np.arange(n_devices - 25, n_devices)  # shared fraud-ring devices

    # --- labels: account compromise bursts + isolated fraud ---------------
    y = np.zeros(n, dtype=np.int8)
    n_fraud = max(2, int(round(n * fraud_rate)))
    compromised = rng.choice(n_accounts, size=max(3, n_fraud // 6), replace=False)
    cand = np.where(np.isin(acct, compromised))[0]
    burst = rng.choice(cand, size=min(len(cand), n_fraud * 2 // 3), replace=False) if len(cand) else np.array([], int)
    y[burst] = 1
    rest = np.where(y == 0)[0]
    y[rng.choice(rest, size=max(0, n_fraud - int(y.sum())), replace=False)] = 1
    fraud = y == 1
    device = own_dev.copy()
    ring_use = fraud & (rng.random(n) < 0.8)
    device[ring_use] = rng.choice(ring_devices, ring_use.sum())

    # --- latent factors -> domain-specific features -----------------------
    drift_mask = np.zeros(n, bool) if drift_at is None else t >= drift_at * horizon
    mu0, mu1 = _latent_means(latent_seed), _latent_means(latent_seed, True)
    z = rng.normal(0, 1, (n, LATENT_DIM))
    z[fraud & ~drift_mask] += mu0
    z[fraud & drift_mask] += mu1
    w_rng = np.random.default_rng(555 + latent_seed + (0 if domain == "target" else 1))
    W = w_rng.normal(0, 1, (LATENT_DIM, k)) / np.sqrt(LATENT_DIM)
    X = z @ W + rng.normal(0, 0.6, (n, k))
    amount = np.exp(rng.normal(3.5, 1.1, n) + 0.9 * fraud * (rng.random(n) < 0.6))

    if domain == "target":
        df = pd.DataFrame(X, columns=[f"V{i + 1}" for i in range(k)])
        df.insert(0, "Time", t)
        df["Amount"] = amount
        df["Class"] = y
    else:
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(k)])
        df.insert(0, "ts", t)
        df["amt"] = amount
        df["Class"] = y
    df["account_id"] = acct
    df["device_id"] = device
    df["is_synthetic"] = True
    return df.reset_index(drop=True)


def schema(domain: str, df: pd.DataFrame) -> dict:
    """Column roles for a generated frame."""
    if domain == "target":
        return {"time": "Time", "amount": "Amount", "label": "Class",
                "features": [c for c in df.columns if c.startswith("V")]}
    return {"time": "ts", "amount": "amt", "label": "Class",
            "features": [c for c in df.columns if c.startswith("f")]}
