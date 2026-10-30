"""Chronological splitting, leakage-safe preprocessing, dataset loading."""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd

KAGGLE_PATH = os.path.join("data", "creditcard.csv")
DAY = 86400.0


def load_kaggle(path: str = KAGGLE_PATH) -> pd.DataFrame | None:
    """Return the Kaggle frame, or None if the (external) file is absent."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    need = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    if need - set(df.columns):
        raise ValueError(f"{path} is not the Kaggle Credit Card dataset; missing {sorted(need - set(df.columns))[:5]}")
    return df


def chrono_split(df: pd.DataFrame, time_col: str, fractions=(0.6, 0.2, 0.2), label_col: str = "Class"):
    """Split by time order: train < val < test. Raises if a split lacks a class.

    Ties are broken by original row order (stable sort) so a split boundary can
    never put a *later* timestamp in an earlier split.
    """
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    d = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    n = len(d)
    a = int(n * fractions[0]); b = int(n * (fractions[0] + fractions[1]))
    parts = d.iloc[:a], d.iloc[a:b], d.iloc[b:]
    for name, p in zip(("train", "val", "test"), parts):
        if p[label_col].nunique() < 2:
            raise ValueError(f"{name} split has a single class; increase data or fraud rate")
    assert parts[0][time_col].max() <= parts[1][time_col].min() <= parts[1][time_col].max() <= parts[2][time_col].min()
    return parts


class FeaturePreprocessor:
    """Fit on TRAIN ONLY. log1p(amount), cyclic hour-of-day, z-scored features.

    Raw elapsed time is deliberately NOT a model input: under a chronological
    split it is extrapolation (test times lie outside the training range).
    """

    def __init__(self, feature_cols, amount_col, time_col):
        self.feature_cols, self.amount_col, self.time_col = list(feature_cols), amount_col, time_col
        self.mean_ = self.scale_ = None

    @property
    def columns(self):
        return self.feature_cols + ["log_amount", "hour_sin", "hour_cos"]

    def _raw(self, df):
        t = df[self.time_col].to_numpy(float)
        ang = 2 * np.pi * (t % DAY) / DAY
        return np.column_stack([df[self.feature_cols].to_numpy(float),
                                np.log1p(df[self.amount_col].to_numpy(float)),
                                np.sin(ang), np.cos(ang)])

    def fit(self, train_df):
        raw = self._raw(train_df)
        self.mean_ = raw.mean(0)
        self.scale_ = np.where(raw.std(0) < 1e-9, 1.0, raw.std(0))
        return self

    def transform(self, df):
        if self.mean_ is None:
            raise RuntimeError("fit() must be called on training data first")
        return ((self._raw(df) - self.mean_) / self.scale_).astype(np.float32)

    def to_dict(self):
        return {"feature_cols": self.feature_cols, "amount_col": self.amount_col, "time_col": self.time_col,
                "mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_dict(cls, d):
        p = cls(d["feature_cols"], d["amount_col"], d["time_col"])
        p.mean_, p.scale_ = np.array(d["mean"]), np.array(d["scale"])
        return p

    def feature_bounds(self, X_train_scaled):
        """Per-feature [min, max] of the (scaled) TRAINING data, for attack clipping."""
        return X_train_scaled.min(0), X_train_scaled.max(0)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
