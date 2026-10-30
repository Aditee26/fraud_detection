"""Causal sequence construction for the LSTM branch.

Row i's window contains only rows of the SAME account (or, when no account id
exists, the same global stream) with time <= time_i, and never any label.
"""
from __future__ import annotations
import numpy as np


def build_sequences(X: np.ndarray, times: np.ndarray, groups: np.ndarray | None, L: int = 8):
    """Return (S, window_times). S: [N, L, F+1] (last channel = log1p(seconds since previous
    event in window)); left-padded with zeros. window_times: [N, L] (NaN padding) for leakage checks."""
    n, f = X.shape
    order = np.argsort(times, kind="stable")
    S = np.zeros((n, L, f + 1), np.float32); WT = np.full((n, L), np.nan)
    g = np.zeros(n, int) if groups is None else np.asarray(groups)
    hist: dict = {}
    for i in order:                                  # chronological pass: only the past is visible
        h = hist.setdefault(g[i], [])
        h.append(i)
        w = h[-L:]
        k = len(w)
        S[i, L - k:, :f] = X[w]
        t = times[w]
        S[i, L - k:, f] = np.log1p(np.concatenate([[0.0], np.diff(t)]))
        WT[i, L - k:] = t
    return S, WT


def check_causal(times, WT, groups=None) -> bool:
    """True iff no window contains a timestamp later than its own row."""
    return bool(np.nanmax(WT - times[:, None]) <= 1e-9)
