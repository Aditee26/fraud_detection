"""Learned late fusion from OUT-OF-SAMPLE base predictions (expanding-window, time-ordered)."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))


def expanding_window_oof(n_train: int, n_folds: int, fit_predict, y):
    """fit_predict(name_idx_train, idx_pred) -> {model: preds}. Block 0 only ever trains; blocks 1..K are
    predicted by models trained on strictly earlier blocks, so each OOF prediction is genuinely out-of-sample
    AND respects time. Returns (oof [n_train, M] with NaN for block 0, names)."""
    edges = np.linspace(0, n_train, n_folds + 2).astype(int)
    oof, names = None, None
    for k in range(1, n_folds + 1):
        tr = np.arange(0, edges[k]); te = np.arange(edges[k], edges[k + 1])
        if y[tr].sum() < 2: continue
        preds = fit_predict(tr, te)
        if oof is None:
            names = list(preds); oof = np.full((n_train, len(names)), np.nan)
        oof[te] = np.column_stack([preds[nm] for nm in names])
    return oof, names


class LateFusion:
    def __init__(self, C: float = 1.0):
        self.lr = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)

    def fit(self, P, y):
        ok = ~np.isnan(P).any(1)
        self.lr.fit(logit(P[ok]), np.asarray(y)[ok]); self.n_fit_ = int(ok.sum())
        return self

    def predict(self, P):
        return self.lr.predict_proba(logit(P))[:, 1]

    @property
    def weights(self):
        return self.lr.coef_.ravel().tolist()
