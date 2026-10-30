"""Imbalance-appropriate metrics and validation-only threshold selection."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             confusion_matrix, matthews_corrcoef)


def evaluate_scores(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    two = len(np.unique(y)) == 2
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1),
            "pr_auc": float(average_precision_score(y, p)) if two else float("nan"),
            "roc_auc": float(roc_auc_score(y, p)) if two else float("nan"),
            "mcc": float(matthews_corrcoef(y, pred)) if (tp + fp + fn) else 0.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "threshold": float(threshold), "n": int(len(y)), "n_fraud": int(y.sum())}


def select_threshold(y_val, p_val, strategy: str = "max_f1", min_precision: float = 0.5) -> float:
    """Choose a threshold on VALIDATION data only (never on test)."""
    pr, rc, th = precision_recall_curve(y_val, p_val)
    pr, rc = pr[:-1], rc[:-1]
    if len(th) == 0:
        return 0.5
    if strategy == "max_f1":
        f = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
        return float(th[int(np.argmax(f))])
    if strategy == "recall_at_precision":
        ok = pr >= min_precision
        return float(th[ok][np.argmax(rc[ok])]) if ok.any() else float(th[int(np.argmax(pr))])
    raise ValueError(strategy)
