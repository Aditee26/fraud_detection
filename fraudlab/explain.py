"""SHAP (tree) + LIME (tabular) explanations and a simple SHAP-vs-LIME agreement check.

Background / training statistics come from the TRAIN split only. Explanations describe the
MODEL, not causal reality; for anonymised V-features they say which components the model used.
"""
from __future__ import annotations
import numpy as np


def shap_values_tree(model, X_explain, X_background=None):
    import shap
    ex = shap.TreeExplainer(model, data=X_background, feature_perturbation="interventional" if X_background is not None else "tree_path_dependent",
                            model_output="raw") if X_background is not None else shap.TreeExplainer(model)
    v = ex.shap_values(X_explain)
    v = v[1] if isinstance(v, list) else v
    return np.asarray(v), float(np.ravel(ex.expected_value)[-1])


def global_importance(sv, names):
    imp = np.abs(sv).mean(0)
    order = np.argsort(-imp)
    return [{"feature": names[i], "mean_abs_shap": float(imp[i])} for i in order]


def lime_explain(predict_proba_1d, X_train, x, names, num_features=8, seed=0):
    """predict_proba_1d: callable (n,F)->(n,) fraud probability."""
    from lime.lime_tabular import LimeTabularExplainer
    ex = LimeTabularExplainer(X_train, feature_names=list(names), class_names=["legit", "fraud"], mode="classification",
                              discretize_continuous=False, random_state=seed)
    f = lambda Z: np.column_stack([1 - predict_proba_1d(Z), predict_proba_1d(Z)])
    e = ex.explain_instance(x, f, num_features=num_features, num_samples=1500)
    return [(n, float(w)) for n, w in e.as_list()]


def topk_agreement(shap_row, lime_pairs, names, k=5):
    s = {names[i] for i in np.argsort(-np.abs(shap_row))[:k]}
    l = {n for n, _ in sorted(lime_pairs, key=lambda t: -abs(t[1]))[:k]}
    return len(s & l) / k
