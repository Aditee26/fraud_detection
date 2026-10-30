import numpy as np, pytest
xgb = pytest.importorskip("xgboost")
from fraudlab import explain


@pytest.fixture(scope="module")
def model_and_data(prepared):
    m = xgb.XGBClassifier(n_estimators=40, max_depth=3, n_jobs=1, verbosity=0).fit(prepared["Xtr"], prepared["ytr"])
    return m, prepared


def test_shap_additivity(model_and_data):
    pytest.importorskip("shap"); m, p = model_and_data
    sv, base = explain.shap_values_tree(m, p["Xte"][:20])
    raw = m.predict(p["Xte"][:20], output_margin=True)
    assert np.allclose(sv.sum(1) + base, raw, atol=1e-3)           # SHAP values + base = model margin


def test_global_importance_sorted(model_and_data):
    pytest.importorskip("shap"); m, p = model_and_data
    sv, _ = explain.shap_values_tree(m, p["Xte"][:30]); imp = explain.global_importance(sv, p["pp"].columns)
    vals = [i["mean_abs_shap"] for i in imp]; assert vals == sorted(vals, reverse=True)


def test_lime_returns_requested_number_of_features(model_and_data):
    pytest.importorskip("lime"); m, p = model_and_data
    out = explain.lime_explain(lambda Z: m.predict_proba(Z)[:, 1], p["Xtr"], p["Xte"][0], p["pp"].columns, 5, seed=0)
    assert 1 <= len(out) <= 5 and all(isinstance(w, float) for _, w in out)


def test_topk_agreement_bounds():
    names = list("abcdef"); sv = np.array([5, 4, 3, 2, 1, 0.])
    assert explain.topk_agreement(sv, [(n, w) for n, w in zip(names, sv)], names, 3) == 1.0
    assert explain.topk_agreement(sv, [("f", 9), ("e", 8), ("d", 7)], names, 3) == 0.0
