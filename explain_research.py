"""SHAP + LIME on flagged test transactions (SYNTHETIC unless data/creditcard.csv exists).
Run: python explain_research.py [--synthetic]"""
from __future__ import annotations
import argparse, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from fraudlab import synthetic
from fraudlab.data import load_kaggle, chrono_split, FeaturePreprocessor, save_json
from fraudlab.metrics import select_threshold
from fraudlab import explain

OUT = os.path.join("artifacts", "research", "explain")


def main(a):
    os.makedirs(OUT, exist_ok=True)
    real = None if a.synthetic else load_kaggle()
    if real is not None: df, origin = real, "KAGGLE_REAL"
    else: df, origin = synthetic.generate_transactions(a.n, "target", fraud_rate=0.02, seed=a.seed), "SYNTHETIC"
    sch = {"features": [c for c in df.columns if c.startswith("V")]}
    tr, va, te = chrono_split(df, "Time", (0.6, 0.2, 0.2))
    pp = FeaturePreprocessor(sch["features"], "Amount", "Time").fit(tr)
    Xtr, Xva, Xte = (pp.transform(d) for d in (tr, va, te)); ytr, yva, yte = tr.Class.values, va.Class.values, te.Class.values
    m = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, n_jobs=1, random_state=a.seed, verbosity=0,
                          scale_pos_weight=(ytr == 0).sum() / max(1, ytr.sum())).fit(Xtr, ytr)
    thr = select_threshold(yva, m.predict_proba(Xva)[:, 1]); names = pp.columns
    pte = m.predict_proba(Xte)[:, 1]
    flagged = np.where((pte >= thr) & (yte == 1))[0][: a.n_explain]          # correctly flagged frauds
    rng = np.random.default_rng(a.seed); bg = Xtr[rng.choice(len(Xtr), 200, replace=False)]
    sv, base = explain.shap_values_tree(m, Xte[flagged], bg)
    imp = explain.global_importance(sv, names)
    pd.DataFrame(imp).to_csv(os.path.join(OUT, "shap_global_importance.csv"), index=False)
    pf = lambda Z: m.predict_proba(Z)[:, 1]
    rows, example = [], None
    for k, i in enumerate(flagged):
        lm = explain.lime_explain(pf, Xtr, Xte[i], names, 8, a.seed)
        rows.append(explain.topk_agreement(sv[k], lm, names, 5))
        if example is None: example = {"test_row": int(i), "fraud_probability": float(pte[i]), "lime": lm,
                                       "shap_top": [(names[j], float(sv[k][j])) for j in np.argsort(-np.abs(sv[k]))[:8]]}
    save_json({"origin": origin, "n_explained": len(flagged), "threshold": thr, "shap_expected_value": base,
               "mean_top5_shap_lime_overlap": float(np.mean(rows)) if rows else None, "example": example,
               "note": "SHAP/LIME explain the model's behaviour; they are not causal claims. LIME is a local linear surrogate and is stochastic."},
              os.path.join(OUT, "explanation_summary.json"))
    top = imp[:12][::-1]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.barh([t["feature"] for t in top], [t["mean_abs_shap"] for t in top])
    ax.set_title(f"Mean |SHAP| (flagged frauds) [{origin}]"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "shap_global_bar.png"), dpi=120); plt.close(fig)
    print(origin, "explained", len(flagged), "mean top-5 SHAP/LIME overlap", np.mean(rows) if rows else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--n-explain", type=int, default=15); ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
