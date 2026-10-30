"""Multi-model architecture: ANN + XGBoost + LSTM + GraphSAGE + learned late fusion.

Dataset: SYNTHETIC temporal transaction stream with account and device ids (the Kaggle
dataset has neither, so the LSTM here is per-account and the graph is a real
account/device graph only on synthetic data). Every result file says SYNTHETIC.

Leakage controls
  * chronological 60/20/20 split; preprocessing fit on train only.
  * LSTM windows and graph edges are causal (checked at runtime) and never use labels.
  * fusion meta-learner is trained ONLY on out-of-sample base predictions from an
    expanding-window scheme inside the training split.
  * decision thresholds are chosen on the validation split; test labels used once.
Run:  python train_multimodel.py --quick
"""
from __future__ import annotations
import argparse, os, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fraudlab import synthetic, nn
from fraudlab.data import chrono_split, FeaturePreprocessor, save_json
from fraudlab.metrics import evaluate_scores, select_threshold
from fraudlab.sequences import build_sequences, check_causal
from fraudlab.graph import build_neighbors, check_edges_causal, sage_inputs, build_graphsage
from fraudlab.fusion import expanding_window_oof, LateFusion

OUT = os.path.join("artifacts", "research", "multimodel")


def build_lstm(L, f, lr=1e-3):
    from keras import layers
    m = nn.keras.Sequential([layers.Input((L, f)), layers.Masking(0.0), layers.LSTM(32), layers.Dropout(0.2),
                             layers.Dense(16, activation="relu"), layers.Dense(1, activation="sigmoid")], name="lstm")
    return nn._compile(m, lr)


def main(a):
    os.makedirs(OUT, exist_ok=True)
    df = synthetic.generate_transactions(a.n, "target", fraud_rate=0.02, seed=a.seed, n_accounts=600)
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    sch = synthetic.schema("target", df)
    n = len(df); a1, a2 = int(n * .6), int(n * .8)
    tr, va, te = chrono_split(df, "Time", (0.6, 0.2, 0.2))          # also validates class presence / ordering
    pp = FeaturePreprocessor(sch["features"], "Amount", "Time").fit(tr)
    X = pp.transform(df); y = df.Class.values; t = df.Time.values
    S, WT = build_sequences(X, t, df.account_id.values, a.L)
    nbr = build_neighbors(t, [df.account_id.values, df.device_id.values], a.max_nb)
    assert check_causal(t, WT), "LSTM windows leak future rows"
    assert check_edges_causal(t, nbr), "graph has edges to the future"
    G = sage_inputs(X, nbr)
    ix = {"tr": np.arange(0, a1), "va": np.arange(a1, a2), "te": np.arange(a2, n)}

    def take(d, i): return {k: v[i] for k, v in d.items()}

    def fit_predict(train_idx, pred_idx, val_idx=None, return_models=False):
        """Fit every base model on train_idx; early-stop on the LAST 15% (chronologically) of train_idx."""
        cut = int(len(train_idx) * .85); fit_i, es_i = train_idx[:cut], train_idx[cut:]
        if y[es_i].sum() < 1: fit_i, es_i = train_idx, train_idx[-max(50, len(train_idx) // 10):]
        preds, models = {}, {}
        nn.set_seed(a.seed)
        m = nn.build_ann(X.shape[1], seed=a.seed); nn.fit_model(m, X[fit_i], y[fit_i], X[es_i], y[es_i], epochs=a.epochs)
        preds["ANN"], models["ANN"] = nn.predict(m, X[pred_idx]), m
        import xgboost as xgb
        xg = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, n_jobs=1, random_state=a.seed, verbosity=0,
                               scale_pos_weight=(y[fit_i] == 0).sum() / max(1, y[fit_i].sum())).fit(X[fit_i], y[fit_i])
        preds["XGBoost"], models["XGBoost"] = xg.predict_proba(X[pred_idx])[:, 1], xg
        l = build_lstm(a.L, S.shape[2]); nn.fit_model(l, S[fit_i], y[fit_i], S[es_i], y[es_i], epochs=a.epochs)
        preds["LSTM"], models["LSTM"] = nn.predict(l, S[pred_idx]), l
        g = build_graphsage(X.shape[1], a.max_nb)
        nn.fit_model(g, take(G, fit_i), y[fit_i], take(G, es_i), y[es_i], epochs=a.epochs)
        preds["GraphSAGE"], models["GraphSAGE"] = g.predict(take(G, pred_idx), batch_size=2048, verbose=0).ravel(), g
        return (preds, models) if return_models else preds

    t0 = time.perf_counter()
    oof, names = expanding_window_oof(len(ix["tr"]), a.folds, lambda tr_i, te_i: fit_predict(tr_i, te_i), y[ix["tr"]])
    print(f"OOF built in {time.perf_counter() - t0:.0f}s; usable rows={int((~np.isnan(oof).any(1)).sum())}", flush=True)
    fusion = LateFusion().fit(oof, y[ix["tr"]])

    # base models refit on the full train split, then scored on val/test
    pred_idx = np.concatenate([ix["va"], ix["te"]])
    base, models = fit_predict(ix["tr"], pred_idx, return_models=True)
    nva = len(ix["va"])
    P = np.column_stack([base[nm] for nm in names])
    out = {nm: (base[nm][:nva], base[nm][nva:]) for nm in names}
    out["Late fusion (learned)"] = (fusion.predict(P[:nva]), fusion.predict(P[nva:]))
    out["Simple average (baseline)"] = (P[:nva].mean(1), P[nva:].mean(1))
    rows = []
    for nm, (pv, pt) in out.items():
        thr = select_threshold(y[ix["va"]], pv)
        r = evaluate_scores(y[ix["te"]], pt, thr); r.update(model=nm, origin="SYNTHETIC"); rows.append(r)
    res = pd.DataFrame(rows)[["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "mcc", "tp", "fp", "fn", "tn", "threshold", "origin"]]
    res.to_csv(os.path.join(OUT, "multimodel_results.csv"), index=False)
    models["ANN"].save(os.path.join(OUT, "ann.keras")); models["LSTM"].save(os.path.join(OUT, "lstm.keras"))
    models["GraphSAGE"].save(os.path.join(OUT, "graphsage.keras")); models["XGBoost"].save_model(os.path.join(OUT, "xgboost.json"))
    save_json({"origin": "SYNTHETIC", "fusion_weights_on_logit": dict(zip(names, fusion.weights)), "fusion_intercept": float(fusion.lr.intercept_[0]),
               "fusion_train_rows": fusion.n_fit_, "folds": a.folds, "L": a.L, "max_neighbors": a.max_nb,
               "causal_checks": {"lstm_windows": True, "graph_edges": True}, "n_test": len(ix["te"]), "n_test_fraud": int(y[ix["te"]].sum()),
               "graph": "account_id + device_id edges to earlier transactions (synthetic ids); NOT available in Kaggle data",
               "caveat": "Base models used for validation/test were refit on the full train split, so their predictions are slightly stronger than the OOF predictions the meta-learner saw (standard stacking caveat)."},
              os.path.join(OUT, "multimodel_config.json"))
    ax = res.set_index("model")[["pr_auc", "f1", "recall", "precision"]].plot.bar(figsize=(10, 4), rot=20)
    ax.set_title("Test metrics - SYNTHETIC temporal graph stream"); ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(OUT, "multimodel_comparison.png"), dpi=120); plt.close(ax.figure)
    print(res.round(3).to_string(index=False))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--L", type=int, default=8); ap.add_argument("--max-nb", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.quick: a.n, a.epochs, a.folds = 6000, 6, 2
    return a


if __name__ == "__main__":
    main(parse())
