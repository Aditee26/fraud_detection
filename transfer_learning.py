"""Transfer-learning experiment: source domain -> target domain, different schemas.

Arms (identical architecture, identical target validation set):
  scratch        random init, trained on target subset
  transfer       shared encoder pretrained on SOURCE; target adapter + head fresh;
                 stage 1 = train adapter+head with encoder frozen,
                 stage 2 = unfreeze everything at a lower learning rate
  xgboost        context baseline (target subset only)
Target label budgets: 10 %, 25 %, 50 %, 100 % of the chronological TRAIN split.

Leakage controls
  * target split is chronological (train < val < test); test labels are used
    only for the final metrics.
  * target scaler is fit on the target training subset actually used.
  * source scaler is fit on source training data only; source labels never mix
    with target data.
  * early stopping + decision threshold use the target VALIDATION split only.

DATA ORIGIN: by default BOTH domains are SYNTHETIC (fraudlab.synthetic). The
numbers therefore demonstrate the method, not real-world benefit. To use real
data pass --source-csv / --target-csv (see REPRODUCIBILITY.md).

Run:  python transfer_learning.py --quick      (about 1-2 min on CPU)
      python transfer_learning.py              (default settings)
"""
from __future__ import annotations
import argparse, os, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fraudlab import synthetic
from fraudlab.data import chrono_split, FeaturePreprocessor, save_json
from fraudlab.metrics import evaluate_scores, select_threshold
from fraudlab import nn

OUT = os.path.join("artifacts", "research", "transfer")
FRACTIONS = (0.10, 0.25, 0.50, 1.00)


def _subsample(df, frac, seed):
    """Stratified subsample of the TRAIN split (always keeps >= 2 frauds)."""
    if frac >= 1.0: return df
    rng = np.random.default_rng(seed)
    pos, neg = df[df.Class == 1], df[df.Class == 0]
    kp = max(2, int(round(len(pos) * frac))); kn = int(round(len(neg) * frac))
    return pd.concat([pos.sample(kp, random_state=int(rng.integers(1e9))),
                      neg.sample(kn, random_state=int(rng.integers(1e9)))]).sort_values(df.columns[0] if "Time" not in df else "Time")


def run(args):
    os.makedirs(OUT, exist_ok=True)
    prov = {"data_origin": "SYNTHETIC" if not (args.source_csv and args.target_csv) else "USER_CSV",
            "note": "Synthetic-domain demonstration; shared latent fraud signal, different feature schemas."}
    if args.source_csv and args.target_csv:
        src, tgt = pd.read_csv(args.source_csv), pd.read_csv(args.target_csv)
        ssch, tsch = _infer_schema(src), _infer_schema(tgt)
    else:
        src = synthetic.generate_transactions(args.n_source, "source", fraud_rate=0.03, seed=args.seed + 1, latent_seed=args.seed)
        tgt = synthetic.generate_transactions(args.n_target, "target", fraud_rate=0.015, seed=args.seed + 2, latent_seed=args.seed)
        ssch, tsch = synthetic.schema("source", src), synthetic.schema("target", tgt)

    # ---- source pretraining (source split chronologically; source labels only touch source) ----
    s_tr, s_va, s_te = chrono_split(src, ssch["time"], (0.7, 0.15, 0.15), ssch["label"])
    sp = FeaturePreprocessor(ssch["features"], ssch["amount"], ssch["time"]).fit(s_tr)
    Xs_tr, Xs_va = sp.transform(s_tr), sp.transform(s_va)
    nn.set_seed(args.seed)
    t0 = time.perf_counter()
    s_model, s_enc = nn.build_domain_net(Xs_tr.shape[1], adapter_dim=args.adapter_dim, seed=args.seed)
    nn.fit_model(s_model, Xs_tr, s_tr[ssch["label"]].values, Xs_va, s_va[ssch["label"]].values, epochs=args.epochs)
    pretrain_s = time.perf_counter() - t0
    # sanity check that pretraining itself works (source-domain held-out data only)
    _pv, _pt = nn.predict(s_model, Xs_va), nn.predict(s_model, sp.transform(s_te))
    _thr = select_threshold(s_va[ssch["label"]].values, _pv)
    source_check = evaluate_scores(s_te[ssch["label"]].values, _pt, _thr)
    enc_weights = s_enc.get_weights()
    s_enc.save_weights(os.path.join(OUT, "source_pretrained_encoder.weights.h5"))

    # ---- target: chronological split, fixed val/test ----
    t_tr_full, t_va, t_te = chrono_split(tgt, tsch["time"], (0.6, 0.2, 0.2), tsch["label"])
    tcols = (tsch["features"], tsch["amount"], tsch["time"])
    rows = []
    for seed in range(args.seeds):
        for frac in FRACTIONS:
            sub = _subsample(t_tr_full, frac, 1000 * seed + int(frac * 100))
            tp = FeaturePreprocessor(*tcols).fit(sub)            # scaler from the subset only
            Xtr, Xva, Xte = tp.transform(sub), tp.transform(t_va), tp.transform(t_te)
            ytr, yva, yte = (d[tsch["label"]].values for d in (sub, t_va, t_te))
            for arm in ("scratch", "transfer", "xgboost"):
                nn.set_seed(seed)
                pva = pte = None                     # guard: every arm must assign its own predictions
                t0 = time.perf_counter()
                if arm == "xgboost":
                    import xgboost as xgb
                    m = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.9,
                                          scale_pos_weight=max(1, (ytr == 0).sum() / max(1, ytr.sum())),
                                          eval_metric="aucpr", n_jobs=1, random_state=seed, verbosity=0)
                    m.fit(Xtr, ytr)
                    pva, pte = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
                elif arm == "scratch":
                    m, _ = nn.build_domain_net(Xtr.shape[1], adapter_dim=args.adapter_dim, seed=seed)
                    nn.fit_model(m, Xtr, ytr, Xva, yva, epochs=args.epochs)
                    pva, pte = nn.predict(m, Xva), nn.predict(m, Xte)
                else:
                    enc = nn.build_shared_encoder(args.adapter_dim + nn.CANONICAL_TAIL)
                    enc.set_weights(enc_weights)
                    m, enc = nn.build_domain_net(Xtr.shape[1], encoder=enc, adapter_dim=args.adapter_dim, lr=1e-3, seed=seed)
                    enc.trainable = False                               # stage 1: adapter + head only
                    m.compile(optimizer=nn.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                              metrics=[nn.keras.metrics.AUC(name="pr_auc", curve="PR")])
                    nn.fit_model(m, Xtr, ytr, Xva, yva, epochs=max(5, args.epochs // 3))
                    enc.trainable = True                                # stage 2: full fine-tune, small LR
                    m.compile(optimizer=nn.keras.optimizers.Adam(2e-4), loss="binary_crossentropy",
                              metrics=[nn.keras.metrics.AUC(name="pr_auc", curve="PR")])
                    nn.fit_model(m, Xtr, ytr, Xva, yva, epochs=args.epochs)
                    pva, pte = nn.predict(m, Xva), nn.predict(m, Xte)
                assert pva is not None and pte is not None, f"arm {arm} produced no predictions"
                secs = time.perf_counter() - t0
                thr = select_threshold(yva, pva)                         # validation only
                r = evaluate_scores(yte, pte, thr)
                r.update(arm=arm, fraction=frac, seed=seed, train_rows=len(sub), train_frauds=int(ytr.sum()),
                         train_seconds=secs, data_origin=prov["data_origin"])
                rows.append(r)
                if seed == 0 and arm in ("scratch", "transfer") and frac in (0.1, 1.0):
                    m.save_weights(os.path.join(OUT, f"{arm}_frac{int(frac * 100)}.weights.h5"))
                print(f"seed={seed} frac={frac:.2f} {arm:8s} PR-AUC={r['pr_auc']:.3f} F1={r['f1']:.3f} R={r['recall']:.3f} P={r['precision']:.3f} {secs:.1f}s", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "transfer_results_per_seed.csv"), index=False)
    metrics = ["precision", "recall", "f1", "pr_auc", "roc_auc", "train_seconds"]
    summ = res.groupby(["arm", "fraction"])[metrics].agg(["mean", "std"]).round(4)
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index(); summ.to_csv(os.path.join(OUT, "transfer_results_summary.csv"), index=False)

    # paired differences (transfer - scratch) per seed, with plain mean/std (no p-values: too few seeds)
    p = res.pivot_table(index=["fraction", "seed"], columns="arm", values=["pr_auc", "f1", "recall", "precision"])
    diffs = []
    for frac, g in p.groupby(level=0):
        d = {"fraction": frac}
        for mname in ("pr_auc", "f1", "recall", "precision"):
            delta = (g[(mname, "transfer")] - g[(mname, "scratch")]).values
            d[f"{mname}_delta_mean"], d[f"{mname}_delta_std"], d[f"{mname}_seeds_transfer_better"] = \
                round(float(delta.mean()), 4), round(float(delta.std(ddof=1)) if len(delta) > 1 else float("nan"), 4), int((delta > 0).sum())
        diffs.append(d)
    pd.DataFrame(diffs).to_csv(os.path.join(OUT, "transfer_vs_scratch_deltas.csv"), index=False)

    save_json({"provenance": prov, "args": vars(args), "source_pretrain_seconds": pretrain_s, "source_domain_test_metrics": source_check,
               "source_schema": ssch["features"][:3] + ["..."], "target_schema": tsch["features"][:3] + ["..."],
               "alignment": "domain-specific Dense adapter per schema + canonical (log_amount, hour_sin, hour_cos) bypass; only the shared encoder is transferred",
               "target_split": "chronological 60/20/20", "threshold": "max-F1 on target validation split",
               "n_target_test": len(t_te), "n_target_test_fraud": int(t_te[tsch["label"]].sum()),
               "tensorflow": nn.tf.__version__}, os.path.join(OUT, "transfer_config.json"))
    _plots(summ)
    return res, summ


def _infer_schema(df):
    t = "Time" if "Time" in df else df.columns[0]
    a = "Amount" if "Amount" in df else [c for c in df.columns if "am" in c.lower()][0]
    feats = [c for c in df.columns if c not in (t, a, "Class", "account_id", "device_id", "is_synthetic")]
    return {"time": t, "amount": a, "label": "Class", "features": feats}


def _plots(summ):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (m, title) in zip(ax, [("pr_auc", "PR-AUC (test)"), ("f1", "F1 at validation-chosen threshold"), ("train_seconds", "Training time (s)")]):
        for arm, g in summ.groupby("arm"):
            a.errorbar(g["fraction"] * 100, g[f"{m}_mean"], yerr=g[f"{m}_std"].fillna(0), marker="o", capsize=3, label=arm)
        a.set_xlabel("% of target training data"); a.set_title(title); a.set_xticks([10, 25, 50, 100]); a.grid(alpha=.3)
    ax[0].legend(); fig.suptitle("Transfer learning vs scratch - SYNTHETIC domains (mean ± std over seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "transfer_curves.png"), dpi=120); plt.close(fig)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller data and fewer epochs (CI / smoke test)")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-source", type=int, default=30000); ap.add_argument("--n-target", type=int, default=20000)
    ap.add_argument("--adapter-dim", type=int, default=48); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source-csv"); ap.add_argument("--target-csv")
    a = ap.parse_args()
    if a.quick: a.seeds, a.epochs, a.n_source, a.n_target = 2, 8, 12000, 8000
    return a


if __name__ == "__main__":
    run(parse())
