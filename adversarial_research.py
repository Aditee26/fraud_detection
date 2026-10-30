"""Clean-vs-adversarial evaluation: FGSM, PGD, random-noise baseline, PGD adversarial training.

Data: Kaggle creditcard.csv if present at data/creditcard.csv (chronological split on `Time`),
      otherwise a SYNTHETIC stream (clearly labelled in every output).
Run:  python adversarial_research.py --quick
"""
from __future__ import annotations
import argparse, os, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fraudlab import synthetic, nn
from fraudlab.data import load_kaggle, chrono_split, FeaturePreprocessor, save_json
from fraudlab.metrics import evaluate_scores, select_threshold
from fraudlab.adversarial import fgsm, pgd, random_noise, mutable_mask, check_constraints, adversarial_train

OUT = os.path.join("artifacts", "research", "adversarial")
NOTE = ("Perturbations are L-inf budgets in standardised feature space. For anonymised PCA features (V1..V28) "
        "they are mathematical robustness tests, not physically realisable transaction edits. "
        "No universal robustness is claimed.")


def main(a):
    os.makedirs(OUT, exist_ok=True)
    real = load_kaggle() if not a.synthetic else None
    if real is not None:
        df, sch, origin = real, {"time": "Time", "amount": "Amount", "label": "Class", "features": [f"V{i}" for i in range(1, 29)]}, "KAGGLE_REAL"
    else:
        df = synthetic.generate_transactions(a.n, "target", fraud_rate=0.02, seed=a.seed)
        sch, origin = synthetic.schema("target", df), "SYNTHETIC"
    tr, va, te = chrono_split(df, sch["time"], (0.6, 0.2, 0.2), sch["label"])
    pp = FeaturePreprocessor(sch["features"], sch["amount"], sch["time"]).fit(tr)
    Xtr, Xva, Xte = pp.transform(tr), pp.transform(va), pp.transform(te)
    ytr, yva, yte = tr[sch["label"]].values, va[sch["label"]].values, te[sch["label"]].values
    lo, hi = pp.feature_bounds(Xtr)                                 # bounds from TRAIN only
    mask = mutable_mask(pp.columns)

    nn.set_seed(a.seed)
    std = nn.build_ann(Xtr.shape[1], seed=a.seed); nn.fit_model(std, Xtr, ytr, Xva, yva, epochs=a.epochs)
    rob = nn.build_ann(Xtr.shape[1], seed=a.seed)
    t0 = time.perf_counter()
    adversarial_train(rob, Xtr, ytr, Xva, yva, eps=a.train_eps, attack="pgd", epochs=a.epochs, steps=5, lo=lo, hi=hi, mask=mask, seed=a.seed)
    at_secs = time.perf_counter() - t0
    import xgboost as xgb
    xg = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, n_jobs=1, random_state=a.seed, verbosity=0,
                           scale_pos_weight=(ytr == 0).sum() / max(1, ytr.sum())).fit(Xtr, ytr)

    models = {"ANN (standard)": (std, lambda X: nn.predict(std, X)),
              "ANN (PGD adversarial training)": (rob, lambda X: nn.predict(rob, X)),
              "XGBoost (transfer attack from standard ANN)": (std, lambda X: xg.predict_proba(X)[:, 1])}
    rows, checks = [], []
    for name, (attack_model, score) in models.items():
        thr = select_threshold(yva, score(Xva))                     # clean validation threshold, fixed under attack
        for eps in a.eps:
            variants = {"clean": Xte} if eps == 0 else {}
            if eps > 0:
                variants["random_noise"] = random_noise(Xte, eps, lo, hi, mask, a.seed)
                variants["fgsm"] = fgsm(attack_model, Xte, yte, eps, lo, hi, mask)
                variants["pgd"] = pgd(attack_model, Xte, yte, eps, steps=a.pgd_steps, lo=lo, hi=hi, mask=mask, seed=a.seed)
                checks.append({"model": name, "eps": eps, **check_constraints(Xte, variants["pgd"], eps, lo, hi, mask)})
            for atk, Xa in variants.items():
                r = evaluate_scores(yte, score(Xa), thr); r.update(model=name, attack=atk, eps=eps, origin=origin); rows.append(r)
    res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, "robustness_results.csv"), index=False)
    pd.DataFrame(checks).to_csv(os.path.join(OUT, "constraint_checks.csv"), index=False)
    save_json({"origin": origin, "note": NOTE, "immutable_features": [c for c, m in zip(pp.columns, mask) if m == 0],
               "train_eps": a.train_eps, "pgd_steps": a.pgd_steps, "adv_training_seconds": at_secs,
               "n_test": len(yte), "n_test_fraud": int(yte.sum()), "threshold_policy": "max-F1 on clean validation split, fixed under attack",
               "all_constraints_satisfied": bool(pd.DataFrame(checks).drop(columns=["model", "eps", "max_abs_perturbation"]).all().all())},
              os.path.join(OUT, "robustness_config.json"))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for m, g in res[res.attack.isin(["clean", "pgd"])].groupby("model"):
        ax[0].plot(g.eps, g.pr_auc, marker="o", label=m); ax[1].plot(g.eps, g.recall, marker="o", label=m)
    ax[0].set_title("PR-AUC under PGD"); ax[1].set_title("Fraud recall under PGD (fixed threshold)")
    for x in ax: x.set_xlabel("eps (standardised units)"); x.grid(alpha=.3)
    ax[0].legend(fontsize=7); fig.suptitle(f"Robustness [{origin}]"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "robustness_curves.png"), dpi=120); plt.close(fig)
    print(res[(res.eps.isin([0, a.eps[-1]]))][["model", "attack", "eps", "precision", "recall", "f1", "pr_auc"]].round(3).to_string(index=False))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--eps", type=float, nargs="+", default=[0, 0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--train-eps", type=float, default=0.1); ap.add_argument("--pgd-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.quick: a.n, a.epochs, a.eps, a.pgd_steps = 8000, 6, [0, 0.1, 0.3], 5
    return a


if __name__ == "__main__":
    main(parse())
