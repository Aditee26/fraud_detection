"""Prequential drift monitoring on a SYNTHETIC chronological stream with a known drift point.

Protocol: every transaction is scored by the CURRENTLY DEPLOYED model using only its own features.
Its true label arrives `--label-delay` transactions later; only then do the detectors (ADWIN, DDM)
see the outcome. Detectors therefore never see future labels. On the first detection a CANDIDATE
model is trained on labels that have already arrived, evaluated on a later labelled hold-out, and
only RECOMMENDED for promotion - the deployed model is left untouched (see fraudlab/registry.py).
"""
from __future__ import annotations
import argparse, os, json, shutil
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from fraudlab import synthetic
from fraudlab.data import FeaturePreprocessor, save_json
from fraudlab.drift import ADWIN, DDM
from fraudlab.registry import ModelRegistry

OUT = os.path.join("artifacts", "research", "drift")


def make_model(y, seed):
    return xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.1, n_jobs=1, random_state=seed, verbosity=0,
                             scale_pos_weight=(y == 0).sum() / max(1, y.sum()))


def main(a):
    os.makedirs(OUT, exist_ok=True)
    log = os.path.join(OUT, "drift_events.jsonl")
    if os.path.exists(log): os.remove(log)
    reg_root = os.path.join(OUT, "registry")
    if os.path.exists(reg_root): shutil.rmtree(reg_root)
    df = synthetic.generate_transactions(a.n, "target", fraud_rate=0.05, seed=a.seed, drift_at=a.drift_at, n_accounts=500)
    sch = synthetic.schema("target", df); df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n_tr = int(len(df) * a.train_frac); drift_idx = int(np.searchsorted(df.Time.values, a.drift_at * df.Time.max()))
    pp = FeaturePreprocessor(sch["features"], "Amount", "Time").fit(df.iloc[:n_tr])
    X, y = pp.transform(df), df.Class.values
    reg = ModelRegistry(reg_root); reg.register_initial(make_model(y[:n_tr], a.seed).fit(X[:n_tr], y[:n_tr]), "trained on pre-drift history")
    model = reg.load_current()
    thr = 0.5
    det = {"error": (ADWIN(), DDM()), "fraud_miss": (ADWIN(), DDM())}
    events, errs, pred = [], [], np.zeros(len(df), int)
    cand_report = None
    for i in range(n_tr, len(df)):
        pred[i] = int(model.predict_proba(X[i:i + 1])[0, 1] >= thr)        # uses features of row i only
        j = i - a.label_delay                                              # label of row j arrives now
        if j < n_tr: continue
        e = int(pred[j] != y[j]); errs.append((j, e))
        streams = [("error", e)] + ([("fraud_miss", int(pred[j] == 0))] if y[j] == 1 else [])
        for sname, val in streams:
            for dn, d in zip(("ADWIN", "DDM"), det[sname]):
                if d.update(val):
                    ev = {"detector": dn, "stream": sname, "stream_index": int(i), "label_index": int(j),
                          "true_drift_index": drift_idx, "pre_drift": bool(j < drift_idx)}
                    events.append(ev); open(log, "a").write(json.dumps(ev) + "\n")
        post = [ev for ev in events if not ev["pre_drift"]]
        if post and cand_report is None:
            # candidate trained ONLY on labels that have arrived (index <= j), most recent window
            lo = max(n_tr, j - a.window); cand = make_model(y[lo:j], a.seed).fit(X[lo:j], y[lo:j])
            cid = reg.add_candidate(cand, f"retrained after {post[0]['detector']}/{post[0]['stream']} at stream index {i}", {"trained_on": [int(lo), int(j)]})
            hold = (j, min(len(df), j + a.window))                          # strictly LATER than candidate training data
            # hold-out labels become available only after the window; evaluated offline in this demo
            cand_report = reg.evaluate_candidate(cid, X[hold[0]:hold[1]], y[hold[0]:hold[1]], thr)
            cand_report.update(candidate_id=cid, trained_on=[int(lo), int(j)], holdout=list(map(int, hold)),
                               deployed_model_changed=False, note="Deployed model left unchanged; promotion needs explicit approval.")
    post_ev = [e for e in events if not e["pre_drift"]]; pre_ev = [e for e in events if e["pre_drift"]]
    summary = {"origin": "SYNTHETIC", "n": len(df), "drift_stream_index": drift_idx, "label_delay": a.label_delay,
               "first_detection_after_drift": {d: next(({"stream": e["stream"], "delay_transactions": e["stream_index"] - drift_idx} for e in post_ev if e["detector"] == d), None) for d in ("ADWIN", "DDM")},
               "false_alarms_before_drift": {d: sum(e["detector"] == d for e in pre_ev) for d in ("ADWIN", "DDM")},
               "events_total": len(events), "candidate_evaluation": cand_report, "current_model_after_run": reg.current,
               "caveat": "Single synthetic stream/seed: detection delays are illustrative, not general performance claims."}
    save_json(summary, os.path.join(OUT, "drift_summary.json"))
    s = pd.DataFrame(errs, columns=["idx", "err"]); roll = s.err.rolling(300).mean()
    fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(s.idx, roll, label="rolling error (300)")
    ax.axvline(drift_idx, color="k", ls="--", label="true drift")
    for e in events: ax.axvline(e["label_index"], color="r" if e["detector"] == "ADWIN" else "orange", alpha=.4)
    ax.set_title("Prequential error with ADWIN (red) / DDM (orange) alarms - SYNTHETIC"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "drift_stream.png"), dpi=120); plt.close(fig)
    print(json.dumps(summary, indent=1, default=str)[:1800])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift-at", type=float, default=0.6); ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--label-delay", type=int, default=100); ap.add_argument("--window", type=int, default=2500)
    main(ap.parse_args())
