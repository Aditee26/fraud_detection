"""Streamlit sections for the research extension. They only READ saved results
(artifacts/research/**); nothing is trained on page load, and every section degrades
to a clear 'not found' message when a result file is missing."""
from __future__ import annotations
import importlib.util, json, os
import pandas as pd
import streamlit as st

R = os.path.join("artifacts", "research")
ORIGIN_NOTE = "Results below are computed on SYNTHETIC data unless the file's own provenance says otherwise. They demonstrate the method, not real-bank performance."


@st.cache_data(show_spinner=False)
def _csv(path, mtime): return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _json(path, mtime):
    with open(path, encoding="utf-8") as f: return json.load(f)


def load_csv(rel):
    p = os.path.join(R, rel); return _csv(p, os.path.getmtime(p)) if os.path.exists(p) else None


def load_json(rel):
    p = os.path.join(R, rel); return _json(p, os.path.getmtime(p)) if os.path.exists(p) else None


def _missing(what, cmd):
    st.warning(f"{what} not found. Generate it with `{cmd}` (saved results are loaded; nothing is retrained here).")


def _img(rel, caption=None):
    p = os.path.join(R, rel)
    if os.path.exists(p): st.image(p, caption=caption)


def component_status():
    def has(m): return importlib.util.find_spec(m) is not None
    rows = []
    for label, mod, role in [("TensorFlow/Keras", "tensorflow", "ANN, LSTM, GraphSAGE, PGD"), ("XGBoost", "xgboost", "XGBoost branch"),
                             ("SHAP", "shap", "tree explanations"), ("LIME", "lime", "local surrogate explanations"),
                             ("river", "river", "optional ADWIN/DDM cross-check"), ("web3", "web3", "optional on-chain mode")]:
        rows.append({"component": label, "role": role, "status": "available" if has(mod) else "optional - not installed"})
    return pd.DataFrame(rows)


def render_transfer_tab():
    st.subheader("Transfer learning: source domain → target domain")
    st.info(ORIGIN_NOTE)
    cfg, summ = load_json("transfer/transfer_config.json"), load_csv("transfer/transfer_results_summary.csv")
    if cfg is None or summ is None:
        _missing("Transfer-learning results", "python transfer_learning.py"); return
    st.caption(f"Data origin: **{cfg['provenance']['data_origin']}** · alignment: {cfg['alignment']} · target split: {cfg['target_split']} · threshold: {cfg['threshold']}")
    sc = cfg.get("source_domain_test_metrics")
    if sc: st.caption(f"Source-domain sanity check (pretraining works on its own domain): PR-AUC {sc['pr_auc']:.3f}, F1 {sc['f1']:.3f}")
    cols = ["arm", "fraction", "precision_mean", "recall_mean", "f1_mean", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "train_seconds_mean"]
    st.dataframe(summ[cols].round(3), hide_index=True)
    _img("transfer/transfer_curves.png")
    d = load_csv("transfer/transfer_vs_scratch_deltas.csv")
    if d is not None:
        st.markdown("**Transfer − scratch (paired by seed)**"); st.dataframe(d[["fraction", "pr_auc_delta_mean", "pr_auc_delta_std", "pr_auc_seeds_transfer_better", "f1_delta_mean", "f1_delta_std"]].round(3), hide_index=True)
        st.caption("Negative values mean transfer was WORSE. No significance test is reported: too few seeds.")
    w = [f for f in os.listdir(os.path.join(R, "transfer")) if f.endswith(".h5")]
    st.caption("Saved weights: " + (", ".join(sorted(w)) if w else "none"))


def render_pgd_tab():
    st.subheader("PGD / FGSM robustness (clean vs adversarial)")
    cfg, res = load_json("adversarial/robustness_config.json"), load_csv("adversarial/robustness_results.csv")
    if cfg is None or res is None:
        _missing("Robustness results", "python adversarial_research.py"); return
    st.warning(cfg["note"]); st.caption(f"Data origin: **{cfg['origin']}** · immutable features: {cfg['immutable_features']} · {cfg['threshold_policy']}")
    models = sorted(res.model.unique()); m = st.selectbox("Model", models)
    g = res[res.model == m]; st.dataframe(g[["attack", "eps", "precision", "recall", "f1", "pr_auc", "roc_auc"]].round(3), hide_index=True)
    _img("adversarial/robustness_curves.png")
    ck = load_csv("adversarial/constraint_checks.csv")
    if ck is not None: st.caption(f"Constraint checks (budget, feature bounds, immutable features) all satisfied: **{cfg['all_constraints_satisfied']}**")


def render_drift_tab():
    st.subheader("Concept-drift monitoring (ADWIN, DDM)")
    s = load_json("drift/drift_summary.json")
    if s is None:
        _missing("Drift-monitoring results", "python drift_demo.py"); return
    st.info(f"Data origin: **{s['origin']}** stream with a known drift point. Labels reach the detectors only after a {s['label_delay']}-transaction delay.")
    c1, c2 = st.columns(2)
    c1.metric("True drift index", s["drift_stream_index"]); c2.metric("Events logged", s["events_total"])
    st.json({"first detection after drift": s["first_detection_after_drift"], "false alarms before drift": s["false_alarms_before_drift"]})
    _img("drift/drift_stream.png")
    ce = s.get("candidate_evaluation")
    if ce:
        st.markdown("**Candidate retraining (safe workflow)**")
        st.write(f"Candidate PR-AUC gain on later hold-out: {ce['pr_auc_gain']:+.3f} · recommended: **{ce['recommend_promote']}** · deployed model changed: **{ce['deployed_model_changed']}** (promotion requires explicit approval)")
    p = os.path.join(R, "drift", "drift_events.jsonl")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f: ev = [json.loads(x) for x in f if x.strip()]
        if ev: st.dataframe(pd.DataFrame(ev), hide_index=True)
    st.caption(s["caveat"])


def render_results_tab():
    st.subheader("Research experiment results")
    st.info(ORIGIN_NOTE)
    st.markdown("**Component status**"); st.dataframe(component_status(), hide_index=True)
    mm, cfg = load_csv("multimodel/multimodel_results.csv"), load_json("multimodel/multimodel_config.json")
    if mm is None: _missing("Multi-model results", "python train_multimodel.py")
    else:
        st.markdown("**Multi-model branches + learned late fusion** (test split, thresholds chosen on validation)")
        st.dataframe(mm[["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "origin"]].round(3), hide_index=True)
        _img("multimodel/multimodel_comparison.png")
        st.caption(f"Graph: {cfg['graph']}. {cfg['caveat']}")
        st.json({"fusion weights (logit space)": cfg["fusion_weights_on_logit"]})
    ex = load_json("explain/explanation_summary.json")
    if ex: st.markdown(f"**Explainability** ({ex['origin']}): mean top-5 SHAP/LIME overlap = {ex['mean_top5_shap_lime_overlap']:.2f} over {ex['n_explained']} flagged frauds"); _img("explain/shap_global_bar.png")
    else: _missing("Explainability summary", "python explain_research.py")
    if os.path.exists("PAPER_CLAIMS_MATRIX.md"):
        with st.expander("Paper-claim → implementation → evidence matrix"):
            st.markdown(open("PAPER_CLAIMS_MATRIX.md", encoding="utf-8").read())
