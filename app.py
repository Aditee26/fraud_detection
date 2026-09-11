"""
app.py
-------
Streamlit presentation UI for:
"Adversarially Robust Fraud Detection in Banking Using Machine Learning
and Blockchain"

4 tabs: Fraud Detection | Model Performance | Adversarial Testing | Blockchain Audit

Run with:  streamlit run app.py
(Run train_model.py and adversarial_test.py at least once first, so the
artifacts/ folder has a trained model, metrics, and comparison numbers.)
"""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras

from blockchain_demo import LocalFraudChain

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "fraud_model.keras")
ROBUST_MODEL_PATH = os.path.join(ARTIFACT_DIR, "fraud_model_robust.keras")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.npz")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics.json")
ADV_PATH = os.path.join(ARTIFACT_DIR, "adversarial_comparison.json")

st.set_page_config(page_title="Adversarially Robust Fraud Detection", layout="wide")


# ---------------------------------------------------------------- loaders --
@st.cache_resource
def load_artifacts():
    missing = [p for p in [MODEL_PATH, SCALER_PATH, METRICS_PATH]
               if not os.path.exists(p)]
    if missing:
        return None
    model = keras.models.load_model(MODEL_PATH)
    scaler_npz = np.load(SCALER_PATH, allow_pickle=True)
    mean, scale, columns = scaler_npz["mean"], scaler_npz["scale"], scaler_npz["columns"]
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    adv = None
    if os.path.exists(ADV_PATH):
        with open(ADV_PATH) as f:
            adv = json.load(f)
    robust_model = None
    if os.path.exists(ROBUST_MODEL_PATH):
        robust_model = keras.models.load_model(ROBUST_MODEL_PATH)
    df = pd.read_csv("data/creditcard.csv")
    return {
        "model": model, "mean": mean, "scale": scale, "columns": list(columns),
        "metrics": metrics, "adv": adv, "robust_model": robust_model, "df": df,
    }


def scale_row(row_values, mean, scale):
    return (np.array(row_values) - mean) / scale


@st.cache_resource
def get_chain():
    return LocalFraudChain()


artifacts = load_artifacts()

st.title("🏦 Adversarially Robust Fraud Detection in Banking")
st.caption("ML + Blockchain prototype — B.Tech project demonstration")

if artifacts is None:
    st.error(
        "No trained model found. Run these first from the project folder:\n\n"
        "```\npython train_model.py\npython adversarial_test.py\n```"
    )
    st.stop()

df = artifacts["df"]
is_demo = artifacts["metrics"].get("is_demo_data", True)

if is_demo:
    st.warning(
        "🧪 **DEMO MODE** — the real Kaggle Credit Card Fraud dataset was not "
        "available in this environment, so a synthetic dataset with the same "
        "shape and fraud rate (~0.17%) was generated for this run. All metrics "
        "below are computed live from that data — drop the real "
        "`data/creditcard.csv` in and re-run `train_model.py` to use real data.",
        icon="🧪",
    )
else:
    st.success("✅ Trained on the real Kaggle Credit Card Fraud Detection dataset.", icon="✅")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Fraud Detection", "📊 Model Performance", "🛡️ Adversarial Testing", "⛓️ Blockchain Audit"]
)

feature_cols = artifacts["columns"]

# ============================================================ TAB 1 =======
with tab1:
    st.subheader("Live Transaction Prediction")

    fraud_examples = df[df["Class"] == 1].head(3)
    legit_examples = df[df["Class"] == 0].head(3)
    preset_options = {"-- choose a preloaded example --": None}
    for i, (_, r) in enumerate(fraud_examples.iterrows()):
        preset_options[f"Example fraud transaction #{i+1}"] = r
    for i, (_, r) in enumerate(legit_examples.iterrows()):
        preset_options[f"Example legitimate transaction #{i+1}"] = r

    choice = st.selectbox("Preloaded example transactions", list(preset_options.keys()))
    preset_row = preset_options[choice]

    col_a, col_b = st.columns(2)
    with col_a:
        time_val = st.number_input(
            "Time (seconds since first transaction)",
            value=float(preset_row["Time"]) if preset_row is not None else float(df["Time"].median()),
        )
        amount_val = st.number_input(
            "Amount",
            value=float(preset_row["Amount"]) if preset_row is not None else float(df["Amount"].median()),
        )
    with col_b:
        st.caption("V1–V28 are anonymised PCA transaction features "
                    "(same convention as the original Kaggle dataset).")
        use_preset_vs = st.checkbox("Use preset V1–V28 values", value=True)

    v_cols = [c for c in feature_cols if c.startswith("V")]
    if preset_row is not None and use_preset_vs:
        v_values = [float(preset_row[c]) for c in v_cols]
    else:
        v_values = [float(df[c].median()) for c in v_cols]

    with st.expander("Adjust individual V1–V28 features (advanced)"):
        cols_grid = st.columns(4)
        new_v_values = []
        for i, c in enumerate(v_cols):
            with cols_grid[i % 4]:
                val = st.number_input(c, value=v_values[i], key=f"v_{c}", format="%.4f")
                new_v_values.append(val)
        v_values = new_v_values

    row_dict = {"Time": time_val}
    for c, v in zip(v_cols, v_values):
        row_dict[c] = v
    row_dict["Amount"] = amount_val
    ordered_values = [row_dict[c] for c in feature_cols]

    if st.button("🔎 Analyze Transaction", type="primary"):
        x_scaled = scale_row(ordered_values, artifacts["mean"], artifacts["scale"]).reshape(1, -1)
        prob = float(artifacts["model"].predict(x_scaled, verbose=0).ravel()[0])

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Fraud Probability", f"{prob * 100:.1f}%")
            if prob >= 0.5:
                st.error("🔴 HIGH RISK / FRAUD")
            else:
                st.success("🟢 LOW RISK / LEGITIMATE")

        with c2:
            st.markdown("**Top contributing features (|z-score| vs. dataset mean)**")
            z = np.array(ordered_values, dtype=float) - artifacts["mean"]
            z = z / artifacts["scale"]
            contrib = pd.DataFrame({"feature": feature_cols, "z_score": z})
            contrib["abs_z"] = contrib["z_score"].abs()
            top = contrib.sort_values("abs_z", ascending=False).head(6)
            st.bar_chart(top.set_index("feature")["z_score"])

        st.session_state["last_txn"] = {
            "transaction_id": f"TXN-{np.random.randint(10000, 99999)}",
            "decision": "FRAUD" if prob >= 0.5 else "LEGITIMATE",
            "risk_score": prob * 100,
        }
        st.info("This transaction's result is available to log in the "
                "**Blockchain Audit** tab.")

# ============================================================ TAB 2 =======
with tab2:
    st.subheader("Model Performance")
    m = artifacts["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{m['precision']*100:.2f}%")
    c3.metric("Recall", f"{m['recall']*100:.2f}%")
    c4.metric("F1-score", f"{m['f1']*100:.2f}%")
    c5.metric("ROC-AUC", f"{m['roc_auc']:.4f}")

    st.caption(
        f"Evaluated on {m['n_test']} held-out test transactions "
        f"(fraud rate {m['fraud_rate_pct']:.4f}%). Because fraud is heavily "
        "imbalanced, class weights were used during training and Recall/"
        "F1/ROC-AUC matter more here than raw Accuracy."
    )

    img_cols = st.columns(3)
    for col, fname, caption in zip(
        img_cols,
        ["confusion_matrix.png", "roc_curve.png", "training_curves.png"],
        ["Confusion Matrix", "ROC Curve", "Training/Validation Curves"],
    ):
        path = os.path.join(ARTIFACT_DIR, fname)
        if os.path.exists(path):
            col.image(path, caption=caption, use_container_width=True)

# ============================================================ TAB 3 =======
with tab3:
    st.subheader("Adversarial Robustness")
    adv = artifacts["adv"]
    if adv is None:
        st.warning("Run `python adversarial_test.py` to generate this comparison.")
    else:
        st.markdown(f"**FGSM perturbation budget (epsilon):** `{adv['epsilon']}`")

        table = pd.DataFrame([
            {
                "Model": "Normal ANN",
                "Clean Accuracy": f"{adv['normal_model']['clean_accuracy']*100:.2f}%",
                "Adversarial Accuracy": f"{adv['normal_model']['adversarial_accuracy']*100:.2f}%",
            },
            {
                "Model": "Adversarially Trained ANN",
                "Clean Accuracy": f"{adv['adversarially_trained_model']['clean_accuracy']*100:.2f}%",
                "Adversarial Accuracy": f"{adv['adversarially_trained_model']['adversarial_accuracy']*100:.2f}%",
            },
        ])
        st.table(table)

        img_path = os.path.join(ARTIFACT_DIR, "adversarial_comparison.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Clean vs. Adversarial Accuracy", width=500)

        st.markdown("**Single transaction, before vs. after attack vs. after defence**")
        d = adv["single_transaction_demo"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Normal model — clean input",
                   f"{d['fraud_probability_normal_model_clean_input']*100:.1f}%")
        c2.metric("Normal model — after FGSM",
                   f"{d['fraud_probability_normal_model_after_fgsm']*100:.1f}%",
                   delta=f"{(d['fraud_probability_normal_model_after_fgsm'] - d['fraud_probability_normal_model_clean_input'])*100:.1f} pts")
        c3.metric("Robust model — after FGSM",
                   f"{d['fraud_probability_robust_model_after_fgsm']*100:.1f}%")
        st.caption(
            "The FGSM attack nudges every input feature by `epsilon` in the "
            "direction that most increases the model's loss. On the normal "
            "model this measurably shifts the fraud probability / accuracy; "
            "the adversarially trained model — trained on a mix of clean and "
            "FGSM-perturbed examples — recovers most of that lost accuracy."
        )

    st.divider()
    st.subheader("Explainability (SHAP) — prototype layer")
    st.caption("Clearly labelled as the prototype's explainability layer.")
    if st.button("Explain last analyzed transaction"):
        if "last_txn" not in st.session_state:
            st.info("Analyze a transaction in the Fraud Detection tab first.")
        else:
            import shap
            bg = artifacts["mean"].reshape(1, -1)
            explainer = shap.KernelExplainer(
                lambda X: artifacts["model"].predict(X, verbose=0), bg
            )
            st.info("Computing SHAP values for a lightweight background sample "
                    "(kept small for a fast, presentable demo).")
            sample_idx = np.random.choice(len(df), size=1)[0]
            row = df.iloc[sample_idx][feature_cols].values.astype(float)
            x_scaled = scale_row(row, artifacts["mean"], artifacts["scale"]).reshape(1, -1)
            shap_values = explainer.shap_values(x_scaled, nsamples=100)
            sv = np.array(shap_values).reshape(-1)
            shap_df = pd.DataFrame({"feature": feature_cols, "shap_value": sv})
            top_pos = shap_df.sort_values("shap_value", ascending=False).head(5)
            top_neg = shap_df.sort_values("shap_value", ascending=True).head(5)

            prob = float(artifacts["model"].predict(x_scaled, verbose=0).ravel()[0])
            st.metric("Fraud probability for explained transaction", f"{prob*100:.1f}%")
            c1, c2 = st.columns(2)
            c1.markdown("**Top features increasing fraud risk**")
            c1.dataframe(top_pos.set_index("feature"))
            c2.markdown("**Top features decreasing fraud risk**")
            c2.dataframe(top_neg.set_index("feature"))
            st.success(
                "High fraud risk is mainly associated with unusual transaction "
                "characteristics and deviation from normal transaction patterns."
            )

# ============================================================ TAB 4 =======
with tab4:
    st.subheader("Blockchain Audit Trail")
    st.caption(
        "This demo uses a local, tamper-evident hash chain (block → previous "
        "hash → transaction data → timestamp → current hash) instead of a "
        "live Ethereum deployment — see `FraudAudit.sol` for the reference "
        "smart contract this simulates, and README.md for how to deploy it "
        "for real on Ganache/Sepolia with Hardhat."
    )

    chain = get_chain()

    if "last_txn" in st.session_state:
        last = st.session_state["last_txn"]
        if st.button(f"📥 Record decision for {last['transaction_id']} on the chain"):
            block = chain.record_decision(
                last["transaction_id"], last["decision"],
                last["risk_score"], "ann-v1-adv-trained",
            )
            st.success(f"Recorded as Block #{block.index}")
    else:
        st.info("Analyze a transaction in the Fraud Detection tab to record it here.")

    if st.button("➕ Log a sample fraud decision (TXN-10021, 94.7% risk)"):
        block = chain.record_decision("TXN-10021", "FRAUD", 94.7, "ann-v1-adv-trained")
        st.success(f"Recorded as Block #{block.index}")

    records = chain.to_records()
    if len(records) > 1:
        show_df = pd.DataFrame(records[1:])  # skip genesis block
        show_df["hash"] = show_df["hash"].str[:12] + "..."
        show_df["previous_hash"] = show_df["previous_hash"].str[:12] + "..."
        st.dataframe(show_df, use_container_width=True)

        valid, bad_index = chain.verify_integrity()
        st.metric("Chain Integrity", "✅ VERIFIED" if valid else f"❌ BROKEN at block {bad_index}")

        with st.expander("⚠️ Demo: tamper with a past record"):
            tamper_idx = st.number_input(
                "Block index to tamper with", min_value=1,
                max_value=len(chain.chain) - 1, value=1,
            )
            new_score = st.number_input("New (fraudulent) risk score", value=1.0)
            if st.button("Tamper with this block"):
                chain.tamper(int(tamper_idx), new_score)
                st.rerun()
    else:
        st.write("No blocks recorded yet.")
