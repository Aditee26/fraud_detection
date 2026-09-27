"""
app.py
-------
Streamlit presentation UI for:
"Adversarially Robust Fraud Detection in Banking Using Machine Learning
and Blockchain"

4 tabs:
    Fraud Detection
    Model Performance
    Adversarial Testing
    Blockchain Audit

Run:
    streamlit run app.py

Required artifacts:
    artifacts/fraud_model.keras
    artifacts/fraud_model_robust.keras
    artifacts/scaler.npz
    artifacts/metrics.json
    artifacts/adversarial_comparison.json
"""

import json
import numpy as np
import streamlit as st
from tensorflow import keras
import os
import pandas as pd
from dataclasses import asdict
from blockchain_demo import LocalFraudChain


# ============================================================
# PATHS
# ============================================================

ARTIFACT_DIR = "artifacts"

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "fraud_model.keras"
)

ROBUST_MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "fraud_model_robust.keras"
)

SCALER_PATH = os.path.join(
    ARTIFACT_DIR,
    "scaler.npz"
)

METRICS_PATH = os.path.join(
    ARTIFACT_DIR,
    "metrics.json"
)

ADV_PATH = os.path.join(
    ARTIFACT_DIR,
    "adversarial_comparison.json"
)

MODEL_COMPARISON_JSON_PATH = os.path.join(
    ARTIFACT_DIR,
    "model_comparison.json"
)

MODEL_COMPARISON_PLOT_PATH = os.path.join(
    ARTIFACT_DIR,
    "model_comparison.png"
)

MODEL_COMPARISON_CM_PLOT_PATH = os.path.join(
    ARTIFACT_DIR,
    "model_comparison_confusion.png"
)

EXPLAIN_DIR = os.path.join(
    ARTIFACT_DIR,
    "explainability"
)

SHAP_SUMMARY_PATH = os.path.join(
    EXPLAIN_DIR,
    "shap_summary.png"
)

SHAP_BAR_PATH = os.path.join(
    EXPLAIN_DIR,
    "shap_bar.png"
)

SHAP_LOCAL_PATH = os.path.join(
    EXPLAIN_DIR,
    "shap_local.png"
)

SHAP_EXPLANATION_JSON_PATH = os.path.join(
    EXPLAIN_DIR,
    "explanation.json"
)

DATA_PATH = os.path.join(
    "data",
    "creditcard.csv"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Adversarially Robust Fraud Detection",
    page_icon="👾",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM THEME (CSS ONLY -- no logic, no data changes)
#
# Streamlit's default look is plain gray boxes. This block re-skins
# it with a dark banking/security palette, a proper heading font,
# card-style metrics, pill-style tabs, and softer borders. Every
# number, table and chart the app shows is completely unchanged --
# this only changes how it LOOKS.
# ============================================================

def load_css(path: str) -> str:
    """Read the external stylesheet and wrap it in a <style> tag."""
    with open(path, "r", encoding="utf-8") as f:
        return f"<style>\n{f.read()}\n</style>"


def load_css(path: str) -> str:
    """Read the external stylesheet and wrap it in a <style> tag."""
    with open(path, "r", encoding="utf-8") as f:
        return f"<style>\n{f.read()}\n</style>"


st.markdown(load_css(os.path.join("assets", "style.css")), unsafe_allow_html=True)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    required = [
        MODEL_PATH,
        SCALER_PATH,
        METRICS_PATH,
        DATA_PATH,
    ]

    missing = [
        path for path in required
        if not os.path.exists(path)
    ]

    if missing:
        return None

    # -------------------------
    # Load normal model
    # -------------------------

    model = keras.models.load_model(
        MODEL_PATH
    )

    # -------------------------
    # Load scaler
    # -------------------------

    scaler_npz = np.load(
        SCALER_PATH,
        allow_pickle=True
    )

    mean = scaler_npz["mean"]
    scale = scaler_npz["scale"]
    columns = scaler_npz["columns"]

    # -------------------------
    # Load metrics
    # -------------------------

    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        metrics = json.load(f)

    # -------------------------
    # Load adversarial metrics
    # -------------------------

    adv = None

    if os.path.exists(ADV_PATH):

        with open(
            ADV_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            adv = json.load(f)

    # -------------------------
    # Load robust model
    # -------------------------

    robust_model = None

    if os.path.exists(ROBUST_MODEL_PATH):

        try:
            robust_model = keras.models.load_model(
                ROBUST_MODEL_PATH
            )
        except Exception:
            robust_model = None

    # -------------------------
    # Load real dataset
    # -------------------------

    df = pd.read_csv(
        DATA_PATH
    )

    return {
        "model": model,
        "robust_model": robust_model,
        "mean": mean,
        "scale": scale,
        "columns": list(columns),
        "metrics": metrics,
        "adv": adv,
        "df": df,
    }


# ============================================================
# HELPERS
# ============================================================

def scale_row(
    row_values,
    mean,
    scale
):
    """
    Standardize one transaction using
    the same scaler used during training.
    """

    values = np.array(
        row_values,
        dtype=float
    )

    return (
        (values - mean) / scale
    )


def metric_value(
    metrics,
    key,
    default=None
):
    """
    Safely retrieve a metric.

    Prevents Streamlit from crashing when
    an older/newer metrics.json has a
    different schema.
    """

    value = metrics.get(key, default)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_percent(value, digits=2):
    """
    Convert decimal metric to percentage.
    """

    if value is None:
        return "N/A"

    return f"{value * 100:.{digits}f}%"


def format_number(value, digits=4):
    """
    Format normal decimal metric.
    """

    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"


@st.cache_resource
def get_chain():
    from blockchain_demo import LocalFraudChain
    return LocalFraudChain()


# ============================================================
# LOAD EVERYTHING
# ============================================================

artifacts = load_artifacts()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="fd-hero">
        <h1>Fraud Detection in Banking</h1>
        <p>Machine Learning · Adversarial Robustness · Explainability · Blockchain Audit</p>
        <div class="fd-badge-row">
            <span class="fd-badge">📊 <b>284,807</b> Transactions</span>
            <span class="fd-badge">🎯 <b>5</b> Models Compared</span>
            <span class="fd-badge">🛡️ FGSM Adversarial Testing</span>
            <span class="fd-badge">🔎 SHAP Explainability</span>
            <span class="fd-badge">⛓️ Blockchain Audit Trail</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MISSING ARTIFACT HANDLING
# ============================================================

if artifacts is None:

    st.error(
        "Required project artifacts are missing."
    )

    st.markdown(
        """
Run these commands from the project folder:

```text
python train_model.py
python adversarial_test.py
streamlit run app.py
    """)
    st.stop()

df = artifacts["df"]

metrics = artifacts["metrics"]

feature_cols = artifacts["columns"]

adv = artifacts["adv"]

is_demo = metrics.get(
"is_demo_data",
False
)

if is_demo:

   st.warning(
    "🧪 DEMO MODE: The model was trained using synthetic data."
)

else:

   st.success(
    "✅ Model trained using the REAL Kaggle Credit Card Fraud Detection dataset."
)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
[
"🔍 Fraud Detection",
"📊 Model Performance",
"🛡️ Adversarial Testing",
"⛓️ Blockchain Audit",
"⚖️ Model Comparison",
"🔎 XGBoost Explainability",
]
)

with tab1:
    st.subheader(
        "Live Transaction Prediction"
    )

    st.write(
        "Enter or select a transaction and let the trained ANN estimate its fraud probability."
    )

    # --------------------------------------------------------
    # PRELOADED EXAMPLES
    # --------------------------------------------------------

    fraud_examples = df[
        df["Class"] == 1
    ].head(3)

    legit_examples = df[
        df["Class"] == 0
    ].head(3)

    preset_options = {
        "-- choose a preloaded example --": None
    }

    for i, (_, row) in enumerate(
        fraud_examples.iterrows()
    ):

        preset_options[
            f"Example fraud transaction #{i + 1}"
        ] = row

    for i, (_, row) in enumerate(
        legit_examples.iterrows()
    ):

        preset_options[
            f"Example legitimate transaction #{i + 1}"
        ] = row

    choice = st.selectbox(
        "Preloaded example transactions",
        list(preset_options.keys())
    )

    preset_row = preset_options[choice]

    # --------------------------------------------------------
    # TIME + AMOUNT
    # --------------------------------------------------------

    col_a, col_b = st.columns(2)

    with col_a:

        default_time = (
            float(preset_row["Time"])
            if preset_row is not None
            else float(df["Time"].median())
        )

        default_amount = (
            float(preset_row["Amount"])
            if preset_row is not None
            else float(df["Amount"].median())
        )

        time_val = st.number_input(
            "Time (seconds since first transaction)",
            value=default_time
        )

        amount_val = st.number_input(
            "Amount",
            value=default_amount,
            min_value=0.0
        )

    with col_b:

        st.info(
            "V1–V28 are anonymised PCA features from the Kaggle dataset."
        )

        use_preset_vs = st.checkbox(
            "Use preset V1–V28 values",
            value=True
        )

    # --------------------------------------------------------
    # V FEATURES
    # --------------------------------------------------------

    v_cols = [
        c for c in feature_cols
        if c.startswith("V")
    ]

    if (
        preset_row is not None
        and use_preset_vs
    ):

        v_values = [
            float(preset_row[c])
            for c in v_cols
        ]

    else:

        v_values = [
            float(df[c].median())
            for c in v_cols
        ]

    # --------------------------------------------------------
    # ADVANCED FEATURE EDITING
    # --------------------------------------------------------

    with st.expander(
        "⚙️ Adjust individual V1–V28 features"
    ):

        cols_grid = st.columns(4)

        new_v_values = []

        for i, c in enumerate(v_cols):

            with cols_grid[i % 4]:

                val = st.number_input(
                    c,
                    value=float(v_values[i]),
                    key=f"v_{c}",
                    format="%.4f"
                )

                new_v_values.append(val)

        v_values = new_v_values

    # --------------------------------------------------------
    # CREATE TRANSACTION
    # --------------------------------------------------------

    row_dict = {
        "Time": time_val
    }

    for c, value in zip(
        v_cols,
        v_values
    ):

        row_dict[c] = value

    row_dict["Amount"] = amount_val

    ordered_values = [
        row_dict[c]
        for c in feature_cols
    ]

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    if st.button(
        "🔎 Analyze Transaction",
        type="primary"
    ):

        x_scaled = scale_row(
            ordered_values,
            artifacts["mean"],
            artifacts["scale"]
        ).reshape(1, -1)

        prob = float(
            artifacts["model"]
            .predict(
                x_scaled,
                verbose=0
            )
            .ravel()[0]
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        c1, c2 = st.columns(
            [1, 2]
        )

        with c1:

            st.metric(
                "Fraud Probability",
                f"{prob * 100:.2f}%"
            )

            if prob >= 0.5:

                st.error(
                    "🔴 HIGH RISK / FRAUD"
                )

            else:

                st.success(
                    "🟢 LOW RISK / LEGITIMATE"
                )

        # ----------------------------------------------------
        # FEATURE CONTRIBUTION PROXY
        # ----------------------------------------------------

        with c2:

            st.markdown(
                "**Most unusual features compared with dataset mean**"
            )

            z = (
                np.array(
                    ordered_values,
                    dtype=float
                )
                - artifacts["mean"]
            )

            z = z / artifacts["scale"]

            contrib = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "z_score": z,
                }
            )

            contrib["abs_z"] = (
                contrib["z_score"].abs()
            )

            top = (
                contrib
                .sort_values(
                    "abs_z",
                    ascending=False
                )
                .head(6)
            )

            st.bar_chart(
                top.set_index(
                    "feature"
                )["z_score"]
            )

        # ----------------------------------------------------
        # SAVE LAST TRANSACTION
        # ----------------------------------------------------

        st.session_state["last_txn"] = {

            "transaction_id":
                f"TXN-{np.random.randint(10000, 99999)}",

            "decision":
                "FRAUD"
                if prob >= 0.5
                else "LEGITIMATE",

            "risk_score":
                prob * 100,

            "probability":
                prob,

            "features":
                ordered_values,
        }

        st.info(
            "This transaction can now be recorded in the Blockchain Audit tab."
        )

with tab2:
    st.subheader(
        "Model Performance"
    )

    m = metrics

    # --------------------------------------------------------
    # CURRENT METRICS
    # --------------------------------------------------------

    accuracy = metric_value(
        m,
        "accuracy"
    )

    precision = metric_value(
        m,
        "precision"
    )

    recall = metric_value(
        m,
        "recall"
    )

    f1 = metric_value(
        m,
        "f1"
    )

    roc_auc = metric_value(
        m,
        "roc_auc"
    )

    pr_auc = metric_value(
        m,
        "pr_auc"
    )

    # --------------------------------------------------------
    # DISPLAY METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Accuracy",
        format_percent(accuracy)
    )

    c2.metric(
        "Precision",
        format_percent(precision)
    )

    c3.metric(
        "Recall",
        format_percent(recall)
    )

    c4.metric(
        "F1-score",
        format_percent(f1)
    )

    c5.metric(
        "ROC-AUC",
        format_number(roc_auc)
    )

    c6.metric(
        "PR-AUC",
        format_number(pr_auc)
    )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "### Dataset & Evaluation"
    )

    n_test = metric_value(
        m,
        "n_test"
    )

    fraud_rate = metric_value(
        m,
        "fraud_rate_pct"
    )

    dataset_cols = st.columns(3)

    with dataset_cols[0]:

        st.metric(
            "Dataset Rows",
            f"{len(df):,}"
        )

    with dataset_cols[1]:

        if n_test is not None:

            st.metric(
                "Test Transactions",
                f"{int(n_test):,}"
            )

        else:

            st.metric(
                "Test Transactions",
                "N/A"
            )

    with dataset_cols[2]:

        if fraud_rate is not None:

            st.metric(
                "Fraud Rate",
                f"{fraud_rate:.4f}%"
            )

        else:

            actual_fraud_rate = (
                df["Class"].mean()
                * 100
            )

            st.metric(
                "Fraud Rate",
                f"{actual_fraud_rate:.4f}%"
            )

    st.info(
        "Fraud detection is a highly imbalanced classification problem. "
        "Therefore, ROC-AUC, PR-AUC, Recall and F1-score are more informative "
        "than accuracy alone."
    )

    # --------------------------------------------------------
    # MODEL TRAINING INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "### Training Result"
    )

    training_info = m.get("training", {})
    validation_info = m.get("validation", {})

    epochs_completed = training_info.get("epochs_completed")
    final_val_accuracy = validation_info.get("final_val_accuracy")
    best_val_auc = validation_info.get("best_val_auc")

    if epochs_completed is not None:

        st.write(
            f"The neural network trained for **{epochs_completed} epochs** "
            f"(early stopping monitored validation ROC-AUC) using the real "
            f"Kaggle Credit Card Fraud Detection dataset."
        )

    else:

        st.write(
            "Training epoch count is not available in metrics.json."
        )

    if final_val_accuracy is not None and best_val_auc is not None:

        st.write(
            f"The final training run reached "
            f"**{final_val_accuracy * 100:.2f}% validation accuracy** and "
            f"**{best_val_auc:.4f} best validation ROC-AUC**. These numbers "
            f"are read directly from artifacts/metrics.json, so they always "
            f"match whatever train_model.py last produced."
        )

    else:

        st.write(
            "Validation accuracy / ROC-AUC are not available in "
            "metrics.json."
        )

    # --------------------------------------------------------
    # IMAGES
    # --------------------------------------------------------

    st.markdown(
        "### Evaluation Plots"
    )

    image_data = [
        (
            "confusion_matrix.png",
            "Confusion Matrix"
        ),
        (
            "roc_curve.png",
            "ROC Curve"
        ),
        (
            "training_curves.png",
            "Training / Validation Curves"
        ),
    ]

    img_cols = st.columns(3)

    for col, (
        filename,
        caption
    ) in zip(
        img_cols,
        image_data
    ):

        path = os.path.join(
            ARTIFACT_DIR,
            filename
        )

        if os.path.exists(path):

            col.image(
                path,
                caption=caption,
                width="stretch"

                                        )
with tab3:
    st.subheader(
        "🛡️ Adversarial Robustness Testing"
    )

    if adv is None:

        st.warning(
            "Run `python adversarial_test.py` first."
        )

    else:

        epsilon = adv.get(
            "epsilon",
            "N/A"
        )

        st.markdown(
            f"**FGSM perturbation budget (epsilon):** `{epsilon}`"
        )

        st.write(
            "FGSM (Fast Gradient Sign Method) creates a small adversarial "
            "perturbation designed to test whether the model's prediction "
            "can be changed."
        )

        # ----------------------------------------------------
        # ACCURACY COMPARISON
        # ----------------------------------------------------

        normal = adv.get(
            "normal_model",
            {}
        )

        robust = adv.get(
            "adversarially_trained_model",
            {}
        )

        normal_clean = normal.get(
            "clean_accuracy"
        )

        normal_adv = normal.get(
            "adversarial_accuracy"
        )

        robust_clean = robust.get(
            "clean_accuracy"
        )

        robust_adv = robust.get(
            "adversarial_accuracy"
        )

        st.markdown(
            "### Clean vs. Adversarial Accuracy"
        )

        table = pd.DataFrame(
            [
                {
                    "Model":
                        "Normal ANN",

                    "Clean Accuracy":
                        format_percent(
                            normal_clean
                        ),

                    "Adversarial Accuracy":
                        format_percent(
                            normal_adv
                        ),
                },
                {
                    "Model":
                        "Adversarially Trained ANN",

                    "Clean Accuracy":
                        format_percent(
                            robust_clean
                        ),

                    "Adversarial Accuracy":
                        format_percent(
                            robust_adv
                        ),
                },
            ]
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # NUMERIC SUMMARY
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3)

        if normal_clean is not None:

            c1.metric(
                "Normal Model - Clean",
                format_percent(
                    normal_clean
                )
            )

        if normal_adv is not None:

            c2.metric(
                "Normal Model - FGSM",
                format_percent(
                    normal_adv
                )
            )

        if robust_adv is not None:

            c3.metric(
                "Robust Model - FGSM",
                format_percent(
                    robust_adv
                )
            )

        # ----------------------------------------------------
        # COMPARISON IMAGE
        # ----------------------------------------------------

        img_path = os.path.join(
            ARTIFACT_DIR,
            "adversarial_comparison.png"
        )

        if os.path.exists(img_path):

            st.image(
                img_path,
                caption="Normal vs. Adversarially Trained Model",
                width=560
            )

        # ----------------------------------------------------
        # SINGLE TRANSACTION DEMO
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            "### Single Transaction Attack Demonstration"
        )

        demo = adv.get(
            "single_transaction_demo",
            {}
        )

        clean_prob = demo.get(
            "fraud_probability_normal_model_clean_input"
        )

        attacked_prob = demo.get(
            "fraud_probability_normal_model_after_fgsm"
        )

        robust_prob = demo.get(
            "fraud_probability_robust_model_after_fgsm"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            if clean_prob is not None:

                st.metric(
                    "Normal Model - Clean",
                    f"{clean_prob * 100:.2f}%"
                )

        with c2:

            if attacked_prob is not None:

                delta = None

                if clean_prob is not None:

                    delta = (
                        attacked_prob
                        - clean_prob
                    ) * 100

                st.metric(
                    "Normal Model - After FGSM",
                    f"{attacked_prob * 100:.2f}%",
                    delta=(
                        f"{delta:.2f} pts"
                        if delta is not None
                        else None
                    )
                )

        with c3:

            if robust_prob is not None:

                st.metric(
                    "Robust Model - After FGSM",
                    f"{robust_prob * 100:.2f}%"
                )

        st.caption(
            "The adversarially trained model was trained using a mixture "
            "of clean and FGSM-perturbed examples. Its purpose is to improve "
            "robustness against small input perturbations."
        )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🔎 Explainability (SHAP)"
    )

    st.caption(
        "Prototype explainability layer using SHAP."
    )

    if st.button(
        "Explain last analyzed transaction"
    ):

        if "last_txn" not in st.session_state:

            st.info(
                "Analyze a transaction in the Fraud Detection tab first."
            )

        else:

            try:

                import shap

                bg = artifacts[
                    "mean"
                ].reshape(1, -1)

                explainer = shap.KernelExplainer(
                    lambda X:
                        artifacts["model"]
                        .predict(
                            X,
                            verbose=0
                        ),
                    bg
                )

                st.info(
                    "Computing SHAP values. This may take a few seconds."
                )

                last = st.session_state[
                    "last_txn"
                ]

                # Explain the transaction the user actually analyzed in
                # the Fraud Detection tab. Older sessions saved before
                # this fix will not have "features" stored -- fall back
                # to a random real row ONLY in that case, and say so.
                if "features" in last:

                    row = np.array(
                        last["features"],
                        dtype=float
                    )

                else:

                    st.warning(
                        "This transaction was analyzed before the "
                        "explainability fix. Showing a random dataset "
                        "row instead -- re-analyze a transaction in the "
                        "Fraud Detection tab for an accurate explanation."
                    )

                    sample_idx = np.random.choice(
                        len(df),
                        size=1
                    )[0]

                    row = (
                        df.iloc[
                            sample_idx
                        ][feature_cols]
                        .values
                        .astype(float)
                    )

                x_scaled = scale_row(
                    row,
                    artifacts["mean"],
                    artifacts["scale"]
                ).reshape(1, -1)

                shap_values = explainer.shap_values(
                    x_scaled,
                    nsamples=100
                )

                sv = np.array(
                    shap_values
                ).reshape(-1)

                shap_df = pd.DataFrame(
                    {
                        "feature":
                            feature_cols,

                        "shap_value":
                            sv,
                    }
                )

                top_pos = (
                    shap_df
                    .sort_values(
                        "shap_value",
                        ascending=False
                    )
                    .head(5)
                )

                top_neg = (
                    shap_df
                    .sort_values(
                        "shap_value",
                        ascending=True
                    )
                    .head(5)
                )

                prob = float(
                    artifacts["model"]
                    .predict(
                        x_scaled,
                        verbose=0
                    )
                    .ravel()[0]
                )

                st.metric(
                    "Explained Transaction Fraud Probability",
                    f"{prob * 100:.2f}%"
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.markdown(
                        "**Features increasing fraud risk**"
                    )

                    st.dataframe(
                        top_pos,
                        use_container_width=True,
                        hide_index=True
                    )

                with c2:

                    st.markdown(
                        "**Features decreasing fraud risk**"
                    )

                    st.dataframe(
                        top_neg,
                        use_container_width=True,
                        hide_index=True
                    )

            except ImportError:

                st.error(
                    "SHAP is not installed. Install it using: "
                    "`pip install shap`"
                )

            except Exception as e:

                st.error(
                    f"SHAP explanation failed: {e}"
                )
# ============================================================
# TAB 4
# BLOCKCHAIN AUDIT
# ============================================================
# ============================================================
# TAB 4: BLOCKCHAIN AUDIT
# ============================================================

with tab4:

    st.subheader("⛓️ Blockchain Audit Trail")

    st.caption(
        "This prototype uses a local tamper-evident hash chain. "
        "Each block stores the previous block's hash, transaction data, "
        "timestamp and its own SHA-256 hash."
    )

    st.info(
        "This is a blockchain-style audit demonstration, not a live "
        "Ethereum network deployment."
    )

    # --------------------------------------------------------
    # GET CHAIN
    # --------------------------------------------------------

    chain = get_chain()

    # Make sure the chain has the expected structure.
    blockchain = getattr(chain, "chain", [])

    # --------------------------------------------------------
    # LAST ANALYZED TRANSACTION
    # --------------------------------------------------------

    if "last_txn" in st.session_state:

        last = st.session_state["last_txn"]

        st.markdown("### Last analyzed transaction")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Transaction",
                last.get("transaction_id", "Unknown")
            )

        with c2:
            st.metric(
                "Decision",
                last.get("decision", "Unknown")
            )

        with c3:
            st.metric(
                "Risk Score",
                f"{float(last.get('risk_score', 0)):.2f}%"
            )

        st.caption(
            "The transaction below can be added to the local "
            "hash-linked audit chain."
        )

        # ----------------------------------------------------
        # ACTUAL CLASS
        # ----------------------------------------------------

        # The live prediction does not necessarily know the real
        # ground-truth class. Therefore, for a live transaction,
        # use -1 to represent "unknown actual class".
        actual_class = int(
            last.get("actual_class", -1)
        )

        if st.button(
            f"📥 Record {last.get('transaction_id', 'transaction')} "
            "on blockchain"
        ):

            block = chain.add_transaction(
                transaction_id=last.get(
                    "transaction_id",
                    "LIVE_TRANSACTION"
                ),
                actual_class=actual_class,
                decision=last.get(
                    "decision",
                    "UNKNOWN"
                ),
                risk_score=float(
                    last.get("risk_score", 0)
                ),
                model_version=last.get(
                    "model_version",
                    "fraud_model.keras"
                )
            )

            # Save updated chain.
            os.makedirs(
                "artifacts",
                exist_ok=True
            )

            chain.save(
                "artifacts/fraud_blockchain.json"
            )

            st.success(
                f"Recorded as Block #{block.index}"
            )

            st.rerun()

    else:

        st.info(
            "Analyze a transaction in the Fraud Detection tab "
            "to record its result here."
        )

    # --------------------------------------------------------
    # SAMPLE FRAUD RECORD
    # --------------------------------------------------------

    st.markdown("### Add Audit Record")

    st.caption(
        "This uses a fixed demonstration record so you can "
        "show how a fraud decision is stored in the hash chain."
    )

    if st.button(
        "➕ Log sample fraud decision "
        "(TXN-10021, 94.7% risk)"
    ):

        block = chain.add_transaction(
            transaction_id="TXN-10021",
            actual_class=1,
            decision="FRAUD",
            risk_score=94.7,
            model_version="fraud_model.keras"
        )

        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        chain.save(
            "artifacts/fraud_blockchain.json"
        )

        st.success(
            f"Recorded as Block #{block.index}"
        )

        st.rerun()

    # --------------------------------------------------------
    # REFRESH CHAIN DATA
    # --------------------------------------------------------

    blockchain = getattr(
        chain,
        "chain",
        []
    )

    # --------------------------------------------------------
    # DISPLAY BLOCKS
    # --------------------------------------------------------

    if len(blockchain) > 1:

        st.markdown("### Recorded Audit Blocks")

        records = []

        for block in blockchain[1:]:

            # Block is a dataclass object in LocalFraudChain.
            if hasattr(block, "__dataclass_fields__"):

                record = asdict(block)

            elif isinstance(block, dict):

                record = block.copy()

            else:

                record = {}

                possible_fields = [
                    "index",
                    "timestamp",
                    "transaction_id",
                    "actual_class",
                    "decision",
                    "risk_score",
                    "model_version",
                    "previous_hash",
                    "hash"
                ]

                for field in possible_fields:

                    if hasattr(block, field):

                        record[field] = getattr(
                            block,
                            field
                        )

            records.append(record)

        if records:

            show_df = pd.DataFrame(records)

            # ------------------------------------------------
            # FORMAT RISK SCORE
            # ------------------------------------------------

            if "risk_score" in show_df.columns:

                show_df["risk_score"] = (
                    show_df["risk_score"]
                    .astype(float)
                    .round(2)
                )

            # ------------------------------------------------
            # SHORTEN HASHES FOR DISPLAY
            # ------------------------------------------------

            if "hash" in show_df.columns:

                show_df["hash"] = (
                    show_df["hash"]
                    .astype(str)
                    .str[:12]
                    + "..."
                )

            if "previous_hash" in show_df.columns:

                show_df["previous_hash"] = (
                    show_df["previous_hash"]
                    .astype(str)
                    .str[:12]
                    + "..."
                )

            st.dataframe(
                show_df,
                width="stretch",
                hide_index=True
            )

        # ----------------------------------------------------
        # BLOCKCHAIN SUMMARY
        # ----------------------------------------------------

        st.divider()

        # LocalFraudChain.verify_integrity()
        # returns ONLY True or False.
        valid = chain.verify_integrity()

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Total Blocks",
                len(blockchain)
            )

        with c2:

            st.metric(
                "Audit Records",
                len(blockchain) - 1
            )

        with c3:

            st.metric(
                "Chain Integrity",
                "VALID" if valid else "BROKEN"
            )

        if valid:

            st.success(
                "✅ Blockchain Integrity: VERIFIED"
            )

        else:

            st.error(
                "❌ Blockchain Integrity: BROKEN"
            )

        # ----------------------------------------------------
        # BLOCK DETAILS
        # ----------------------------------------------------

        st.markdown("### 🔗 Block Details")

        selected_block_index = st.selectbox(
            "Select block",
            options=list(
                range(
                    1,
                    len(blockchain)
                )
            ),
            format_func=lambda x: (
                f"Block #{x} - "
                f"{blockchain[x].transaction_id}"
            )
        )

        selected_block = blockchain[
            selected_block_index
        ]

        block_data = asdict(
            selected_block
        )

        st.json(
            block_data
        )

        # ----------------------------------------------------
        # HASH VERIFICATION EXPLANATION
        # ----------------------------------------------------

        st.markdown("### 🔐 Hash Verification")

        st.write(
            "Each block contains the SHA-256 hash of its own "
            "contents and the hash of the previous block."
        )

        st.code(
            f"Block #{selected_block.index}\n"
            f"Transaction: {selected_block.transaction_id}\n"
            f"Previous Hash: {selected_block.previous_hash}\n"
            f"Current Hash:  {selected_block.hash}",
            language="text"
        )

        # ----------------------------------------------------
        # TAMPER DEMONSTRATION
        # ----------------------------------------------------

        with st.expander(
            "⚠️ Demonstrate Blockchain Tampering"
        ):

            st.warning(
                "The current LocalFraudChain class does not "
                "provide a tamper() method. This demonstration "
                "manually changes a block in memory and then "
                "runs verify_integrity() to show that the "
                "stored hash no longer matches the block data."
            )

            tamper_idx = st.number_input(
                "Block index to tamper with",
                min_value=1,
                max_value=max(
                    1,
                    len(blockchain) - 1
                ),
                value=1,
                step=1
            )

            new_score = st.number_input(
                "New fraudulent risk score",
                min_value=0.0,
                max_value=100.0,
                value=99.9,
                step=0.1
            )

            if st.button(
                "⚠️ Tamper with this block"
            ):

                target_block = blockchain[
                    int(tamper_idx)
                ]

                # Store original value so the user can see
                # exactly what was changed.
                original_score = (
                    target_block.risk_score
                )

                target_block.risk_score = round(
                    float(new_score),
                    4
                )

                tampered_valid = (
                    chain.verify_integrity()
                )

                if tampered_valid:

                    st.error(
                        "Unexpected result: integrity still "
                        "appears valid."
                    )

                else:

                    st.error(
                        f"❌ Tampering detected! "
                        f"Block #{int(tamper_idx)} was changed "
                        f"from {original_score:.2f}% to "
                        f"{new_score:.2f}%."
                    )

                st.code(
                    f"Original risk score: "
                    f"{original_score:.4f}%\n"
                    f"Modified risk score: "
                    f"{new_score:.4f}%\n\n"
                    f"Blockchain integrity: "
                    f"{'VALID' if tampered_valid else 'BROKEN'}",
                    language="text"
                )

    else:

        st.write(
            "No audit blocks recorded yet."
        )

        st.caption(
            "Analyze a transaction and record its decision "
            "to create the first audit block."
        )

# ============================================================
# TAB 5
# MODEL COMPARISON (Phase 1)
# ============================================================

with tab5:

    st.subheader(
        "⚖️ Model Comparison"
    )

    st.caption(
        "Logistic Regression, Decision Tree, Random Forest and XGBoost, "
        "trained and scored on the exact same test set as the ANN above. "
        "No model here is retrained by the dashboard -- these numbers "
        "come from running  python model_comparison.py  and are read "
        "directly from artifacts/model_comparison.json."
    )

    if not os.path.exists(MODEL_COMPARISON_JSON_PATH):

        st.warning(
            "No comparison results found yet. Run this from the project "
            "folder, then reload this page:"
        )

        st.code(
            "python model_comparison.py",
            language="text"
        )

    else:

        with open(
            MODEL_COMPARISON_JSON_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            comparison = json.load(f)

        results = comparison.get("results", [])

        protocol = comparison.get("protocol", {})

        # ------------------------------------------------------
        # PROTOCOL SUMMARY
        # ------------------------------------------------------

        st.markdown("### Evaluation Protocol")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.metric(
                "Test Transactions",
                f"{protocol.get('test_rows', 'N/A'):,}"
                if isinstance(protocol.get("test_rows"), int)
                else "N/A"
            )

        with p2:
            st.metric(
                "Fraud in Test Set",
                protocol.get("test_fraud_rows", "N/A")
            )

        with p3:
            st.metric(
                "Decision Threshold",
                protocol.get("decision_threshold", "N/A")
            )

        split_ok = protocol.get("split_matches_saved_scaler")

        if split_ok is True:

            st.success(
                "✅ Verified: this comparison uses the same train/test "
                "split as the ANN, so the ANN row below did not see "
                "these test rows during training."
            )

        elif split_ok is False:

            st.error(
                "⚠️ The split used for this comparison did NOT match "
                "artifacts/scaler.npz when it was generated. Re-run "
                "train_model.py then model_comparison.py before "
                "trusting the ANN row below."
            )

        # ------------------------------------------------------
        # RESULTS TABLE
        # ------------------------------------------------------

        st.markdown("### Results Table")

        st.caption(
            "Accuracy is shown for completeness only. With fraud at "
            "roughly 0.17% of transactions, a model that always predicts "
            "'legitimate' would already score about 99.8% accuracy -- "
            "Precision, Recall, F1, PR-AUC and MCC are what actually "
            "separate these models."
        )

        if results:

            table_rows = []

            for r in results:

                table_rows.append(
                    {
                        "Model": r.get("model"),
                        "Accuracy": format_percent(r.get("accuracy")),
                        "Precision": format_percent(r.get("precision")),
                        "Recall": format_percent(r.get("recall")),
                        "F1": format_number(r.get("f1")),
                        "ROC-AUC": format_number(r.get("roc_auc")),
                        "PR-AUC": format_number(r.get("pr_auc")),
                        "MCC": format_number(r.get("mcc")),
                        "Fraud Caught": (
                            f"{r.get('true_positive', 'N/A')}"
                        ),
                        "False Alarms": (
                            f"{r.get('false_positive', 'N/A'):,}"
                            if isinstance(r.get("false_positive"), int)
                            else "N/A"
                        ),
                        "Fraud Missed": (
                            f"{r.get('false_negative', 'N/A')}"
                        ),
                    }
                )

            comparison_df = pd.DataFrame(table_rows)

            st.dataframe(
                comparison_df,
                width="stretch",
                hide_index=True
            )

            # --------------------------------------------------
            # BEST-BY-METRIC CALLOUTS
            #
            # These are factual ("highest X was model Y"), never a
            # single "best overall" claim -- the trade-off between
            # precision and recall is a business decision, not
            # something this dashboard should decide for the user.
            # --------------------------------------------------

            def best_by(key):
                valid = [r for r in results if r.get(key) is not None]
                if not valid:
                    return None
                return max(valid, key=lambda r: r[key])

            st.markdown("### Reading the Results")

            b1, b2, b3, b4 = st.columns(4)

            best_recall = best_by("recall")
            best_precision = best_by("precision")
            best_pr_auc = best_by("pr_auc")
            best_mcc = best_by("mcc")

            with b1:
                if best_recall:
                    st.metric(
                        "Highest Recall",
                        best_recall["model"],
                        format_percent(best_recall["recall"])
                    )

            with b2:
                if best_precision:
                    st.metric(
                        "Highest Precision",
                        best_precision["model"],
                        format_percent(best_precision["precision"])
                    )

            with b3:
                if best_pr_auc:
                    st.metric(
                        "Highest PR-AUC",
                        best_pr_auc["model"],
                        format_number(best_pr_auc["pr_auc"])
                    )

            with b4:
                if best_mcc:
                    st.metric(
                        "Highest MCC",
                        best_mcc["model"],
                        format_number(best_mcc["mcc"])
                    )

            st.info(
                "Recall and precision trade against each other: the "
                "model that catches the most fraud (highest recall) is "
                "not the model with the fewest false alarms (highest "
                "precision). Which trade-off is right depends on the "
                "relative cost of a missed fraud versus an analyst "
                "reviewing a false alarm -- a threshold/cost analysis "
                "(a later phase of this project) turns that judgement "
                "call into a number instead of a guess."
            )

        # ------------------------------------------------------
        # PLOTS
        # ------------------------------------------------------

        st.markdown("### Charts")

        chart_cols = st.columns(2)

        with chart_cols[0]:

            if os.path.exists(MODEL_COMPARISON_PLOT_PATH):

                st.image(
                    MODEL_COMPARISON_PLOT_PATH,
                    caption="Metric-by-metric comparison",
                    width=560
                )

        with chart_cols[1]:

            if os.path.exists(MODEL_COMPARISON_CM_PLOT_PATH):

                st.image(
                    MODEL_COMPARISON_CM_PLOT_PATH,
                    caption="Confusion matrix per model",
                    width=560
                )

        st.caption(
            "To refresh these results after retraining any model, "
            "run  python model_comparison.py  again and reload this page."
        )

# ============================================================
# TAB 6
# XGBOOST EXPLAINABILITY (Phase 3 - SHAP)
# ============================================================

with tab6:

    st.subheader(
        "🔎 XGBoost Explainability (SHAP)"
    )

    st.caption(
        "SHAP breaks down the XGBoost model's fraud score into a "
        "per-feature contribution: which features pushed a prediction "
        "toward fraud, which pushed it toward legitimate, and by how "
        "much. Generated by  python explainability.py  -- not "
        "recomputed live by the dashboard."
    )

    st.warning(
        "SHAP values explain what influenced the MODEL'S prediction. "
        "They are not proof that a transaction is fraudulent."
    )

    if not os.path.exists(SHAP_EXPLANATION_JSON_PATH):

        st.warning(
            "No explainability results found yet. Run this from the "
            "project folder (requires artifacts/xgboost_model.json, "
            "produced by model_comparison.py), then reload this page:"
        )

        st.code(
            "python model_comparison.py\n"
            "python explainability.py",
            language="text"
        )

    else:

        with open(
            SHAP_EXPLANATION_JSON_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            explanation = json.load(f)

        # ------------------------------------------------------
        # GLOBAL: TOP FEATURES
        # ------------------------------------------------------

        st.markdown("### Global Feature Importance")

        st.write(
            "Averaged across a sample of test transactions, ranked by "
            "how much each feature typically moves the model's "
            "prediction (in either direction)."
        )

        global_info = explanation.get("global_explanation", {})

        top_features = global_info.get(
            "top_features_by_mean_abs_shap", {}
        )

        if top_features:

            top_df = pd.DataFrame(
                [
                    {"Feature": k, "Mean |SHAP value|": round(v, 4)}
                    for k, v in top_features.items()
                ]
            )

            st.dataframe(
                top_df,
                width="stretch",
                hide_index=True
            )

        chart_cols = st.columns(2)

        with chart_cols[0]:

            if os.path.exists(SHAP_BAR_PATH):

                st.image(
                    SHAP_BAR_PATH,
                    caption="Global feature importance",
                    width=560
                )

        with chart_cols[1]:

            if os.path.exists(SHAP_SUMMARY_PATH):

                st.image(
                    SHAP_SUMMARY_PATH,
                    caption=(
                        "SHAP summary -- color shows the feature's "
                        "actual value (red = high, blue = low), "
                        "position shows its impact"
                    ),
                    width=560
                )

        # ------------------------------------------------------
        # LOCAL: ONE SPECIFIC TRANSACTION
        # ------------------------------------------------------

        st.divider()

        st.markdown("### Local Explanation — One Real Fraud Transaction")

        local_info = explanation.get("local_explanation_example", {})

        prob = local_info.get("model_probability")

        if prob is not None:

            st.metric(
                "Model's Fraud Probability for This Transaction",
                f"{prob * 100:.2f}%"
            )

        contributions = local_info.get("top_contributing_features", [])

        if contributions:

            contrib_df = pd.DataFrame(
                [
                    {
                        "Feature": c["feature"],
                        "Scaled Value": round(c["scaled_value"], 3),
                        "SHAP Value": round(c["shap_value"], 4),
                        "Direction": c["direction"],
                    }
                    for c in contributions
                ]
            )

            st.dataframe(
                contrib_df,
                width="stretch",
                hide_index=True
            )

        if os.path.exists(SHAP_LOCAL_PATH):

            st.image(
                SHAP_LOCAL_PATH,
                caption=(
                    "Red bars pushed this transaction's score toward "
                    "fraud. Blue bars pushed it toward legitimate."
                ),
                width=560
            )

        st.caption(
            "To explain a different transaction, edit the selection "
            "logic in explainability.py and re-run it, then reload "
            "this page."
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="fd-footer">
        Built with <b>XGBoost</b> · <b>Keras</b> · <b>SHAP</b> ·
        <b>Streamlit</b> — B.Tech final-year project,
        Fraud Detection in Banking using Machine Learning and Blockchain
    </div>
    """,
    unsafe_allow_html=True,
)