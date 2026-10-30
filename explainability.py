"""
explainability.py
------------------
PHASE 3 of the roadmap: explain individual fraud predictions using SHAP.

BEGINNER EXPLANATION -- SHAP
-----------------------------

WHAT is it?
    SHAP (SHapley Additive exPlanations) assigns each feature of a
    transaction a number that says how much that feature pushed the
    model's prediction up (toward fraud) or down (toward legitimate).

WHY do we need it?
    A probability like "94% fraud" tells an analyst almost nothing about
    WHY the model thinks that. SHAP turns the black-box score into a
    list like "Amount contributed +0.31, V14 contributed +0.22,
    V4 contributed -0.08", which an analyst can actually act on and a
    regulator can actually audit.

WHAT problem does it solve?
    It solves the "black box" problem: a highly accurate model that
    cannot explain itself is not usable in banking, where a blocked
    transaction has to be justified.

HOW does it work (short version)?
    SHAP comes from cooperative game theory. Imagine the features are
    "players" who together produce the model's output as their
    "payout". SHAP fairly splits that payout among the players (the
    features) based on how much each one changes the prediction, on
    average, across every possible subset of the other features being
    present or absent. For tree-based models (XGBoost, Random Forest),
    this exact calculation can be done efficiently using the tree
    structure itself -- that is what shap.TreeExplainer does. It does
    NOT need to approximate with random sampling the way KernelExplainer
    (used for the ANN in app.py) does, so it is both exact and fast.

WHAT input does it take?
    A trained tree model, and one or more transactions (already scaled
    the same way the model was trained on).

WHAT output does it produce?
    One SHAP value per feature per transaction. Positive = pushed the
    prediction toward fraud. Negative = pushed it toward legitimate.
    The SHAP values for one transaction always sum to
    (that transaction's prediction) - (the model's average prediction).

HOW does it connect to fraud detection?
    Every high-risk alert this project raises can now come with a
    plain list of "why", instead of just a percentage. This is exactly
    what the research paper (Section III-G) and synopsis (Section 9)
    require.

IMPORTANT (per project rules):
    SHAP does NOT prove that a transaction is fraudulent. It explains
    which features influenced the MODEL'S prediction, not the ground
    truth. A feature can have a large positive SHAP value on a
    transaction that turns out to be legitimate.

WHY THIS MODULE USES XGBOOST, NOT THE ANN
    artifacts/xgboost_model.json is the model this module explains.
    Reasons:
      1. TreeExplainer is exact and fast for tree models. The ANN
         would need KernelExplainer, which is an approximation and is
         slow (app.py's existing ANN-SHAP code already shows this --
         it only explains one transaction at a time and still takes a
         few seconds).
      2. Phase 1 (model_comparison.py) measured that XGBoost has
         considerably better precision, recall and MCC than the ANN
         on this dataset, so it is the more relevant model to explain.

Outputs (artifacts/explainability/):
    shap_summary.png    global summary (beeswarm) -- all features,
                         all sampled transactions, at once
    shap_bar.png         global feature importance -- mean |SHAP value|
                         per feature, ranked
    shap_local.png       local explanation for ONE specific transaction
    explanation.json     the raw numbers behind both plots, in JSON

Run:
    python explainability.py

Requires:
    artifacts/xgboost_model.json   (created by model_comparison.py --
                                     run that first if this is missing)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
#
# SEED and TEST_SIZE match train_model.py / model_comparison.py so
# that X_test here is the exact same held-out data XGBoost was
# scored on, not data it was trained on.
# ============================================================

DATA_PATH = "data/creditcard.csv"

ARTIFACT_DIR = "artifacts"

EXPLAIN_DIR = os.path.join(ARTIFACT_DIR, "explainability")

XGBOOST_MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgboost_model.json")

SEED = 42

TEST_SIZE = 0.20

# How many test transactions to use for the GLOBAL plots (summary +
# bar chart). TreeExplainer is fast, but computing SHAP values for
# all ~57,000 test rows is unnecessary for a summary plot and slows
# things down for no visual benefit. A random sample is standard
# practice for SHAP summary plots.
GLOBAL_SAMPLE_SIZE = 2000

SHAP_SUMMARY_PATH = os.path.join(EXPLAIN_DIR, "shap_summary.png")
SHAP_BAR_PATH = os.path.join(EXPLAIN_DIR, "shap_bar.png")
SHAP_LOCAL_PATH = os.path.join(EXPLAIN_DIR, "shap_local.png")
EXPLANATION_JSON_PATH = os.path.join(EXPLAIN_DIR, "explanation.json")


# ============================================================
# Data loading (mirrors model_comparison.py exactly)
# ============================================================

def load_dataset():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\nDataset not found.\nExpected: {os.path.abspath(DATA_PATH)}\n\n"
            f"Place the Kaggle creditcard.csv file inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    print(f"[explain] Loaded {len(df):,} rows")

    return df


def build_test_split(df):
    """
    Reproduce the exact same train/test split used everywhere else in
    this project. We only need X_test / y_test here -- SHAP explains
    predictions on data the model did not train on.
    """

    feature_columns = [c for c in df.columns if c != "Class"]

    X = df[feature_columns].values.astype(np.float32)
    y = df["Class"].astype(np.int32).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    scaler = StandardScaler().fit(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"[explain] Test rows: {len(X_test_s):,}")

    return feature_columns, X_test_s, y_test


def load_xgboost_model():
    """
    Load the XGBoost model saved by model_comparison.py.

    This does NOT retrain anything -- Phase 3 explains a model that
    already exists, it does not create a new one.
    """

    if not os.path.exists(XGBOOST_MODEL_PATH):
        raise SystemExit(
            f"\n{XGBOOST_MODEL_PATH} not found.\n"
            f"Run  python model_comparison.py  first -- it trains and "
            f"saves the XGBoost model this script explains."
        )

    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(XGBOOST_MODEL_PATH)

    print(f"[explain] Loaded XGBoost model <- {XGBOOST_MODEL_PATH}")

    return model


# ============================================================
# SHAP computation
# ============================================================

def compute_shap_values(model, X):
    """
    TreeExplainer computes exact SHAP values for tree-based models by
    walking the tree structure directly -- no sampling, no
    approximation, and much faster than KernelExplainer.

    Returns:
        shap_values: array of shape (n_rows, n_features)
        base_value:  the model's average prediction (in log-odds space
                     for XGBoost's default output), i.e. what the model
                     would predict with no information about any
                     specific feature
    """

    import shap

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    base_value = explainer.expected_value

    # XGBClassifier binary output sometimes returns expected_value as
    # a length-1 array/list depending on the SHAP/XGBoost version.
    # Normalize to a plain float so the rest of this script is simple.
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.array(base_value).reshape(-1)[0])
    else:
        base_value = float(base_value)

    return np.array(shap_values), base_value


# ============================================================
# Plots
# ============================================================

def save_summary_plot(shap_values, X_sample, feature_columns, output_path):
    """
    The "beeswarm" summary plot: one dot per (transaction, feature)
    pair. Position on the x-axis is the SHAP value (how much that
    feature pushed this transaction's prediction toward fraud or
    legitimate). Color is the feature's actual value (red = high,
    blue = low). Features are ranked top-to-bottom by importance.

    Reading it: if a feature shows red dots clustered on the positive
    (right) side, it means HIGH values of that feature push predictions
    toward fraud.
    """

    import shap

    plt.figure(figsize=(9, 8))

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_columns,
        show=False,
        max_display=15,
    )

    plt.title("SHAP Summary — Global Feature Impact (XGBoost)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()

    print(f"[explain] Saved -> {output_path}")


def save_bar_plot(shap_values, feature_columns, output_path):
    """
    Simple ranked bar chart: mean absolute SHAP value per feature,
    across all sampled transactions. This answers "which features
    matter most to this model overall", without saying which
    direction (fraud or legitimate) they push in -- that is what the
    summary plot above is for.
    """

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    order = np.argsort(mean_abs_shap)[::-1][:15]

    top_features = [feature_columns[i] for i in order]
    top_values = mean_abs_shap[order]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(top_features[::-1], top_values[::-1], color="#3B6FA0")
    ax.set_xlabel("Mean |SHAP value|  (average impact on model output)")
    ax.set_title("Global Feature Importance (XGBoost)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)

    print(f"[explain] Saved -> {output_path}")

    return {
        feature_columns[i]: float(mean_abs_shap[i])
        for i in order
    }


def save_local_plot(
    shap_row, base_value, feature_row, feature_columns, output_path,
    title,
):
    """
    Local explanation for ONE transaction: a horizontal bar per
    feature showing its SHAP contribution for this specific row,
    ordered by absolute impact. Positive bars (pushed toward fraud)
    and negative bars (pushed toward legitimate) are colored
    differently so the direction is immediately visible.
    """

    order = np.argsort(np.abs(shap_row))[::-1][:12]

    labels = [
        f"{feature_columns[i]} = {feature_row[i]:.3f}" for i in order
    ]
    values = shap_row[order]

    colors = ["#D64545" if v > 0 else "#3B6FA0" for v in values]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(
        "SHAP value  (red = pushes toward FRAUD, blue = pushes toward "
        "LEGITIMATE)"
    )
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)

    print(f"[explain] Saved -> {output_path}")


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 68)
    print("PHASE 3 - SHAP EXPLAINABILITY")
    print("=" * 68)

    os.makedirs(EXPLAIN_DIR, exist_ok=True)

    np.random.seed(SEED)

    # --- 1. Data and model ---------------------------------------

    df = load_dataset()

    feature_columns, X_test_s, y_test = build_test_split(df)

    model = load_xgboost_model()

    # --- 2. Pick a representative sample for the GLOBAL plots ------
    #
    # A random sample is standard for SHAP summary plots -- using the
    # full ~57,000-row test set would take much longer to compute and
    # to render, without materially changing the visual pattern.

    sample_size = min(GLOBAL_SAMPLE_SIZE, len(X_test_s))

    sample_idx = np.random.choice(
        len(X_test_s), size=sample_size, replace=False
    )

    X_sample = X_test_s[sample_idx]

    print(f"\n[explain] Computing SHAP values for {sample_size:,} "
          f"sampled test transactions (global plots) ...")

    shap_values_sample, base_value = compute_shap_values(model, X_sample)

    # --- 3. Global plots --------------------------------------------

    save_summary_plot(
        shap_values_sample, X_sample, feature_columns, SHAP_SUMMARY_PATH
    )

    top_features = save_bar_plot(
        shap_values_sample, feature_columns, SHAP_BAR_PATH
    )

    # --- 4. Local explanation for ONE genuine fraud transaction -----
    #
    # We pick the fraud transaction the model was MOST confident about,
    # so the explanation is easy to read (a clear-cut case rather than
    # a borderline one).

    fraud_test_idx = np.where(y_test == 1)[0]

    if len(fraud_test_idx) == 0:
        raise SystemExit(
            "No fraud transactions found in the test set -- cannot "
            "produce a local fraud explanation."
        )

    fraud_probs = model.predict_proba(X_test_s[fraud_test_idx])[:, 1]

    most_confident_local = fraud_test_idx[np.argmax(fraud_probs)]

    print(
        f"\n[explain] Explaining one specific fraud transaction "
        f"(test row index {most_confident_local}, model probability "
        f"{fraud_probs.max():.4f}) ..."
    )

    x_local = X_test_s[most_confident_local:most_confident_local + 1]

    shap_values_local, _ = compute_shap_values(model, x_local)

    shap_row = shap_values_local[0]
    feature_row = x_local[0]

    save_local_plot(
        shap_row,
        base_value,
        feature_row,
        feature_columns,
        SHAP_LOCAL_PATH,
        title=(
            f"Local Explanation — Fraud Transaction "
            f"(model probability {fraud_probs.max() * 100:.2f}%)"
        ),
    )

    # --- 5. Save the raw numbers --------------------------------------

    local_contributions = sorted(
        [
            {
                "feature": feature_columns[i],
                "scaled_value": float(feature_row[i]),
                "shap_value": float(shap_row[i]),
                "direction": (
                    "pushes toward FRAUD"
                    if shap_row[i] > 0
                    else "pushes toward LEGITIMATE"
                ),
            }
            for i in range(len(feature_columns))
        ],
        key=lambda d: abs(d["shap_value"]),
        reverse=True,
    )

    payload = {
        "phase": "3 - SHAP explainability",
        "model_explained": "XGBoost (artifacts/xgboost_model.json)",
        "method": "shap.TreeExplainer (exact, not sampled)",
        "note": (
            "SHAP values explain what influenced the MODEL'S prediction. "
            "They are not proof that a transaction is fraudulent, and a "
            "feature can have a large positive SHAP value on a "
            "transaction that turns out to be legitimate."
        ),
        "global_explanation": {
            "sample_size": int(sample_size),
            "base_value": base_value,
            "base_value_meaning": (
                "The model's average prediction (log-odds space) before "
                "any feature information is considered -- the starting "
                "point every transaction's SHAP values are added to."
            ),
            "top_features_by_mean_abs_shap": top_features,
        },
        "local_explanation_example": {
            "test_row_index": int(most_confident_local),
            "actual_class": "FRAUD",
            "model_probability": float(fraud_probs.max()),
            "base_value": base_value,
            "top_contributing_features": local_contributions[:12],
        },
    }

    with open(EXPLANATION_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[explain] Saved -> {EXPLANATION_JSON_PATH}")

    # --- 6. Print a readable summary ----------------------------------

    print("\n" + "=" * 68)
    print("TOP GLOBAL FEATURES (most influential overall)")
    print("=" * 68)

    for feature, value in list(top_features.items())[:8]:
        print(f"  {feature:<10} mean |SHAP| = {value:.4f}")

    print("\n" + "=" * 68)
    print(
        f"WHY WAS TEST ROW {most_confident_local} FLAGGED "
        f"({fraud_probs.max() * 100:.2f}% fraud probability)?"
    )
    print("=" * 68)

    for item in local_contributions[:6]:
        arrow = "+" if item["shap_value"] > 0 else "-"
        print(
            f"  {arrow} {item['feature']:<8} "
            f"(scaled value {item['scaled_value']:+.3f})  "
            f"SHAP = {item['shap_value']:+.4f}  "
            f"({item['direction']})"
        )

    print(
        "\nReminder: this shows what the MODEL used to reach its "
        "decision, not proof the transaction is actually fraud."
    )

    print("\nDone.\n")


if __name__ == "__main__":
    main()