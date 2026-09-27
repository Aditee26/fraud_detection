"""
model_comparison.py
-------------------
PHASE 1 of the research-paper roadmap: compare several fraud-detection
models on EXACTLY the same test set, using metrics that are meaningful
for a highly imbalanced problem.

Models compared:

    1. Logistic Regression      (interpretable linear baseline)
    2. Decision Tree            (simple non-linear baseline)
    3. Random Forest            (bagged tree ensemble)
    4. XGBoost                  (gradient-boosted trees, optional import)
    5. Existing trained ANN     (loaded from artifacts/, NOT retrained)

IMPORTANT DESIGN DECISIONS
--------------------------

1. The train/test split reproduces train_model.py exactly
   (same SEED, same TEST_SIZE, same stratify=y). That means the ANN
   saved in artifacts/fraud_model.keras never saw these test rows,
   so comparing it against the new models on this split is fair.

2. The StandardScaler is re-fitted on X_train. Because the split is
   identical, this reproduces artifacts/scaler.npz. The script checks
   that and warns loudly if it does not match -- a mismatch would mean
   the ANN comparison is invalid.

3. Every model is given class-imbalance handling of some kind
   (class_weight="balanced" for sklearn, scale_pos_weight for XGBoost,
   class weights for the already-trained ANN). Otherwise the comparison
   would just measure "which model ignores the minority class least".

4. NO SMOTE here. SMOTE is Phase 2, in smote_training.py.

Outputs:
    artifacts/model_comparison.csv
    artifacts/model_comparison.json
    artifacts/model_comparison.png
    artifacts/model_comparison_confusion.png
    artifacts/xgboost_model.json          (trained XGBoost model)
    artifacts/random_forest_model.joblib  (trained Random Forest model)

Run:
    python model_comparison.py

Requires (in addition to requirements.txt):
    pip install joblib
"""

import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# Configuration
#
# These MUST match train_model.py, otherwise the saved ANN
# would be evaluated on rows it was trained on.
# ============================================================

DATA_PATH = "data/creditcard.csv"

ARTIFACT_DIR = "artifacts"

ANN_MODEL_PATH = os.path.join(ARTIFACT_DIR, "fraud_model.keras")

SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.npz")

SEED = 42

TEST_SIZE = 0.20

DECISION_THRESHOLD = 0.50

# Random Forest size. 100 trees is enough for a stable comparison
# and keeps runtime reasonable on a laptop CPU.
RF_N_ESTIMATORS = 100

CSV_PATH = os.path.join(ARTIFACT_DIR, "model_comparison.csv")
JSON_PATH = os.path.join(ARTIFACT_DIR, "model_comparison.json")
PLOT_PATH = os.path.join(ARTIFACT_DIR, "model_comparison.png")
CM_PLOT_PATH = os.path.join(ARTIFACT_DIR, "model_comparison_confusion.png")

# Saved model files. These let later phases (SHAP explainability in
# Phase 3, threshold analysis in Phase 7) load the best tree model
# without retraining it from scratch every time.
XGBOOST_MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgboost_model.json")
RANDOM_FOREST_MODEL_PATH = os.path.join(
    ARTIFACT_DIR, "random_forest_model.joblib"
)


# ============================================================
# Data loading and validation
# ============================================================

def load_dataset():
    """
    Load the real Kaggle Credit Card Fraud Detection dataset
    and check that it has the columns we expect.
    """

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\nDataset not found.\n"
            f"Expected: {os.path.abspath(DATA_PATH)}\n\n"
            f"Place the Kaggle creditcard.csv file inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount", "Class"]
    )

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )

    if df.isna().any().any():
        raise ValueError(
            f"Dataset contains {int(df.isna().sum().sum())} missing values."
        )

    print(f"[compare] Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"[compare] Fraud rows: {int(df['Class'].sum()):,}")
    print(f"[compare] Fraud rate: {df['Class'].mean() * 100:.4f}%")

    return df


def build_split(df):
    """
    Reproduce the exact train/test split used by train_model.py.

    stratify=y keeps the same fraud percentage in train and test,
    which matters a lot when only ~0.17% of rows are fraud.
    """

    feature_columns = [c for c in df.columns if c != "Class"]

    X = df[feature_columns].values.astype(np.float32)
    y = df["Class"].astype(np.int32).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=SEED,
    )

    print(f"\n[compare] Train rows: {len(X_train):,}")
    print(f"[compare] Test rows:  {len(X_test):,}")
    print(f"[compare] Fraud in test set: {int(y_test.sum()):,}")

    return feature_columns, X_train, X_test, y_train, y_test


def build_scaler(X_train, feature_columns):
    """
    Fit a StandardScaler on the training rows only (never on the test
    rows -- that would leak test information into training).

    Then verify it reproduces artifacts/scaler.npz. If it does not,
    the saved ANN was trained on a different split and comparing it
    here would be misleading, so we say so clearly.
    """

    scaler = StandardScaler().fit(X_train)

    if not os.path.exists(SCALER_PATH):
        print(
            "\n[compare] WARNING: artifacts/scaler.npz not found. "
            "Cannot verify that this split matches train_model.py."
        )
        return scaler, False

    saved = np.load(SCALER_PATH, allow_pickle=True)

    mean_matches = np.allclose(
        scaler.mean_, saved["mean"], rtol=1e-4, atol=1e-6
    )

    scale_matches = np.allclose(
        scaler.scale_, saved["scale"], rtol=1e-4, atol=1e-6
    )

    columns_match = (
        [str(c) for c in saved["columns"]] == list(feature_columns)
    )

    matches = mean_matches and scale_matches and columns_match

    if matches:
        print(
            "\n[compare] Split verified: this scaler reproduces "
            "artifacts/scaler.npz, so the saved ANN did not see these "
            "test rows during training."
        )
    else:
        print(
            "\n[compare] WARNING: this scaler does NOT match "
            "artifacts/scaler.npz.\n"
            "           The saved ANN may have been trained on a "
            "different split.\n"
            "           Re-run  python train_model.py  before trusting "
            "the ANN row of this comparison."
        )

    return scaler, matches


# ============================================================
# Metric calculation
# ============================================================

def compute_metrics(name, y_true, y_prob, threshold, train_seconds):
    """
    Turn predicted probabilities into the full metric set.

    y_prob is a probability of fraud in [0, 1].
    y_pred applies the decision threshold (default 0.50).

    Why each metric is here:

    accuracy   -- included only for completeness. On a dataset that is
                  99.83% legitimate, a model predicting "legitimate"
                  every single time scores about 99.83%. Accuracy alone
                  tells you almost nothing here.

    precision  -- of the transactions we flagged, how many were really
                  fraud. Low precision means analysts drown in false
                  alarms.

    recall     -- of the real frauds, how many did we catch. Low recall
                  means fraud goes through undetected.

    f1         -- harmonic mean of precision and recall. Punishes a model
                  that is good at one and terrible at the other.

    roc_auc    -- ranking quality across all thresholds. Can look
                  flattering under heavy imbalance because the huge
                  number of true negatives dominates.

    pr_auc     -- average precision. Focuses on the minority (fraud)
                  class, so it is the more honest ranking metric here.

    mcc        -- Matthews Correlation Coefficient. Uses all four cells
                  of the confusion matrix (TP, TN, FP, FN) and stays
                  meaningful under extreme imbalance. +1 perfect,
                  0 = no better than random, -1 = perfectly wrong.
    """

    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "alerts_raised": int(tp + fp),
        "confusion_matrix": cm.tolist(),
        "threshold": float(threshold),
        "train_seconds": round(float(train_seconds), 2),
    }


# ============================================================
# Individual models
# ============================================================

def run_logistic_regression(X_train_s, y_train, X_test_s):
    """
    Logistic Regression.

    WHAT: a linear model. It learns one weight per feature and squashes
          the weighted sum through a sigmoid to get a probability.
    WHY:  it is the interpretability baseline. If a far more complex
          model cannot beat it, the complexity is not paying for itself.
    NOTE: class_weight="balanced" makes each fraud row count roughly
          580x more than a legitimate row during training, because fraud
          is roughly 580x rarer. Without it the model would essentially
          ignore fraud.
    """

    print("\n[compare] Training Logistic Regression ...")

    start = time.time()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train_s, y_train)

    elapsed = time.time() - start

    y_prob = model.predict_proba(X_test_s)[:, 1]

    return y_prob, elapsed


def run_decision_tree(X_train_s, y_train, X_test_s):
    """
    Decision Tree.

    WHAT: a flowchart of if/else splits on feature values.
    WHY:  simplest non-linear baseline, and it shows what happens when a
          single unpruned tree memorises the training data.
    NOTE: max_depth is limited so the tree does not grow until every leaf
          is pure, which on 227k rows would overfit badly.
    """

    print("[compare] Training Decision Tree ...")

    start = time.time()

    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=SEED,
    )

    model.fit(X_train_s, y_train)

    elapsed = time.time() - start

    y_prob = model.predict_proba(X_test_s)[:, 1]

    return y_prob, elapsed


def run_random_forest(X_train_s, y_train, X_test_s):
    """
    Random Forest.

    WHAT: many decision trees, each trained on a random sample of rows
          and a random subset of features. Their votes are averaged.
    WHY:  averaging many slightly different trees cancels out the
          overfitting a single tree suffers from.
    NOTE: n_jobs=-1 uses all CPU cores. This is the slowest model here.
    """

    print(f"[compare] Training Random Forest ({RF_N_ESTIMATORS} trees) ...")

    start = time.time()

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    model.fit(X_train_s, y_train)

    elapsed = time.time() - start

    y_prob = model.predict_proba(X_test_s)[:, 1]

    # Save the trained model so later phases (threshold analysis,
    # calibration) can reload it without retraining. joblib is
    # scikit-learn's recommended format for this.
    try:
        import joblib
        joblib.dump(model, RANDOM_FOREST_MODEL_PATH)
        print(f"[compare]   Saved -> {RANDOM_FOREST_MODEL_PATH}")
    except ImportError:
        print(
            "[compare]   joblib not installed, Random Forest model was "
            "not saved to disk (metrics were still computed normally). "
            "Install with: pip install joblib"
        )

    return y_prob, elapsed


def run_xgboost(X_train_s, y_train, X_test_s):
    """
    XGBoost.

    WHAT: gradient-boosted trees. Trees are built one after another, and
          each new tree is trained to correct the mistakes the previous
          ones made.
    WHY:  this is the model family the research paper's transaction
          branch actually specifies (Hashemi et al. [4]), and it is
          generally the strongest choice for structured/tabular data.
    NOTE: XGBoost has no class_weight parameter. The equivalent is
          scale_pos_weight = (number of legitimate rows) / (number of
          fraud rows), which scales the gradient of the positive class.

    Returns (None, None) if xgboost is not installed, so the rest of the
    comparison still runs instead of crashing.
    """

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print(
            "\n[compare] xgboost is not installed. Skipping XGBoost.\n"
            "          Install it with:  pip install xgboost\n"
            "          then re-run this script."
        )
        return None, None

    print("[compare] Training XGBoost ...")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / max(positive_count, 1)

    print(f"[compare]   scale_pos_weight = {scale_pos_weight:.2f}")

    start = time.time()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(X_train_s, y_train)

    elapsed = time.time() - start

    y_prob = model.predict_proba(X_test_s)[:, 1]

    # XGBoost has its own native save format (a portable JSON file that
    # does not depend on the exact library version used to train it).
    # This is the model Phase 3 (SHAP) will load and explain.
    model.save_model(XGBOOST_MODEL_PATH)
    print(f"[compare]   Saved -> {XGBOOST_MODEL_PATH}")

    return y_prob, elapsed


def run_existing_ann(X_test_s):
    """
    Load the ANN already trained by train_model.py and score the test set.

    The model is NOT retrained here. Retraining would change the existing
    artifact and make this comparison inconsistent with metrics.json.

    Returns (None, None) if the model file is missing.
    """

    if not os.path.exists(ANN_MODEL_PATH):
        print(
            f"\n[compare] {ANN_MODEL_PATH} not found. Skipping ANN.\n"
            f"          Run:  python train_model.py"
        )
        return None, None

    print("[compare] Loading existing ANN (not retraining) ...")

    try:
        from tensorflow import keras
    except ImportError:
        print("[compare] TensorFlow not available. Skipping ANN.")
        return None, None

    model = keras.models.load_model(ANN_MODEL_PATH)

    y_prob = model.predict(X_test_s, verbose=0).ravel()

    # train_seconds is 0.0 because the training happened in a previous
    # run of train_model.py, not here. It is not a measurement.
    return y_prob, 0.0


# ============================================================
# Plots
# ============================================================

def save_comparison_plot(results, output_path):
    """
    Grouped bar chart of the metrics that matter under imbalance.

    Accuracy is deliberately left OUT of this chart: every model scores
    in the high 0.9s and the bars would all look identical, which is
    exactly the misleading impression this project should avoid.
    """

    metric_keys = ["precision", "recall", "f1", "pr_auc", "roc_auc", "mcc"]

    labels = [r["model"] for r in results]

    x = np.arange(len(metric_keys))

    width = 0.8 / max(len(results), 1)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, result in enumerate(results):

        values = [result[k] for k in metric_keys]

        offset = (i - (len(results) - 1) / 2) * width

        bars = ax.bar(x + offset, values, width, label=result["model"])

        ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in metric_keys])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(
        f"Model Comparison on identical test set "
        f"(threshold = {DECISION_THRESHOLD})"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)

    print(f"[compare] Saved -> {output_path}")


def save_confusion_grid(results, output_path):
    """
    One confusion matrix per model, side by side.

    Reading a confusion matrix here:

        top-left      TN  legitimate, correctly allowed
        top-right     FP  legitimate, wrongly flagged  -> analyst workload
        bottom-left   FN  fraud, missed                -> financial loss
        bottom-right  TP  fraud, correctly caught
    """

    n = len(results)

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))

    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results):

        cm = np.array(result["confusion_matrix"])

        ax.imshow(cm, cmap="Blues")

        ax.set_title(result["model"], fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Legit", "Fraud"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Legit", "Fraud"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)

        limit = cm.max() / 2

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cm[i, j] > limit else "black",
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)

    print(f"[compare] Saved -> {output_path}")


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 68)
    print("PHASE 1 - MODEL COMPARISON")
    print("=" * 68)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    np.random.seed(SEED)

    # --- 1. Data -------------------------------------------------

    df = load_dataset()

    feature_columns, X_train, X_test, y_train, y_test = build_split(df)

    scaler, split_verified = build_scaler(X_train, feature_columns)

    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- 2. Train and score each model ---------------------------

    results = []

    runners = [
        ("Logistic Regression",
         lambda: run_logistic_regression(X_train_s, y_train, X_test_s)),

        ("Decision Tree",
         lambda: run_decision_tree(X_train_s, y_train, X_test_s)),

        ("Random Forest",
         lambda: run_random_forest(X_train_s, y_train, X_test_s)),

        ("XGBoost",
         lambda: run_xgboost(X_train_s, y_train, X_test_s)),

        ("ANN (existing)",
         lambda: run_existing_ann(X_test_s)),
    ]

    for name, runner in runners:

        y_prob, elapsed = runner()

        if y_prob is None:
            continue

        results.append(
            compute_metrics(
                name, y_test, y_prob, DECISION_THRESHOLD, elapsed
            )
        )

    if not results:
        raise SystemExit(
            "No models produced results. Check the messages above."
        )

    # --- 3. Save results ----------------------------------------

    table = pd.DataFrame(results)[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "mcc",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "alerts_raised",
            "train_seconds",
        ]
    ]

    table.to_csv(CSV_PATH, index=False)

    payload = {
        "phase": "1 - model comparison",
        "dataset": {
            "path": DATA_PATH,
            "total_rows": int(len(df)),
            "fraud_rows": int(df["Class"].sum()),
            "fraud_rate_pct": float(df["Class"].mean() * 100),
        },
        "protocol": {
            "split": "stratified random",
            "test_size": TEST_SIZE,
            "random_state": SEED,
            "decision_threshold": DECISION_THRESHOLD,
            "scaling": "StandardScaler fitted on training rows only",
            "imbalance_handling": (
                "class_weight='balanced' for sklearn models, "
                "scale_pos_weight for XGBoost, "
                "class weights for the pre-trained ANN. No SMOTE."
            ),
            "split_matches_saved_scaler": bool(split_verified),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_fraud_rows": int(y_test.sum()),
        },
        "note": (
            "A random split is used here to stay consistent with "
            "train_model.py. The research paper recommends a temporal "
            "split; that is a documented limitation, not an oversight."
        ),
        "results": results,
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[compare] Saved -> {CSV_PATH}")
    print(f"[compare] Saved -> {JSON_PATH}")

    if os.path.exists(XGBOOST_MODEL_PATH):
        print(
            f"[compare] Saved -> {XGBOOST_MODEL_PATH} "
            f"(reusable for Phase 3 SHAP explainability)"
        )

    if os.path.exists(RANDOM_FOREST_MODEL_PATH):
        print(f"[compare] Saved -> {RANDOM_FOREST_MODEL_PATH}")

    save_comparison_plot(results, PLOT_PATH)
    save_confusion_grid(results, CM_PLOT_PATH)

    # --- 4. Print the table --------------------------------------

    print("\n" + "=" * 68)
    print("RESULTS ON THE TEST SET")
    print("=" * 68)

    display = table.copy()

    for column in [
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"
    ]:
        display[column] = display[column].map(lambda v: f"{v:.4f}")

    print(display.to_string(index=False))

    # --- 5. Factual trade-off commentary -------------------------
    #
    # Everything below is derived from the numbers just measured.
    # No model is declared "best" -- the right choice depends on the
    # cost of a missed fraud versus the cost of a false alarm, which
    # is decided in Phase 7 (threshold_analysis.py).

    print("\n" + "=" * 68)
    print("HOW TO READ THIS")
    print("=" * 68)

    fraud_in_test = int(y_test.sum())

    print(
        f"\nThe test set holds {len(y_test):,} transactions, of which "
        f"{fraud_in_test:,} are fraud ({fraud_in_test / len(y_test) * 100:.3f}%)."
    )

    print(
        f"A model that predicted LEGITIMATE for every single row would "
        f"score {(1 - fraud_in_test / len(y_test)) * 100:.2f}% accuracy "
        f"and catch zero fraud. That is why accuracy is reported but not "
        f"used to rank anything."
    )

    best_recall = max(results, key=lambda r: r["recall"])
    best_precision = max(results, key=lambda r: r["precision"])
    best_pr_auc = max(results, key=lambda r: r["pr_auc"])
    best_mcc = max(results, key=lambda r: r["mcc"])

    print(
        f"\nHighest recall   : {best_recall['model']} "
        f"({best_recall['recall']:.4f}) -- caught "
        f"{best_recall['true_positive']} of {fraud_in_test} frauds, "
        f"while raising {best_recall['false_positive']:,} false alarms."
    )

    print(
        f"Highest precision: {best_precision['model']} "
        f"({best_precision['precision']:.4f}) -- of its "
        f"{best_precision['alerts_raised']:,} alerts, "
        f"{best_precision['true_positive']} were real fraud, but it missed "
        f"{best_precision['false_negative']}."
    )

    print(
        f"Highest PR-AUC   : {best_pr_auc['model']} "
        f"({best_pr_auc['pr_auc']:.4f}) -- best ranking of fraud above "
        f"legitimate across all thresholds."
    )

    print(
        f"Highest MCC      : {best_mcc['model']} "
        f"({best_mcc['mcc']:.4f}) -- best balance across all four "
        f"confusion-matrix cells."
    )

    print(
        "\nThese are four different questions, so they may well have four "
        "different answers. Raising recall almost always costs precision "
        "(more alerts, more of them wrong) and raising precision almost "
        "always costs recall (fewer alerts, more fraud missed). "
        "Phase 7 turns that trade-off into a cost function."
    )

    if not split_verified:
        print(
            "\nWARNING: the split could not be verified against "
            "artifacts/scaler.npz. Treat the ANN row with caution until "
            "train_model.py has been re-run."
        )

    print("\nDone.\n")


if __name__ == "__main__":
    main()