"""
train_model.py
--------------
Train the fraud detection ANN using the REAL Kaggle Credit Card Fraud
Detection dataset.

Dataset:
    data/creditcard.csv

Expected columns:
    Time, V1 ... V28, Amount, Class

Class:
    0 = legitimate transaction
    1 = fraudulent transaction

Pipeline:
    Real dataset
        ↓
    Validation
        ↓
    Train/test split
        ↓
    StandardScaler
        ↓
    Class-weight calculation
        ↓
    ANN training
        ↓
    Evaluation
        ↓
    Save model + scaler + metrics + plots
"""

import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from keras import layers

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    average_precision_score,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/creditcard.csv"

ARTIFACT_DIR = "artifacts"

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "fraud_model.keras"
)

SCALER_PATH = os.path.join(
    ARTIFACT_DIR,
    "scaler.npz"
)

METRICS_PATH = os.path.join(
    ARTIFACT_DIR,
    "metrics.json"
)

SEED = 42

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.20

EPOCHS = 40

BATCH_SIZE = 256

PREDICTION_THRESHOLD = 0.50


# ============================================================
# Setup
# ============================================================

os.makedirs(ARTIFACT_DIR, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Load REAL DATA
# ============================================================

def load_data():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"\nReal dataset not found.\n"
            f"Expected file:\n{os.path.abspath(DATA_PATH)}\n\n"
            f"Download the Kaggle Credit Card Fraud Detection dataset "
            f"and place creditcard.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    print(
        f"[train] Loaded REAL dataset: "
        f"{DATA_PATH}"
    )

    print(
        f"[train] Rows: {len(df):,}"
    )

    print(
        f"[train] Columns: {len(df.columns)}"
    )

    return df


# ============================================================
# Validate dataset
# ============================================================

def validate_dataset(df):

    required_columns = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
        "Class",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Dataset is missing required columns:\n"
            + ", ".join(missing_columns)
        )

    if df["Class"].isna().any():

        raise ValueError(
            "The Class column contains missing values."
        )

    unique_classes = set(
        df["Class"].unique()
    )

    if not unique_classes.issubset({0, 1}):

        raise ValueError(
            f"Class column must contain only 0 and 1. "
            f"Found: {unique_classes}"
        )

    if df.isna().any().any():

        missing_count = int(
            df.isna().sum().sum()
        )

        raise ValueError(
            f"Dataset contains {missing_count} missing values."
        )

    print(
        "[train] Dataset validation passed."
    )

    legitimate_count = int(
        (df["Class"] == 0).sum()
    )

    fraud_count = int(
        (df["Class"] == 1).sum()
    )

    fraud_rate = (
        fraud_count / len(df) * 100
    )

    print(
        f"[train] Legitimate: "
        f"{legitimate_count:,}"
    )

    print(
        f"[train] Fraud: "
        f"{fraud_count:,}"
    )

    print(
        f"[train] Fraud rate: "
        f"{fraud_rate:.4f}%"
    )


# ============================================================
# Build ANN
# ============================================================

def build_model(input_dim):

    model = keras.Sequential(
        [

            keras.Input(
                shape=(input_dim,)
            ),

            layers.Dense(
                64,
                activation="relu"
            ),

            layers.Dropout(
                0.30
            ),

            layers.Dense(
                32,
                activation="relu"
            ),

            layers.Dropout(
                0.30
            ),

            layers.Dense(
                16,
                activation="relu"
            ),

            layers.Dense(
                1,
                activation="sigmoid"
            ),
        ]
    )

    model.compile(

        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),

        loss="binary_crossentropy",

        metrics=[
            keras.metrics.AUC(
                name="auc"
            ),

            keras.metrics.Precision(
                name="precision"
            ),

            keras.metrics.Recall(
                name="recall"
            ),

            "accuracy",
        ],
    )

    return model


# ============================================================
# Plot confusion matrix
# ============================================================

def save_confusion_matrix(
    cm,
    output_path
):

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    ax.imshow(
        cm,
        cmap="Blues"
    )

    ax.set_title(
        "Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels(
        ["Legit", "Fraud"]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_yticklabels(
        ["Legit", "Fraud"]
    )

    threshold = cm.max() / 2

    for i in range(2):

        for j in range(2):

            value = cm[i, j]

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=130
    )

    plt.close(fig)


# ============================================================
# Plot ROC curve
# ============================================================

def save_roc_curve(
    fpr,
    tpr,
    roc_auc,
    output_path
):

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    ax.plot(
        fpr,
        tpr,
        label=(
            f"ANN "
            f"(AUC = {roc_auc:.3f})"
        ),
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray",
        label="Random",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=130
    )

    plt.close(fig)


# ============================================================
# Plot training curves
# ============================================================

def save_training_curves(
    history,
    output_path
):

    history_dict = history.history

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9, 4)
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    axes[0].plot(
        history_dict["loss"],
        label="train"
    )

    axes[0].plot(
        history_dict["val_loss"],
        label="validation"
    )

    axes[0].set_title(
        "Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].legend()


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    axes[1].plot(
        history_dict["accuracy"],
        label="train"
    )

    axes[1].plot(
        history_dict["val_accuracy"],
        label="validation"
    )

    axes[1].set_title(
        "Accuracy"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Accuracy"
    )

    axes[1].legend()


    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=130
    )

    plt.close(fig)


# ============================================================
# Main training pipeline
# ============================================================

def main():

    print(
        "\n"
        + "=" * 65
    )

    print(
        "REAL CREDIT CARD FRAUD DETECTION - ANN TRAINING"
    )

    print(
        "=" * 65
    )


    # --------------------------------------------------------
    # 1. Load REAL dataset
    # --------------------------------------------------------

    df = load_data()

    validate_dataset(df)


    # --------------------------------------------------------
    # 2. Separate features and target
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column != "Class"
    ]

    X = df[
        feature_columns
    ].values.astype(
        np.float32
    )

    y = df[
        "Class"
    ].astype(
        np.int32
    ).values


    print(
        f"\n[train] Feature count: "
        f"{len(feature_columns)}"
    )


    # --------------------------------------------------------
    # 3. Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        stratify=y,

        random_state=SEED,
    )


    print(
        f"[train] Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"[train] Testing rows: "
        f"{len(X_test):,}"
    )


    # --------------------------------------------------------
    # 4. Scale features
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(
        X_train
    )

    X_test_s = scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    np.savez(

        SCALER_PATH,

        mean=scaler.mean_,

        scale=scaler.scale_,

        columns=np.array(
            feature_columns
        ),
    )


    print(
        f"[train] Saved scaler -> "
        f"{SCALER_PATH}"
    )


    # --------------------------------------------------------
    # 5. Calculate class weights
    # --------------------------------------------------------

    classes = np.unique(
        y_train
    )

    weights = compute_class_weight(

        class_weight="balanced",

        classes=classes,

        y=y_train,
    )

    class_weight = {

        int(c): float(w)

        for c, w in zip(
            classes,
            weights
        )

    }


    print(
        f"[train] Class weights: "
        f"{class_weight}"
    )


    # --------------------------------------------------------
    # 6. Build ANN
    # --------------------------------------------------------

    model = build_model(
        X_train_s.shape[1]
    )


    print(
        "\n[train] Model architecture:"
    )

    model.summary()


    # --------------------------------------------------------
    # 7. Early stopping
    # --------------------------------------------------------

    early_stop = keras.callbacks.EarlyStopping(

        monitor="val_auc",

        mode="max",

        patience=6,

        restore_best_weights=True,

        verbose=1,
    )


    # --------------------------------------------------------
    # 8. Train
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 65
    )

    print(
        "STARTING TRAINING"
    )

    print(
        "=" * 65
    )


    history = model.fit(

        X_train_s,

        y_train,

        validation_split=VALIDATION_SIZE,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weight,

        callbacks=[
            early_stop
        ],

        verbose=2,
    )


    # --------------------------------------------------------
    # 9. Best epoch
    # --------------------------------------------------------

    val_auc_history = history.history.get(
        "val_auc",
        []
    )

    if val_auc_history:

        best_epoch_index = int(
            np.argmax(
                val_auc_history
            )
        )

        best_epoch = (
            best_epoch_index + 1
        )

        best_val_auc = float(
            val_auc_history[
                best_epoch_index
            ]
        )

    else:

        best_epoch = len(
            history.history[
                "loss"
            ]
        )

        best_val_auc = None


    print(
        f"\n[train] Best epoch: "
        f"{best_epoch}"
    )

    if best_val_auc is not None:

        print(
            f"[train] Best validation AUC: "
            f"{best_val_auc:.6f}"
        )


    # --------------------------------------------------------
    # 10. Predictions
    # --------------------------------------------------------

    y_prob = model.predict(
        X_test_s,
        verbose=0
    ).ravel()

    y_pred = (
        y_prob >= PREDICTION_THRESHOLD
    ).astype(
        np.int32
    )


    # --------------------------------------------------------
    # 11. Evaluation metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # 12. Additional statistics
    # --------------------------------------------------------

    true_negative = int(
        cm[0, 0]
    )

    false_positive = int(
        cm[0, 1]
    )

    false_negative = int(
        cm[1, 0]
    )

    true_positive = int(
        cm[1, 1]
    )

    test_fraud_count = int(
        (y_test == 1).sum()
    )

    test_legitimate_count = int(
        (y_test == 0).sum()
    )

    predicted_fraud_count = int(
        (y_pred == 1).sum()
    )

    predicted_legitimate_count = int(
        (y_pred == 0).sum()
    )

    test_fraud_rate = (
        test_fraud_count
        / len(y_test)
        * 100
    )


    # --------------------------------------------------------
    # 13. Create metrics JSON
    #
    # IMPORTANT:
    # Metrics are stored both:
    #
    #   metrics["performance"]["accuracy"]
    #
    # AND
    #
    #   metrics["accuracy"]
    #
    # This makes the file compatible with both the
    # current Streamlit app and older code.
    # --------------------------------------------------------

    performance = {

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "roc_auc": float(
            roc_auc
        ),

        "pr_auc": float(
            pr_auc
        ),

        "true_negative": true_negative,

        "false_positive": false_positive,

        "false_negative": false_negative,

        "true_positive": true_positive,

        "confusion_matrix": cm.tolist(),
    }


    metrics = {

        # ----------------------------------------------------
        # Compatibility fields
        # ----------------------------------------------------

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "roc_auc": float(
            roc_auc
        ),

        "pr_auc": float(
            pr_auc
        ),

        "n_test": int(
            len(y_test)
        ),

        "test_rows": int(
            len(y_test)
        ),

        "train_rows": int(
            len(X_train)
        ),

        "fraud_rate_pct": float(
            df["Class"].mean() * 100
        ),

        # ----------------------------------------------------
        # Dataset information
        # ----------------------------------------------------

        "dataset": {

            "name":
                "Kaggle Credit Card Fraud Detection",

            "path":
                DATA_PATH,

            "source_type":
                "REAL_DATASET",

            "total_rows":
                int(len(df)),

            "total_columns":
                int(len(df.columns)),

            "feature_count":
                int(len(feature_columns)),

            "fraud_transactions":
                int(
                    (df["Class"] == 1).sum()
                ),

            "legitimate_transactions":
                int(
                    (df["Class"] == 0).sum()
                ),

            "fraud_rate_pct":
                float(
                    df["Class"].mean() * 100
                ),
        },

        # ----------------------------------------------------
        # Training information
        # ----------------------------------------------------

        "training": {

            "train_rows":
                int(len(X_train)),

            "test_rows":
                int(len(X_test)),

            "validation_split":
                float(VALIDATION_SIZE),

            "feature_count":
                int(X_train_s.shape[1]),

            "epochs_requested":
                int(EPOCHS),

            "epochs_completed":
                int(
                    len(
                        history.history[
                            "loss"
                        ]
                    )
                ),

            "best_epoch":
                int(best_epoch),

            "batch_size":
                int(BATCH_SIZE),

            "random_seed":
                int(SEED),

            "prediction_threshold":
                float(
                    PREDICTION_THRESHOLD
                ),

            "class_weight":
                class_weight,
        },

        # ----------------------------------------------------
        # Model information
        # ----------------------------------------------------

        "model": {

            "type":
                "Artificial Neural Network",

            "architecture":
                "64-32-16-1",

            "input_features":
                feature_columns,

            "optimizer":
                "Adam",

            "learning_rate":
                0.001,

            "loss":
                "binary_crossentropy",

            "prediction_threshold":
                float(
                    PREDICTION_THRESHOLD
                ),
        },

        # ----------------------------------------------------
        # Performance information
        # ----------------------------------------------------

        "performance":
            performance,

        # ----------------------------------------------------
        # Test set information
        # ----------------------------------------------------

        "test_set": {

            "rows":
                int(len(y_test)),

            "legitimate_transactions":
                test_legitimate_count,

            "fraud_transactions":
                test_fraud_count,

            "fraud_rate_pct":
                float(test_fraud_rate),

            "predicted_legitimate":
                predicted_legitimate_count,

            "predicted_fraud":
                predicted_fraud_count,
        },

        # ----------------------------------------------------
        # Validation training result
        # ----------------------------------------------------

        "validation": {

            "best_epoch":
                int(best_epoch),

            "best_val_auc":
                (
                    float(best_val_auc)
                    if best_val_auc is not None
                    else None
                ),

            "final_val_accuracy":
                float(
                    history.history[
                        "val_accuracy"
                    ][-1]
                ),

            "final_val_auc":
                float(
                    history.history[
                        "val_auc"
                    ][-1]
                ),

            "final_val_loss":
                float(
                    history.history[
                        "val_loss"
                    ][-1]
                ),
        },
    }


    # --------------------------------------------------------
    # 14. Print final metrics
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 65
    )

    print(
        "REAL DATA TEST METRICS"
    )

    print(
        "=" * 65
    )

    print(
        f"Accuracy  : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{pr_auc:.4f}"
    )

    print(
        f"\nTest rows: "
        f"{len(y_test):,}"
    )

    print(
        f"Actual fraud: "
        f"{test_fraud_count:,}"
    )

    print(
        f"Predicted fraud: "
        f"{predicted_fraud_count:,}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )


    # --------------------------------------------------------
    # 15. Save confusion matrix plot
    # --------------------------------------------------------

    save_confusion_matrix(

        cm,

        os.path.join(
            ARTIFACT_DIR,
            "confusion_matrix.png"
        ),
    )


    # --------------------------------------------------------
    # 16. Save ROC curve
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    save_roc_curve(

        fpr,

        tpr,

        roc_auc,

        os.path.join(
            ARTIFACT_DIR,
            "roc_curve.png"
        ),
    )


    # --------------------------------------------------------
    # 17. Save training curves
    # --------------------------------------------------------

    save_training_curves(

        history,

        os.path.join(
            ARTIFACT_DIR,
            "training_curves.png"
        ),
    )


    # --------------------------------------------------------
    # 18. Save training history
    # --------------------------------------------------------

    history_json = {

        key: [
            float(value)
            for value in values
        ]

        for key, values
        in history.history.items()
    }

    metrics[
        "training_history"
    ] = history_json


    # --------------------------------------------------------
    # 19. Save model
    # --------------------------------------------------------

    model.save(

        MODEL_PATH,

        include_optimizer=False
    )


    # --------------------------------------------------------
    # 20. Save metrics JSON
    # --------------------------------------------------------

    with open(
        METRICS_PATH,
        "w"
    ) as f:

        json.dump(

            metrics,

            f,

            indent=2
        )


    # --------------------------------------------------------
    # 21. Final output
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 65
    )

    print(
        "ARTIFACTS SAVED"
    )

    print(
        "=" * 65
    )

    print(
        f"Model   -> {MODEL_PATH}"
    )

    print(
        f"Scaler  -> {SCALER_PATH}"
    )

    print(
        f"Metrics -> {METRICS_PATH}"
    )

    print(
        "Plots   -> artifacts/"
    )

    print(
        "\nTraining completed successfully "
        "using the REAL Kaggle dataset."
    )

    print(
        "=" * 65
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()