"""
train_model.py
----------------
Trains the core ANN fraud-detection model.

Pipeline: dataset -> preprocessing -> scaling -> class-imbalance handling
(class weights) -> train/test split -> ANN -> fraud probability -> metrics.

Uses data/creditcard.csv if present (real Kaggle Credit Card Fraud dataset).
Otherwise auto-generates a clearly-labelled synthetic DEMO dataset with the
same shape/imbalance so the pipeline still runs end-to-end. All metrics
below are computed live on whatever dataset is loaded -- nothing here is a
made-up number.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import keras
from keras import layers


DATA_PATH = "data/creditcard.csv"
ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "fraud_model.keras")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.npz")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics.json")
SEED = 42

os.makedirs(ARTIFACT_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)


def load_data():
    is_demo = not os.path.exists(DATA_PATH)
    if is_demo:
        from generate_demo_data import generate
        print("[train] data/creditcard.csv not found -> generating synthetic DEMO dataset")
        df = generate(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"[train] loaded real dataset from {DATA_PATH} ({len(df)} rows)")
    return df, is_demo


def build_model(input_dim):
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model


def main():
    df, is_demo = load_data()

    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    np.savez(SCALER_PATH, mean=scaler.mean_, scale=scaler.scale_,
              columns=np.array(df.drop(columns=["Class"]).columns))

    # Class imbalance handling: class weights (fraud is a small minority).
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"[train] class weights: {class_weight}")

    model = build_model(X_train_s.shape[1])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=6, restore_best_weights=True
    )

    history = model.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=40,
        batch_size=256,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )

    # ---- Evaluation ----
    y_prob = model.predict(X_test_s, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "is_demo_data": bool(is_demo),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "fraud_rate_pct": float(y.mean() * 100),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    print("[train] test metrics:", json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # ---- Plots ----
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Legit", "Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ANN (AUC = {metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "roc_curve.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "training_curves.png"), dpi=130)
    plt.close(fig)

    model.save(MODEL_PATH, include_optimizer=False)
    metrics["confusion_matrix"] = cm.tolist()
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] saved model -> {MODEL_PATH}")
    print(f"[train] saved metrics -> {METRICS_PATH}")
    print("[train] done.")


if __name__ == "__main__":
    main()
