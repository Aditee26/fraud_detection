"""
Self-supervised transfer-learning baseline for credit-card fraud detection.

Stage 1: train an autoencoder on transaction features without using labels.
Stage 2: transfer the pretrained encoder into a fraud classifier and fine-tune
it using the training labels. This is transfer from an unsupervised feature-
reconstruction task, not transfer from an external bank dataset.

Run: python train_transfer.py
"""
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix)

DATA_PATH = "data/creditcard.csv"
ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "transfer_model.keras")
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "pretrained_encoder.keras")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "transfer_metrics.json")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "transfer_scaler.npz")
SEED = 42
EPOCHS_PRETRAIN = 25
EPOCHS_FINETUNE = 35
BATCH_SIZE = 256

def score_metrics(y, p):
    pred = (p >= 0.5).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c != "Class"]
    if "Class" not in df or len(feature_cols) != 30:
        raise ValueError("Expected the Kaggle dataset with 30 features and Class target.")
    X = df[feature_cols].to_numpy(np.float32)
    y = df["Class"].to_numpy(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    np.savez(SCALER_PATH, mean=scaler.mean_, scale=scaler.scale_, columns=np.array(feature_cols))

    # Pretraining is label-free: reconstruct the input feature vector.
    inputs = keras.Input(shape=(X_train.shape[1],), name="transaction_input")
    encoded = layers.Dense(64, activation="relu", name="encoder_dense_64")(inputs)
    encoded = layers.Dense(16, activation="relu", name="encoder_embedding")(encoded)
    decoded = layers.Dense(64, activation="relu")(encoded)
    decoded = layers.Dense(X_train.shape[1], activation="linear")(decoded)
    autoencoder = keras.Model(inputs, decoded, name="transaction_autoencoder")
    encoder = keras.Model(inputs, encoded, name="pretrained_encoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    pretrain_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                                   restore_best_weights=True, verbose=1)
    print("[Transfer] Stage 1/2: unsupervised autoencoder pretraining")
    pretrain_history = autoencoder.fit(
        X_train, X_train, validation_split=0.2,
        epochs=EPOCHS_PRETRAIN, batch_size=BATCH_SIZE,
        callbacks=[pretrain_stop], verbose=2
    )
    encoder.save(ENCODER_PATH)

    # Transfer the learned encoder weights to a supervised fraud classifier.
    clf_input = keras.Input(shape=(X_train.shape[1],), name="transaction_features")
    x = encoder(clf_input)
    x = layers.Dense(32, activation="relu", name="transfer_classifier_dense")(x)
    x = layers.Dropout(0.25)(x)
    output = layers.Dense(1, activation="sigmoid", name="fraud_probability")(x)
    model = keras.Model(clf_input, output, name="transfer_learning_fraud_classifier")
    model.compile(optimizer=keras.optimizers.Adam(5e-4),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"), "accuracy"])
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    fine_stop = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                              restore_best_weights=True, verbose=1)
    print("[Transfer] Stage 2/2: transfer encoder and fine-tune fraud classifier")
    model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS_FINETUNE,
              batch_size=BATCH_SIZE, class_weight=class_weight,
              callbacks=[fine_stop], verbose=2)
    probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    metrics = score_metrics(y_test, probabilities)
    metrics.update({
        "model": "Transfer Learning (autoencoder-pretrained encoder)",
        "pretraining_task": "Unsupervised reconstruction of transaction features",
        "dataset_rows": int(len(df)), "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)), "test_fraud_rows": int(y_test.sum()),
        "pretraining_epochs_completed": int(len(pretrain_history.history["loss"])),
        "threshold": 0.5, "seed": SEED,
        "note": "Encoder pretrained on unlabeled features from the training split, then fine-tuned for fraud classification.",
    })
    model.save(MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"[Transfer] Saved model: {MODEL_PATH}\n[Transfer] Saved metrics: {METRICS_PATH}")

if __name__ == "__main__":
    main()
