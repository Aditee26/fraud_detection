"""
Graph-neighbour aggregation model for credit-card fraud detection.

This is a GraphSAGE-inspired inductive model: each transaction is connected to
its nearest training transactions in standardized feature space, and their
features are aggregated. The source dataset has no real account/card/merchant
IDs, so these are similarity edges, NOT real banking relationships.

Run: python train_gnn.py
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
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix)

DATA_PATH = "data/creditcard.csv"
ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "gnn_model.keras")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "gnn_metrics.json")
REFERENCE_PATH = os.path.join(ARTIFACT_DIR, "gnn_reference.npz")
SEED = 42
TEST_SIZE = 0.20
K_NEIGHBORS = 8
EPOCHS = 35
BATCH_SIZE = 256

def neighbor_mean(query_x, reference_x, k, exclude_self=False):
    """Mean feature vector of k nearest reference rows for each query row."""
    k_eff = min(k + (1 if exclude_self else 0), len(reference_x))
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=-1)
    nn.fit(reference_x)
    indices = nn.kneighbors(query_x, return_distance=False)
    if exclude_self:
        # For train rows, query and reference arrays are the same and self is first.
        indices = indices[:, 1:]
    else:
        indices = indices[:, :min(k, indices.shape[1])]
    return reference_x[indices].mean(axis=1).astype(np.float32)

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
    required = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df[[c for c in df.columns if c != "Class"]].to_numpy(np.float32)
    y = df["Class"].to_numpy(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print(f"[GNN] Train={len(X_train):,}; test={len(X_test):,}; k={K_NEIGHBORS}")
    print("[GNN] Building nearest-neighbour aggregates; edges are feature-similarity links.")
    train_neighbours = neighbor_mean(X_train, X_train, K_NEIGHBORS, exclude_self=True)
    test_neighbours = neighbor_mean(X_test, X_train, K_NEIGHBORS, exclude_self=False)

    # Two-view message passing: the model sees the transaction and aggregated
    # neighbour features, then learns a fraud score.
    own_input = keras.Input(shape=(X_train.shape[1],), name="transaction_features")
    nbr_input = keras.Input(shape=(X_train.shape[1],), name="neighbour_aggregate")
    own = layers.Dense(64, activation="relu")(own_input)
    nbr = layers.Dense(64, activation="relu")(nbr_input)
    merged = layers.Concatenate()([own, nbr])
    merged = layers.Dense(64, activation="relu")(merged)
    merged = layers.Dropout(0.25)(merged)
    merged = layers.Dense(32, activation="relu")(merged)
    output = layers.Dense(1, activation="sigmoid")(merged)
    model = keras.Model([own_input, nbr_input], output, name="gnn_neighbor_aggregation")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"), "accuracy"])
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    stop = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                         restore_best_weights=True, verbose=1)
    model.fit([X_train, train_neighbours], y_train, validation_split=0.2,
              epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight,
              callbacks=[stop], verbose=2)
    probabilities = model.predict([X_test, test_neighbours], batch_size=BATCH_SIZE, verbose=0).ravel()
    metrics = score_metrics(y_test, probabilities)
    metrics.update({
        "model": "GNN (nearest-neighbour aggregation)",
        "architecture": "GraphSAGE-inspired two-view feature aggregation",
        "dataset_rows": int(len(df)), "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)), "test_fraud_rows": int(y_test.sum()),
        "k_neighbors": K_NEIGHBORS, "threshold": 0.5, "seed": SEED,
        "graph_semantics": "Feature-similarity edges only; not real account/customer relationships.",
    })
    model.save(MODEL_PATH)
    np.savez_compressed(REFERENCE_PATH, train_scaled=X_train,
                        columns=np.array([c for c in df.columns if c != "Class"]),
                        mean=scaler.mean_, scale=scaler.scale_)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"[GNN] Saved model: {MODEL_PATH}\n[GNN] Saved metrics: {METRICS_PATH}")

if __name__ == "__main__":
    main()
