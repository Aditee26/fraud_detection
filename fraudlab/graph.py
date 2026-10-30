"""Temporal transaction graph + GraphSAGE (mean aggregator, 2 layers, neighbour sampling).

Nodes = transactions. Edges connect a transaction to EARLIER transactions that
share an account_id or a device_id. Edges therefore never point to the future,
and node inputs are features only - labels are never aggregated.

This needs genuine relationship columns. The Kaggle Credit Card dataset has none,
so `build_neighbors` refuses to run without them rather than fabricate a graph.
"""
from __future__ import annotations
from collections import deque
import numpy as np


def build_neighbors(times, key_arrays: list, max_nb: int = 8) -> np.ndarray:
    if not key_arrays:
        raise ValueError("No relationship columns (account/device/merchant ids): a genuine transaction graph "
                         "cannot be built for this dataset. Use a dataset with entity ids or the labelled synthetic generator.")
    n = len(times)
    order = np.argsort(times, kind="stable")
    recent = [dict() for _ in key_arrays]
    idx = np.full((n, max_nb), -1, np.int64)
    for i in order:
        cand = []
        for d, keys in zip(recent, key_arrays):
            dq = d.get(keys[i])
            if dq: cand.extend(dq)
        if cand:
            cand = sorted(set(cand), reverse=True)[:max_nb]            # most recent earlier neighbours
            idx[i, :len(cand)] = cand
        for d, keys in zip(recent, key_arrays):
            d.setdefault(keys[i], deque(maxlen=max_nb)).append(i)
    return idx


def check_edges_causal(times, idx) -> bool:
    ok = idx >= 0
    nt = np.where(ok, np.asarray(times)[np.where(ok, idx, 0)], -np.inf)
    return bool((nt <= np.asarray(times)[:, None] + 1e-9).all())


def mean_neighbors(X, idx):
    ok = (idx >= 0)[..., None]
    g = X[np.where(idx >= 0, idx, 0)] * ok
    return (g.sum(1) / np.maximum(ok.sum(1), 1)).astype(np.float32)


def sage_inputs(X, idx):
    """Everything a 2-layer GraphSAGE needs for each target node (features only)."""
    m = mean_neighbors(X, idx)
    safe = np.where(idx >= 0, idx, 0)
    mask = (idx >= 0).astype(np.float32)
    return {"x": X.astype(np.float32), "m": m, "nbr_x": (X[safe] * mask[..., None]).astype(np.float32),
            "nbr_m": (m[safe] * mask[..., None]).astype(np.float32), "mask": mask}


def build_graphsage(f: int, s: int, hidden: int = 48, lr: float = 1e-3):
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    x = keras.Input((f,), name="x"); m = keras.Input((f,), name="m")
    nx = keras.Input((s, f), name="nbr_x"); nm = keras.Input((s, f), name="nbr_m"); mask = keras.Input((s,), name="mask")
    W1 = layers.Dense(hidden, activation="relu", name="sage1")             # layer-1 weights shared by all nodes
    h1_i = W1(layers.Concatenate()([x, m]))
    h1_j = W1(layers.Concatenate()([nx, nm]))                              # [B, S, hidden]
    agg = layers.Lambda(lambda t: tf.reduce_sum(t[0] * t[1][..., None], 1) / tf.maximum(tf.reduce_sum(t[1], 1, keepdims=True), 1.0),
                        name="mean_agg")([h1_j, mask])
    h2 = layers.Dense(hidden, activation="relu", name="sage2")(layers.Concatenate()([h1_i, agg]))
    out = layers.Dense(1, activation="sigmoid")(layers.Dropout(0.2)(h2))
    mdl = keras.Model([x, m, nx, nm, mask], out, name="graphsage")
    mdl.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                metrics=[keras.metrics.AUC(name="pr_auc", curve="PR")])
    return mdl
