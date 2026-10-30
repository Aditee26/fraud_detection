"""Small Keras building blocks shared by the research experiments (CPU only)."""
from __future__ import annotations
import os, random, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

CANONICAL_TAIL = 3   # log_amount, hour_sin, hour_cos: semantically identical in every domain


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def class_weights(y):
    y = np.asarray(y).astype(int); pos = max(1, int(y.sum())); neg = max(1, len(y) - pos)
    return {0: len(y) / (2 * neg), 1: len(y) / (2 * pos)}


def _compile(m, lr):
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(name="pr_auc", curve="PR")])
    return m


def build_ann(input_dim: int, lr: float = 1e-3, seed: int | None = None):
    """Same 64-32-16-1 topology as the original train_model.py ANN."""
    if seed is not None: set_seed(seed)
    m = keras.Sequential([layers.Input((input_dim,)), layers.Dense(64, activation="relu"), layers.Dropout(0.2),
                          layers.Dense(32, activation="relu"), layers.Dense(16, activation="relu"),
                          layers.Dense(1, activation="sigmoid")], name="ann")
    return _compile(m, lr)


def build_shared_encoder(in_dim: int, hidden: int = 64, latent: int = 32):
    inp = keras.Input((in_dim,))
    x = layers.Dense(hidden, activation="relu", name="enc_dense")(inp)
    x = layers.Dense(latent, activation="relu", name="enc_latent")(x)
    return keras.Model(inp, x, name="shared_encoder")


def build_domain_net(input_dim: int, encoder: keras.Model | None = None, adapter_dim: int = 48,
                     hidden: int = 64, latent: int = 32, lr: float = 1e-3, seed: int | None = None):
    """Schema-aware network.

    domain features --Dense adapter (domain specific)--> adapter_dim
    [adapter ; canonical features (log_amount, hour_sin, hour_cos)] --> SHARED encoder --> head

    The adapter learns an alignment of each schema into a common space; the
    canonical tail (same meaning in every dataset) bypasses the adapter.
    Only the shared encoder is transferred between domains.
    """
    if seed is not None: set_seed(seed)
    inp = keras.Input((input_dim,), name="features")
    raw = layers.Lambda(lambda t: t[:, :-CANONICAL_TAIL], name="domain_features")(inp)
    canon = layers.Lambda(lambda t: t[:, -CANONICAL_TAIL:], name="canonical_features")(inp)
    a = layers.Dense(adapter_dim, activation="relu", name="adapter")(raw)
    enc = encoder or build_shared_encoder(adapter_dim + CANONICAL_TAIL, hidden, latent)
    z = enc(layers.Concatenate()([a, canon]))
    z = layers.Dropout(0.2)(z)
    out = layers.Dense(1, activation="sigmoid", name="head")(z)
    return _compile(keras.Model(inp, out, name="domain_net"), lr), enc


def fit_model(model, Xtr, ytr, Xva, yva, epochs=25, batch=256, patience=5, use_class_weight=True, verbose=0):
    """Early-stop on VALIDATION PR-AUC; returns (history, seconds)."""
    t0 = time.perf_counter()
    cb = keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=patience,
                                       restore_best_weights=True, verbose=0)
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch,
                  class_weight=class_weights(ytr) if use_class_weight else None, callbacks=[cb], verbose=verbose)
    return h, time.perf_counter() - t0


def predict(model, X, batch=2048):
    return model.predict(X, batch_size=batch, verbose=0).ravel()
