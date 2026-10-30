"""Gradient attacks (FGSM, PGD) with feature-bound and immutable-feature enforcement.

All perturbation budgets are L-infinity in the *standardised* feature space.
IMPORTANT: for the Kaggle dataset the V1..V28 columns are anonymised PCA
components. A perturbation of V7 is a mathematical robustness probe; it does
NOT correspond to a transaction change an attacker could physically make.
Robustness numbers here do not imply universal robustness - they are
white-box results against specific attacks, budgets and models.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow import keras


def mutable_mask(columns, immutable=("hour_sin", "hour_cos")) -> np.ndarray:
    """1.0 where a feature may be perturbed, 0.0 for protected features."""
    return np.array([0.0 if c in set(immutable) else 1.0 for c in columns], dtype=np.float32)


def _loss_grad(model, x, y, class_w=None):
    with tf.GradientTape() as tape:
        tape.watch(x)
        p = model(x, training=False)
        loss = keras.losses.binary_crossentropy(y, p)
        if class_w is not None:
            loss = loss * tf.where(y[:, 0] > 0.5, class_w[1], class_w[0])
        loss = tf.reduce_sum(loss)
    return tape.gradient(loss, x)


def _project(x_adv, x0, eps_vec, lo, hi, mask):
    delta = tf.clip_by_value(x_adv - x0, -eps_vec, eps_vec) * mask   # budget + immutability
    out = x0 + delta
    if lo is not None:
        # stay inside the training feature range, but never move a clean point that already lies outside it
        out = tf.clip_by_value(out, tf.minimum(lo, x0), tf.maximum(hi, x0))
    return out


def _prep(X, y, eps, lo, hi, mask):
    x0 = tf.convert_to_tensor(X, tf.float32)
    yt = tf.convert_to_tensor(np.asarray(y).reshape(-1, 1), tf.float32)
    n = X.shape[1]
    m = tf.constant(np.ones(n, np.float32) if mask is None else mask)
    e = tf.constant(np.broadcast_to(np.asarray(eps, np.float32), (n,)).copy())  # scalar or per-feature budget
    l = None if lo is None else tf.constant(lo, tf.float32); h = None if hi is None else tf.constant(hi, tf.float32)
    return x0, yt, e, l, h, m


def fgsm(model, X, y, eps, lo=None, hi=None, mask=None, class_w=None):
    x0, yt, e, l, h, m = _prep(X, y, eps, lo, hi, mask)
    g = _loss_grad(model, x0, yt, class_w)
    return _project(x0 + e * tf.sign(g), x0, e, l, h, m).numpy()


def pgd(model, X, y, eps, steps=10, alpha=None, random_start=True, lo=None, hi=None, mask=None, seed=0, class_w=None):
    """Projected gradient descent (Madry et al.). alpha defaults to 2.5*eps/steps."""
    x0, yt, e, l, h, m = _prep(X, y, eps, lo, hi, mask)
    a = (2.5 / steps) * e if alpha is None else tf.constant(np.broadcast_to(np.asarray(alpha, np.float32), (X.shape[1],)).copy())
    x = x0
    if random_start:
        rng = np.random.default_rng(seed)
        noise = tf.constant(rng.uniform(-1, 1, X.shape).astype(np.float32)) * e
        x = _project(x0 + noise, x0, e, l, h, m)
    for _ in range(steps):
        g = _loss_grad(model, x, yt, class_w)
        x = _project(x + a * tf.sign(g), x0, e, l, h, m)
    return x.numpy()


def random_noise(X, eps, lo=None, hi=None, mask=None, seed=0):
    """Sanity baseline: random sign noise with the same budget."""
    rng = np.random.default_rng(seed)
    n = X.shape[1]
    d = rng.choice([-1.0, 1.0], X.shape).astype(np.float32) * np.broadcast_to(np.asarray(eps, np.float32), (n,))
    if mask is not None: d = d * mask
    out = X + d
    return np.clip(out, np.minimum(lo, X), np.maximum(hi, X)) if lo is not None else out


def check_constraints(X, X_adv, eps, lo=None, hi=None, mask=None, tol=1e-5) -> dict:
    """Verify an attack respected its budget, bounds and immutable features."""
    d = np.abs(X_adv - X)
    res = {"max_abs_perturbation": float(d.max()), "within_budget": bool(d.max() <= np.max(eps) + tol)}
    if lo is not None:
        res["within_bounds"] = bool((X_adv >= np.minimum(lo, X) - tol).all() and (X_adv <= np.maximum(hi, X) + tol).all())
    if mask is not None:
        res["immutable_unchanged"] = bool(d[:, mask == 0].max(initial=0.0) <= tol)
    return res


def adversarial_train(model, Xtr, ytr, Xva, yva, eps, attack="pgd", epochs=12, steps=5, batch=256,
                      lo=None, hi=None, mask=None, patience=4, seed=0):
    """Madry-style adversarial training: each epoch, craft adversarial examples
    against the CURRENT model from the TRAINING split only, then train on
    clean + adversarial. Model selection uses clean validation PR-AUC."""
    from .nn import class_weights
    cw = class_weights(ytr); best, best_w, bad = -1.0, None, 0
    for ep in range(epochs):
        if attack == "pgd":
            adv = pgd(model, Xtr, ytr, eps, steps=steps, lo=lo, hi=hi, mask=mask, seed=seed + ep)
        else:
            adv = fgsm(model, Xtr, ytr, eps, lo=lo, hi=hi, mask=mask)
        model.fit(np.vstack([Xtr, adv]), np.concatenate([ytr, ytr]), batch_size=batch, epochs=1,
                  class_weight=cw, verbose=0, shuffle=True)
        score = float(keras.metrics.AUC(curve="PR")(yva, model.predict(Xva, verbose=0).ravel()).numpy())
        if score > best: best, best_w, bad = score, model.get_weights(), 0
        else:
            bad += 1
            if bad >= patience: break
    model.set_weights(best_w)
    return model
