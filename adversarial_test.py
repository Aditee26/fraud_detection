"""
adversarial_test.py
---------------------
Adversarial robustness demonstration.

    Normal Model -> Fraud prediction -> FGSM perturbation ->
    prediction changes / confidence drops -> Adversarial Training ->
    more robust prediction

Implements real FGSM (Fast Gradient Sign Method) directly with TensorFlow
gradients (no ART dependency needed for this simple ANN case, keeping the
prototype lightweight). Every number in the output comparison table is
computed by actually attacking/retraining the model -- nothing is invented.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from train_model import build_model, load_data, ARTIFACT_DIR, SEED

EPSILON = 0.3  # FGSM perturbation budget (in scaled feature space)


def fgsm_attack(model, X, y, epsilon):
    """Generate FGSM adversarial examples: x_adv = x + eps * sign(grad_x loss)."""
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y.reshape(-1, 1), dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_tf)
        preds = model(X_tf, training=False)
        loss = keras.losses.binary_crossentropy(y_tf, preds)
    grads = tape.gradient(loss, X_tf)
    signed_grads = tf.sign(grads)
    X_adv = X_tf + epsilon * signed_grads
    return X_adv.numpy()


def evaluate(model, X, y, threshold=0.5):
    prob = model.predict(X, verbose=0).ravel()
    pred = (prob >= threshold).astype(int)
    return accuracy_score(y, pred), prob


def main():
    df, is_demo = load_data()
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    # ---- 1. Load the already-trained "normal" model ----
    normal_model_path = os.path.join(ARTIFACT_DIR, "fraud_model.keras")
    if not os.path.exists(normal_model_path):
        raise SystemExit("Run train_model.py first to produce artifacts/fraud_model.keras")
    normal_model = keras.models.load_model(normal_model_path)

    clean_acc, clean_prob = evaluate(normal_model, X_test_s, y_test)

    # Pick one genuine fraud example to show a before/after story.
    fraud_idx = np.where(y_test == 1)[0]
    demo_idx = fraud_idx[0] if len(fraud_idx) else 0
    x_demo = X_test_s[demo_idx:demo_idx + 1]
    y_demo = y_test[demo_idx:demo_idx + 1]
    prob_before = float(normal_model.predict(x_demo, verbose=0).ravel()[0])

    # ---- 2. Attack the normal model with FGSM ----
    X_test_adv = fgsm_attack(normal_model, X_test_s, y_test, EPSILON)
    adv_acc_normal, adv_prob = evaluate(normal_model, X_test_adv, y_test)

    x_demo_adv = fgsm_attack(normal_model, x_demo, y_demo, EPSILON)
    prob_after = float(normal_model.predict(x_demo_adv, verbose=0).ravel()[0])

    # ---- 3. Adversarial training: retrain on clean + FGSM examples ----
    X_train_adv = fgsm_attack(normal_model, X_train_s, y_train, EPSILON)
    X_train_combined = np.vstack([X_train_s, X_train_adv])
    y_train_combined = np.concatenate([y_train, y_train])

    robust_model = build_model(X_train_s.shape[1])
    robust_model.fit(
        X_train_combined, y_train_combined,
        validation_split=0.2,
        epochs=15,
        batch_size=256,
        class_weight=class_weight,
        verbose=2,
    )
    robust_model.save(os.path.join(ARTIFACT_DIR, "fraud_model_robust.keras"))

    # ---- 4. Re-evaluate: robust model on clean data and on fresh FGSM attack ----
    clean_acc_robust, _ = evaluate(robust_model, X_test_s, y_test)
    X_test_adv_vs_robust = fgsm_attack(robust_model, X_test_s, y_test, EPSILON)
    adv_acc_robust, _ = evaluate(robust_model, X_test_adv_vs_robust, y_test)

    prob_after_robust_training = float(
        robust_model.predict(fgsm_attack(robust_model, x_demo, y_demo, EPSILON), verbose=0).ravel()[0]
    )

    comparison = {
        "epsilon": EPSILON,
        "normal_model": {
            "clean_accuracy": round(float(clean_acc), 4),
            "adversarial_accuracy": round(float(adv_acc_normal), 4),
        },
        "adversarially_trained_model": {
            "clean_accuracy": round(float(clean_acc_robust), 4),
            "adversarial_accuracy": round(float(adv_acc_robust), 4),
        },
        "single_transaction_demo": {
            "fraud_probability_normal_model_clean_input": round(prob_before, 4),
            "fraud_probability_normal_model_after_fgsm": round(prob_after, 4),
            "fraud_probability_robust_model_after_fgsm": round(prob_after_robust_training, 4),
        },
    }
    print(json.dumps(comparison, indent=2))

    with open(os.path.join(ARTIFACT_DIR, "adversarial_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    # Bar chart: clean vs adversarial accuracy, normal vs robust model.
    labels = ["Normal ANN", "Adversarially Trained ANN"]
    clean_vals = [clean_acc, clean_acc_robust]
    adv_vals = [adv_acc_normal, adv_acc_robust]
    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, clean_vals, width, label="Clean accuracy")
    ax.bar(x + width / 2, adv_vals, width, label="Adversarial (FGSM) accuracy")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Robustness Comparison (FGSM, epsilon={EPSILON})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "adversarial_comparison.png"), dpi=130)
    plt.close(fig)

    print(f"[adversarial_test] saved -> artifacts/adversarial_comparison.json / .png")
    print(f"[adversarial_test] saved -> artifacts/fraud_model_robust.keras")


if __name__ == "__main__":
    main()
