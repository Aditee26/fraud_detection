import argparse
import hashlib
import json
import os
import pandas as pd
from dataclasses import asdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np



# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "artifacts/fraud_model.keras"
SCALER_PATH = "artifacts/scaler.npz"

DEFAULT_THRESHOLD = 0.5
DEFAULT_ROWS = 5


# Block / LocalFraudChain now live in audit_chain.py (TensorFlow-free, unit-tested).
from audit_chain import Block, LocalFraudChain, verify_blocks, verify_against_anchor  # noqa: F401


# -------------------------------------------------------------------
# Load real dataset
# -------------------------------------------------------------------

def load_real_data():
    """
    Loads the actual Kaggle Credit Card Fraud Detection dataset.

    No fallback dataset is generated.
    """

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\nReal dataset not found:\n{DATA_PATH}\n\n"
            "Place the Kaggle creditcard.csv file inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'Class' column."
        )

    if len(df) == 0:
        raise ValueError("The dataset is empty.")

    return df


# -------------------------------------------------------------------
# Load model + scaler
# -------------------------------------------------------------------

def load_model_and_scaler():
    """
    Loads the trained fraud model and the scaler created during training.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\nTrained model not found:\n{MODEL_PATH}\n\n"
            "Run train_model.py first."
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"\nScaler not found:\n{SCALER_PATH}\n\n"
            "Run train_model.py first."
        )

    import tensorflow as tf  # lazy: only needed to score real rows
    model = tf.keras.models.load_model(MODEL_PATH)

    scaler_data = np.load(SCALER_PATH)

    mean = scaler_data["mean"]
    scale = scaler_data["scale"]
    columns = scaler_data["columns"]

    return model, mean, scale, columns


# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------

def predict_transactions(
    df,
    model,
    mean,
    scale,
    feature_columns,
    threshold
):
    """
    Generates fraud predictions for real dataset rows.
    """

    feature_columns = [str(column) for column in feature_columns]

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing features used during training: "
            + ", ".join(missing_columns)
        )

    X = df[feature_columns].copy()

    # Make sure values are numeric.
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        raise ValueError(
            "Dataset contains missing or non-numeric feature values."
        )

    X = X.to_numpy(dtype=np.float32)

    # Apply exactly the same scaling used during training.
    X_scaled = (X - mean) / scale

    probabilities = model.predict(
        X_scaled,
        verbose=0
    ).ravel()

    predictions = (probabilities >= threshold).astype(int)

    return probabilities, predictions


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Create a hash-chain audit log from real fraud predictions."
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help="Number of real dataset rows to process."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Fraud probability threshold."
    )

    parser.add_argument(
        "--output",
        default="artifacts/fraud_blockchain.json",
        help="Output JSON file."
    )

    args = parser.parse_args()

    if args.rows <= 0:
        raise ValueError("--rows must be greater than 0.")

    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1.")

    print("\n========================================")
    print(" REAL FRAUD BLOCKCHAIN AUDIT")
    print("========================================")

    # ---------------------------------------------------------------
    # Load real data
    # ---------------------------------------------------------------

    print("\n[1/5] Loading real dataset...")

    df = load_real_data()

    print(f"[blockchain] Dataset rows: {len(df):,}")
    print(f"[blockchain] Dataset columns: {len(df.columns)}")

    # ---------------------------------------------------------------
    # Load trained model
    # ---------------------------------------------------------------

    print("\n[2/5] Loading trained model and scaler...")

    model, mean, scale, feature_columns = load_model_and_scaler()

    print(f"[blockchain] Model: {MODEL_PATH}")
    print(f"[blockchain] Features: {len(feature_columns)}")

    # ---------------------------------------------------------------
    # Select real rows
    # ---------------------------------------------------------------

    rows_to_process = min(args.rows, len(df))

    selected_df = df.iloc[:rows_to_process].copy()

    print(
        f"\n[3/5] Processing {rows_to_process} real dataset rows..."
    )

    # ---------------------------------------------------------------
    # Predict
    # ---------------------------------------------------------------

    probabilities, predictions = predict_transactions(
        selected_df,
        model,
        mean,
        scale,
        feature_columns,
        args.threshold
    )

    # ---------------------------------------------------------------
    # Build blockchain
    # ---------------------------------------------------------------

    print("\n[4/5] Creating hash-linked blocks...")

    chain = LocalFraudChain()

    model_version = os.path.basename(MODEL_PATH)

    for position, ((row_index, row), probability, prediction) in enumerate(
        zip(
            selected_df.iterrows(),
            probabilities,
            predictions
        )
    ):

        # The ULB Kaggle dataset does not have a TransactionID column.
        # Therefore we use the actual CSV row index as the identifier.
        #
        # This is NOT a fabricated transaction ID.
        transaction_id = f"DATASET_ROW_{row_index}"

        decision = (
            "FRAUD"
            if prediction == 1
            else "LEGITIMATE"
        )

        risk_score = float(probability * 100)

        block = chain.add_transaction(
            transaction_id=transaction_id,
            actual_class=int(row["Class"]),
            decision=decision,
            risk_score=risk_score,
            model_version=model_version
        )

        print(
            f"  Block #{block.index} | "
            f"{transaction_id} | "
            f"{decision:<10} | "
            f"Risk: {risk_score:6.2f}%"
        )

    # ---------------------------------------------------------------
    # Verify chain
    # ---------------------------------------------------------------

    print("\n[5/5] Verifying blockchain integrity...")

    is_valid = chain.verify_integrity()

    print(
        "[blockchain] Integrity:",
        "VALID" if is_valid else "INVALID"
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True
    )

    chain.save(args.output)

    print(f"\n[blockchain] Saved audit chain -> {args.output}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    fraud_predictions = int(predictions.sum())
    legitimate_predictions = int(len(predictions) - fraud_predictions)

    actual_fraud = int(selected_df["Class"].sum())
    actual_legitimate = int(len(selected_df) - actual_fraud)

    print("\n========================================")
    print(" SUMMARY")
    print("========================================")

    print(f"Rows processed       : {len(selected_df)}")
    print(f"Actual fraud         : {actual_fraud}")
    print(f"Actual legitimate    : {actual_legitimate}")
    print(f"Predicted fraud      : {fraud_predictions}")
    print(f"Predicted legitimate : {legitimate_predictions}")
    print(f"Threshold            : {args.threshold}")
    print(f"Chain blocks         : {len(chain.chain)}")
    print(f"Integrity             : {'VALID' if is_valid else 'INVALID'}")

    print("\nDone.")


if __name__ == "__main__":
    main()