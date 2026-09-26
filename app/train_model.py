# app/train_model.py
"""
Trains a gateway-success-prediction model from historical transaction data.
Run this as a standalone script: python -m app.train_model
"""
import json
import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib

from app.database import SessionLocal, Transaction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TRAIN] %(message)s")
logger = logging.getLogger(__name__)

MODEL_OUTPUT_PATH = "app/models/gateway_success_model.pkl"
ENCODERS_OUTPUT_PATH = "app/models/encoders.pkl"


def load_attempt_level_data() -> pd.DataFrame:
    """
    Each transaction row has an `attempts_log` JSON column containing
    EVERY gateway attempt made (not just the final outcome). We flatten
    this into one row PER ATTEMPT, because that's the real unit the
    routing model needs to predict on: "if I send THIS transaction to
    THIS specific gateway right now, will it succeed?"
    """
    db = SessionLocal()
    try:
        transactions = db.query(Transaction).filter(
            Transaction.attempts_log.isnot(None)
        ).all()

        rows = []
        for txn in transactions:
            try:
                attempts = json.loads(txn.attempts_log)
            except (json.JSONDecodeError, TypeError):
                continue

            for attempt in attempts:
                rows.append({
                    "gateway_id": attempt["gateway_id"],
                    "method": txn.method,
                    "amount": txn.amount,
                    "hour_of_day": txn.created_at.hour,
                    "day_of_week": txn.created_at.weekday(),
                    "success": 1 if attempt["success"] else 0,
                })

        logger.info(f"Loaded {len(transactions)} transactions -> {len(rows)} attempt-level rows")
        return pd.DataFrame(rows)
    finally:
        db.close()


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Converts categorical columns to numeric, adds a derived
    'is_night' flag and 'is_high_value' flag (mirrors the logic
    the simulator itself uses, so the model can rediscover those
    same patterns from data instead of being told them directly).
    """
    df = df.copy()
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if 1 <= h < 5 else 0)
    df["is_high_value"] = df["amount"].apply(lambda a: 1 if a >= 30000 else 0)

    encoders = {}
    for col in ["gateway_id", "method"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def train():
    df = load_attempt_level_data()

    if len(df) < 500:
        logger.error(f"Only {len(df)} attempt-rows found — not enough to train reliably. Aborting.")
        return

    df, encoders = engineer_features(df)

    feature_cols = ["gateway_id_enc", "method_enc", "amount", "hour_of_day",
                     "day_of_week", "is_night", "is_high_value"]
    X = df[feature_cols]
    y = df["success"]

    logger.info(f"Class balance -> success: {y.mean():.3f}, failure: {1 - y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=20,
        # class_weight="balanced" REMOVED — we need calibrated
        # probabilities for ranking gateways against each other,
        # not a balanced-threshold classifier. Balancing distorts
        # predict_proba() output, which is exactly what routing
        # decisions depend on.
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    logger.info("Evaluation on held-out test set:")
    print(classification_report(y_test, y_pred, target_names=["failure", "success"]))
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    logger.info(f"Feature importances:\n{importances}")

    import os
    os.makedirs("app/models", exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    joblib.dump(encoders, ENCODERS_OUTPUT_PATH)
    logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")
    logger.info(f"Encoders saved to {ENCODERS_OUTPUT_PATH}")


if __name__ == "__main__":
    train()