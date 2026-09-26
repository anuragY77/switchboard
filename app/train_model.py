# app/train_model.py
"""
Trains a gateway-success-prediction model from historical transaction data.
Uses a TIME-BASED train/test split (not random) — the model trains only on
the earlier 80% of transactions (by timestamp) and is evaluated on the later
20%, which it has never seen. This mirrors how the model will actually be
used in production (predict on transactions that happen after training) and
avoids the model appearing to "know" patterns from data it was literally
fit on, which a random split can hide when features like recent_success_rate
correlate with the exact historical window used for training.
Run this as a standalone script: python -m app.train_model
"""
import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

from app.config import settings
from app.data_loader import load_attempt_level_data
from app.features import add_static_features, add_rolling_feature, FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TRAIN] %(message)s")
logger = logging.getLogger(__name__)

MODEL_OUTPUT_PATH = "app/models/gateway_success_model.pkl"
ENCODERS_OUTPUT_PATH = "app/models/encoders.pkl"

TEST_SPLIT_FRACTION = 0.2  # last 20% of transactions BY TIME held out as test set


def engineer_features(df: pd.DataFrame):
    df = add_static_features(df)
    df = add_rolling_feature(df, window=settings.rolling_window_size)

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

    # --- Time-based split, not random ---
    # add_rolling_feature() already computed recent_success_rate using only
    # PAST outcomes for every row (train and test alike) — that part was
    # already point-in-time-correct. What's being fixed here is that the
    # MODEL ITSELF must never be fit on rows from the held-out time window.
    df = df.sort_values("created_at").reset_index(drop=True)
    cutoff_idx = int(len(df) * (1 - TEST_SPLIT_FRACTION))
    cutoff_timestamp = df.iloc[cutoff_idx]["created_at"]

    train_df = df[df["created_at"] < cutoff_timestamp]
    test_df = df[df["created_at"] >= cutoff_timestamp]

    logger.info(
        f"Time-based split at {cutoff_timestamp} -> "
        f"train={len(train_df)} rows, test={len(test_df)} rows"
    )

    X_train, y_train = train_df[FEATURE_COLS], train_df["success"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["success"]

    logger.info(f"Train class balance -> success: {y_train.mean():.3f}, failure: {1 - y_train.mean():.3f}")
    logger.info(f"Test class balance  -> success: {y_test.mean():.3f}, failure: {1 - y_test.mean():.3f}")

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    logger.info("Evaluation on TIME-HELD-OUT test set (model never saw this during training):")
    print(classification_report(y_test, y_pred, target_names=["failure", "success"]))
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    logger.info(f"Feature importances:\n{importances}")

    encoders["meta"] = {
        "default_recent_success_rate": df["success"].mean(),
        "rolling_window_size": settings.rolling_window_size,
        "test_cutoff_timestamp": cutoff_timestamp.isoformat(),
    }

    import os
    os.makedirs("app/models", exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    joblib.dump(encoders, ENCODERS_OUTPUT_PATH)
    logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")
    logger.info(f"Encoders saved to {ENCODERS_OUTPUT_PATH}")
    logger.info(
        f"Test cutoff timestamp saved: {cutoff_timestamp.isoformat()} — "
        f"evaluate_routing.py will use this to backtest ONLY on unseen data."
    )


if __name__ == "__main__":
    train()
