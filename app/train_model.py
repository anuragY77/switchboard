"""
Trains a gateway-success-prediction model from historical transaction data.
Run this as a standalone script: python -m app.train_model
"""
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib

from app.config import settings
from app.data_loader import load_attempt_level_data
from app.features import add_static_features, add_rolling_feature, FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TRAIN] %(message)s")
logger = logging.getLogger(__name__)

MODEL_OUTPUT_PATH = "app/models/gateway_success_model.pkl"
ENCODERS_OUTPUT_PATH = "app/models/encoders.pkl"


def engineer_features(df):
    df = add_static_features(df)
    df = add_rolling_feature(df, window=settings.rolling_window_size)

    encoders = {}
    for col in ["gateway_id", "method"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    encoders["meta"] = {
        "default_recent_success_rate": df["success"].mean(),
        "rolling_window_size": settings.rolling_window_size,
    }
    return df, encoders


def train():
    df = load_attempt_level_data()

    if len(df) < 500:
        logger.error(f"Only {len(df)} attempt-rows found — not enough to train reliably. Aborting.")
        return

    df, encoders = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df["success"]

    logger.info(f"Class balance -> success: {y.mean():.3f}, failure: {1 - y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=20,
        # No class_weight="balanced" — we need calibrated probabilities
        # for RANKING gateways against each other, not a balanced-threshold
        # classifier. Balancing distorts predict_proba() output.
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    logger.info("Evaluation on held-out test set:")
    print(classification_report(y_test, y_pred, target_names=["failure", "success"]))
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    import pandas as pd
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    logger.info(f"Feature importances:\n{importances}")

    import os
    os.makedirs("app/models", exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    joblib.dump(encoders, ENCODERS_OUTPUT_PATH)
    logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")
    logger.info(f"Encoders saved to {ENCODERS_OUTPUT_PATH}")


if __name__ == "__main__":
    train()
