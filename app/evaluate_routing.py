# app/evaluate_routing.py
"""
Evaluates ROUTING DECISION QUALITY, not per-attempt classification accuracy.
For each historical transaction, computes the TRUE expected success rate
for every eligible gateway at that transaction's exact context (method,
amount, timestamp), then compares:
  - what the model would have chosen
  - what the rule-based baseline chose
  - what a random choice would achieve
against the TRUE BEST possible choice.

This answers the question that actually matters: "does the model pick
good routes?" — not "can it predict a single noisy coin-flip?"
"""
import json
import logging

import joblib
import numpy as np
import pandas as pd

from app.database import SessionLocal, Transaction
from app.gateways import GATEWAYS, get_gateways_for_method, compute_expected_success_rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EVAL] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "app/models/gateway_success_model.pkl"
ENCODERS_PATH = "app/models/encoders.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, encoders


def build_feature_row(gateway_id: str, method: str, amount: float, timestamp, encoders) -> pd.DataFrame:
    hour = timestamp.hour
    dow = timestamp.weekday()
    is_night = 1 if 1 <= hour < 5 else 0
    is_high_value = 1 if amount >= 30000 else 0

    gw_enc = encoders["gateway_id"].transform([gateway_id])[0]
    method_enc = encoders["method"].transform([method])[0]

    return pd.DataFrame([{
        "gateway_id_enc": gw_enc,
        "method_enc": method_enc,
        "amount": amount,
        "hour_of_day": hour,
        "day_of_week": dow,
        "is_night": is_night,
        "is_high_value": is_high_value,
    }])


def model_choose_gateway(method: str, amount: float, timestamp, model, encoders) -> str:
    """Model scores every eligible gateway, picks the one with highest predicted success probability."""
    candidates = get_gateways_for_method(method)
    best_gw, best_prob = None, -1.0

    for gw in candidates:
        features = build_feature_row(gw.id, method, amount, timestamp, encoders)
        prob = model.predict_proba(features)[0][1]  # P(success)
        if prob > best_prob:
            best_prob, best_gw = prob, gw.id

    return best_gw


def true_best_gateway_and_rate(method: str, amount: float, timestamp) -> tuple[str, float]:
    """Ground truth: uses the actual simulator formula (no random jitter) to find the REAL best gateway."""
    candidates = get_gateways_for_method(method)
    scored = [
        (gw.id, compute_expected_success_rate(gw, amount, timestamp))  # <-- deterministic
        for gw in candidates
    ]
    scored.sort(key=lambda p: p[1], reverse=True)
    return scored[0]


def true_rate_for_gateway(gateway_id: str, method: str, amount: float, timestamp) -> float:
    gw = GATEWAYS[gateway_id]
    return compute_expected_success_rate(gw, amount, timestamp)  # <-- deterministic


def run_evaluation(sample_size: int = 1000):
    from datetime import datetime

    model, encoders = load_model()
    db = SessionLocal()
    try:
        transactions = db.query(Transaction).filter(
            Transaction.attempts_log.isnot(None)
        ).limit(sample_size).all()
    finally:
        db.close()

    model_rates, baseline_rates, true_best_rates, random_rates = [], [], [], []
    model_matches_best = 0

    import random as pyrandom

    for txn in transactions:
        ts = txn.created_at
        method, amount = txn.method, txn.amount

        true_best_id, true_best_rate = true_best_gateway_and_rate(method, amount, ts)

        model_choice = model_choose_gateway(method, amount, ts, model, encoders)
        model_rate = true_rate_for_gateway(model_choice, method, amount, ts)

        try:
            attempts = json.loads(txn.attempts_log)
            baseline_choice = attempts[0]["gateway_id"]  # first attempt = what rule-based router picked
            baseline_rate = true_rate_for_gateway(baseline_choice, method, amount, ts)
        except (json.JSONDecodeError, IndexError, KeyError, TypeError):
            continue

        random_choice = pyrandom.choice(get_gateways_for_method(method))
        random_rate = true_rate_for_gateway(random_choice.id, method, amount, ts)

        model_rates.append(model_rate)
        baseline_rates.append(baseline_rate)
        true_best_rates.append(true_best_rate)
        random_rates.append(random_rate)

        if model_choice == true_best_id:
            model_matches_best += 1

    n = len(model_rates)
    logger.info(f"Evaluated {n} transactions\n")
    logger.info(f"Model's avg achieved success-rate:      {np.mean(model_rates):.4f}")
    logger.info(f"Rule-based baseline avg success-rate:    {np.mean(baseline_rates):.4f}")
    logger.info(f"True BEST possible avg success-rate:     {np.mean(true_best_rates):.4f}")
    logger.info(f"Random-choice avg success-rate:          {np.mean(random_rates):.4f}\n")
    logger.info(f"Model picked the TRUE BEST gateway:      {model_matches_best}/{n} ({100*model_matches_best/n:.1f}%)")
    logger.info(f"Model vs Random uplift:                  {100*(np.mean(model_rates) - np.mean(random_rates)):.2f} percentage points")
    logger.info(f"Model vs Rule-based baseline:             {100*(np.mean(model_rates) - np.mean(baseline_rates)):+.2f} percentage points")


if __name__ == "__main__":
    run_evaluation()