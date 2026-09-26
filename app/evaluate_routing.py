# app/evaluate_routing.py
"""
Backtests routing decision quality using the SAME point-in-time-correct
rolling-history logic that live routing uses — replayed offline against
historical data, so this is a faithful simulation of "what would the
ML router have chosen at that exact moment," without any data leakage
from the future.
"""
import json
import logging
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.database import SessionLocal, Transaction
from app.data_loader import load_attempt_level_data
from app.features import GatewayRollingHistory, FEATURE_COLS
from app.gateways import GATEWAYS, get_gateways_for_method, compute_expected_success_rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EVAL] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "app/models/gateway_success_model.pkl"
ENCODERS_PATH = "app/models/encoders.pkl"


def load_model():
    return joblib.load(MODEL_PATH), joblib.load(ENCODERS_PATH)


def model_choose_gateway(method, amount, timestamp, model, encoders, history: GatewayRollingHistory, default_rate: float):
    candidates = get_gateways_for_method(method)
    hour, dow = timestamp.hour, timestamp.weekday()
    is_night = 1 if 1 <= hour < 5 else 0
    is_high_value = 1 if amount >= 30000 else 0
    method_enc = encoders["method"].transform([method])[0]

    best_gw, best_prob = None, -1.0
    for gw in candidates:
        gw_enc = encoders["gateway_id"].transform([gw.id])[0]
        recent_rate = history.rolling_rate_before(gw.id, timestamp, default=default_rate)

        features = pd.DataFrame([{
            "gateway_id_enc": gw_enc, "method_enc": method_enc, "amount": amount,
            "hour_of_day": hour, "day_of_week": dow,
            "is_night": is_night, "is_high_value": is_high_value,
            "recent_success_rate": recent_rate,
        }])[FEATURE_COLS]

        prob = model.predict_proba(features)[0][1]
        if prob > best_prob:
            best_prob, best_gw = prob, gw.id

    return best_gw


def true_best_gateway_and_rate(method, amount, timestamp):
    candidates = get_gateways_for_method(method)
    scored = [(gw.id, compute_expected_success_rate(gw, amount, timestamp)) for gw in candidates]
    scored.sort(key=lambda p: p[1], reverse=True)
    return scored[0]


def true_rate_for_gateway(gateway_id, method, amount, timestamp):
    return compute_expected_success_rate(GATEWAYS[gateway_id], amount, timestamp)


def run_evaluation_by_method(sample_size: int = 1000):
    import random as pyrandom

    model, encoders = load_model()

    # Build the full historical replay context ONCE — same rolling-history
    # object the model's recent_success_rate feature depends on.
    full_df = load_attempt_level_data()
    history = GatewayRollingHistory(full_df, window=settings.rolling_window_size)
    default_rate = full_df["success"].mean()

    db = SessionLocal()
    try:
        transactions = db.query(Transaction).filter(
            Transaction.attempts_log.isnot(None)
        ).limit(sample_size).all()
    finally:
        db.close()

    by_method = defaultdict(lambda: {"model": [], "baseline": [], "best": [], "random": []})
    model_matches_best = 0
    total = 0

    for txn in transactions:
        ts, method, amount = txn.created_at, txn.method, txn.amount

        true_best_id, true_best_rate = true_best_gateway_and_rate(method, amount, ts)
        model_choice = model_choose_gateway(method, amount, ts, model, encoders, history, default_rate)
        model_rate = true_rate_for_gateway(model_choice, method, amount, ts)

        try:
            attempts = json.loads(txn.attempts_log)
            baseline_choice = attempts[0]["gateway_id"]
            baseline_rate = true_rate_for_gateway(baseline_choice, method, amount, ts)
        except (json.JSONDecodeError, IndexError, KeyError, TypeError):
            continue

        random_choice = pyrandom.choice(get_gateways_for_method(method))
        random_rate = true_rate_for_gateway(random_choice.id, method, amount, ts)

        by_method[method]["model"].append(model_rate)
        by_method[method]["baseline"].append(baseline_rate)
        by_method[method]["best"].append(true_best_rate)
        by_method[method]["random"].append(random_rate)

        if model_choice == true_best_id:
            model_matches_best += 1
        total += 1

    logger.info(f"\n=== Overall (n={total}) ===")
    logger.info(f"Model picked TRUE BEST gateway: {model_matches_best}/{total} ({100*model_matches_best/total:.1f}%)")

    logger.info("\n=== By method ===")
    for method, data in by_method.items():
        n = len(data["model"])
        num_gw = len(get_gateways_for_method(method))
        logger.info(
            f"\n{method} (n={n}, {num_gw} eligible gateways):\n"
            f"  Model:    {np.mean(data['model']):.4f}\n"
            f"  Baseline: {np.mean(data['baseline']):.4f}\n"
            f"  Best:     {np.mean(data['best']):.4f}\n"
            f"  Random:   {np.mean(data['random']):.4f}"
        )


if __name__ == "__main__":
    run_evaluation_by_method()
