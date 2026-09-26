# app/ml_router.py
"""
The live ML-based routing decision-maker. For every eligible gateway,
builds the same feature vector the model was trained on (including
the LIVE rolling success rate from Redis) and picks whichever gateway
the model scores highest for predicted success probability.
"""
from datetime import datetime

import joblib
import pandas as pd

from app.gateways import get_gateways_for_method, Gateway
from app.gateway_health import get_rolling_success_rate
from app.features import FEATURE_COLS

MODEL_PATH = "app/models/gateway_success_model.pkl"
ENCODERS_PATH = "app/models/encoders.pkl"

_model = None
_encoders = None


def _load_artifacts():
    global _model, _encoders
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _encoders = joblib.load(ENCODERS_PATH)
    return _model, _encoders


def select_best_gateway_ml(
    method: str,
    amount: float,
    timestamp: datetime,
    exclude_ids: set[str] | None = None,
) -> Gateway | None:
    model, encoders = _load_artifacts()
    exclude_ids = exclude_ids or set()

    candidates = [gw for gw in get_gateways_for_method(method) if gw.id not in exclude_ids]
    if not candidates:
        return None

    default_rate = encoders["meta"]["default_recent_success_rate"]
    hour = timestamp.hour
    dow = timestamp.weekday()
    is_night = 1 if 1 <= hour < 5 else 0
    is_high_value = 1 if amount >= 30000 else 0
    method_enc = encoders["method"].transform([method])[0]

    best_gateway, best_prob = None, -1.0

    for gw in candidates:
        gw_enc = encoders["gateway_id"].transform([gw.id])[0]
        recent_rate = get_rolling_success_rate(gw.id, default=default_rate)

        features = pd.DataFrame([{
            "gateway_id_enc": gw_enc,
            "method_enc": method_enc,
            "amount": amount,
            "hour_of_day": hour,
            "day_of_week": dow,
            "is_night": is_night,
            "is_high_value": is_high_value,
            "recent_success_rate": recent_rate,
        }])[FEATURE_COLS]

        prob = model.predict_proba(features)[0][1]
        if prob > best_prob:
            best_prob, best_gateway = prob, gw

    return best_gateway
