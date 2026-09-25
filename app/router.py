# app/router.py
import random
from datetime import datetime

from app.config import settings
from app.gateways import get_gateways_for_method, compute_effective_success_rate, Gateway


def select_best_gateway(
    method: str,
    amount: float,
    timestamp: datetime,
    exclude_ids: set[str] | None = None,
) -> Gateway | None:
    """
    Picks a gateway for this transaction.

    Uses epsilon-greedy exploration: most of the time (1 - epsilon),
    picks the highest computed-success-rate gateway (exploitation).
    But with probability `epsilon`, picks a RANDOM eligible gateway
    instead (exploration) — this is essential during data collection,
    because always picking the "best known" gateway creates a
    selection-biased dataset where weaker/riskier combinations are
    almost never observed, so the ML model trained on that data
    never learns their true failure patterns.
    """
    exclude_ids = exclude_ids or set()
    candidates = [
        gw for gw in get_gateways_for_method(method)
        if gw.id not in exclude_ids
    ]

    if not candidates:
        return None

    if random.random() < settings.exploration_epsilon:
        return random.choice(candidates)

    scored = [
        (gw, compute_effective_success_rate(gw, amount, timestamp))
        for gw in candidates
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return scored[0][0]
