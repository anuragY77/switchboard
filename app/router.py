# app/router.py
from datetime import datetime

from app.gateways import get_gateways_for_method, compute_effective_success_rate, Gateway


def select_best_gateway(method: str, amount: float, exclude_ids: set[str] | None = None) -> Gateway | None:
    """
    Picks the gateway with the highest CURRENT effective success rate
    for this method/amount/time combination.

    This is a rule-based baseline — Phase 3 replaces this function's
    internals with a trained ML model, but the function signature
    stays the same so nothing else in the codebase needs to change.
    """
    exclude_ids = exclude_ids or set()
    candidates = [
        gw for gw in get_gateways_for_method(method)
        if gw.id not in exclude_ids
    ]

    if not candidates:
        return None

    now = datetime.utcnow()
    scored = [
        (gw, compute_effective_success_rate(gw, amount, now))
        for gw in candidates
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return scored[0][0]