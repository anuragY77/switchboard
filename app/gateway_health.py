# app/gateway_health.py
"""
Tracks each gateway's LIVE rolling success rate in Redis, updated as
real transactions are processed. This is what lets the model react
to a gateway degrading RIGHT NOW (e.g. a bank's server starting to
fail) — something a static time-of-day formula can never capture,
because it only knows about patterns baked in ahead of time.
"""
import redis

from app.config import settings

_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
    return _redis_client


def _health_key(gateway_id: str) -> str:
    return f"health:{gateway_id}"


def record_outcome(gateway_id: str, success: bool) -> None:
    """Push the latest outcome onto this gateway's rolling window (most recent first)."""
    r = get_redis_client()
    key = _health_key(gateway_id)
    r.lpush(key, "1" if success else "0")
    r.ltrim(key, 0, settings.rolling_window_size - 1)


def get_rolling_success_rate(gateway_id: str, default: float) -> float:
    """
    Returns this gateway's current rolling success rate.
    Falls back to `default` when there's no data yet (cold start —
    e.g. right after a fresh deploy, before any live outcomes exist).
    """
    r = get_redis_client()
    key = _health_key(gateway_id)
    values = r.lrange(key, 0, settings.rolling_window_size - 1)

    if not values:
        return default

    successes = sum(int(v) for v in values)
    return successes / len(values)
