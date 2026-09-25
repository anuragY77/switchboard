# app/idempotency.py
import redis
import logging

from app.config import settings

logger = logging.getLogger(__name__)

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


IDEMPOTENCY_TTL_SECONDS = 60 * 60 * 24  # 24 hours — matches typical payment-industry idempotency window


def is_duplicate(idempotency_key: str) -> bool:
    """
    Checks if this idempotency_key has already been seen.
    Uses Redis SETNX (SET if Not eXists) — this is atomic, so two
    concurrent requests with the same key can never both pass this check.
    This is THE mechanism that prevents double-charging a customer.
    """
    r = get_redis_client()
    key = f"idem:{idempotency_key}"

    # SET with NX (only if not exists) + EX (expiry) — atomic operation
    was_set = r.set(key, "processing", nx=True, ex=IDEMPOTENCY_TTL_SECONDS)

    if was_set:
        # We just claimed this key — NOT a duplicate, safe to process
        return False
    else:
        # Key already existed — this IS a duplicate
        logger.warning(f"Duplicate idempotency_key detected: {idempotency_key}")
        return True


def mark_processed(idempotency_key: str, transaction_id: str) -> None:
    """After successful processing, store the final transaction_id
    against the idempotency key so future duplicate calls can look up the result."""
    r = get_redis_client()
    key = f"idem:{idempotency_key}"
    r.set(key, transaction_id, ex=IDEMPOTENCY_TTL_SECONDS)