# app/retry_policy.py

MAX_ATTEMPTS = 3
BASE_DELAY_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0


def get_backoff_delay(attempt_number: int) -> float:
    """
    attempt_number starts at 1.
    attempt 1 -> 1s, attempt 2 -> 2s, attempt 3 -> 4s
    """
    return BASE_DELAY_SECONDS * (BACKOFF_MULTIPLIER ** (attempt_number - 1))


def should_retry(attempt_number: int) -> bool:
    return attempt_number < MAX_ATTEMPTS