"""
Shared feature-engineering logic used by BOTH training and evaluation,
so there is no train/eval mismatch (a common source of silent bugs in
ML systems — if training computes a feature one way and evaluation
computes it slightly differently, results become meaningless).
"""
import bisect
from collections import defaultdict

import pandas as pd

FEATURE_COLS = [
    "gateway_id_enc",
    "method_enc",
    "amount",
    "hour_of_day",
    "day_of_week",
    "is_night",
    "is_high_value",
    "recent_success_rate",
]


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if 1 <= h < 5 else 0)
    df["is_high_value"] = df["amount"].apply(lambda a: 1 if a >= 30000 else 0)
    return df


class GatewayRollingHistory:
    """
    For each gateway, keeps a chronologically sorted list of
    (timestamp, success) outcomes. Given any timestamp, answers:
    "what was this gateway's rolling success rate using ONLY outcomes
    that happened strictly BEFORE this timestamp?"

    This point-in-time correctness matters a lot: if we let a
    transaction's own outcome (or a future one) leak into its own
    "recent success rate" feature, the model would appear to perform
    far better than it actually would in a real live deployment.
    """

    def __init__(self, df: pd.DataFrame, window: int = 50):
        self.window = window
        self._history: dict[str, list[tuple]] = defaultdict(list)
        for gw_id, group in df.sort_values("created_at").groupby("gateway_id"):
            self._history[gw_id] = list(zip(group["created_at"].tolist(), group["success"].tolist()))

    def rolling_rate_before(self, gateway_id: str, timestamp, default: float) -> float:
        history = self._history.get(gateway_id, [])
        if not history:
            return default

        timestamps = [h[0] for h in history]
        idx = bisect.bisect_left(timestamps, timestamp)
        window_slice = history[max(0, idx - self.window):idx]

        if not window_slice:
            return default

        successes = [s for _, s in window_slice]
        return sum(successes) / len(successes)


def add_rolling_feature(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    df = df.copy()
    history = GatewayRollingHistory(df, window=window)
    default_rate = df["success"].mean()

    df["recent_success_rate"] = df.apply(
        lambda row: history.rolling_rate_before(row["gateway_id"], row["created_at"], default=default_rate),
        axis=1,
    )
    return df
