# app/data_loader.py
import json
import logging

import pandas as pd

from app.database import SessionLocal, Transaction

logger = logging.getLogger(__name__)


def load_attempt_level_data(limit: int | None = None) -> pd.DataFrame:
    """
    Flattens each transaction's attempts_log JSON into one row per
    gateway attempt — this is the unit both training and evaluation
    need: "if this transaction had gone to this specific gateway at
    this specific time, did it succeed?"
    """
    db = SessionLocal()
    try:
        query = db.query(Transaction).filter(Transaction.attempts_log.isnot(None))
        if limit:
            query = query.limit(limit)
        transactions = query.all()

        rows = []
        for txn in transactions:
            try:
                attempts = json.loads(txn.attempts_log)
            except (json.JSONDecodeError, TypeError):
                continue

            for attempt in attempts:
                rows.append({
                    "gateway_id": attempt["gateway_id"],
                    "method": txn.method,
                    "amount": txn.amount,
                    "created_at": txn.created_at,
                    "hour_of_day": txn.created_at.hour,
                    "day_of_week": txn.created_at.weekday(),
                    "success": 1 if attempt["success"] else 0,
                })

        logger.info(f"Loaded {len(transactions)} transactions -> {len(rows)} attempt-level rows")
        return pd.DataFrame(rows)
    finally:
        db.close()
