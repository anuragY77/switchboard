# app/database.py
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

from app.config import settings

engine = create_engine(settings.postgres_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Transaction(Base):
    """
    Core transaction ledger. This IS the state machine:
    status moves: pending -> routing -> success/failed -> reconciled
    """
    __tablename__ = "transactions"

    transaction_id = Column(String, primary_key=True)
    idempotency_key = Column(String, unique=True, nullable=False, index=True)
    merchant_id = Column(String, nullable=False, index=True)
    merchant_name = Column(String)
    customer_id = Column(String)
    amount = Column(Float, nullable=False)
    currency = Column(String, default="INR")
    method = Column(String, nullable=False)

    status = Column(String, default="pending", index=True)
    # pending -> routing -> success | failed -> reconciled

    chosen_gateway_id = Column(String, nullable=True)
    attempt_count = Column(Integer, default=0)
    last_decline_code = Column(String, nullable=True)
    last_latency_ms = Column(Integer, nullable=True)

    attempts_log = Column(Text, nullable=True)  # JSON string of all attempts made

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    reconciled_at = Column(DateTime, nullable=True)


def init_db():
    """Creates all tables if they don't exist yet."""
    Base.metadata.create_all(bind=engine)


def get_db_session():
    """Yields a DB session, ensures it's closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()