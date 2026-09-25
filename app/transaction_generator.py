# app/transaction_generator.py
import random
import uuid
from datetime import datetime, timedelta
from faker import Faker

from app.gateways import PaymentMethod

fake = Faker("en_IN")

METHOD_WEIGHTS = {"UPI": 0.55, "CARD": 0.25, "NETBANKING": 0.10, "WALLET": 0.10}

MERCHANT_POOL = [
    "TechMart Electronics", "QuickGrocery", "StyleHub Fashion",
    "FoodExpress", "BookNest", "TravelEase", "HomeDecor Plus",
    "FitGear Sports", "PetCare Store", "EduLearn Courses",
]


def generate_synthetic_timestamp() -> datetime:
    """
    Generates a random timestamp spread across the last 30 days and
    across all 24 hours — this is what the transaction is PRETENDING
    happened, regardless of when the simulator script is actually running.

    Why: a 15-minute real-time simulator run only ever produces
    real wall-clock timestamps within that same 15-minute window,
    so a model trained on that data never sees enough hour/day
    variance to learn time-based patterns (like night-time gateway
    outages). Synthetic timestamps let one short run cover a full
    24-hour x 7-day cycle instead.
    """
    now = datetime.utcnow()
    days_back = random.randint(0, 30)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)

    dt = now - timedelta(days=days_back)
    return dt.replace(hour=hour, minute=minute, second=second, microsecond=0)


def generate_transaction() -> dict:
    method: PaymentMethod = random.choices(
        list(METHOD_WEIGHTS.keys()), weights=list(METHOD_WEIGHTS.values()), k=1
    )[0]

    if random.random() < 0.85:
        amount = round(random.uniform(100, 5000), 2)
    else:
        amount = round(random.uniform(5000, 100000), 2)

    synthetic_ts = generate_synthetic_timestamp()

    return {
        "transaction_id": str(uuid.uuid4()),
        "idempotency_key": str(uuid.uuid4()),
        "merchant_id": f"merch_{random.randint(1000, 1050)}",
        "merchant_name": random.choice(MERCHANT_POOL),
        "customer_id": f"cust_{random.randint(10000, 99999)}",
        "amount": amount,
        "currency": "INR",
        "method": method,
        "card_last4": fake.credit_card_number()[-4:] if method == "CARD" else None,
        "created_at": synthetic_ts.isoformat(),  # <-- ab synthetic, real wall-clock nahi
    }