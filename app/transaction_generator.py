import random
import uuid
from datetime import datetime
from faker import Faker

from app.gateways import PaymentMethod

fake = Faker("en_IN")

METHOD_WEIGHTS = {"UPI": 0.55, "CARD": 0.25, "NETBANKING": 0.10, "WALLET": 0.10}

MERCHANT_POOL = [
    "TechMart Electronics", "QuickGrocery", "StyleHub Fashion",
    "FoodExpress", "BookNest", "TravelEase", "HomeDecor Plus",
    "FitGear Sports", "PetCare Store", "EduLearn Courses",
]


def generate_transaction() -> dict:
    method: PaymentMethod = random.choices(
        list(METHOD_WEIGHTS.keys()), weights=list(METHOD_WEIGHTS.values()), k=1
    )[0]

    if random.random() < 0.85:
        amount = round(random.uniform(100, 5000), 2)
    else:
        amount = round(random.uniform(5000, 100000), 2)

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
        "created_at": datetime.utcnow().isoformat(),
    }