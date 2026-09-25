# app/processor_service.py
import json
import time
import logging
from datetime import datetime

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from app.config import settings
from app.database import init_db, SessionLocal, Transaction
from app.idempotency import is_duplicate, mark_processed
from app.router import select_best_gateway
from app.retry_policy import get_backoff_delay, should_retry, MAX_ATTEMPTS
from app.gateways import simulate_gateway_attempt, GATEWAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SWITCHBOARD-PROCESSOR] %(message)s")
logger = logging.getLogger(__name__)


def get_kafka_consumer() -> KafkaConsumer:
    for attempt in range(5):
        try:
            return KafkaConsumer(
                settings.kafka_topic_transactions,
                bootstrap_servers=settings.kafka_bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                group_id="switchboard-processor-group",
            )
        except NoBrokersAvailable:
            logger.warning(f"Redpanda not ready, retrying... ({attempt + 1}/5)")
            time.sleep(3)
    raise ConnectionError("Could not connect to Redpanda after 5 attempts.")


def process_transaction(txn: dict) -> None:
    """
    Runs one transaction through the full state machine:
    pending -> routing -> (attempt gateway, retry on failure) -> success/failed
    """
    idempotency_key = txn["idempotency_key"]

    # --- Idempotency check (Step 2's function) ---
    if is_duplicate(idempotency_key):
        logger.info(f"Skipping duplicate transaction (idem_key={idempotency_key[:8]}...)")
        return

    db = SessionLocal()
    attempts_log = []
    excluded_gateways: set[str] = set()

    try:
        # --- Create the record: status=pending ---
        record = Transaction(
            transaction_id=txn["transaction_id"],
            idempotency_key=idempotency_key,
            merchant_id=txn["merchant_id"],
            merchant_name=txn["merchant_name"],
            customer_id=txn["customer_id"],
            amount=txn["amount"],
            currency=txn["currency"],
            method=txn["method"],
            status="pending",
        )
        db.add(record)
        db.commit()

        # --- Move to routing ---
        record.status = "routing"
        db.commit()

        final_status = "failed"
        chosen_gateway_id = None

        for attempt_number in range(1, MAX_ATTEMPTS + 1):
            gateway = select_best_gateway(
                method=txn["method"],
                amount=txn["amount"],
                exclude_ids=excluded_gateways,
            )

            if gateway is None:
                logger.error(f"No available gateway for method={txn['method']}")
                break

            result = simulate_gateway_attempt(gateway, txn["amount"], datetime.utcnow())
            attempts_log.append({"attempt": attempt_number, **result})

            record.attempt_count = attempt_number
            record.last_latency_ms = result["latency_ms"]

            if result["success"]:
                final_status = "success"
                chosen_gateway_id = gateway.id
                break
            else:
                record.last_decline_code = result["decline_code"]
                excluded_gateways.add(gateway.id)  # don't retry same failing gateway

                if should_retry(attempt_number):
                    delay = get_backoff_delay(attempt_number)
                    logger.info(
                        f"Attempt {attempt_number} failed on {gateway.name} "
                        f"({result['decline_code']}), retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.warning(f"Max attempts reached for txn {txn['transaction_id'][:8]}...")

        # --- Final state ---
        record.status = final_status
        record.chosen_gateway_id = chosen_gateway_id
        record.attempts_log = json.dumps(attempts_log)
        db.commit()

        mark_processed(idempotency_key, txn["transaction_id"])

        logger.info(
            f"Transaction {txn['transaction_id'][:8]}... -> {final_status.upper()} "
            f"(gateway={chosen_gateway_id}, attempts={len(attempts_log)})"
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Error processing transaction {txn.get('transaction_id', '?')[:8]}...: {e}")
    finally:
        db.close()


def run_processor():
    """Main loop: consumes from Redpanda, processes each transaction through the state machine."""
    init_db()
    consumer = get_kafka_consumer()

    logger.info("Switchboard processor started. Listening on topic 'transactions_raw'...")

    try:
        for message in consumer:
            txn = message.value
            process_transaction(txn)
    except KeyboardInterrupt:
        logger.info("Processor stopped.")
    finally:
        consumer.close()


if __name__ == "__main__":
    run_processor()