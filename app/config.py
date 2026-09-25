# app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_transactions: str = "transactions_raw"
    redis_host: str = "localhost"
    redis_port: int = 6379
    postgres_url: str
    simulator_tps: float = 2.0
    exploration_epsilon: float = 0.15   # <-- naya: 15% random exploration

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()