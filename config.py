import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    WANDB_API_KEY: str = os.getenv("WANDB_API_KEY", "")
    NUM_WORKERS: int = os.getenv("NUM_WORKERS", 6)
    TOKENIZER_NAME: str = os.getenv("TOKENIZER_NAME", "gpt2")
    DATA_LENGTH: int = os.getenv("DATA_LENGTH", 128)
    BLOCK_SIZE: int = os.getenv("BLOCK_SIZE", 512)
    BATCH_SIZE: int = os.getenv("BATCH_SIZE", 16)
    MAX_EPOCHS: int = os.getenv("MAX_EPOCHS", 3)
    LR: float = os.getenv("LR", 3e-4)
    IS_RESUME: bool = os.getenv("IS_RESUME", False)
    CHECKPOINT_PATH: str = os.getenv("CHECKPOINT_PATH", "")
    MAX_TRAINING_HOURS: int = os.getenv("MAX_TRAINING_HOURS", 5)
    AWS_ENABLED: bool = os.getenv("AWS_ENABLED", False)
    AWS_ENDPOINT_URL: str = os.getenv("AWS_ENDPOINT_URL", "")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_BUCKET_NAME: str = os.getenv("AWS_BUCKET_NAME", "deep-delta-learning-lm")
    S3_MODEL_PATH: str = os.getenv("S3_MODEL_PATH", "")

    class Config:
        env_file = ".env"


@lru_cache
def get_settings():
    return Settings()
