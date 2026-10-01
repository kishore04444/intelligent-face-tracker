import logging
import os

LOG_PATH = "logs/events.log"

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("face_tracker")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_event(message):
    logger.info(message)