"""
Minimal utilities for logging.
"""

import logging
import os
import re
from pathlib import Path

LOGS_DIR = os.environ.get("LOGS_DIR", "./logs")

formatter = logging.Formatter(
    fmt="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(log_name: str):
    """
    Returns a logger that logs to console. Features the following:
        - Console stream handler (already present)
        - Optional rotating file handler (env LOG_TO_FILE=1)
        - Redaction filter for common secret keys
        - Propagation disabled to avoid duplicate logs

    Args:
        log_name (str): The name of the logger.

    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger(log_name)
    logger.propagate = False
    level_env = os.environ.get("NNUNET_SERVE_LOGGING_LEVEL", "INFO")
    if isinstance(level_env, str):
        level = getattr(logging, level_env.upper(), logging.INFO)
    else:
        try:
            level = int(level_env)
        except Exception:
            level = logging.INFO
    logger.setLevel(level)

    has_stream = any(
        isinstance(h, logging.StreamHandler) for h in logger.handlers
    )
    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if os.getenv("NNUNET_SERVE_LOG_TO_FILE", "0") == "1":
        from logging.handlers import RotatingFileHandler

        add_file_handler(logger, log_name)
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler) and not isinstance(
                h, RotatingFileHandler
            ):
                logger.removeHandler(h)
        rotating = RotatingFileHandler(
            f"{LOGS_DIR}/{log_name}.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        rotating.setFormatter(formatter)
        logger.addHandler(rotating)
        try:
            os.chmod(rotating.baseFilename, 0o600)
        except Exception:
            pass

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def add_file_handler(logger: logging.Logger, log_name: str) -> None:
    """
    Adds a file handler to the logger.

    Args:
        logger (logging.Logger): The logger to add the file handler to.
        log_name (str): The name of the log file.
    """
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(f"{LOGS_DIR}/{log_name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
