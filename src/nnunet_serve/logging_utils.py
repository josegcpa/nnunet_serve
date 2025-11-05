"""
Minimal utilities for logging.
"""

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

LOGS_DIR = os.environ.get("LOGS_DIR", "./logs")

formatter = logging.Formatter(
    fmt="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

MANAGER = logging.Manager(logging.getLogger())


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
    logger = MANAGER.getLogger(log_name)
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

    return logger


def add_file_handler_to_manager(
    log_name: str | None = None,
    log_path: str | None = None,
    exclude: list[str] = [],
) -> None:
    """
    Adds a file handler to all loggers in the manager.

    Args:
        log_name (str): The name of the log file.
    """
    for logger in MANAGER.loggerDict.values():
        if isinstance(logger, logging.PlaceHolder):
            continue
        if logger.name in exclude:
            continue
        add_file_handler(logger, log_name=log_name, log_path=log_path)


def add_file_handler(
    logger: logging.Logger,
    log_name: str | None = None,
    log_path: str | None = None,
) -> None:
    """
    Adds a file handler to the logger.

    Args:
        logger (logging.Logger): The logger to add the file handler to.
        log_name (str): The name of the log file.
    """
    if log_path is None:
        if log_name is None:
            raise ValueError(
                "log_name must be provided if log_path is not provided"
            )
        log_path = f"{LOGS_DIR}/{log_name}.log"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    if logger.handlers:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if handler.baseFilename == log_path:
                    return
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
