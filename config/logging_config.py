"""
Logging configuration for Chennai Flood Access Risk Mapper.
Provides structured console and file logging with Windows path compatibility.
"""
import os
import sys
import logging
import logging.config
from pathlib import Path
from typing import Optional


LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def get_logging_dict(log_level: str = "INFO", log_to_file: bool = True) -> dict:
    """Build logging configuration dictionary."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = str(LOGS_DIR / "floodmap.log")

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s %(process)d:%(threadName)s %(pathname)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "console": {
                "format": "%(levelname)-8s %(name)s: %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "console",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console"],
                "level": log_level,
                "propagate": True,
            },
            "floodmap": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            "drainwatch": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
        },
    }

    if log_to_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "standard",
            "filename": log_file_path,
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "encoding": "utf-8",
        }
        config["loggers"][""]["handlers"].append("file")
        config["loggers"]["floodmap"]["handlers"].append("file")
        config["loggers"]["drainwatch"]["handlers"].append("file")

    return config


def setup_logging(
    level: Optional[str] = None,
    log_to_file: bool = True,
) -> None:
    """Initialize logging subsystem."""
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level not in valid_levels:
        log_level = "INFO"

    config_dict = get_logging_dict(log_level=log_level, log_to_file=log_to_file)
    logging.config.dictConfig(config_dict)


def get_logger(name: str = "floodmap") -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)
