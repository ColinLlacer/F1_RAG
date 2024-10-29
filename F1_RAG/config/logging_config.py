"""Unified logging configuration for F1 RAG system."""

import logging
import logging.config
import sys
from pathlib import Path
import warnings

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

warnings.filterwarnings("ignore", message="PipelineMaxLoops is deprecated*")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "%(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple", 
            "stream": sys.stdout,
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "default",
            "filename": LOGS_DIR / "f1_rag.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "wiki_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "default",
            "filename": LOGS_DIR / "wiki_downloader.log",
            "maxBytes": 10485760,
            "backupCount": 5,
            "encoding": "utf8"
        }
    },
    "loggers": {
        "F1_RAG": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "propagate": False
        },
        "F1_RAG.wiki_downloader": {
            "level": "INFO",
            "handlers": ["console", "wiki_file"],
            "propagate": False
        },
        "haystack": {
            "level": "INFO", 
            "handlers": ["console", "file"],
            "propagate": False
        },
        "urllib3": {
            "level": "WARNING",
            "handlers": ["console"],
            "propagate": False
        }
    }
}

def setup_logging():
    """Initialize logging configuration."""
    try:
        logging.config.dictConfig(LOGGING_CONFIG)
    except Exception as e:
        print(f"Error setting up logging configuration: {str(e)}")
        sys.exit(1) 