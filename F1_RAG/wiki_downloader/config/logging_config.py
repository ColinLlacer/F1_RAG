import logging
from .defaults import CONFIG

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=CONFIG['LOG_LEVEL'],
        format=CONFIG['LOG_FORMAT'],
        handlers=[
            logging.FileHandler(CONFIG['LOG_FILE']),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)