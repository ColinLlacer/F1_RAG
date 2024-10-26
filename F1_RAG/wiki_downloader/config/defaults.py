import logging
from pathlib import Path

# Create base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
LOG_DIR = BASE_DIR / 'logs'
ARTICLES_DIR = DATA_DIR / 'articles'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
ARTICLES_DIR.mkdir(exist_ok=True)

CONFIG = {
    'USER_AGENT': 'F1_RAG (colin.llacer@gmail.com)',
    'LANGUAGE': 'en',
    'MAX_DEPTH': 3,  # How deep to traverse subcategories
    'CATEGORY': 'Formula_One_races',
    'ARTICLES_DIR': ARTICLES_DIR,
    'LOG_LEVEL': logging.INFO,
    'LOG_FORMAT': '%(asctime)s - %(levelname)s - %(message)s',
    'LOG_FILE': LOG_DIR / 'wikipedia_scraper.log'
}