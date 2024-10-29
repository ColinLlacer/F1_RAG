import logging
from pathlib import Path

# Create base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
ARTICLES_DIR = DATA_DIR / 'articles'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
ARTICLES_DIR.mkdir(exist_ok=True)

CONFIG = {
    'USER_AGENT': 'F1_RAG (your-email@gmail.com)',
    'LANGUAGE': 'en',
    'MAX_DEPTH': 3,  # How deep to traverse subcategories
    'CATEGORY': 'Formula_One_races',
    'ARTICLES_DIR': ARTICLES_DIR
}