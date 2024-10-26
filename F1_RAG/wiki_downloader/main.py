"""
Wikipedia Article Downloader Module

This module provides functionality to download articles from Wikipedia based on categories.
It uses the Wikipedia API to fetch articles and their contents, and saves them locally.

The module reads configuration from config/defaults.py including:
- USER_AGENT: User agent string for Wikipedia API requests
- LANGUAGE: Language code for Wikipedia articles
- CATEGORY: The Wikipedia category to download articles from
- MAX_DEPTH: Maximum depth to traverse category tree
- ARTICLES_DIR: Directory to save downloaded articles

This will download all articles from the configured category and save them
to the specified directory.
"""

import wikipediaapi
from config.defaults import CONFIG
from config.logging_config import setup_logging
from wiki_downloader import WikiCategoryDownloader

logger = setup_logging()

def main():
    """Main function to orchestrate the download process."""
    
    logger.info("Initializing Wikipedia API")
    wiki = wikipediaapi.Wikipedia(
        user_agent=CONFIG['USER_AGENT'],
        language=CONFIG['LANGUAGE']
    )
    
    downloader = WikiCategoryDownloader(wiki, CONFIG['ARTICLES_DIR'])
    
    logger.info(f"Fetching articles from category: {CONFIG['CATEGORY']}")
    articles = downloader.get_categorymembers(
        CONFIG['CATEGORY'], 
        max_depth=CONFIG['MAX_DEPTH']
    )
    
    logger.info(f"Found {len(articles)} articles")
    logger.info("Downloading and saving article contents...")
    
    downloader.save_all_articles(articles)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)