"""
Wikipedia Category Downloader Module

This module provides functionality to download Wikipedia articles from specified categories.
It includes a WikiCategoryDownloader class that can recursively traverse category hierarchies,
download articles, and save them to local files.

The module handles:
- Recursive category traversal with configurable depth
- Article content extraction including title, URL, summary and full text
- Safe file naming and storage
- Duplicate article and category detection
- Logging of operations

Classes:
    WikiCategoryDownloader: Main class for downloading and saving Wikipedia category articles

Dependencies:
    - wikipediaapi: For accessing Wikipedia content
    - pathlib: For file path handling
    - logging: For operation logging
    - re: For filename sanitization
"""

import wikipediaapi
import logging
from pathlib import Path
from typing import List, Set
import re

logger = logging.getLogger("F1_RAG.wiki_downloader")

class WikiCategoryDownloader:
    """Class to handle downloading Wikipedia articles from a category."""
    
    def __init__(self, wiki_api: wikipediaapi.Wikipedia, articles_dir: Path):
        """
        Initialize the downloader with a Wikipedia API instance.
        
        Args:
            wiki_api: Initialized Wikipedia API object
            articles_dir: Directory to save article files
        """
        self.wiki_api = wiki_api
        self.articles_dir = articles_dir
        self.seen_pages: Set[str] = set()
        self.seen_categories: Set[str] = set()

    def get_categorymembers(self, category_name: str, max_depth: int = 1) -> List[wikipediaapi.WikipediaPage]:
        """
        Recursively get all articles from a category and its subcategories.
        
        Args:
            category_name: Name of the category to process
            max_depth: Maximum depth to traverse subcategories
            
        Returns:
            List of Wikipedia page objects
        """
        articles = []
        category = self.wiki_api.page(f"Category:{category_name}")
        
        if not category.exists():
            logger.error(f"Category '{category_name}' does not exist!")
            return articles
            
        def process_category(category: wikipediaapi.WikipediaPage, level: int = 0) -> None:
            if level > max_depth:
                return
                
            if category.title in self.seen_categories:
                return
                
            self.seen_categories.add(category.title)
            logger.debug(f"Processing category: {category.title} at level {level}")
            
            for member in category.categorymembers.values():
                if member.ns == wikipediaapi.Namespace.CATEGORY:
                    process_category(member, level + 1)
                else:
                    if member.title not in self.seen_pages:
                        self.seen_pages.add(member.title)
                        articles.append(member)
                        logger.debug(f"Added article: {member.title}")
        
        process_category(category)
        return articles

    def save_article(self, article: wikipediaapi.WikipediaPage) -> None:
        """
        Save a single article to a file.
        
        Args:
            article: Wikipedia page object to save
        """
        # Create a safe filename from the article title
        safe_filename = self._create_safe_filename(article.title)
        file_path = self.articles_dir / f"{safe_filename}.txt"
        
        try:
            with file_path.open('w', encoding='utf-8') as f:
                f.write(f"Title: {article.title}\n")
                f.write(f"URL: {article.fullurl}\n\n")
                f.write("=== Summary ===\n")
                f.write(f"{article.summary}\n\n")
                f.write("=== Full Text ===\n")
                f.write(article.text)
            logger.debug(f"Saved article: {article.title}")
        except IOError as e:
            logger.error(f"Error saving article {article.title}: {e}")

    def save_all_articles(self, articles: List[wikipediaapi.WikipediaPage]) -> None:
        """
        Save all downloaded articles to individual files.
        
        Args:
            articles: List of Wikipedia page objects to save
        """
        logger.debug(f"Saving {len(articles)} articles to {self.articles_dir}")
        for article in articles:
            self.save_article(article)
        logger.debug("Finished saving all articles")

    @staticmethod
    def _create_safe_filename(title: str) -> str:
        """
        Create a safe filename from an article title.
        
        Args:
            title: Article title
            
        Returns:
            Safe filename string
        """
        # Remove unsafe characters and replace spaces with underscores
        safe_name = re.sub(r'[^\w\s-]', '', title)
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        return safe_name.strip('-_')