"""
Test module for WikiCategoryDownloader class.

This module contains unit tests for the WikiCategoryDownloader class functionality,
including category traversal, article saving, and filename sanitization.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from F1_RAG.wiki_downloader.wiki_downloader import WikiCategoryDownloader

# Test configuration
TEST_CATEGORY = "TEST_CATEGORY"
TEST_ARTICLE_TITLE = "Test Article"
TEST_ARTICLE_URL = "https://en.wikipedia.org/wiki/Test_Article"
TEST_ARTICLE_SUMMARY = "This is a test summary"
TEST_ARTICLE_TEXT = "This is the full article text"

@pytest.fixture
def mock_wiki_api():
    """Fixture to create a mock Wikipedia API."""
    mock_api = Mock()
    return mock_api

@pytest.fixture
def temp_dir(tmp_path):
    """Fixture to create a temporary directory for test files."""
    return tmp_path

@pytest.fixture
def downloader(mock_wiki_api, temp_dir):
    """Fixture to create a WikiCategoryDownloader instance."""
    return WikiCategoryDownloader(mock_wiki_api, temp_dir)

def create_mock_article(title=TEST_ARTICLE_TITLE):
    """Helper function to create a mock Wikipedia article."""
    mock_article = Mock()
    mock_article.title = title
    mock_article.fullurl = TEST_ARTICLE_URL
    mock_article.summary = TEST_ARTICLE_SUMMARY
    mock_article.text = TEST_ARTICLE_TEXT
    mock_article.ns = 0  # Regular article namespace
    return mock_article

def test_init(downloader):
    """Test initialization of WikiCategoryDownloader."""
    assert isinstance(downloader.articles_dir, Path)
    assert len(downloader.seen_pages) == 0
    assert len(downloader.seen_categories) == 0

def test_create_safe_filename():
    """Test filename sanitization."""
    test_cases = [
        ("Simple Title", "Simple_Title"),
        ("Title with !@#$%^", "Title_with"),
        ("  Spaces  ", "Spaces"),
        ("Mixed-Case-Title!", "Mixed_Case_Title"),
    ]
    
    for input_title, expected in test_cases:
        result = WikiCategoryDownloader._create_safe_filename(input_title)
        assert result == expected

def test_get_categorymembers_nonexistent(downloader):
    """Test handling of non-existent categories."""
    downloader.wiki_api.page.return_value.exists.return_value = False
    articles = downloader.get_categorymembers(TEST_CATEGORY)
    assert len(articles) == 0

def test_get_categorymembers_basic(downloader):
    """Test basic category member retrieval."""
    mock_category = Mock()
    mock_category.title = TEST_CATEGORY
    mock_category.exists.return_value = True
    
    mock_article = create_mock_article()
    mock_category.categorymembers.values.return_value = [mock_article]
    
    downloader.wiki_api.page.return_value = mock_category
    
    articles = downloader.get_categorymembers(TEST_CATEGORY, max_depth=1)
    assert len(articles) == 1
    assert articles[0].title == TEST_ARTICLE_TITLE

@patch('pathlib.Path.open', new_callable=mock_open)
def test_save_article(mock_file, downloader):
    """Test article saving functionality."""
    mock_article = create_mock_article()
    
    downloader.save_article(mock_article)
    
    # Verify file was opened with correct name
    mock_file.assert_called_once_with('w', encoding='utf-8')
    
    # Verify write calls
    handle = mock_file()
    write_calls = handle.write.call_args_list
    assert any(TEST_ARTICLE_TITLE in str(call) for call in write_calls)
    assert any(TEST_ARTICLE_URL in str(call) for call in write_calls)
    assert any(TEST_ARTICLE_SUMMARY in str(call) for call in write_calls)
    assert any(TEST_ARTICLE_TEXT in str(call) for call in write_calls)

@patch('pathlib.Path.open')
def test_save_article_io_error(mock_file, downloader):
    """Test handling of IOError during article saving."""
    mock_file.side_effect = IOError("Test error")
    mock_article = create_mock_article()
    
    # Should not raise exception but log error
    downloader.save_article(mock_article)

def test_save_all_articles(downloader):
    """Test saving multiple articles."""
    mock_articles = [create_mock_article(f"Article_{i}") for i in range(3)]
    
    with patch.object(downloader, 'save_article') as mock_save:
        downloader.save_all_articles(mock_articles)
        assert mock_save.call_count == 3

def test_duplicate_prevention(downloader):
    """Test prevention of duplicate article processing."""
    mock_category = Mock()
    mock_category.title = TEST_CATEGORY
    mock_category.exists.return_value = True
    
    # Create two identical articles
    mock_article = create_mock_article()
    mock_category.categorymembers.values.return_value = [mock_article, mock_article]
    
    downloader.wiki_api.page.return_value = mock_category
    
    articles = downloader.get_categorymembers(TEST_CATEGORY)
    assert len(articles) == 1  # Should only include one copy

if __name__ == '__main__':
    pytest.main(['-v'])
