"""
Tests for document_processor.py

This module contains tests for the document processing functionality, including:
- Document validation
- Batch processing
- Directory scanning
- Error handling
"""

import pytest
from pathlib import Path
from unittest.mock import Mock
from haystack import Pipeline

from F1_RAG.RAG.document_processor import (
    validate_documents,
    process_batch,
    index_documents,
    get_documents_from_directory
)

# Test fixtures
@pytest.fixture
def mock_pipeline():
    return Mock(spec=Pipeline)

@pytest.fixture
def temp_directory(tmp_path):
    # Create test files
    test_file1 = tmp_path / "test1.txt"
    test_file1.write_text("Test content 1")
    test_file2 = tmp_path / "test2.txt"
    test_file2.write_text("Test content 2")
    
    # Create subdirectory with files
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    test_file3 = subdir / "test3.txt"
    test_file3.write_text("Test content 3")
    
    return tmp_path

# Test validate_documents
def test_validate_documents_with_valid_files(temp_directory):
    files = [temp_directory / "test1.txt", temp_directory / "test2.txt"]
    validated = validate_documents(files)
    assert len(validated) == 2
    assert all(isinstance(path, str) for path in validated)

def test_validate_documents_with_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        validate_documents([Path("nonexistent.txt")])

def test_validate_documents_with_directory(temp_directory):
    with pytest.raises(FileNotFoundError):
        validate_documents([temp_directory])  # Directory instead of file

# Test process_batch
def test_process_batch_success(mock_pipeline):
    batch = ["doc1.txt", "doc2.txt"]
    process_batch(mock_pipeline, batch)
    mock_pipeline.run.assert_called_once_with({"sources": batch})

def test_process_batch_failure(mock_pipeline):
    mock_pipeline.run.side_effect = Exception("Pipeline error")
    with pytest.raises(RuntimeError):
        process_batch(mock_pipeline, ["doc1.txt"])

# Test index_documents
def test_index_documents_empty_list(mock_pipeline):
    index_documents(mock_pipeline, [], batch_size=2)
    mock_pipeline.run.assert_not_called()

def test_index_documents_invalid_batch_size(mock_pipeline, temp_directory):
    files = [temp_directory / "test1.txt"]
    with pytest.raises(ValueError):
        index_documents(mock_pipeline, files, batch_size=0)

def test_index_documents_successful_batching(mock_pipeline, temp_directory):
    files = [
        temp_directory / "test1.txt",
        temp_directory / "test2.txt",
        temp_directory / "subdir" / "test3.txt"
    ]
    index_documents(mock_pipeline, files, batch_size=2)
    assert mock_pipeline.run.call_count == 2  # Should make 2 calls with batch_size=2

# Test get_documents_from_directory
def test_get_documents_from_directory_recursive(temp_directory):
    docs = get_documents_from_directory(temp_directory, pattern="*.txt")
    assert len(docs) == 3  # Should find all 3 text files

def test_get_documents_from_directory_non_recursive(temp_directory):
    docs = get_documents_from_directory(temp_directory, pattern="*.txt", recursive=False)
    assert len(docs) == 2  # Should only find top-level text files

def test_get_documents_from_directory_nonexistent():
    with pytest.raises(FileNotFoundError):
        get_documents_from_directory(Path("nonexistent_dir"))

def test_get_documents_from_directory_not_directory(temp_directory):
    with pytest.raises(NotADirectoryError):
        get_documents_from_directory(temp_directory / "test1.txt")

def test_get_documents_from_directory_no_matches(temp_directory):
    docs = get_documents_from_directory(temp_directory, pattern="*.pdf")
    assert len(docs) == 0
