"""
Minimal tests for RAG pipeline configuration validation
"""

import pytest
from unittest.mock import patch
from haystack.document_stores.in_memory import InMemoryDocumentStore
from F1_RAG.RAG.RAG_pipeline import create_rag_pipeline

@pytest.fixture
def document_store():
    return InMemoryDocumentStore()

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables needed for the pipeline"""
    with patch.dict('os.environ', {'HUGGINGFACE_API_KEY': 'mock-api-key'}):
        yield

def test_missing_required_params():
    """Test that missing required parameters raise appropriate errors"""
    with pytest.raises(ValueError, match="document_store is required"):
        create_rag_pipeline(prompt_template="test")

    with pytest.raises(ValueError, match="prompt_template is required"):
        create_rag_pipeline(document_store=InMemoryDocumentStore())