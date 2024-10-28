"""
Tests for indexing_pipeline.py focusing on configuration and integration points
"""

import pytest
import logging
from haystack.document_stores.in_memory import InMemoryDocumentStore
from F1_RAG.RAG.indexing_pipeline import create_indexing_pipeline

@pytest.fixture
def document_store():
    return InMemoryDocumentStore()

def test_pipeline_structure(document_store):
    """Test that pipeline is constructed with correct components and connections"""
    pipeline = create_indexing_pipeline(document_store)
    
    # Verify all required components are present
    assert pipeline.get_component("converter") is not None
    assert pipeline.get_component("embedder") is not None
    assert pipeline.get_component("writer") is not None

def test_custom_model_configuration(document_store):
    """Test that custom model name is properly configured"""
    custom_model = "custom/model-name"
    pipeline = create_indexing_pipeline(document_store, model_name=custom_model)
    
    embedder = pipeline.get_component("embedder")
    assert embedder.model == custom_model

def test_logging(document_store, caplog):
    """Test that important operations are properly logged"""
    with caplog.at_level(logging.INFO):
        create_indexing_pipeline(document_store)
        
    assert "Creating indexing pipeline..." in caplog.text
    assert "Indexing pipeline created successfully" in caplog.text
