"""
Tests for main.py focusing on RAGSystem initialization and error handling
"""

import pytest
from unittest.mock import Mock, patch
from F1_RAG.RAG.main import RAGSystem

@pytest.fixture
def temp_docs_dir(tmp_path):
    # Create a test document
    doc_file = tmp_path / "test.txt"
    doc_file.write_text("Test content")
    return tmp_path

def test_initialization_with_empty_directory(temp_docs_dir):
    """Test system initialization with empty directory"""
    with patch('F1_RAG.RAG.main.get_documents_from_directory', return_value=[]):
        
        rag_system = RAGSystem(docs_dir=str(temp_docs_dir))
        rag_system.initialize()
        
        assert rag_system.rag_pipeline is None

def test_query_without_initialization(temp_docs_dir):
    """Test querying before initialization"""
    rag_system = RAGSystem(docs_dir=str(temp_docs_dir))
    
    with pytest.raises(RuntimeError, match="RAG pipeline not initialized"):
        rag_system.query("test question")

def test_successful_query(temp_docs_dir):
    """Test successful query processing"""
    # Create a test document file
    test_doc = temp_docs_dir / "test_doc.txt"
    test_doc.write_text("Test content")
    
    with patch('F1_RAG.RAG.main.create_rag_pipeline') as mock_create_pipeline, \
         patch('F1_RAG.RAG.main.get_documents_from_directory') as mock_get_docs:
        
        # Mock documents to return the actual test file path
        mock_get_docs.return_value = [test_doc]
        
        # Create a mock pipeline that returns a specific response
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {
            "generator": {
                "replies": ["Test answer"]
            }
        }
        mock_create_pipeline.return_value = mock_pipeline
        
        # Create and initialize RAG system
        rag_system = RAGSystem(docs_dir=str(temp_docs_dir))
        rag_system.initialize()
        
        # Test the query
        result = rag_system.query("test question")
        assert result == "Test answer"
        
        # Verify the pipeline was called correctly
        mock_pipeline.run.assert_called_once()

def test_query_error_handling(temp_docs_dir):
    """Test error handling during query processing"""
    test_doc = temp_docs_dir / "test.txt"
    
    with patch('F1_RAG.RAG.main.create_rag_pipeline') as mock_create_pipeline, \
         patch('F1_RAG.RAG.main.get_documents_from_directory', return_value=[str(test_doc)]), \
         patch('F1_RAG.RAG.main.index_documents') as mock_index:
        
        # Configure mock pipeline to raise an exception
        mock_pipeline = Mock()
        mock_pipeline.run.side_effect = Exception("Pipeline error")
        mock_create_pipeline.return_value = mock_pipeline
        
        # Initialize and query
        rag_system = RAGSystem(docs_dir=str(temp_docs_dir))
        rag_system.initialize()
        answer = rag_system.query("test question")
        
        assert answer is None
        # Verify index_documents was called
        mock_index.assert_called_once()

def test_custom_configuration(temp_docs_dir):
    """Test RAGSystem initialization with custom configuration"""
    custom_embedder = "custom/embedder"
    custom_llm = "custom/llm"
    custom_batch_size = 64
    
    rag_system = RAGSystem(
        docs_dir=str(temp_docs_dir),
        embedder_model=custom_embedder,
        llm_model=custom_llm,
        batch_size=custom_batch_size
    )
    
    assert rag_system.embedder_model == custom_embedder
    assert rag_system.llm_model == custom_llm
    assert rag_system.batch_size == custom_batch_size