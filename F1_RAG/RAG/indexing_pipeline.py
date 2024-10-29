"""
Document Indexing Pipeline Module

This module provides functionality to create and configure document indexing pipelines
using Haystack components. The pipeline processes text documents by:

1. Converting text files to document objects
2. Generating embeddings using sentence transformers
3. Writing processed documents to a document store

The module uses configuration from environment variables including:
- HUGGINGFACE_API_KEY: API key for accessing Hugging Face models

Components:
- TextFileToDocument: Converts raw text files to Haystack document objects
- SentenceTransformersDocumentEmbedder: Generates document embeddings
- DocumentWriter: Writes processed documents to document store

The pipeline can optionally generate a visualization of its structure.
"""

import logging

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

# Configure logging
logger = logging.getLogger("F1_RAG.RAG.indexing_pipeline")

def create_indexing_pipeline(
    document_store: InMemoryDocumentStore,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Pipeline:
    """
    Create indexing pipeline for document processing.
    
    Args:
        document_store: Document store instance
        model_name: Name of the embedding model
        
    Returns:
        Configured indexing pipeline
        
    Raises:
        RuntimeError: If pipeline creation fails
    """
    try:
        logger.info("Creating indexing pipeline...")
        
        # Initialize pipeline components
        try:
            text_converter = TextFileToDocument()
            embedder = SentenceTransformersDocumentEmbedder(model=model_name)
            writer = DocumentWriter(document_store=document_store)
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {str(e)}")
            raise RuntimeError(f"Pipeline component initialization failed: {str(e)}")

        # Create pipeline
        try:
            indexing_pipeline = Pipeline()
            indexing_pipeline.add_component("converter", text_converter)
            indexing_pipeline.add_component("embedder", embedder)
            indexing_pipeline.add_component("writer", writer)

            # Connect components
            indexing_pipeline.connect("converter", "embedder")
            indexing_pipeline.connect("embedder", "writer")
        except Exception as e:
            logger.error(f"Failed to create pipeline: {str(e)}")
            raise RuntimeError(f"Unexpected error creating indexing pipeline: {str(e)}")

        logger.info("Indexing pipeline created successfully")
        return indexing_pipeline
        
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating indexing pipeline: {str(e)}")
        raise RuntimeError(f"Unexpected error creating indexing pipeline: {str(e)}")
