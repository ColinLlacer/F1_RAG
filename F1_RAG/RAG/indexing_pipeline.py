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

import os
import logging
from typing import Optional, Any

from haystack import Pipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

logger = logging.getLogger(__name__)

def create_indexing_pipeline(document_store: Optional[Any] = None,
                           model_name: str = "intfloat/e5-large-v2",
                           draw_pipeline: bool = False) -> Pipeline:
    """
    Creates and configures an indexing pipeline for document processing.
    
    Args:
        document_store: Optional document store instance that implements required Haystack 
                       document store interface (write_documents method). If None, creates 
                       a new InMemoryDocumentStore
        model_name: Name of the sentence transformer model to use for embeddings
        draw_pipeline: Whether to save a visualization of the pipeline
        
    Returns:
        Configured Pipeline instance
        
    Raises:
        ValueError: If HUGGINGFACE_API_KEY environment variable is not set
        TypeError: If document_store doesn't implement required interface
        RuntimeError: If pipeline component initialization fails
    """
    try:
        logger.info("Creating indexing pipeline...")
        
        if document_store is not None:
            # Validate document store interface, since BaseDocumentStore doesn't exist in Haystack 2.0
            required_methods = ['write_documents']
            missing_methods = [method for method in required_methods 
                             if not hasattr(document_store, method)]
            
            if missing_methods:
                logger.error(f"Document store missing required methods: {missing_methods}")
                raise TypeError(
                    f"Document store must implement the following methods: {required_methods}. "
                    f"Missing methods: {missing_methods}"
                )
        else:
            logger.debug("No document store provided, creating new InMemoryDocumentStore")
            document_store = InMemoryDocumentStore()
        
        huggingface_token = os.getenv("HUGGINGFACE_API_KEY")
        if not huggingface_token:
            logger.error("HUGGINGFACE_API_KEY environment variable not set")
            raise ValueError("HUGGINGFACE_API_KEY environment variable must be set")
            
        logger.debug("Initializing pipeline components")
        try:
            converter = TextFileToDocument()
            embedder = SentenceTransformersDocumentEmbedder(
                model=model_name,
                token=huggingface_token
            )
            writer = DocumentWriter(document_store=document_store)
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {str(e)}")
            raise RuntimeError(f"Pipeline component initialization failed: {str(e)}")

        logger.debug("Creating pipeline and connecting components")
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("converter", converter)
        indexing_pipeline.add_component("embedder", embedder)
        indexing_pipeline.add_component("writer", writer)

        indexing_pipeline.connect("converter", "embedder")
        indexing_pipeline.connect("embedder", "writer")

        if draw_pipeline:
            logger.info("Generating pipeline visualization")
            indexing_pipeline.draw('pipeline.png')
            
        logger.info("Indexing pipeline created successfully")
        return indexing_pipeline
        
    except Exception as e:
        logger.error(f"Unexpected error creating indexing pipeline: {str(e)}")
        raise
