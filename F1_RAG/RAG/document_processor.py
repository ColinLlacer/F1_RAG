"""
Document Processing Module

This module handles the execution of document indexing pipelines, including:
- Document validation and path handling
- Batch processing of documents
- Error handling and logging for document processing

Components:
- index_documents: Main function to process and index documents through a pipeline
- validate_documents: Helper function to validate document paths
"""

import logging
from pathlib import Path
from typing import List, Union
from haystack import Pipeline

# Configure logging
logger = logging.getLogger(__name__)

def validate_documents(documents: List[Union[str, Path]]) -> List[str]:
    """
    Validate document paths and convert them to strings.
    
    Args:
        documents: List of document paths (strings or Path objects)
        
    Returns:
        List of validated document paths as strings
        
    Raises:
        FileNotFoundError: If any document path is invalid
    """
    doc_paths = []
    for doc in documents:
        path = Path(doc)
        if not path.exists():
            logger.error(f"Document not found: {path}")
            raise FileNotFoundError(f"Document not found: {path}")
        if not path.is_file():
            logger.error(f"Path is not a file: {path}")
            raise FileNotFoundError(f"Path is not a file: {path}")
        doc_paths.append(str(path))
    return doc_paths

def process_batch(pipeline, batch):
    """Process a batch of documents through the indexing pipeline."""
    try:
        pipeline.run({
            "sources": batch
        })
    except Exception as e:
        logger.error(f"Error during document indexing: {str(e)}")
        raise RuntimeError(f"Pipeline execution failed: {str(e)}")

def index_documents(
    pipeline: Pipeline, 
    documents: List[Union[str, Path]], 
    batch_size: int = 32
) -> None:
    """
    Index a list of documents using the provided pipeline.
    
    Args:
        pipeline: Configured indexing pipeline
        documents: List of paths to documents (can be string or Path objects)
        batch_size: Number of documents to process in each batch
        
    Raises:
        FileNotFoundError: If any document path is invalid
        RuntimeError: If pipeline execution fails
        ValueError: If batch_size is less than 1
    """
    if not documents:
        logger.warning("No documents provided for indexing")
        return
        
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
        
    try:
        logger.info(f"Starting to index {len(documents)} documents")
        
        # Validate all document paths
        doc_paths = validate_documents(documents)
        
        # Process documents in batches
        total_batches = (len(doc_paths) + batch_size - 1) // batch_size
        for i in range(0, len(doc_paths), batch_size):
            batch = doc_paths[i:i + batch_size]
            current_batch = (i // batch_size) + 1
            logger.debug(
                f"Processing batch {current_batch}/{total_batches}, "
                f"size: {len(batch)}"
            )
            
            process_batch(pipeline, batch)
            
            logger.debug(
                f"Completed batch {current_batch}/{total_batches} "
                f"({(current_batch/total_batches)*100:.1f}%)"
            )
                
        logger.info("Document indexing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during document indexing: {str(e)}")
        raise

def get_documents_from_directory(
    directory: Union[str, Path], 
    pattern: str = "*.txt",
    recursive: bool = True
) -> List[Path]:
    """
    Get all documents matching the pattern from a directory.
    
    Args:
        directory: Directory to search for documents
        pattern: Glob pattern to match files (e.g., "*.txt", "*.pdf")
        recursive: Whether to search subdirectories
        
    Returns:
        List of Path objects for matching documents
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
        
    logger.debug(f"Searching for documents matching '{pattern}' in {directory}")
    
    if recursive:
        files = directory.rglob(pattern)
    else:
        files = directory.glob(pattern)
        
    documents = sorted(files)  # Sort for consistent ordering
    logger.info(f"Found {len(documents)} documents matching pattern '{pattern}'")
    
    return documents
