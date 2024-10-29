"""
Main Module for F1 RAG System

This module orchestrates the complete RAG system workflow:
1. Creates document store and indexing pipeline
2. Processes and indexes documents from a directory
3. Creates and configures RAG pipeline for querying
4. Provides interface for asking questions

The module uses components from:
- indexing_pipeline.py: For document processing and indexing
- document_processor.py: For document handling and batch processing
- RAG_pipeline.py: For question answering using RAG

Note: The LLM model must be available on the Hugging Face Serverless Inference API.
See https://huggingface.co/inference-api for available models.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from haystack.document_stores.in_memory import InMemoryDocumentStore
from F1_RAG.RAG.indexing_pipeline import create_indexing_pipeline
from F1_RAG.RAG.document_processor import get_documents_from_directory, index_documents
from F1_RAG.RAG.RAG_pipeline import create_rag_pipeline
from F1_RAG.RAG.config.config import (
    prompt,
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TOP_K
)
from F1_RAG.config.logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger("F1_RAG.RAG.main")

# Load environment variables from .env file
load_dotenv()

class RAGSystem:
    def __init__(
        self,
        docs_dir: str,
        embedder_model: str = DEFAULT_EMBEDDER_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        """
        Initialize RAG system with configuration parameters.
        
        Args:
            docs_dir: Directory containing documents to index
            embedder_model: Model name for embeddings
            llm_model: Model name for LLM (must be available on HF Serverless Inference API)
            batch_size: Batch size for document processing
        """
        self.docs_dir = Path(docs_dir)
        self.embedder_model = embedder_model
        self.llm_model = llm_model
        self.batch_size = batch_size
        self.document_store = InMemoryDocumentStore()
        self.rag_pipeline = None

    def initialize(self) -> None:
        """
        Initialize the system by indexing documents and creating RAG pipeline.
        
        Raises:
            FileNotFoundError: If documents directory doesn't exist
            RuntimeError: If pipeline creation fails or if LLM model is not available
                         on Hugging Face Serverless Inference API
        """
        try:
            logger.info("Initializing RAG system...")
            
            # Create and run indexing pipeline
            indexing_pipeline = create_indexing_pipeline(
                document_store=self.document_store,
                model_name=self.embedder_model
            )
            
            # Get and process documents
            documents = get_documents_from_directory(self.docs_dir)
            if not documents:
                logger.warning("No documents found to index")
                return
                
            index_documents(indexing_pipeline, documents, self.batch_size)
            
            # Create RAG pipeline
            self.rag_pipeline = create_rag_pipeline(
                embedder_model=self.embedder_model,
                llm_model=self.llm_model,
                document_store=self.document_store,
                prompt_template=prompt,
                top_k=DEFAULT_TOP_K
            )
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise

    def query(self, question: str) -> Optional[str]:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask the system
            
        Returns:
            Generated answer or None if processing fails
            
        Raises:
            RuntimeError: If RAG pipeline is not initialized
        """
        if not self.rag_pipeline:
            raise RuntimeError("RAG pipeline not initialized. Call initialize() first.")
            
        try:
            logger.info(f"Processing question: {question}")
            result = self.rag_pipeline.run(
                data={
                    "query_embedder": {"text": question},
                    "prompt_builder": {"query": question}
                }
            )
            # Extract and return only the replies
            replies = result.get("generator", {}).get("replies", [])
            if replies:
                return replies[0]  # Return the first reply
            else:
                logger.warning("No replies generated.")
                return None
                
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return None

def main():
    """Main function to demonstrate RAG system usage."""
    try:
        # Initialize system
        docs_dir = os.getenv("DOCS_DIR", "F1_RAG/wiki_downloader/data/articles")
        rag_system = RAGSystem(docs_dir=docs_dir)
        rag_system.initialize()
        
        # Example usage
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
                
            answer = rag_system.query(question)
            if answer:
                print(f"\n{answer}")
            else:
                print("\nFailed to generate answer")
                
    except Exception as e:
        logger.critical(f"System error: {str(e)}", exc_info=True)
        
if __name__ == "__main__":
    main()
