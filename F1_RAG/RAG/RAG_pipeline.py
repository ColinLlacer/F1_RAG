"""
RAG Pipeline Module

This module provides functionality to create and configure a Retrieval Augmented Generation (RAG) pipeline.
The pipeline combines document retrieval with language model generation to provide accurate, context-aware responses.

Key components:
- SentenceTransformersTextEmbedder: Generates embeddings for input queries
- InMemoryEmbeddingRetriever: Retrieves relevant documents based on embeddings
- PromptBuilder: Constructs prompts combining query and retrieved context
- HuggingFaceAPIGenerator: Generates responses using LLM

The module uses configuration from environment variables including:
- HUGGINGFACE_API_KEY: API key for accessing Hugging Face models

Important:
- The LLM model must be available on the Hugging Face Serverless Inference API
- See https://huggingface.co/inference-api for available models

The pipeline performs the following steps:
1. Embeds user query using sentence transformers
2. Retrieves relevant documents from document store
3. Builds prompt combining query and retrieved context
4. Generates response using LLM via Hugging Face Serverless Inference API
"""


from typing import Optional
import logging

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder 
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack import Pipeline
from haystack.utils import Secret

logger = logging.getLogger(__name__)

def create_rag_pipeline(
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    llm_model: str = "HuggingFaceH4/zephyr-7b-beta",
    document_store: Optional[object] = None,
    prompt_template: str = None,
    top_k: int = 3 #1-3 to limit the number of input tokens with the free serverless inference API
) -> Pipeline:
    """
    Creates and configures a RAG (Retrieval Augmented Generation) pipeline.
    
    Args:
        embedder_model (str): Name of the sentence transformer model for embeddings
        llm_model (str): Name of the LLM model to use for generation. Must be available on 
                        Hugging Face Serverless Inference API
        document_store (object): Document store containing the knowledge base
        prompt_template (str): Template string for prompt construction
        
    Returns:
        Pipeline: Configured RAG pipeline
        
    Raises:
        ValueError: If required parameters are missing
        RuntimeError: If pipeline creation or connection fails, or if LLM model is not
                     available on Hugging Face Serverless Inference API
    """
    
    # Validate inputs
    if document_store is None:
        raise ValueError("document_store is required")
    if prompt_template is None:
        raise ValueError("prompt_template is required")
        
    try:
        # Initialize components
        query_embedder = SentenceTransformersTextEmbedder(model=embedder_model)
        retriever = InMemoryEmbeddingRetriever(
            document_store=document_store,
            top_k=top_k  # Add this parameter to limit retrieved documents
        )
        prompt_builder = PromptBuilder(template=prompt_template)
        
        generator = HuggingFaceAPIGenerator(
            api_type="serverless_inference_api",  # Using the free Serverless Inference API
            api_params={
                "model": llm_model,
            },
            token=Secret.from_env_var("HUGGINGFACE_API_KEY"),
            generation_kwargs={
                "max_new_tokens": 512,
                "temperature": 0.7,
                "do_sample": True,
            }
        )
        
        # Create pipeline
        rag_pipeline = Pipeline()
        rag_pipeline.add_component("query_embedder", query_embedder)
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("generator", generator)

        # Connect components
        rag_pipeline.connect("query_embedder", "retriever")
        rag_pipeline.connect("retriever", "prompt_builder")
        rag_pipeline.connect("prompt_builder", "generator")

        logger.info("RAG pipeline created successfully")
        return rag_pipeline

    except Exception as e:
        logger.error(f"Failed to create RAG pipeline: {str(e)}")
        raise RuntimeError(f"Pipeline creation failed: {str(e)}")
