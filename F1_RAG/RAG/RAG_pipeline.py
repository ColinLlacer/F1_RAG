from typing import Optional
import logging
import os
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder 
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline
from haystack.errors import PipelineError

logger = logging.getLogger(__name__)

def create_rag_pipeline(
    embedder_model: str = "intfloat/e5-large-v2",
    llm_model: str = "groq/llama3-8b-8192",
    document_store: Optional[object] = None,
    prompt_template: str = None             
) -> Pipeline:
    """
    Creates and configures a RAG (Retrieval Augmented Generation) pipeline.
    
    Args:
        embedder_model (str): Name of the sentence transformer model for embeddings
        llm_model (str): Name of the LLM model to use for generation
        document_store (object): Document store containing the knowledge base
        prompt_template (str): Template string for prompt construction
        
    Returns:
        Pipeline: Configured RAG pipeline
        
    Raises:
        ValueError: If required parameters are missing
        PipelineError: If pipeline creation or connection fails
    """
    
    # Validate inputs
    if document_store is None:
        raise ValueError("document_store is required")
    if prompt_template is None:
        raise ValueError("prompt_template is required")
        
    try:
        # Initialize components
        query_embedder = SentenceTransformersTextEmbedder(model_name=embedder_model)
        retriever = InMemoryEmbeddingRetriever(document_store=document_store)
        prompt_builder = PromptBuilder(prompt=prompt_template)
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
            
        generator = OpenAIGenerator(model_name=llm_model, api_key=api_key)

        # Create and configure pipeline
        rag_pipeline = Pipeline()
        rag_pipeline.add_component("query_embedder", query_embedder)
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("generator", generator)

        # Connect components
        rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "generator")

        logger.info("RAG pipeline created successfully")
        return rag_pipeline

    except Exception as e:
        logger.error(f"Failed to create RAG pipeline: {str(e)}")
        raise PipelineError(f"Pipeline creation failed: {str(e)}")