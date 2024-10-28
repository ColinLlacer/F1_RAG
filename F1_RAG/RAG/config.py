# Prompt for the RAG pipeline
prompt = """
Answer the question based on the provided context.
Context:
{% for doc in documents %}
   {{ doc.content }} 
{% endfor %}
Question: {{ query }}
"""

# Default model configurations
DEFAULT_EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Pipeline configurations
DEFAULT_BATCH_SIZE = 1028
DEFAULT_TOP_K = 3  # Number of documents to retrieve
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7

# LLM generation parameters
GENERATION_KWARGS = {
    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    "temperature": DEFAULT_TEMPERATURE,
    "do_sample": True,
}
