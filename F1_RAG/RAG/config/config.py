# Prompt for the RAG pipeline
prompt = """
Answer the question based on the provided context. Mention the name of the document in your answer.
Context:
{% for doc in documents %}
   {{ doc.content }} 
{% endfor %}
Question: {{ query }}
"""

# Default model configurations
DEFAULT_EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2" #A chat model would be better, but choice of model is very limited on the serverless API

# Pipeline configurations
DEFAULT_BATCH_SIZE = 1028
DEFAULT_TOP_K = 1
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1

# LLM generation parameters
GENERATION_KWARGS = {
    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    "temperature": DEFAULT_TEMPERATURE,
    "do_sample": True,
}
