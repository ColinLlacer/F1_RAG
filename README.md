# F1 RAG System

A Retrieval-Augmented Generation (RAG) system specialized in Formula 1 knowledge, built using FastAPI, Haystack, and Hugging Face's models.

## Overview

This system combines the power of Large Language Models with a specialized knowledge base of Formula 1 information to provide accurate and contextual answers to F1-related queries. It uses RAG architecture to ground LLM responses in factual F1 data.

## Features

- **RAG Pipeline**: Combines document retrieval with language model generation
- **FastAPI Backend**: RESTful API for easy integration
- **Document Processing**: Automatic processing and indexing of F1-related documents
- **Semantic Search**: Uses embeddings for accurate document retrieval
- **Error Handling**: Comprehensive error handling and logging
- **Modular Architecture**: Clean separation of concerns for easy maintenance

## Technical Stack

- **RAG Framework**: Haystack
- **Embeddings**: Any sentence-transformer model from HuggingFace
- **LLM**: Any LLM model from the serverless inference API from HuggingFace
- **Document Store**: In-Memory Document Store (easily scalable to a database)
- **Package Management**: Poetry

## Installation

1. Clone the repository: 

```bash
git clone https://github.com/yourusername/F1-RAG.git
cd F1-RAG
```

2. Install dependencies using Poetry:

```bash
poetry install
```

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configurations
```

Edit .env with your configurations

4. Prepare the document database:

```bash
poetry run python -m F1_RAG.wiki_downloader.downloader
```

5. Run the app using Poetry (An API will be implemented in the future):

```bash
poetry run python F1_RAG/api/run.py
```


