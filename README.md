# F1 RAG System

A Retrieval-Augmented Generation (RAG) system specialized in Formula 1 knowledge, built using Haystack and Hugging Face's models.

## Overview

This system combines the power of LLMs with a specialized knowledge base of Formula 1 information to provide accurate and contextual answers to F1-related queries. It uses a basic RAG architecture to ground LLM responses in factual F1 data.

## Features

- **Document Processing**: Automatic downloading, processing and indexing of F1-related wikipedia articles.
  
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
```

4. Prepare the document database using the wikipedia-downloader module:

```bash
poetry run python -m F1_RAG.wiki_downloader.downloader
```

5. Run the app using Poetry (An API will be implemented in the future):

```bash
poetry run python F1_RAG/RAG/main.py
```

6: Optional - Change the retrievable documents or models using the variables in the different configuration files.

7: Ask your questions about F1!

