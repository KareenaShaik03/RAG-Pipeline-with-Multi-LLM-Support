# RAG Pipeline with Multi-LLM Support

**Tech Stack:** LangChain · ChromaDB · Sentence Transformers · OpenAI · Groq (LLaMA 3.3-70B) · Gemini · Python · FastAPI

---

## Overview

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that ingests documents, stores vector embeddings in ChromaDB, and answers user queries using any of three LLM backends — OpenAI, Groq's LLaMA 3.3-70B, or Google Gemini — via a unified LangChain interface.

---

## Features

- **Document ingestion** — Load and chunk text documents; embed using `sentence-transformers/all-MiniLM-L6-v2`
- **Vector storage** — Persist embeddings in ChromaDB for fast semantic retrieval
- **Top-k retrieval** — Configurable top-k similarity search to fetch the most relevant context chunks
- **Multi-LLM support** — Switch between OpenAI GPT, Groq LLaMA 3.3-70B, and Gemini with a single config flag
- **Dynamic prompt templates** — Context-aware prompts built at runtime using LangChain's `PromptTemplate`
- **Modular architecture** — Ingestor, retriever, and LLM chain are fully decoupled

---

## Project Structure

```
rag-pipeline/
├── ingest.py           # Document loading, chunking, embedding, ChromaDB storage
├── retriever.py        # Top-k semantic search from ChromaDB
├── chain.py            # LangChain chain with dynamic prompt + LLM routing
├── llm_config.py       # LLM backend selection (OpenAI / Groq / Gemini)
├── app.py              # FastAPI endpoint for query handling
├── data/               # Input documents
├── chroma_store/       # Persisted ChromaDB vector store
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/KareenaShaik03/rag-pipeline
cd rag-pipeline
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
LLM_BACKEND=groq          # options: openai | groq | gemini
```

---

## Usage

**Step 1 — Ingest documents**
```bash
python ingest.py --input data/
```

**Step 2 — Run the API**
```bash
uvicorn app:app --reload
```

**Step 3 — Query**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

---

## How It Works

```
User Query
    │
    ▼
Sentence Transformer (embed query)
    │
    ▼
ChromaDB (top-k similarity search)
    │
    ▼
Retrieved Chunks → Dynamic Prompt Template
    │
    ▼
LLM (OpenAI / Groq LLaMA / Gemini)
    │
    ▼
Answer
```

---

## Tech Highlights

| Component | Tool |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector DB | ChromaDB (persistent local store) |
| LLM Orchestration | LangChain |
| LLM Backends | OpenAI GPT-4o, Groq LLaMA 3.3-70B, Gemini 1.5 |
| API Layer | FastAPI |

---

## Why Groq as Default?

OpenAI's free-tier quota limitations made Groq's LLaMA 3.3-70B the preferred backend — it offers fast inference, generous free limits, and strong reasoning quality comparable to GPT-4-class models.

---

## Author

**Kareena Shaik** · [GitHub](https://github.com/KareenaShaik03) · [LinkedIn](https://linkedin.com/in/kareena-shaik)
