#  RAG 

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using LangChain, Qdrant (vector database), and NLP preprocessing.  
It allows users to query custom documents using semantic search and LLM-based contextual answers.

---

##  Overview

The pipeline consists of two main components:

- **`ingest.py`** — Loads, preprocesses, and embeds documents, then stores the embeddings in **Qdrant**.  
- **`query.py`** — Retrieves relevant document chunks from Qdrant and generates responses using an **LLM** via LangChain.

Qdrant is hosted locally, ensuring fast, private, and efficient vector search.

---

##  Architecture
```
    ┌────────────┐
    │  Documents │
    └──────┬─────┘
           │
     (1) Ingest
           │
           ▼
 ┌────────────────────┐
 │    Embeddings      │
 └────────────────────┘
           │
           ▼
    ┌────────────┐
    │   Qdrant   │  ← Local vector DB
    └────────────┘
           │
     (2) Query
           │
           ▼
 ┌────────────────────┐
 │    LangChain LLM   │
 └────────────────────┘
           │
           ▼
     ┌────────────┐
     │  Response  │
     └────────────┘

```

