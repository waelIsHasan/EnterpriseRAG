# EnterpriseRAG: Context-Aware Document Retrieval API 🚀

An enterprise-grade, asynchronous RESTful API built with **FastAPI**, **LangChain**, and **ChromaDB** to enable semantic search and context-aware question answering over technical documentation using LLMs.

## 🎯 Business Value
Reduces the time engineering and support teams spend searching through internal documentation by providing exact, natural-language answers backed by source citations. It eliminates LLM hallucinations by strictly grounding generated responses in the ingested vector store.

## ⚙️ Tech Stack
- **Backend Framework:** FastAPI, Uvicorn, Pydantic
- **RAG Orchestration:** LangChain
- **Vector Database:** ChromaDB (Persistent Local Storage)
- **Embeddings:** Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
- **LLM Inference:** Hugging Face Serverless Endpoints (Mistral-7B)
- **Environment:** Ubuntu Linux, Python 3.12+

## 🏗️ System Architecture
1. **Document Ingestion:** Reads unorganized knowledge base files.
2. **Semantic Chunking:** Splits text iteratively with precise overlap to maintain context boundaries.
3. **Vectorization:** Converts text chunks into dense vector embeddings.
4. **Retrieval & Generation:** Intercepts user queries via API, performs similarity search in ChromaDB, and constructs a strict prompt for the LLM to generate the final response.

## 📡 API Endpoints
- `POST /ingest`: Upload and vectorize new documents into the database dynamically.
- `POST /ask`: Query the knowledge base and receive a generated answer with source document excerpts.