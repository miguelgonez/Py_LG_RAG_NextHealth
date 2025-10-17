# RAG NextHealth - Replit Configuration

## Overview

RAG NextHealth is an advanced clinical search system for Spanish-language medical queries. It combines semantic search, intelligent routing, BGE re-ranking, and SQL capabilities to provide accurate, source-backed medical information. The system uses multilingual embeddings optimized for Spanish, ChromaDB for vector storage, and implements multiple RAG techniques (RAG-Fusion, HYDE, RAPTOR, CRAG) with clinical guardrails to ensure safe, informative responses without providing medical diagnoses or prescriptions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core RAG Pipeline

**Multi-format Document Ingestion**
- **Document Directory**: `./docs` (default location for uploading clinical documents)
- **Incremental Ingestion**: SQLite-based state tracking detects new/modified files and processes only changes
- **Change Detection**: SHA256 hashing, file size, and mtime comparison for accurate change detection
- **Duplicate Prevention**: Automatically removes old chunks when files are updated
- **Cleanup**: Removes embeddings from vector store when source files are deleted
- Supports PDF, HTML, TXT, and MD files via LangChain loaders
- Uses `RecursiveCharacterTextSplitter` for semantic chunking
- Adds comprehensive metadata (source file, file type, ingestion date) to all documents
- Stores processed documents in ChromaDB with persistence
- **Ingestion Modes**:
  - *Incremental* (default): Processes only new/modified files, maintains existing embeddings
  - *Full Reindex*: Clears all data and reprocesses entire corpus from scratch

**Embedding & Vector Storage**
- Primary embedding model: `multilingual-e5-base` (HuggingFace) optimized for Spanish clinical text
- Vector database: ChromaDB with local persistence (`persist_dir`)
- Fallback reranker model: `BAAI/bge-reranker-base` when FlagEmbedding unavailable

**Advanced Retrieval Techniques**
- Multi-query retrieval: Generates reformulated queries to improve recall
- RAG-Fusion: Combines results from multiple query variations using Reciprocal Rank Fusion (RRF)
- HYDE (Hypothetical Document Embeddings): Generates hypothetical documents to find semantically similar content
- RAPTOR: Hierarchical summarization with KMeans clustering for long-context queries
- CRAG (Corrective RAG): Evaluates relevance and re-retrieves if quality is below threshold

**Re-ranking Pipeline**
- Initial retrieval: Fetches k=8 documents from vectorstore
- BGE cross-encoder re-ranking: Uses `FlagReranker` or `bge-reranker-base` to reorder by relevance
- Final selection: Returns top k=4 most relevant documents after re-ranking

### Intelligent Routing System

**Query Classification**
- LangChain-based routing chain decides between three paths:
  - **Vectorstore**: Semantic/content-based questions
  - **SQL**: Structured queries (ICPC-3 codes, mappings, counts, classifications)
  - **Hybrid**: Queries requiring both approaches
- SQL keyword detection for mapping/code queries (ICPC, ICD-10, SNOMED)
- Confidence scoring and reasoning for routing decisions

**Text-to-SQL Capabilities**
- SQLAlchemy + SQLite backend for ICPC-3 clinical codes and ICD-10 mappings
- LangChain SQL chain converts natural language to SQL queries
- Tables include: `icpc_codes`, `icpc_icd10_mapping` with clinical terminology

### Response Generation & Safety

**Clinical Guardrails (MDR/AI Act Compliance)**
- Strict policy enforcement: No diagnoses, no medication dosages, no medical advice
- Mandatory disclaimer on all responses
- Source citation requirements (minimum 2 documents)
- Response length limits (max 2000 characters)
- Policy instructions embedded in generation prompts

**LLM Configuration**
- Primary model: OpenAI GPT-5 (default as of August 2025)
- Temperature varies by task: 0.0 for evaluation, 0.7 for HYDE generation, configurable for main responses
- Prompt templates enforce clinical safety and citation requirements

### Evaluation Framework

**Metrics Implementation**
- nDCG@k (Normalized Discounted Cumulative Gain) using scikit-learn
- Recall@k for retrieval quality measurement
- Precision@k calculations
- Per-query and aggregate metrics
- Latency tracking for each pipeline stage (retrieval, re-ranking, generation)

**Test Dataset Format**
- CSV-based with columns: `question`, `relevant_docs`
- Supports histogram analysis of nDCG scores
- Identifies low-performing queries for iteration

### Application Architecture

**Streamlit Frontend (3-Tab Interface)**
- Tab 1: RAG Search with configurable k values, similarity thresholds, retrieval mode selection
- Tab 2: SQL Console with query editor, sample templates, and result pagination
- Tab 3: Evaluation Dashboard with CSV upload, metric visualization, and performance analysis

**Modular Code Structure**
```
src/
├── ingest.py          # Document loading & vectorstore creation
├── retrievers.py      # Multi-query, fusion, RAPTOR, advanced RAG
├── reranker.py        # BGE cross-encoder re-ranking
├── routing.py         # Intelligent query routing logic
├── tools.py           # Text-to-SQL, database utilities
├── graph.py           # LangChain orchestration & routing chain
├── policies.py        # Clinical guardrails & response formatting
├── evaluation.py      # nDCG, recall, metric calculations
├── advanced_rag.py    # RAG-Fusion, HYDE implementations
├── raptor.py          # RAPTOR hierarchical retrieval
└── crag.py            # CRAG corrective retrieval
```

## External Dependencies

**Core Framework**
- LangChain + LangGraph for RAG orchestration and routing
- Streamlit for web interface

**Vector & Embedding**
- ChromaDB for vector storage with local persistence
- HuggingFace Transformers for `multilingual-e5-base` embeddings
- FlagEmbedding for `bge-reranker-base` cross-encoder re-ranking

**LLM & AI**
- OpenAI API (GPT-5) for generation, query reformulation, and routing decisions
- Requires `OPENAI_API_KEY` environment variable

**Data Processing**
- PyPDF for PDF parsing
- UnstructuredFileLoader for HTML/other formats
- pandas for data manipulation and CSV handling
- scikit-learn for KMeans clustering (RAPTOR) and nDCG calculations

**Database**
- SQLite for ICPC-3 code storage
- SQLAlchemy for ORM and Text-to-SQL

**Python Requirements**
- Python 3.8+
- sentence-transformers for embeddings
- numpy for numerical operations
- torch (CPU/CUDA) for model inference

**Environment Variables**
- `OPENAI_API_KEY`: Required for GPT-5 access
- `OPENAI_CHAT_MODEL`: Optional, defaults to "gpt-5"
- `RERANKER_MODEL`: Optional, defaults to "BAAI/bge-reranker-base"

**System Requirements**
- 4GB+ RAM recommended for embedding models
- 2GB+ disk space for vector storage and models
- CUDA optional for GPU acceleration