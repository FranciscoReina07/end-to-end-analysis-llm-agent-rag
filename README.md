# Customer Support Intelligence Engine ✈️🤖

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Production_API-009688?style=for-the-badge&logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-LangGraph-orange?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn)

An **End-to-End AI System** designed to analyze, diagnose, and automate customer support interactions for an airline. This project integrates a robust **ETL pipeline**, **Predictive Machine Learning models**, and a **Generative AI Agent (RAG)** exposed via a production-ready **REST API**.

---

## 🎯 Key Features

### 1. 🧠 Autonomous AI Agent (LangGraph)
- **Architecture**: State Graph with cyclic logic and conditional routing using `LangGraph`.
- **Router**: Automated Conversation Analyst that classifies sentiment and intent, then routes to specialized nodes (escalation, technical draft, summary, or general assistant).
- **Tools**: 7 dynamic tools for retrieving metrics, top threads, escalation rates, and conversation history.
- **Memory**: Session-based `MemorySaver` for multi-turn conversations without cross-contamination.

### 2. 📚 Semantic Memory (RAG)
- **Vector Database**: **FAISS** for local, cost-free vector storage and retrieval.
- **Embeddings**: `nomic-embed-text` via **Ollama** (local, free) for high-performance semantic search.
- **Function**: Retrieves historical Q&A context to augment the agent's responses (Retrieval-Augmented Generation).

### 3. 🔮 Predictive Analytics (ML)
- **Escalation Prediction**: Logistic Regression with `class_weight='balanced'` to predict ticket escalation.
- **Evaluation**: Classification report, confusion matrix, ROC-AUC, PR-AUC, and cost-sensitive analysis (false negatives weighted 5x).
- **Persistence**: Model versioning with `joblib` and embedding cache with `Parquet`.

### 4. ⚙️ Data Engineering (ETL)
- **Processing**: Automated cleaning, normalization, and master table creation from raw JSON data using `Pandas`.
- **Text Pipeline**: Content normalization, text chunking, and SHA-256 hash-based embedding cache to avoid redundant vectorization.
- **Persistence**: Optimized I/O with **Parquet** format.

### 5. 🌐 Production API (FastAPI)
- **Endpoints**: `POST /agent/ask`, `GET /metrics`, `GET /health`.
- **Security**: API Key validation via `X-API-Key` header.
- **Rate Limiting**: `slowapi` integration (10 req/min per client).
- **Health Check**: Deep diagnostics (LLM reachability, graph status, memory status, active sessions).

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Core** | Python 3.10+ |
| **API** | FastAPI, Uvicorn, Pydantic |
| **GenAI** | LangChain, LangGraph, OpenAI (gpt-4o-mini), Ollama |
| **Data & ML** | Pandas, Scikit-learn, NumPy, Matplotlib, Seaborn |
| **Vector Store** | FAISS (CPU) |
| **Embeddings** | Ollama (nomic-embed-text) |

---

## 📂 Project Structure

```
├── data/                  # Raw dataset (GitIgnored for privacy/size)
├── memory/
│   └── faiss_memory.py    # FAISS semantic memory module
├── models/                # Serialized ML models (GitIgnored)
├── outputs/               # Embeddings cache, FAISS index, metrics (GitIgnored)
├── Aerolinea.ipynb        # Main Notebook (ETL + ML + Agent + API)
├── .env                   # Environment variables (GitIgnored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- OpenAI API Key (only for agent conversations, minimal cost)

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/end-to-end-analysis-llm-agent-rag.git
cd end-to-end-analysis-llm-agent-rag
pip install -r requirements.txt
```

### 2. Pull the embedding model

```bash
ollama pull nomic-embed-text
```

### 3. Configure environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
AERYA_API_KEY=your-api-key-for-endpoints
```

### 4. Add your data

Place `Threads.json` and `Messages.json` inside a `data/` folder at the project root.

### 5. Run the notebook

Open `Aerolinea.ipynb` and execute all cells sequentially. The notebook will:
1. Load and profile the raw data (EDA)
2. Build the master table
3. Generate embeddings and train the ML model
4. Initialize the LangGraph agent
5. Run evaluation and start the FastAPI server

### 6. Access the API

Once the server cell is running:
- **Interactive docs**: `http://localhost:8000/docs`
- **Health check**: `GET http://localhost:8000/health`
- **Metrics**: `GET http://localhost:8000/metrics` (requires `X-API-Key` header)
- **Ask the agent**: `POST http://localhost:8000/agent/ask` (requires `X-API-Key` header)

---

## 📊 Key Decisions

| Decision | Rationale |
|---|---|
| OpenAI gpt-4o-mini for agent | Speed and quality for structured JSON responses during live demo. Minimal cost. |
| Ollama for embeddings | Local execution, free, no API dependency for batch vectorization. |
| FAISS as semantic memory | Local, free, CPU-friendly. Production alternative: managed vector DB. |
| Closure pattern for tools | Eliminates global state. Each tool captures the DataFrame via closure. |
| Notebook as main deliverable | Demonstrates the full analytical pipeline. Production deployment would extract to `.py` modules. |

---

## ⚠️ Assumptions & Limitations

- `Threads.json` and `Messages.json` represent real airline customer interactions.
- `escalate_conversation` is binary and reliable as a classification target.
- The ML model (Logistic Regression) is a baseline. Production would benchmark alternatives (XGBoost, Random Forest).
- FAISS memory is local. Production would migrate to a distributed vector store.
- The agent depends on LLM quality for structured JSON output. Malformed responses are gracefully handled.
- Embeddings are generated locally on CPU, which limits batch vectorization speed.
