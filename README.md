# Customer Support Intelligence Engine ✈️🤖

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-LangGraph-orange?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn)

An **End-to-End AI System** designed to analyze, diagnose, and automate customer support interactions. This project integrates a robust **ETL pipeline**, **Predictive Machine Learning models**, and a **Generative AI Agent (RAG)** exposed via a production-ready **REST API**.

##  Key Features

### 1. 🧠 Autonomous AI Agent (LangGraph)
- **Architecture**: State Graph with cyclic logic using `LangGraph`.
- **Router**: Semantic routing to classify intent (Technical Support vs. Booking vs. Claims).
- **Tools**: Dynamic tool calling for retrieving specific booking or technical data.

### 2. 📚 Semantic Memory (RAG)
- **Vector Database**: Implements **FAISS** for local vector storage.
- **Embeddings**: Uses `nomic-embed-text` via Ollama for cost-effective, high-performance retrieval.
- **Function**: Retrieves historical context to augment the agent's responses (Retrieval-Augmented Generation).

### 3. 🔮 Predictive Analytics (ML)
- **Escalation Prediction**: Logistic Regression model to predict if a ticket will escalate to a human agent.
- **Sentiment Analysis**: KPI extraction from unstructured conversation logs.

### 4. ⚙️ Robust Data Engineering (ETL)
- **Processing**: Automated cleaning and normalization of raw JSON/CSV data using `Pandas`.
- **Persistence**: Optimized storage using **Parquet** format for high-performance I/O.

### 5. 🌐 Production API (FastAPI)
- **Endpoints**: RESTful endpoints for real-time querying.
- **Security**: API Key validation and Rate Limiting (`slowapi`) to prevent abuse.
- **Health Checks**: Integrated system diagnostics.

---

## 🛠️ Tech Stack

- **Core**: Python 3.10+
- **API Framework**: FastAPI, Uvicorn, Pydantic
- **GenAI Framework**: LangChain, LangGraph, Ollama
- **Data & ML**: Pandas, Scikit-learn, NumPy
- **Vector Store**: FAISS
- **Testing**: Pytest (QA Framework included)

---

## 📂 Project Structure

```bash
├── data/                  # Raw dataset (GitIgnored for privacy/size)
├── memory/                # Vector Database Logic (FAISS wrapper)
├── models/                # Serialized ML models (.pkl)
├── outputs/               # Generated reports and logs
├── Aerolinea.ipynb        # Main Notebook (ETL + Training + Prototyping)
├── .env                   # Environment variables (API Keys)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation