# Motor de Inteligencia para Soporte al Cliente ✈️🤖

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Production_API-009688?style=for-the-badge&logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-LangGraph-orange?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn)

Sistema de **IA End-to-End** disenado para analizar, diagnosticar y automatizar interacciones de soporte al cliente en una aerolinea. Integra un pipeline robusto de **ETL**, modelos **predictivos de Machine Learning** y un **Agente de IA Generativa (RAG)** expuesto a traves de una **API REST** lista para produccion.

---

## 🎯 Funcionalidades Clave

### 1. 🧠 Agente Autonomo de IA (LangGraph)
- **Arquitectura**: State Graph con logica ciclica y enrutamiento condicional usando `LangGraph`.
- **Router**: Analista de Conversaciones Automatizado que clasifica sentimiento e intencion, luego enruta a nodos especializados (escalamiento, borrador tecnico, resumen o asistente general).
- **Tools**: 7 herramientas dinamicas para consultar metricas, hilos principales, tasas de escalamiento e historial de conversaciones.
- **Memoria**: `MemorySaver` basado en sesion para conversaciones multi-turno sin contaminacion cruzada.

### 2. 📚 Memoria Semantica (RAG)
- **Base Vectorial**: **FAISS** para almacenamiento y busqueda vectorial local, sin costo.
- **Embeddings**: `nomic-embed-text` via **Ollama** (local, gratuito) para busqueda semantica de alto rendimiento.
- **Funcion**: Recupera contexto historico de preguntas y respuestas para enriquecer las respuestas del agente (Retrieval-Augmented Generation).

### 3. 🔮 Analitica Predictiva (ML)
- **Prediccion de Escalamiento**: Regresion Logistica con `class_weight='balanced'` para predecir si un ticket sera escalado.
- **Evaluacion**: Reporte de clasificacion, matriz de confusion, ROC-AUC, PR-AUC y analisis sensible al costo (falsos negativos ponderados 5x).
- **Persistencia**: Versionado de modelos con `joblib` y cache de embeddings en `Parquet`.

### 4. ⚙️ Ingenieria de Datos (ETL)
- **Procesamiento**: Limpieza automatizada, normalizacion y creacion de tabla maestra desde datos crudos en JSON usando `Pandas`.
- **Pipeline de Texto**: Normalizacion de contenido, fragmentacion de texto y cache de embeddings basado en hash SHA-256 para evitar vectorizacion redundante.
- **Persistencia**: I/O optimizado con formato **Parquet**.

### 5. 🌐 API de Produccion (FastAPI)
- **Endpoints**: `POST /agent/ask`, `GET /metrics`, `GET /health`.
- **Seguridad**: Validacion de API Key via header `X-API-Key`.
- **Rate Limiting**: Integracion con `slowapi` (10 req/min por cliente).
- **Health Check**: Diagnostico profundo (alcanzabilidad del LLM, estado del grafo, estado de la memoria, sesiones activas).

---

## 🛠️ Stack Tecnologico

| Categoria | Tecnologias |
|---|---|
| **Core** | Python 3.10+ |
| **API** | FastAPI, Uvicorn, Pydantic |
| **IA Generativa** | LangChain, LangGraph, OpenAI (gpt-4o-mini), Ollama |
| **Datos y ML** | Pandas, Scikit-learn, NumPy, Matplotlib, Seaborn |
| **Base Vectorial** | FAISS (CPU) |
| **Embeddings** | Ollama (nomic-embed-text) |

---

## 📂 Estructura del Proyecto

```
├── data/                  # Dataset crudo (GitIgnored por privacidad/tamano)
├── memory/
│   └── faiss_memory.py    # Modulo de memoria semantica FAISS
├── models/                # Modelos ML serializados (GitIgnored)
├── outputs/               # Cache de embeddings, indice FAISS, metricas (GitIgnored)
├── Aerolinea.ipynb        # Notebook Principal (ETL + ML + Agente + API)
├── .env                   # Variables de entorno (GitIgnored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Como Ejecutar

### Prerrequisitos

- Python 3.10+
- [Ollama](https://ollama.com/) instalado y corriendo
- API Key de OpenAI (solo para conversaciones del agente, costo minimo)

### 1. Clonar e instalar

```bash
git clone https://github.com/<tu-usuario>/end-to-end-analysis-llm-agent-rag.git
cd end-to-end-analysis-llm-agent-rag
pip install -r requirements.txt
```

### 2. Descargar el modelo de embeddings

```bash
ollama pull nomic-embed-text
```

### 3. Configurar el entorno

Crear un archivo `.env` en la raiz del proyecto:

```
OPENAI_API_KEY=sk-tu-clave-aqui
AERYA_API_KEY=tu-api-key-para-endpoints
```

### 4. Agregar los datos

Colocar `Threads.json` y `Messages.json` dentro de la carpeta `data/` en la raiz del proyecto.

### 5. Ejecutar el notebook

Abrir `Aerolinea.ipynb` y ejecutar todas las celdas en orden. El notebook realizara:
1. Carga y perfilamiento de datos crudos (EDA)
2. Construccion de la tabla maestra
3. Generacion de embeddings y entrenamiento del modelo ML
4. Inicializacion del agente LangGraph
5. Evaluacion y arranque del servidor FastAPI

### 6. Acceder a la API

Una vez que la celda del servidor este corriendo:
- **Documentacion interactiva**: `http://localhost:8000/docs`
- **Health check**: `GET http://localhost:8000/health`
- **Metricas**: `GET http://localhost:8000/metrics` (requiere header `X-API-Key`)
- **Consultar al agente**: `POST http://localhost:8000/agent/ask` (requiere header `X-API-Key`)

---

## 📊 Decisiones Clave

| Decision | Justificacion |
|---|---|
| OpenAI gpt-4o-mini para el agente | Velocidad y calidad para respuestas JSON estructuradas en demo en vivo. Costo minimo. |
| Ollama para embeddings | Ejecucion local, gratuito, sin dependencia de API para vectorizacion masiva. |
| FAISS como memoria semantica | Local, gratuito, amigable con CPU. Alternativa en produccion: base vectorial gestionada. |
| Patron closure para tools | Elimina estado global. Cada tool captura el DataFrame via closure. |
| Notebook como entregable principal | Demuestra el pipeline analitico completo. Despliegue en produccion extraeria a modulos `.py`. |

---

## ⚠️ Supuestos y Limitaciones

- `Threads.json` y `Messages.json` representan interacciones reales de soporte al cliente de una aerolinea.
- `escalate_conversation` es binario y confiable como variable objetivo de clasificacion.
- El modelo ML (Regresion Logistica) es un baseline. En produccion se evaluarian alternativas (XGBoost, Random Forest).
- La memoria FAISS es local. En produccion se migraria a una base vectorial distribuida.
- El agente depende de la calidad del LLM para generar JSON estructurado. Respuestas malformadas se manejan de forma segura.
- Los embeddings se generan localmente en CPU, lo cual limita la velocidad de vectorizacion masiva.
