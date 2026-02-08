# Aerya - AI Agent para Analisis de Conversaciones de Aerolinea

## Objetivo

Sistema de inteligencia artificial que analiza conversaciones de soporte al cliente de una aerolinea. Combina Machine Learning clasico (clasificacion de escalamiento), embeddings semanticos y un agente conversacional con LangGraph para generar insights accionables.

Desarrollado como proyecto tecnico para demostrar competencias en AI Engineering: diseno de agentes, integracion de LLMs, manejo de datos y arquitectura production-ready.

## Arquitectura

```
airline_ai_agent/
  app/
    config/settings.py      -- Configuracion centralizada (LLM, rutas, constantes)
    data/
      loaders.py             -- Carga y validacion de archivos JSON
      preprocessing.py       -- Tabla maestra (merge threads + messages)
    metrics/
      business_metrics.py    -- Metricas de impacto (escalamiento, duracion, volumen)
    prompts/
      agent_prompts.py       -- Todos los prompts como constantes reutilizables
    memory/
      faiss_memory.py        -- Memoria semantica local con FAISS
    agent/
      context.py             -- AgentContext (dataclass, elimina globals)
      state.py               -- AgentState (TypedDict con Annotated messages)
      tools.py               -- Tools como factory (closure sobre DataFrame)
      nodes.py               -- Nodos del grafo (router, assistant, escalar, tecnico, resumen)
      graph.py               -- Construccion del StateGraph y session management
    evaluation/
      quality.py             -- Evaluacion de calidad de respuestas del agente
    main.py                  -- Entry point (carga datos -> agente -> demo)
  data/                      -- Archivos JSON de entrada
  outputs/                   -- Cache de embeddings, FAISS index, metricas
  models/                    -- Modelos ML persistidos (joblib)
  requirements.txt
  README.md
```

## Decisiones Tecnicas Clave

| Decision | Razon |
|---|---|
| **OpenAI gpt-4o-mini para el agente** | Velocidad y calidad en respuestas JSON estructuradas. Costo minimo. |
| **Ollama (nomic-embed-text) para embeddings** | Ejecucion local, gratuito, sin dependencia de API para vectorizacion masiva. |
| **FAISS como memoria semantica** | Local, gratuito, CPU-friendly. Alternativa a vector stores cloud para este scope. |
| **Closure pattern en tools** | Elimina variables globales. Cada tool captura el DataFrame via closure. |
| **Nodos como factories** | `make_nodo_router(llm, df)` inyecta dependencias explicitamente. Testeable. |
| **AgentContext dataclass** | Contenedor inmutable. Ningun modulo accede a `df_master` global. |
| **Prompts como constantes** | Centralizados en un modulo. Facilita auditoria y versionado. |

## Como Ejecutar

### Requisitos

- Python 3.9+
- Ollama instalado y corriendo (`ollama serve`)
- Modelo de embeddings: `ollama pull nomic-embed-text`
- API Key de OpenAI en archivo `.env`

### Instalacion

```bash
cd airline_ai_agent
pip install -r requirements.txt
```

### Configuracion

Crear archivo `.env` en la raiz del proyecto:

```
OPENAI_API_KEY=sk-...
AERYA_API_KEY=tu-api-key-para-la-api
```

### Ejecucion (demo)

```bash
cd airline_ai_agent
python -m app.main
```

### Ejecucion (API)

```bash
cd airline_ai_agent
uvicorn app.api:api --host 0.0.0.0 --port 8000 --reload
```

La documentacion interactiva estara en `http://localhost:8000/docs`.

## Supuestos y Limitaciones

- Los archivos Threads.json y Messages.json representan interacciones reales con clientes.
- El campo `escalate_conversation` es binario y fiable como target de clasificacion.
- El modelo de clasificacion (Logistic Regression) es un baseline. En produccion se evaluarian alternativas con benchmark comparativo.
- La memoria FAISS es local. En produccion se migraria a un vector store distribuido.
- El agente depende de la calidad del LLM para generar JSON estructurado.
- Los embeddings se generan localmente con CPU, lo cual limita velocidad en vectorizacion masiva.
