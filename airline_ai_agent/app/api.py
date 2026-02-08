"""
API FastAPI para el agente de analisis de conversaciones.

Endpoints:
- GET  /health     -- Health check profundo (LLM reachable, memory, graph)
- GET  /metrics    -- Metricas de negocio de la aerolinea
- POST /agent/ask  -- Consulta al agente con rate limiting y timeout
"""

import json
import logging
import signal
import uuid
from contextlib import contextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.graph import (
    get_active_sessions,
    get_agent_app,
    initialize_graph,
)
from app.config.settings import (
    AGENT_TIMEOUT_S,
    API_KEY,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    MAX_QUESTION_LENGTH,
    MESSAGES_PATH,
    THREADS_PATH,
    get_embedder,
    get_llm,
    logger,
)
from app.data.loaders import cargar_datos
from app.data.preprocessing import crear_tabla_maestra
from app.memory.faiss_memory import FaissMemory
from app.metrics.business_metrics import (
    construir_resumen_contexto,
    metricas_impacto_aerolinea,
)
from app.prompts.agent_prompts import SYSTEM_PROMPT_TEMPLATE

# Suprimir logs verbosos de FAISS
logging.getLogger("faiss.loader").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=MAX_QUESTION_LENGTH
    )
    session_id: Optional[str] = Field(None, max_length=128)
    thread_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    session_id: str


class HealthResponse(BaseModel):
    status: str
    graph_loaded: bool
    memory_enabled: bool
    llm_reachable: bool
    active_sessions: int


class MetricsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class ErrorResponse(BaseModel):
    error: str


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

class AgentTimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds: int):
    """Timeout context manager (solo Unix). En Windows funciona como fallback sin efecto."""
    if hasattr(signal, "SIGALRM"):
        def handler(signum, frame):
            raise AgentTimeoutError(
                f"El agente excedio el timeout de {seconds}s."
            )

        old = signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        yield


# ---------------------------------------------------------------------------
# App State (inicializado al arrancar)
# ---------------------------------------------------------------------------

_df_master = None
_memory = None
_memory_enabled = False
_graph_initialized = False


def _startup():
    """Inicializa datos, grafo y memoria al arrancar la API."""
    global _df_master, _memory, _memory_enabled, _graph_initialized

    logger.info("Inicializando API...")

    df_threads, df_messages = cargar_datos(THREADS_PATH, MESSAGES_PATH)
    _df_master = crear_tabla_maestra(df_threads, df_messages)
    logger.info("Tabla maestra cargada: %s", _df_master.shape)

    initialize_graph(_df_master)
    _graph_initialized = True
    logger.info("Grafo del agente inicializado.")

    try:
        embedder = get_embedder()
        _memory = FaissMemory(
            index_path=FAISS_INDEX_PATH,
            metadata_path=FAISS_METADATA_PATH,
            embeddings=embedder,
        )
        _memory_enabled = True
        logger.info("Memoria FAISS activa.")
    except Exception as exc:
        _memory_enabled = False
        logger.warning("Memoria FAISS deshabilitada: %s", exc)


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

api = FastAPI(
    title="Aerya API",
    description="API para el agente de analisis de conversaciones de la aerolinea.",
    version="1.0.0",
)

api.state.limiter = limiter
api.add_middleware(SlowAPIMiddleware)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.on_event("startup")
def on_startup():
    _startup()


# ---------------------------------------------------------------------------
# API Key validation
# ---------------------------------------------------------------------------

def validate_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> str:
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401, detail="API key invalida o ausente."
        )
    return x_api_key


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@api.get("/health", response_model=HealthResponse)
def health_check():
    llm_ok = True
    try:
        get_llm().invoke("ping")
    except Exception:
        llm_ok = False

    return HealthResponse(
        status="ok" if llm_ok else "degraded",
        graph_loaded=_graph_initialized,
        memory_enabled=_memory_enabled,
        llm_reachable=llm_ok,
        active_sessions=get_active_sessions(),
    )


@api.get("/metrics", response_model=MetricsResponse)
def get_metrics(
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    validate_api_key(x_api_key)
    try:
        metrics_df = metricas_impacto_aerolinea(_df_master)
        return metrics_df.to_dict(orient="records")[0]
    except Exception as exc:
        logger.error("Error en /metrics", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc))


@api.post("/agent/ask", response_model=AskResponse)
@limiter.limit("10/minute")
def agent_ask(
    request: Request,
    req: AskRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    validate_api_key(x_api_key)

    session_id = req.session_id or str(uuid.uuid4())

    logger.info(
        "agent_request",
        extra={"session_id": session_id, "thread_id": req.thread_id},
    )

    try:
        contexto = construir_resumen_contexto(_df_master)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(contexto=contexto)

        # Contexto semantico
        if _memory is not None:
            hits = _memory.search(req.question, k=3)
            if hits:
                semantic_ctx = "\n".join(
                    [f"- {h['text']}" for h in hits]
                )
                system_prompt = (
                    f"{system_prompt}\n\n"
                    f"Contexto semantico relevante:\n{semantic_ctx}"
                )

        app_graph = get_agent_app(session_id)

        with timeout(AGENT_TIMEOUT_S):
            result = app_graph.invoke(
                {
                    "messages": [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=req.question),
                    ],
                    "tool_calls_count": 0,
                    "thread_id": req.thread_id,
                    "route": None,
                    "analysis": {},
                },
                config={
                    "configurable": {"thread_id": session_id},
                    "recursion_limit": 6,
                },
            )

        answer = result["messages"][-1].content

        # Guardar en memoria semantica si es JSON valido
        try:
            json.loads(answer)
            if _memory is not None:
                _memory.save(
                    text=f"Pregunta: {req.question}\nRespuesta: {answer}",
                    metadata={
                        "thread_id": req.thread_id,
                        "session_id": session_id,
                    },
                )
        except (json.JSONDecodeError, ValueError):
            logger.debug(
                "Respuesta no es JSON valido, no se guarda en memoria."
            )

        logger.info(
            "agent_response",
            extra={
                "session_id": session_id,
                "answer_length": len(answer),
            },
        )

        return AskResponse(answer=answer, session_id=session_id)

    except AgentTimeoutError:
        logger.warning(
            "Timeout en /agent/ask",
            extra={"session_id": session_id},
        )
        raise HTTPException(
            status_code=504,
            detail=f"El agente excedio el timeout de {AGENT_TIMEOUT_S}s.",
        )

    except Exception as exc:
        logger.error(
            "Error en /agent/ask",
            extra={"session_id": session_id, "error": str(exc)},
        )
        raise HTTPException(status_code=500, detail=str(exc))
