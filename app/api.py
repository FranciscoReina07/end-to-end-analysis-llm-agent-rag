"""
API FastAPI de produccion.

Endpoints:
- GET  /health      -> diagnostico profundo del sistema.
- GET  /metrics     -> metricas de negocio de la aerolinea.
- POST /agent/ask   -> consulta al agente LangGraph.

Seguridad:
- API Key via header X-API-Key.
- Rate limiting con slowapi (10 req/min).
- CORS configurado.
- Timeout explicito en el agente.
- Validacion de entrada con Pydantic V2.
"""

import json
import signal
import uuid
from contextlib import contextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import (
    AERYA_API_KEY,
    AGENT_TIMEOUT_S,
    MAX_QUESTION_LENGTH,
    AgentTimeoutError,
    logger,
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
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
# Timeout helper (Unix: SIGALRM, Windows: no-op)
# ---------------------------------------------------------------------------


@contextmanager
def timeout(seconds: int):
    """Context manager con timeout. Solo funciona en Unix (SIGALRM)."""
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
# API Key validation
# ---------------------------------------------------------------------------


def validate_api_key(x_api_key: str) -> None:
    if x_api_key != AERYA_API_KEY:
        raise HTTPException(status_code=401, detail="API key invalida o ausente.")


# ---------------------------------------------------------------------------
# Factory: crea la app FastAPI con dependencias inyectadas
# ---------------------------------------------------------------------------


def create_api(
    df_master: pd.DataFrame,
    session_manager,
    semantic_memory,
    llm,
    graph_template,
    construir_system_prompt_fn,
    construir_resumen_contexto_fn,
    metricas_impacto_fn,
) -> FastAPI:
    """
    Factory que crea la app FastAPI inyectando todas las dependencias.
    Elimina estado global: todo se pasa explicitamente.
    """

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

    # --- Health ---

    @api.get("/health", response_model=HealthResponse)
    def health_check():
        llm_ok = True
        try:
            llm.invoke("ping")
        except Exception:
            llm_ok = False

        return HealthResponse(
            status="ok" if llm_ok else "degraded",
            graph_loaded=graph_template is not None,
            memory_enabled=semantic_memory.enabled,
            llm_reachable=llm_ok,
            active_sessions=session_manager.active_count,
        )

    # --- Metrics ---

    @api.get("/metrics", response_model=MetricsResponse)
    def get_metrics(
        x_api_key: str = Header(..., alias="X-API-Key"),
    ):
        validate_api_key(x_api_key)
        try:
            metrics_df = metricas_impacto_fn(df_master)
            return metrics_df.to_dict(orient="records")[0]
        except Exception as exc:
            logger.error("Error en /metrics: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    # --- Agent ask ---

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
            "agent_request session=%s thread=%s", session_id, req.thread_id
        )

        try:
            contexto = construir_resumen_contexto_fn(df_master)
            system_prompt = construir_system_prompt_fn(contexto)

            sem_ctx = semantic_memory.get_context(req.question, k=3)
            if sem_ctx:
                system_prompt = (
                    f"{system_prompt}\n\nContexto semantico relevante:\n{sem_ctx}"
                )

            app_graph = session_manager.get(session_id)

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
                semantic_memory.save(
                    question=req.question,
                    answer=answer,
                    thread_id=req.thread_id,
                    session_id=session_id,
                )
            except (json.JSONDecodeError, ValueError):
                logger.debug("Respuesta no es JSON valido, no se guarda.")

            logger.info(
                "agent_response session=%s len=%d", session_id, len(answer)
            )
            return AskResponse(answer=answer, session_id=session_id)

        except AgentTimeoutError:
            logger.warning("Timeout en /agent/ask session=%s", session_id)
            raise HTTPException(
                status_code=504,
                detail=f"El agente excedio el timeout de {AGENT_TIMEOUT_S}s.",
            )
        except Exception as exc:
            logger.error("Error en /agent/ask: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    return api
