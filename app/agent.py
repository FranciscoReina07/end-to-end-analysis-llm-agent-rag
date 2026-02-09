"""
Agente LangGraph: estado, nodos, tools (closure), grafo y gestion de sesiones.

Responsabilidades:
- Definir AgentState con tipado fuerte y add_messages.
- Crear tools via closure (sin estado global).
- Definir nodos: router, escalar, tecnico, resumen, asistente.
- Construir el StateGraph con enrutamiento condicional.
- Gestionar sesiones aisladas (un MemorySaver por session_id).
- Memoria semantica FAISS integrada.
"""

import json
import re
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_TOOL_CALLS,
    AgentError,
    logger,
)
from app.data import construir_resumen_contexto, get_conversation_text
from memory.faiss_memory import FaissMemory

# ---------------------------------------------------------------------------
# JSON extraction (robusto contra LLM "charlatan")
# ---------------------------------------------------------------------------

_JSON_REGEX = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> Optional[str]:
    """Extrae el primer bloque JSON de un texto con posible texto alrededor."""
    match = _JSON_REGEX.search(text)
    return match.group(0) if match else None


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Intenta parsear JSON, incluso si el LLM agrega texto extra."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        extracted = _extract_json(text) if text else None
        if extracted:
            try:
                return json.loads(extracted)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        logger.debug("Respuesta no es JSON valido: %.80s", text)
        return {}


# ---------------------------------------------------------------------------
# LLM Factory (lazy, sin estado global)
# ---------------------------------------------------------------------------


def create_llm() -> ChatOpenAI:
    """Crea una instancia del LLM. No pasa api_key; usa os.environ."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        request_timeout=20,
    )


# ---------------------------------------------------------------------------
# Tools (closure pattern -- cada tool captura el DataFrame)
# ---------------------------------------------------------------------------


def _safe_datetime_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(
        series.map(lambda x: x.get("$date") if isinstance(x, dict) else x),
        errors="coerce",
    )


def build_tools(df: pd.DataFrame) -> list:
    """Construye las tools del agente capturando el DataFrame via closure."""

    @tool
    def obtener_metricas_resumen() -> str:
        """Devuelve un resumen de metricas clave de la aerolinea."""
        return construir_resumen_contexto(df)

    @tool
    def top_hilos_por_mensajes(n: int = 5) -> str:
        """Devuelve los top hilos con mas mensajes y su conteo."""
        if "msg_count" not in df.columns:
            return "No existe la columna msg_count."
        top = df[["thread_id", "msg_count"]].sort_values(
            "msg_count", ascending=False
        ).head(n)
        return top.to_string(index=False)

    @tool
    def top_hilos_por_duracion(n: int = 5) -> str:
        """Devuelve los top hilos con mayor duracion (min)."""
        if "duration_minutes" not in df.columns:
            return "No existe la columna duration_minutes."
        top = df[["thread_id", "duration_minutes"]].sort_values(
            "duration_minutes", ascending=False
        ).head(n)
        return top.to_string(index=False)

    @tool
    def tasa_escalamiento_por_categoria(
        col: str = "status", top_n: int = 8
    ) -> str:
        """Calcula la tasa de escalamiento por categoria."""
        if col not in df.columns or "escalate_conversation" not in df.columns:
            return f"No existe la columna {col} o escalate_conversation."
        tmp = df[[col, "escalate_conversation"]].copy()
        tmp["escalate_conversation"] = tmp["escalate_conversation"].fillna(0).astype(int)
        rate = (
            tmp.groupby(col)["escalate_conversation"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        return rate.to_string(index=False)

    @tool
    def resumen_tiempos_operativos() -> str:
        """Devuelve estadisticas de duracion y volumen de mensajes."""
        if "duration_minutes" not in df.columns or "msg_count" not in df.columns:
            return "No existen columnas de duracion o mensajes."
        stats = {
            "duracion_promedio_min": df["duration_minutes"].mean(),
            "duracion_p90_min": df["duration_minutes"].quantile(0.90),
            "mensajes_promedio": df["msg_count"].mean(),
            "mensajes_p90": df["msg_count"].quantile(0.90),
        }
        return pd.DataFrame([stats]).to_string(index=False)

    @tool
    def tendencia_escalamiento_mensual() -> str:
        """Muestra la tendencia mensual de escalamiento."""
        if "created_at" not in df.columns or "escalate_conversation" not in df.columns:
            return "No existen columnas requeridas para tendencia mensual."
        tmp = df[["created_at", "escalate_conversation"]].copy()
        tmp["created_at"] = _safe_datetime_series(tmp["created_at"])
        tmp = tmp.dropna(subset=["created_at"])
        tmp["month"] = tmp["created_at"].dt.to_period("M").astype(str)
        tmp["escalate_conversation"] = tmp["escalate_conversation"].fillna(0).astype(int)
        trend = tmp.groupby("month")["escalate_conversation"].mean().reset_index()
        return trend.to_string(index=False)

    @tool
    def obtener_conversacion_por_thread(thread_id: str) -> str:
        """Devuelve la conversacion completa para un thread_id."""
        text = get_conversation_text(df, thread_id)
        if not text:
            return "No se encontro conversacion para ese thread_id."
        return text

    return [
        obtener_metricas_resumen,
        top_hilos_por_mensajes,
        top_hilos_por_duracion,
        tasa_escalamiento_por_categoria,
        resumen_tiempos_operativos,
        tendencia_escalamiento_mensual,
        obtener_conversacion_por_thread,
    ]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = (
    "Rol: Analista de operaciones de una aerolinea.\n"
    "Objetivo: Responder preguntas de negocio con base en metricas.\n"
    "Restricciones: No revelar razonamiento interno. No usar emojis.\n"
    "Formato de salida: JSON con llaves summary, insights, risks, actions, metrics_used.\n"
    "Ejemplo de salida:\n"
    '{{\n'
    '  "summary": "La tasa de escalamiento es elevada en ciertos estados.",\n'
    '  "insights": ["Los hilos con mas mensajes muestran mayor escalamiento."],\n'
    '  "risks": ["Falsos negativos impactan la satisfaccion del cliente."],\n'
    '  "actions": ["Priorizar mejora en los estados con mayor tasa."],\n'
    '  "metrics_used": ["tasa_escalamiento", "mensajes_promedio"]\n'
    '}}\n\n'
    "Contexto:\n{contexto}"
)


def construir_system_prompt(contexto: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(contexto=contexto)


# ---------------------------------------------------------------------------
# Agent State (tipado fuerte con add_messages)
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_calls_count: int
    thread_id: Optional[str]
    route: Optional[str]
    analysis: Dict[str, Any]


# ---------------------------------------------------------------------------
# Nodos del grafo
# ---------------------------------------------------------------------------


def _build_router_node(llm: ChatOpenAI, df: pd.DataFrame):
    """Factory: crea el nodo router capturando LLM y df."""

    def nodo_router(state: AgentState) -> dict:
        if not state.get("messages"):
            raise ValueError("Estado invalido: messages vacio")

        thread_id = state.get("thread_id")
        conversation_text = get_conversation_text(df, thread_id)

        router_prompt = (
            "Rol: Analista de conversaciones automatizado.\n"
            "Tarea: Clasificar sentimiento y motivo, y decidir la ruta.\n"
            "Reglas de ruteo:\n"
            "- Si el sentimiento es negativo o hay queja fuerte: route = escalar\n"
            "- Si la consulta es tecnica: route = tecnico\n"
            "- Si la conversacion parece resuelta: route = resumen\n"
            "- Si no hay thread_id o conversacion: route = assistant\n"
            "Devuelve SOLO JSON con llaves: route, sentiment, motivo, rationale.\n"
        )

        last_user = ""
        if state["messages"]:
            last = state["messages"][-1]
            last_user = getattr(last, "content", "")

        router_input = (
            f"thread_id: {thread_id}\n"
            f"conversacion:\n{conversation_text}\n\n"
            f"pregunta_actual:\n{last_user}"
        )

        response = llm.invoke([
            SystemMessage(content=router_prompt),
            HumanMessage(content=router_input),
        ])

        analysis = _parse_json_response(getattr(response, "content", ""))
        route = analysis.get("route", "assistant")

        return {
            "messages": [response],
            "tool_calls_count": state.get("tool_calls_count", 0),
            "thread_id": thread_id,
            "route": route,
            "analysis": analysis,
        }

    return nodo_router


def _build_specialized_node(llm: ChatOpenAI, df: pd.DataFrame, prompt: str):
    """Factory: crea un nodo especializado (escalar, tecnico, resumen)."""

    def nodo(state: AgentState) -> dict:
        analysis = state.get("analysis", {})
        thread_id = state.get("thread_id")
        conversation_text = get_conversation_text(df, thread_id)

        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(
                content=f"analisis: {analysis}\n\nconversacion:\n{conversation_text}"
            ),
        ])

        return {
            "messages": [response],
            "tool_calls_count": state.get("tool_calls_count", 0),
        }

    return nodo


def _build_assistant_node(llm_with_tools: ChatOpenAI):
    """Factory: crea el nodo asistente con tools vinculadas."""

    def nodo_asistente(state: AgentState) -> dict:
        if not state.get("messages"):
            raise ValueError("Estado invalido: messages vacio")

        try:
            response = llm_with_tools.invoke(
                state["messages"]
                + [
                    SystemMessage(
                        content="IMPORTANTE: Devuelve SOLO un JSON valido con las claves solicitadas."
                    )
                ]
            )
        except Exception as exc:
            logger.error("Error invocando LLM: %s", exc)
            raise AgentError(f"Fallo del LLM: {exc}") from exc

        tool_calls_count = state.get("tool_calls_count", 0)
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls_count += 1

        return {
            "messages": [response],
            "tool_calls_count": tool_calls_count,
            "thread_id": state.get("thread_id"),
            "route": state.get("route"),
            "analysis": state.get("analysis", {}),
        }

    return nodo_asistente


# ---------------------------------------------------------------------------
# Construccion del grafo
# ---------------------------------------------------------------------------


def construir_grafo(
    llm: ChatOpenAI,
    tools: list,
    df: pd.DataFrame,
) -> StateGraph:
    """Construye el StateGraph completo del agente."""
    llm_with_tools = llm.bind_tools(tools)
    tools_node = ToolNode(tools)

    def route_after_router(state: AgentState) -> str:
        ruta = state.get("route", "assistant")
        if ruta in ("escalar", "tecnico", "resumen"):
            return ruta
        return "assistant"

    def route_after_assistant(state: AgentState) -> str:
        if state.get("tool_calls_count", 0) >= MAX_TOOL_CALLS:
            return END
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)

    # Nodos
    graph.add_node("router", _build_router_node(llm, df))
    graph.add_node(
        "escalar",
        _build_specialized_node(
            llm,
            df,
            "Rol: Analista de conversaciones automatizado.\n"
            "Tarea: Generar salida de escalamiento a humano.\n"
            "Devuelve SOLO JSON con llaves: action, sentiment, motivo, summary, next_steps.\n"
            "action debe ser 'escalar_humano'.",
        ),
    )
    graph.add_node(
        "tecnico",
        _build_specialized_node(
            llm,
            df,
            "Rol: Analista de conversaciones automatizado.\n"
            "Tarea: Generar borrador de respuesta tecnica.\n"
            "Devuelve SOLO JSON con llaves: action, sentiment, motivo, respuesta, assumptions.\n"
            "action debe ser 'borrador_tecnico'.",
        ),
    )
    graph.add_node(
        "resumen",
        _build_specialized_node(
            llm,
            df,
            "Rol: Analista de conversaciones automatizado.\n"
            "Tarea: Generar resumen final de conversacion resuelta.\n"
            "Devuelve SOLO JSON con llaves: action, sentiment, motivo, summary, resolution.\n"
            "action debe ser 'resumen'.",
        ),
    )
    graph.add_node("assistant", _build_assistant_node(llm_with_tools))
    graph.add_node("tools", tools_node)

    # Entry point
    graph.set_entry_point("router")

    # Edges
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "escalar": "escalar",
            "tecnico": "tecnico",
            "resumen": "resumen",
            "assistant": "assistant",
        },
    )
    graph.add_edge("escalar", END)
    graph.add_edge("tecnico", END)
    graph.add_edge("resumen", END)
    graph.add_conditional_edges(
        "assistant",
        route_after_assistant,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "assistant")

    return graph


# ---------------------------------------------------------------------------
# Session manager (un MemorySaver por session_id)
# ---------------------------------------------------------------------------


class SessionManager:
    """Gestiona sesiones aisladas del agente. Thread-safe por diseno."""

    def __init__(self, graph_template: StateGraph) -> None:
        self._graph = graph_template
        self._sessions: Dict[str, Any] = {}

    def get(self, session_id: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = self._graph.compile(
                checkpointer=MemorySaver()
            )
        return self._sessions[session_id]

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# ---------------------------------------------------------------------------
# Memoria semantica FAISS
# ---------------------------------------------------------------------------


class SemanticMemory:
    """Wrapper sobre FaissMemory con inicializacion lazy."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self._memory: Optional[FaissMemory] = None

    def initialize(
        self,
        index_path: Path = Path("outputs/faiss.index"),
        metadata_path: Path = Path("outputs/faiss_metadata.json"),
        model_name: str = EMBEDDING_MODEL,
    ) -> None:
        try:
            embeddings = OllamaEmbeddings(model=model_name)
            self._memory = FaissMemory(
                index_path=index_path,
                metadata_path=metadata_path,
                embeddings=embeddings,
            )
            self.enabled = self._memory.enabled
            if self.enabled:
                logger.info("Memoria semantica FAISS inicializada.")
            else:
                logger.warning("FAISS no disponible. Memoria deshabilitada.")
        except Exception as exc:
            self.enabled = False
            self._memory = None
            logger.warning("Memoria FAISS deshabilitada: %s", exc)

    def get_context(self, query: str, k: int = 3) -> str:
        if not self.enabled or self._memory is None or not query:
            return ""
        hits = self._memory.search(query, k=k)
        if not hits:
            return ""
        return "\n".join([f"- {h['text']}" for h in hits])

    def save(
        self,
        question: str,
        answer: str,
        thread_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        if not self.enabled or self._memory is None or not answer:
            return
        self._memory.save(
            text=f"Pregunta: {question}\nRespuesta: {answer}",
            metadata={"thread_id": thread_id, "session_id": session_id},
        )
