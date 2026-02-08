"""
Nodos del grafo LangGraph.

Cada funcion recibe AgentState y devuelve AgentState.
Todos los prompts vienen de app.prompts.agent_prompts.
El contexto de datos se accede via _get_conversation_text con el df
inyectado, NO via variables globales.
"""

import json
import logging
from typing import Any, Dict, Optional

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.agent.tools import _get_conversation_text
from app.prompts.agent_prompts import (
    ASSISTANT_JSON_INSTRUCTION,
    ESCALATION_PROMPT,
    ROUTER_PROMPT,
    SUMMARY_PROMPT,
    TECHNICAL_PROMPT,
)

logger = logging.getLogger("aerya")


class AgentError(Exception):
    pass


def _parse_json_response(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Factories: cada factory recibe llm y df y devuelve un nodo cerrado
# ---------------------------------------------------------------------------


def make_nodo_router(llm: Any, df: pd.DataFrame):
    """Crea el nodo router. Recibe LLM y df explicitamente."""

    def nodo_router(state: AgentState) -> AgentState:
        if "messages" not in state or not state["messages"]:
            raise ValueError("Estado invalido: messages vacio")

        thread_id = state.get("thread_id")
        conversation_text = (
            _get_conversation_text(df, thread_id) if df is not None else ""
        )

        last_user = (
            state["messages"][-1].content
            if hasattr(state["messages"][-1], "content")
            else ""
        )
        router_input = (
            f"thread_id: {thread_id}\n"
            f"conversacion:\n{conversation_text}\n\n"
            f"pregunta_actual:\n{last_user}"
        )

        response = llm.invoke(
            [
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(content=router_input),
            ]
        )

        analysis = _parse_json_response(
            getattr(response, "content", "")
        )
        route = analysis.get("route", "assistant")

        return {
            "messages": state["messages"] + [response],
            "tool_calls_count": state.get("tool_calls_count", 0),
            "thread_id": thread_id,
            "route": route,
            "analysis": analysis,
        }

    return nodo_router


def make_nodo_escalar(llm: Any, df: pd.DataFrame):
    """Crea el nodo de escalamiento a humano."""

    def nodo_escalar_humano(state: AgentState) -> AgentState:
        analysis = state.get("analysis", {})
        thread_id = state.get("thread_id")
        conversation_text = (
            _get_conversation_text(df, thread_id) if df is not None else ""
        )

        response = llm.invoke(
            [
                SystemMessage(content=ESCALATION_PROMPT),
                HumanMessage(
                    content=(
                        f"analisis: {analysis}\n\n"
                        f"conversacion:\n{conversation_text}"
                    )
                ),
            ]
        )

        return {
            "messages": state["messages"] + [response],
            "tool_calls_count": state.get("tool_calls_count", 0),
        }

    return nodo_escalar_humano


def make_nodo_tecnico(llm: Any, df: pd.DataFrame):
    """Crea el nodo de respuesta tecnica."""

    def nodo_respuesta_tecnica(state: AgentState) -> AgentState:
        analysis = state.get("analysis", {})
        thread_id = state.get("thread_id")
        conversation_text = (
            _get_conversation_text(df, thread_id) if df is not None else ""
        )

        response = llm.invoke(
            [
                SystemMessage(content=TECHNICAL_PROMPT),
                HumanMessage(
                    content=(
                        f"analisis: {analysis}\n\n"
                        f"conversacion:\n{conversation_text}"
                    )
                ),
            ]
        )

        return {
            "messages": state["messages"] + [response],
            "tool_calls_count": state.get("tool_calls_count", 0),
        }

    return nodo_respuesta_tecnica


def make_nodo_resumen(llm: Any, df: pd.DataFrame):
    """Crea el nodo de resumen de conversacion resuelta."""

    def nodo_resumen_resuelto(state: AgentState) -> AgentState:
        analysis = state.get("analysis", {})
        thread_id = state.get("thread_id")
        conversation_text = (
            _get_conversation_text(df, thread_id) if df is not None else ""
        )

        response = llm.invoke(
            [
                SystemMessage(content=SUMMARY_PROMPT),
                HumanMessage(
                    content=(
                        f"analisis: {analysis}\n\n"
                        f"conversacion:\n{conversation_text}"
                    )
                ),
            ]
        )

        return {
            "messages": state["messages"] + [response],
            "tool_calls_count": state.get("tool_calls_count", 0),
        }

    return nodo_resumen_resuelto


def make_nodo_asistente(llm_with_tools: Any):
    """Crea el nodo asistente general (con tools bindeados)."""

    def nodo_asistente(state: AgentState) -> AgentState:
        if "messages" not in state or not state["messages"]:
            raise ValueError("Estado invalido: messages vacio")

        try:
            response = llm_with_tools.invoke(
                state["messages"]
                + [SystemMessage(content=ASSISTANT_JSON_INSTRUCTION)]
            )
        except Exception as exc:
            logger.error("Error invocando LLM: %s", exc)
            raise AgentError(f"Fallo del LLM: {exc}") from exc

        tool_calls_count = state.get("tool_calls_count", 0)
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls_count += 1

        return {
            "messages": state["messages"] + [response],
            "tool_calls_count": tool_calls_count,
            "thread_id": state.get("thread_id"),
            "route": state.get("route"),
            "analysis": state.get("analysis", {}),
        }

    return nodo_asistente
