"""
Construccion del StateGraph de LangGraph.

Separa la definicion del grafo de los nodos.
El grafo se construye una vez y se compila por sesion con MemorySaver.
"""

import logging
from typing import Any, Dict, List

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent.nodes import (
    make_nodo_asistente,
    make_nodo_escalar,
    make_nodo_resumen,
    make_nodo_router,
    make_nodo_tecnico,
)
from app.agent.state import AgentState
from app.agent.tools import build_tools
from app.config.settings import MAX_TOOL_CALLS, get_llm

logger = logging.getLogger("aerya")


def construir_grafo(df: pd.DataFrame) -> StateGraph:
    """Construye el StateGraph completo del agente.

    Args:
        df: DataFrame maestro. Se inyecta en tools y nodos via closure.

    Returns:
        StateGraph sin compilar (se compila por sesion con checkpointer).
    """
    llm = get_llm()
    tools = build_tools(df)
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    tools_node = ToolNode(tools)

    # -- Nodos (creados via factories con dependencias inyectadas) ---------
    nodo_router = make_nodo_router(llm, df)
    nodo_asistente = make_nodo_asistente(llm_with_tools)
    nodo_escalar = make_nodo_escalar(llm, df)
    nodo_tecnico = make_nodo_tecnico(llm, df)
    nodo_resumen = make_nodo_resumen(llm, df)

    # -- Routing functions -------------------------------------------------

    def route_after_router(state: AgentState) -> str:
        """Decide el siguiente nodo segun la ruta del router."""
        ruta = state.get("route", "assistant")
        if ruta == "escalar":
            return "escalar"
        if ruta == "tecnico":
            return "tecnico"
        if ruta == "resumen":
            return "resumen"
        return "assistant"

    def route_after_assistant(state: AgentState):
        """Decide si el asistente necesita tools o termina."""
        if state.get("tool_calls_count", 0) >= MAX_TOOL_CALLS:
            return END
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    # -- Construir grafo ---------------------------------------------------

    graph = StateGraph(AgentState)

    graph.add_node("router", nodo_router)
    graph.add_node("assistant", nodo_asistente)
    graph.add_node("tools", tools_node)
    graph.add_node("escalar", nodo_escalar)
    graph.add_node("tecnico", nodo_tecnico)
    graph.add_node("resumen", nodo_resumen)

    graph.set_entry_point("router")

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
# Session management
# ---------------------------------------------------------------------------

_AGENT_APPS: Dict[str, Any] = {}
_GRAPH_TEMPLATE: StateGraph = None  # type: ignore[assignment]


def initialize_graph(df: pd.DataFrame) -> None:
    """Construye el grafo template una vez. Debe llamarse al inicio."""
    global _GRAPH_TEMPLATE
    _GRAPH_TEMPLATE = construir_grafo(df)
    logger.info("Grafo del agente construido correctamente.")


def get_agent_app(session_id: str):
    """Devuelve un grafo compilado con MemorySaver para la sesion dada."""
    if _GRAPH_TEMPLATE is None:
        raise RuntimeError(
            "El grafo no ha sido inicializado. Llama a initialize_graph() primero."
        )
    if session_id not in _AGENT_APPS:
        _AGENT_APPS[session_id] = _GRAPH_TEMPLATE.compile(
            checkpointer=MemorySaver()
        )
    return _AGENT_APPS[session_id]


def get_active_sessions() -> int:
    """Devuelve la cantidad de sesiones activas."""
    return len(_AGENT_APPS)
