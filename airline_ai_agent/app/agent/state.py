"""
Estado del agente LangGraph.

Define AgentState con tipado fuerte. Usa Annotated[list, add_messages]
para que LangGraph acumule mensajes en lugar de sobrescribirlos.
"""

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_calls_count: int
    thread_id: Optional[str]
    route: Optional[str]
    analysis: Dict[str, Any]
