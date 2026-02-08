"""
Entry point del proyecto Aerya.

Carga datos, construye el contexto, inicializa el agente
y ejecuta una consulta de ejemplo.

Facilmente convertible a API FastAPI (ver seccion al final).
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.context import AgentContext
from app.agent.graph import get_agent_app, initialize_graph
from app.config.settings import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    MESSAGES_PATH,
    THREADS_PATH,
    get_embedder,
    logger,
)
from app.data.loaders import DataValidationError, cargar_datos
from app.data.preprocessing import crear_tabla_maestra
from app.memory.faiss_memory import FaissMemory
from app.metrics.business_metrics import construir_resumen_contexto
from app.prompts.agent_prompts import SYSTEM_PROMPT_TEMPLATE

# Suprimir logs verbosos de FAISS loader
logging.getLogger("faiss.loader").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Memoria semantica
# ---------------------------------------------------------------------------

def _init_memory() -> tuple:
    """Inicializa FAISS memory. Retorna (memory, enabled)."""
    try:
        embedder = get_embedder()
        memory = FaissMemory(
            index_path=FAISS_INDEX_PATH,
            metadata_path=FAISS_METADATA_PATH,
            embeddings=embedder,
        )
        return memory, True
    except Exception as exc:
        logger.warning("Memoria FAISS deshabilitada: %s", exc)
        return None, False


def get_semantic_context(memory, query: str, k: int = 3) -> str:
    """Recupera contexto semantico relevante."""
    if memory is None or not query:
        return ""
    hits = memory.search(query, k=k)
    if not hits:
        return ""
    return "\n".join([f"- {h['text']}" for h in hits])


def save_semantic_memory(
    memory, question: str, answer: str,
    thread_id: str = None, session_id: str = None,
) -> None:
    """Guarda un par pregunta-respuesta en memoria semantica."""
    if memory is None or not answer:
        return
    memory.save(
        text=f"Pregunta: {question}\nRespuesta: {answer}",
        metadata={"thread_id": thread_id, "session_id": session_id},
    )


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_agent_demo():
    """Ejecuta el pipeline completo como demostracion."""

    # 1. Cargar datos
    logger.info("=== Cargando datos ===")
    df_threads, df_messages = cargar_datos(THREADS_PATH, MESSAGES_PATH)

    # 2. Crear tabla maestra
    logger.info("=== Creando tabla maestra ===")
    df_master = crear_tabla_maestra(df_threads, df_messages)
    logger.info("Tabla maestra: %s", df_master.shape)

    # 3. Construir contexto (dataclass inmutable)
    context = AgentContext(df=df_master)

    # 4. Inicializar grafo del agente
    logger.info("=== Inicializando agente ===")
    initialize_graph(context.df)

    # 5. Inicializar memoria FAISS
    memory, memory_enabled = _init_memory()
    logger.info("Memoria FAISS: %s", "activa" if memory_enabled else "deshabilitada")

    # 6. Construir prompt
    contexto_str = construir_resumen_contexto(context.df)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(contexto=contexto_str)

    # 7. Ejecutar consulta de ejemplo
    session_id = "aerya-demo"
    app = get_agent_app(session_id)

    thread_id = None
    if "thread_id" in context.df.columns and not context.df.empty:
        thread_id = str(context.df["thread_id"].iloc[0])

    user_question = (
        "Cual es la tasa de escalamiento por estado "
        "y cuales son los hilos con mayor duracion?"
    )

    # Inyectar contexto semantico si hay historial
    semantic_ctx = get_semantic_context(memory, user_question, k=3)
    if semantic_ctx:
        system_prompt = (
            f"{system_prompt}\n\n"
            f"Contexto semantico relevante:\n{semantic_ctx}"
        )

    logger.info("=== Ejecutando agente ===")
    print("Iniciando flujo del agente...")

    for event in app.stream(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_question),
            ],
            "tool_calls_count": 0,
            "thread_id": thread_id,
            "route": None,
            "analysis": {},
        },
        config={
            "configurable": {"thread_id": session_id},
            "recursion_limit": 6,
        },
    ):
        for value in event.values():
            if "messages" in value:
                last_msg = value["messages"][-1]
                if (
                    hasattr(last_msg, "tool_calls")
                    and last_msg.tool_calls
                ):
                    print("\nAgente solicita herramientas:")
                    for tool_call in last_msg.tool_calls:
                        print(f"  - {tool_call['name']}")
                elif (
                    hasattr(last_msg, "content")
                    and last_msg.content
                    and last_msg.content.strip()
                ):
                    print("\nRespuesta:")
                    print(last_msg.content)

                    # Guardar en memoria semantica si es JSON valido
                    try:
                        json.loads(last_msg.content)
                        save_semantic_memory(
                            memory,
                            question=user_question,
                            answer=last_msg.content,
                            thread_id=thread_id,
                            session_id=session_id,
                        )
                    except (json.JSONDecodeError, ValueError):
                        logger.debug(
                            "Respuesta no es JSON valido, "
                            "no se guarda en memoria."
                        )


if __name__ == "__main__":
    run_agent_demo()
