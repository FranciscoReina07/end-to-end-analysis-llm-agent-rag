"""
Entry point de la aplicacion.

Orquesta la inicializacion completa:
1. Valida entorno y rutas.
2. Carga datos y construye tabla maestra.
3. Inicializa LLM, tools, grafo y memoria semantica.
4. Crea y lanza la API FastAPI.

Uso:
    python main.py
    uvicorn main:api --host 0.0.0.0 --port 8000 --reload
"""

import uvicorn

from app.config import (
    AGENT_TIMEOUT_S,
    DF_MASTER_PATH,
    DataValidationError,
    EmbeddingError,
    ModelTrainingError,
    ensure_dirs,
    logger,
    validate_data_paths,
    validate_environment,
)
from app.data import (
    cargar_datos,
    cargar_df_master_si_existe,
    construir_resumen_contexto,
    crear_tabla_maestra,
    guardar_df_master,
    metricas_impacto_aerolinea,
)
from app.agent import (
    SessionManager,
    SemanticMemory,
    build_tools,
    construir_grafo,
    construir_system_prompt,
    create_llm,
)
from app.api import create_api

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap():
    """Inicializa todos los componentes y devuelve la app FastAPI."""

    # 1. Validaciones de arranque
    logger.info("Iniciando bootstrap...")
    validate_environment()
    validate_data_paths()
    ensure_dirs()

    # 2. Datos
    df_master = cargar_df_master_si_existe(DF_MASTER_PATH)
    if df_master is None:
        logger.info("Cache de df_master no encontrado. Construyendo desde JSON...")
        df_threads, df_messages = cargar_datos()
        df_master = crear_tabla_maestra(df_threads, df_messages)
        guardar_df_master(df_master, DF_MASTER_PATH)
    else:
        logger.info("df_master cargado desde cache.")

    logger.info("df_master: %d filas x %d columnas", *df_master.shape)

    # 3. LLM + Tools + Grafo
    llm = create_llm()
    tools = build_tools(df_master)
    graph_template = construir_grafo(llm, tools, df_master)
    session_manager = SessionManager(graph_template)

    logger.info("Agente LangGraph inicializado con %d tools.", len(tools))

    # 4. Memoria semantica
    semantic_memory = SemanticMemory()
    semantic_memory.initialize()

    # 5. API
    app = create_api(
        df_master=df_master,
        session_manager=session_manager,
        semantic_memory=semantic_memory,
        llm=llm,
        graph_template=graph_template,
        construir_system_prompt_fn=construir_system_prompt,
        construir_resumen_contexto_fn=construir_resumen_contexto,
        metricas_impacto_fn=metricas_impacto_aerolinea,
    )

    logger.info("API FastAPI lista.")
    return app


# ---------------------------------------------------------------------------
# App global (para uvicorn main:api)
# ---------------------------------------------------------------------------

api = bootstrap()

# ---------------------------------------------------------------------------
# Ejecucion directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Iniciando servidor en http://localhost:8000")
    logger.info("Documentacion interactiva: http://localhost:8000/docs")
    uvicorn.run(api, host="0.0.0.0", port=8000)
