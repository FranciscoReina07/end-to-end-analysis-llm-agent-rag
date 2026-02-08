"""
Configuracion centralizada del proyecto.

Todas las constantes, rutas y factories de modelos viven aqui.
Ningun modulo debe instanciar LLM ni definir rutas por su cuenta.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("aerya")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # airline_ai_agent/
DATA_DIR = BASE_DIR / "data"

# Fallback: si data/ no existe dentro del proyecto, buscar en el directorio padre
# (compatibilidad con la estructura original del workspace)
if not DATA_DIR.exists():
    _PARENT_DATA = BASE_DIR.parent / "data"
    if _PARENT_DATA.exists():
        DATA_DIR = _PARENT_DATA
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

THREADS_PATH = DATA_DIR / "Threads.json"
MESSAGES_PATH = DATA_DIR / "Messages.json"
EMBEDDINGS_CACHE_PATH = OUTPUTS_DIR / "embeddings_cache.parquet"
DF_MASTER_PATH = OUTPUTS_DIR / "df_master.parquet"
FAISS_INDEX_PATH = OUTPUTS_DIR / "faiss.index"
FAISS_METADATA_PATH = OUTPUTS_DIR / "faiss_metadata.json"

OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0
EMBEDDING_MODEL = "nomic-embed-text"

# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------

MAX_TOOL_CALLS = 3
RECURSION_LIMIT = 6
AGENT_TIMEOUT_S = 30

# ---------------------------------------------------------------------------
# Embeddings / ML
# ---------------------------------------------------------------------------

MAX_CHARS = 1000
CHUNK_SIZE = 500
EMBEDDING_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("AERYA_API_KEY")
if not API_KEY:
    logger.warning("AERYA_API_KEY no definida. Usando key de desarrollo.")
    API_KEY = "dev-key-local"

MAX_QUESTION_LENGTH = 2000

# ---------------------------------------------------------------------------
# Validacion de entorno
# ---------------------------------------------------------------------------

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY no esta definida en .env")


def get_llm():
    """Factory del LLM. Centraliza la creacion para evitar instancias dispersas."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


def get_embedder():
    """Factory de embeddings locales con Ollama."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=EMBEDDING_MODEL)
