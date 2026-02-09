"""
Configuracion centralizada, logging estructurado y excepciones personalizadas.

Todas las constantes, rutas y variables de entorno se gestionan aqui.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Variables de entorno
# ---------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
AERYA_API_KEY: str = os.getenv("AERYA_API_KEY", "dev-key-local")

# ---------------------------------------------------------------------------
# 2. Rutas del proyecto
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
OUTPUT_DIR: Path = BASE_DIR / "outputs"
MODEL_DIR: Path = BASE_DIR / "models"

THREADS_PATH: Path = DATA_DIR / "Threads.json"
MESSAGES_PATH: Path = DATA_DIR / "Messages.json"

EMBEDDINGS_CACHE_PATH: Path = OUTPUT_DIR / "embeddings_cache.parquet"
DF_MASTER_PATH: Path = OUTPUT_DIR / "df_master.parquet"

# ---------------------------------------------------------------------------
# 3. Constantes de negocio
# ---------------------------------------------------------------------------

MAX_CHARS: int = 1000
CHUNK_SIZE: int = 500
EMBEDDING_MODEL: str = "nomic-embed-text"
EMBEDDING_BATCH_SIZE: int = 64

LLM_MODEL: str = "gpt-4o-mini"
LLM_TEMPERATURE: float = 0.0

MAX_TOOL_CALLS: int = 3
AGENT_TIMEOUT_S: int = 30
MAX_QUESTION_LENGTH: int = 2000

# ---------------------------------------------------------------------------
# 4. Logging estructurado
# ---------------------------------------------------------------------------


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configura logging una sola vez para toda la aplicacion."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Silenciar logs verbosos de terceros
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger("aerya")


logger: logging.Logger = setup_logging()

# ---------------------------------------------------------------------------
# 5. Excepciones personalizadas
# ---------------------------------------------------------------------------


class DataValidationError(Exception):
    """Error en validacion o estructura de datos."""


class EmbeddingError(Exception):
    """Error en generacion de embeddings."""


class ModelTrainingError(Exception):
    """Error en entrenamiento del modelo ML."""


class AgentError(Exception):
    """Error en la ejecucion del agente LangGraph."""


class AgentTimeoutError(Exception):
    """El agente excedio el timeout configurado."""


# ---------------------------------------------------------------------------
# 6. Validaciones de arranque
# ---------------------------------------------------------------------------


def validate_environment() -> None:
    """Valida que el entorno tenga lo minimo necesario para arrancar."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY no esta definida en el entorno.")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    logger.info("Entorno validado: API Key detectada.")


def validate_data_paths() -> None:
    """Valida que existan los archivos de datos."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"La carpeta de datos no existe: {DATA_DIR}")
    if not THREADS_PATH.exists() or not MESSAGES_PATH.exists():
        raise FileNotFoundError(f"Faltan archivos JSON en {DATA_DIR}")
    logger.info("Archivos de datos encontrados.")


def ensure_dirs() -> None:
    """Crea directorios de salida si no existen."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
