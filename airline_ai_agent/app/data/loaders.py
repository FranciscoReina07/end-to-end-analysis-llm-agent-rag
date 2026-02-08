"""
Carga de datos desde archivos JSON.

Funciones puras que reciben rutas y devuelven DataFrames.
No dependen de estado global ni de configuracion implicitai.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger("aerya")


class DataValidationError(Exception):
    pass


def cargar_datos(
    ruta_threads: Path, ruta_messages: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga Threads.json y Messages.json y devuelve dos DataFrames.

    Raises:
        FileNotFoundError: si alguno de los archivos no existe.
        DataValidationError: si el JSON es invalido o algun DataFrame queda vacio.
    """
    if not ruta_threads.exists() or not ruta_messages.exists():
        raise FileNotFoundError("No se encuentran los archivos JSON.")

    logger.info("Cargando datos desde %s y %s ...", ruta_threads, ruta_messages)

    try:
        df_threads = pd.read_json(ruta_threads)
        df_messages = pd.read_json(ruta_messages)
    except ValueError as exc:
        raise DataValidationError(
            f"JSON malformado o estructura incompatible: {exc}"
        ) from exc

    if df_threads.empty or df_messages.empty:
        raise DataValidationError("Alguno de los JSON esta vacio.")

    logger.info(
        "Datos cargados: threads=%d filas, messages=%d filas",
        len(df_threads),
        len(df_messages),
    )
    return df_threads, df_messages


def validar_columnas(
    df: pd.DataFrame, required: set, contexto: str
) -> None:
    """Valida que el DataFrame contenga las columnas requeridas."""
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Faltan columnas en {contexto}: {missing}")
