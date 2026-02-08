"""
Preprocessing y creacion de la tabla maestra.

Transforma los DataFrames crudos en un DataFrame unificado
con conversaciones concatenadas, duraciones y metricas temporales.
"""

import json
import logging
import re
from typing import Any, Optional

import pandas as pd

from app.data.loaders import DataValidationError, validar_columnas

logger = logging.getLogger("aerya")

# Regex precompilada para snake_case
_RE_WHITESPACE = re.compile(r"[\s\-]+")
_RE_NON_ALNUM = re.compile(r"[^0-9a-zA-Z_]+")
_RE_MULTI_UNDERSCORE = re.compile(r"_+")


def _to_snake_case(name: str) -> str:
    name = name.strip()
    name = _RE_WHITESPACE.sub("_", name)
    name = _RE_NON_ALNUM.sub("", name)
    name = _RE_MULTI_UNDERSCORE.sub("_", name)
    return name.lower()


def _normalize_text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _extract_date_value(value: Any) -> Optional[str]:
    if isinstance(value, dict) and "$date" in value:
        return value["$date"]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_to_snake_case(col) for col in df.columns]
    return df


def crear_tabla_maestra(
    df_t: pd.DataFrame, df_m: pd.DataFrame
) -> pd.DataFrame:
    """Crea la tabla maestra uniendo threads con mensajes agrupados.

    Args:
        df_t: DataFrame de threads (crudo).
        df_m: DataFrame de messages (crudo).

    Returns:
        DataFrame con conversaciones concatenadas, duracion y conteo.

    Raises:
        DataValidationError: si faltan columnas o el resultado queda vacio.
    """
    df_t = normalizar_columnas(df_t)
    df_m = normalizar_columnas(df_m)

    validar_columnas(df_t, {"thread_id"}, "threads")
    validar_columnas(df_m, {"thread_id", "content"}, "messages")

    if "created_at" in df_t.columns:
        df_t["created_at"] = pd.to_datetime(
            df_t["created_at"].map(_extract_date_value),
            errors="coerce",
            utc=True,
        )

    col_fecha_msg = (
        "created_at" if "created_at" in df_m.columns else "createdat"
    )
    if col_fecha_msg in df_m.columns:
        df_m["msg_timestamp"] = pd.to_datetime(
            df_m[col_fecha_msg].map(_extract_date_value),
            errors="coerce",
            utc=True,
        )
    else:
        df_m["msg_timestamp"] = pd.NaT

    df_m = df_m.sort_values(by=["thread_id", "msg_timestamp"])
    df_m["clean_content"] = df_m["content"].apply(_normalize_text_content)

    df_grouped = (
        df_m.groupby("thread_id")
        .agg(
            full_conversation=(
                "clean_content",
                lambda x: " | ".join(x[x != ""]),
            ),
            msg_count=("clean_content", "count"),
            first_msg=("msg_timestamp", "min"),
            last_msg=("msg_timestamp", "max"),
        )
        .reset_index()
    )

    if df_grouped.empty:
        raise DataValidationError(
            "No se pudieron agrupar mensajes por thread_id."
        )

    threads_antes = len(df_t)
    df_master = pd.merge(df_t, df_grouped, on="thread_id", how="inner")
    threads_perdidos = threads_antes - len(df_master)

    if df_master.empty:
        raise DataValidationError(
            "La tabla maestra quedo vacia tras el merge."
        )

    if threads_perdidos > 0:
        logger.warning(
            "Merge: %d threads sin mensajes fueron descartados (%d -> %d)",
            threads_perdidos,
            threads_antes,
            len(df_master),
        )

    df_master["duration_minutes"] = (
        (df_master["last_msg"] - df_master["first_msg"]).dt.total_seconds()
        / 60
    ).fillna(0)

    return df_master
