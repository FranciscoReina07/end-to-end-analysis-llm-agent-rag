"""
Carga de datos, EDA, tabla maestra y metricas de impacto.

Responsabilidades:
- Leer JSON crudos y validar estructura.
- Perfilamiento (faltantes, duplicados, unicos).
- Construir tabla maestra (merge threads + messages).
- Calcular metricas de negocio para la aerolinea.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config import (
    DataValidationError,
    THREADS_PATH,
    MESSAGES_PATH,
    logger,
)

# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

_SNAKE_RE_1 = re.compile(r"[\s\-]+")
_SNAKE_RE_2 = re.compile(r"[^0-9a-zA-Z_]+")
_SNAKE_RE_3 = re.compile(r"_+")


def _to_snake_case(name: str) -> str:
    name = _SNAKE_RE_1.sub("_", name.strip())
    name = _SNAKE_RE_2.sub("", name)
    name = _SNAKE_RE_3.sub("_", name)
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


def _get_hashable_df(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    obj_cols = temp.select_dtypes(include=["object"]).columns
    temp[obj_cols] = temp[obj_cols].astype(str).replace("nan", pd.NA)
    return temp


def _safe_datetime_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(
        series.map(lambda x: x.get("$date") if isinstance(x, dict) else x),
        errors="coerce",
    )


# ---------------------------------------------------------------------------
# Validacion
# ---------------------------------------------------------------------------


def validar_columnas(df: pd.DataFrame, required: set, contexto: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Faltan columnas en {contexto}: {missing}")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


def cargar_datos(
    ruta_threads: Path = THREADS_PATH,
    ruta_messages: Path = MESSAGES_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los archivos JSON y devuelve (df_threads, df_messages)."""
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


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------


def profile_missing(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    missing = df.isna().sum().sort_values(ascending=False)
    return missing[missing > 0].head(top_n).to_frame(name="missing_count")


def profile_uniques(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    safe = _get_hashable_df(df)
    uniques = safe.nunique(dropna=True).sort_values(ascending=False).head(top_n)
    return uniques.to_frame(name="unique_values")


def safe_duplicated_count(df: pd.DataFrame) -> int:
    safe = _get_hashable_df(df)
    return int(safe.duplicated().sum())


def analizar_dataset(df: pd.DataFrame, nombre: str) -> None:
    """Imprime un perfilamiento rapido del dataset."""
    print(f"ANALISIS DEL DATASET: {nombre}")
    print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")

    n_dup = safe_duplicated_count(df)
    perc = (n_dup / len(df)) * 100 if len(df) else 0
    print(f"Duplicados: {n_dup} ({perc:.2f}%)")

    print("\nTop 15 columnas con faltantes")
    missing_df = profile_missing(df, top_n=15)
    if missing_df.empty:
        print("No hay valores faltantes.")
    else:
        print(missing_df)

    print("\nTop 15 columnas con valores unicos")
    print(profile_uniques(df, top_n=15))
    print("-" * 60)


# ---------------------------------------------------------------------------
# Tabla maestra
# ---------------------------------------------------------------------------


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_to_snake_case(col) for col in df.columns]
    return df


def crear_tabla_maestra(
    df_t: pd.DataFrame, df_m: pd.DataFrame
) -> pd.DataFrame:
    """Merge threads + messages agrupados en una tabla maestra."""
    df_t = normalizar_columnas(df_t)
    df_m = normalizar_columnas(df_m)

    validar_columnas(df_t, {"thread_id"}, "threads")
    validar_columnas(df_m, {"thread_id", "content"}, "messages")

    if "created_at" in df_t.columns:
        df_t["created_at"] = pd.to_datetime(
            df_t["created_at"].map(_extract_date_value), errors="coerce", utc=True
        )

    col_fecha_msg = "created_at" if "created_at" in df_m.columns else "createdat"
    if col_fecha_msg in df_m.columns:
        df_m["msg_timestamp"] = pd.to_datetime(
            df_m[col_fecha_msg].map(_extract_date_value), errors="coerce", utc=True
        )
    else:
        df_m["msg_timestamp"] = pd.NaT

    df_m = df_m.sort_values(by=["thread_id", "msg_timestamp"])
    df_m["clean_content"] = df_m["content"].apply(_normalize_text_content)

    df_grouped = (
        df_m.groupby("thread_id")
        .agg(
            full_conversation=("clean_content", lambda x: " | ".join(x[x != ""])),
            msg_count=("clean_content", "count"),
            first_msg=("msg_timestamp", "min"),
            last_msg=("msg_timestamp", "max"),
        )
        .reset_index()
    )

    if df_grouped.empty:
        raise DataValidationError("No se pudieron agrupar mensajes por thread_id.")

    threads_antes = len(df_t)
    df_master = pd.merge(df_t, df_grouped, on="thread_id", how="inner")
    threads_perdidos = threads_antes - len(df_master)

    if df_master.empty:
        raise DataValidationError("La tabla maestra quedo vacia tras el merge.")

    if threads_perdidos > 0:
        logger.warning(
            "Merge: %d threads sin mensajes descartados (%d -> %d)",
            threads_perdidos,
            threads_antes,
            len(df_master),
        )

    df_master["duration_minutes"] = (
        (df_master["last_msg"] - df_master["first_msg"]).dt.total_seconds() / 60
    ).fillna(0)

    return df_master


# ---------------------------------------------------------------------------
# Metricas de impacto
# ---------------------------------------------------------------------------


def metricas_impacto_aerolinea(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula metricas clave de negocio para la aerolinea."""
    if df.empty:
        raise DataValidationError("No hay datos para calcular metricas.")

    required = {"escalate_conversation", "duration_minutes", "msg_count", "status"}
    available = required.intersection(df.columns)

    metrics: Dict[str, Any] = {}

    if "escalate_conversation" in available:
        target = df["escalate_conversation"].fillna(0).astype(int)
        metrics["tasa_escalamiento"] = float(target.mean())

    if "duration_minutes" in available:
        metrics["duracion_promedio_min"] = float(df["duration_minutes"].mean())
        metrics["duracion_p90_min"] = float(df["duration_minutes"].quantile(0.90))

    if "msg_count" in available:
        metrics["mensajes_promedio"] = float(df["msg_count"].mean())
        metrics["mensajes_p90"] = float(df["msg_count"].quantile(0.90))

    if "status" in available:
        metrics["estado_top"] = json.dumps(
            df["status"].value_counts().head(5).to_dict()
        )

    return pd.DataFrame([metrics])


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------


def guardar_df_master(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("df_master guardado: %s", path)


def cargar_df_master_si_existe(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        logger.info("Cargando df_master desde cache: %s", path)
        return pd.read_parquet(path)
    return None


# ---------------------------------------------------------------------------
# Texto para el agente
# ---------------------------------------------------------------------------


def construir_resumen_contexto(df: pd.DataFrame) -> str:
    """Construye un resumen textual de metricas clave para el system prompt."""
    partes: List[str] = []
    if "escalate_conversation" in df.columns:
        tasa = df["escalate_conversation"].fillna(0).astype(int).mean()
        partes.append(f"Tasa de escalamiento: {tasa:.3f}")
    if "duration_minutes" in df.columns:
        partes.append(f"Duracion promedio (min): {df['duration_minutes'].mean():.2f}")
    if "msg_count" in df.columns:
        partes.append(f"Mensajes promedio: {df['msg_count'].mean():.2f}")
    return "\n".join(partes)


def get_conversation_text(df: pd.DataFrame, thread_id: Optional[str]) -> str:
    """Recupera la conversacion completa de un thread_id."""
    if not thread_id or "thread_id" not in df.columns:
        return ""
    match = df[df["thread_id"].astype(str) == str(thread_id)]
    if match.empty:
        return ""
    if "full_conversation" in df.columns:
        return str(match["full_conversation"].iloc[0])
    text_cols = [c for c in ["content", "message_text", "text", "body"] if c in df.columns]
    if text_cols:
        return "\n".join(match[text_cols[0]].astype(str).tolist())
    return ""
