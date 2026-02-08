"""
Metricas de impacto para la aerolinea.

Funciones puras que reciben DataFrames y devuelven resultados.
"""

import json
import logging
from typing import Any, Dict

import pandas as pd

from app.data.loaders import DataValidationError

logger = logging.getLogger("aerya")


def metricas_impacto_aerolinea(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula metricas de negocio a partir de la tabla maestra.

    Returns:
        DataFrame de una fila con las metricas calculadas.
    """
    if df.empty:
        raise DataValidationError(
            "No hay datos para calcular metricas."
        )

    required = {
        "escalate_conversation",
        "duration_minutes",
        "msg_count",
        "status",
    }
    available = required.intersection(df.columns)

    metrics: Dict[str, Any] = {}

    if "escalate_conversation" in available:
        target = df["escalate_conversation"].fillna(0).astype(int)
        metrics["tasa_escalamiento"] = target.mean()

    if "duration_minutes" in available:
        metrics["duracion_promedio_min"] = df["duration_minutes"].mean()
        metrics["duracion_p90_min"] = df["duration_minutes"].quantile(0.90)

    if "msg_count" in available:
        metrics["mensajes_promedio"] = df["msg_count"].mean()
        metrics["mensajes_p90"] = df["msg_count"].quantile(0.90)

    if "status" in available:
        metrics["estado_top"] = json.dumps(
            df["status"].value_counts().head(5).to_dict()
        )

    metrics_df = pd.DataFrame([metrics])
    return metrics_df


def construir_resumen_contexto(df: pd.DataFrame) -> str:
    """Genera un string de contexto con metricas clave para inyectar al LLM."""
    resumen = []
    if "escalate_conversation" in df.columns:
        tasa = df["escalate_conversation"].fillna(0).astype(int).mean()
        resumen.append(f"Tasa de escalamiento: {tasa:.3f}")
    if "duration_minutes" in df.columns:
        resumen.append(
            f"Duracion promedio (min): {df['duration_minutes'].mean():.2f}"
        )
    if "msg_count" in df.columns:
        resumen.append(
            f"Mensajes promedio: {df['msg_count'].mean():.2f}"
        )
    return "\n".join(resumen)
