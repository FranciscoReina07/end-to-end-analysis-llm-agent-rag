"""
Herramientas del agente como factory.

build_tools(df) recibe el DataFrame explicitamente y devuelve una lista
de herramientas LangChain. Cada tool captura df via closure, evitando
depender de variables globales.

Por que closures y no estado global:
- Las tools se compilan una vez al inicio con los datos correctos.
- En un entorno multi-tenant, cada sesion podria recibir datos distintos.
- Facilita testing: se puede pasar un DataFrame mock sin monkeypatching.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.tools import tool

logger = logging.getLogger("aerya")


def _safe_datetime_series(series: pd.Series) -> pd.Series:
    """Convierte una serie a datetime, manejando formatos MongoDB {$date: ...}."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(
        series.map(
            lambda x: x.get("$date") if isinstance(x, dict) else x
        ),
        errors="coerce",
    )


def _get_conversation_text(
    df: pd.DataFrame, thread_id: Optional[str]
) -> str:
    """Extrae el texto de conversacion para un thread_id dado."""
    if not thread_id or "thread_id" not in df.columns:
        return ""

    match = df[df["thread_id"].astype(str) == str(thread_id)]
    if match.empty:
        return ""

    if "full_conversation" in df.columns:
        return str(match["full_conversation"].iloc[0])

    text_cols = [
        c
        for c in ["content", "message_text", "text", "body"]
        if c in df.columns
    ]
    if text_cols:
        return "\n".join(match[text_cols[0]].astype(str).tolist())

    return ""


def build_tools(df: pd.DataFrame) -> List:
    """Construye las herramientas del agente usando el df capturado por closure.

    Args:
        df: DataFrame maestro con las conversaciones y metricas.

    Returns:
        Lista de tools listas para bind_tools.
    """

    @tool
    def obtener_metricas_resumen() -> str:
        """Devuelve un resumen de metricas clave de la aerolinea."""
        resumen = []
        if "escalate_conversation" in df.columns:
            tasa = (
                df["escalate_conversation"].fillna(0).astype(int).mean()
            )
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

    @tool
    def top_hilos_por_mensajes(n: int = 5) -> str:
        """Devuelve los top hilos con mas mensajes y su conteo."""
        if "msg_count" not in df.columns:
            return "No existe la columna msg_count."
        top_df = (
            df[["thread_id", "msg_count"]]
            .sort_values("msg_count", ascending=False)
            .head(n)
        )
        return top_df.to_string(index=False)

    @tool
    def top_hilos_por_duracion(n: int = 5) -> str:
        """Devuelve los top hilos con mayor duracion (min)."""
        if "duration_minutes" not in df.columns:
            return "No existe la columna duration_minutes."
        top_df = (
            df[["thread_id", "duration_minutes"]]
            .sort_values("duration_minutes", ascending=False)
            .head(n)
        )
        return top_df.to_string(index=False)

    @tool
    def tasa_escalamiento_por_categoria(
        col: str = "status", top_n: int = 8
    ) -> str:
        """Calcula la tasa de escalamiento por categoria (status, platform, source)."""
        if (
            col not in df.columns
            or "escalate_conversation" not in df.columns
        ):
            return f"No existe la columna {col} o escalate_conversation."
        tmp = df[[col, "escalate_conversation"]].copy()
        tmp["escalate_conversation"] = (
            tmp["escalate_conversation"].fillna(0).astype(int)
        )
        rate = (
            tmp.groupby(col)["escalate_conversation"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        return rate.to_string(index=False)

    @tool
    def resumen_tiempos_operativos() -> str:
        """Devuelve estadisticas de duracion y volumen de mensajes."""
        if (
            "duration_minutes" not in df.columns
            or "msg_count" not in df.columns
        ):
            return "No existen columnas de duracion o mensajes."
        stats: Dict[str, float] = {
            "duracion_promedio_min": df["duration_minutes"].mean(),
            "duracion_p90_min": df["duration_minutes"].quantile(0.90),
            "mensajes_promedio": df["msg_count"].mean(),
            "mensajes_p90": df["msg_count"].quantile(0.90),
        }
        return pd.DataFrame([stats]).to_string(index=False)

    @tool
    def tendencia_escalamiento_mensual() -> str:
        """Muestra la tendencia mensual de escalamiento."""
        if (
            "created_at" not in df.columns
            or "escalate_conversation" not in df.columns
        ):
            return "No existen columnas requeridas para tendencia mensual."
        tmp = df[["created_at", "escalate_conversation"]].copy()
        tmp["created_at"] = _safe_datetime_series(tmp["created_at"])
        tmp = tmp.dropna(subset=["created_at"])
        tmp["month"] = (
            tmp["created_at"].dt.to_period("M").astype(str)
        )
        tmp["escalate_conversation"] = (
            tmp["escalate_conversation"].fillna(0).astype(int)
        )
        trend = (
            tmp.groupby("month")["escalate_conversation"]
            .mean()
            .reset_index()
        )
        return trend.to_string(index=False)

    @tool
    def obtener_conversacion_por_thread(thread_id: str) -> str:
        """Devuelve la conversacion completa para un thread_id."""
        text = _get_conversation_text(df, thread_id)
        if not text:
            return "No se encontro conversacion para ese thread_id."
        return text

    return [
        obtener_metricas_resumen,
        top_hilos_por_mensajes,
        top_hilos_por_duracion,
        tasa_escalamiento_por_categoria,
        resumen_tiempos_operativos,
        tendencia_escalamiento_mensual,
        obtener_conversacion_por_thread,
    ]
