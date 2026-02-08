"""
Evaluacion de calidad de las respuestas del agente.

Mide si el agente devuelve JSON valido con la estructura esperada.
Sesiones aisladas por pregunta para evaluar capacidad en frio.
"""

import json
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.graph import get_agent_app
from app.metrics.business_metrics import construir_resumen_contexto
from app.prompts.agent_prompts import SYSTEM_PROMPT_TEMPLATE

logger = logging.getLogger("aerya")

EXPECTED_KEYS = {"summary", "insights", "risks", "actions", "metrics_used"}

_JSON_REGEX = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> Optional[str]:
    """Extrae el primer bloque JSON de un texto, incluso si tiene texto alrededor."""
    match = _JSON_REGEX.search(text)
    return match.group(0) if match else None


def evaluar_respuesta_agente(raw_response: str) -> Dict[str, Any]:
    """Evalua la calidad de una respuesta del agente.

    Verifica:
    - Si es JSON valido
    - Si contiene las claves esperadas
    - Si insights y actions no estan vacios
    """
    result: Dict[str, Any] = {
        "json_valido": False,
        "claves_presentes": [],
        "claves_faltantes": [],
        "score_estructura": 0.0,
        "insights_no_vacios": False,
        "actions_no_vacios": False,
    }

    json_str = _extract_json(raw_response) if raw_response else None

    try:
        parsed = (
            json.loads(json_str)
            if json_str
            else json.loads(raw_response)
        )
        result["json_valido"] = True
    except (json.JSONDecodeError, ValueError, TypeError):
        result["claves_faltantes"] = list(EXPECTED_KEYS)
        return result

    presentes = EXPECTED_KEYS.intersection(parsed.keys())
    faltantes = EXPECTED_KEYS - presentes

    result["claves_presentes"] = sorted(presentes)
    result["claves_faltantes"] = sorted(faltantes)
    result["score_estructura"] = len(presentes) / len(EXPECTED_KEYS)

    if "insights" in parsed and isinstance(parsed["insights"], list):
        result["insights_no_vacios"] = len(parsed["insights"]) > 0

    if "actions" in parsed and isinstance(parsed["actions"], list):
        result["actions_no_vacios"] = len(parsed["actions"]) > 0

    return result


def ejecutar_evaluacion_agente(
    preguntas: List[str],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Ejecuta multiples preguntas en sesiones aisladas y evalua las respuestas.

    Cada pregunta se ejecuta en una sesion independiente (uuid unico)
    para medir capacidad del agente en frio.
    """
    resultados = []

    contexto = construir_resumen_contexto(df)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(contexto=contexto)

    for pregunta in preguntas:
        eval_session = f"eval-{uuid.uuid4().hex[:8]}"
        eval_app = get_agent_app(eval_session)

        t0 = time.time()
        try:
            res = eval_app.invoke(
                {
                    "messages": [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=pregunta),
                    ],
                    "tool_calls_count": 0,
                    "thread_id": None,
                    "route": None,
                    "analysis": {},
                },
                config={
                    "configurable": {"thread_id": eval_session},
                    "recursion_limit": 6,
                },
            )
            answer = res["messages"][-1].content
        except Exception as exc:
            answer = str(exc)

        elapsed = time.time() - t0
        evaluacion = evaluar_respuesta_agente(answer)
        evaluacion["pregunta"] = pregunta[:80]
        evaluacion["tiempo_s"] = round(elapsed, 2)
        resultados.append(evaluacion)

    return pd.DataFrame(resultados)
