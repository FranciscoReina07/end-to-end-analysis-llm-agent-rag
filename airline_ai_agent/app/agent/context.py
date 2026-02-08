"""
Contexto explicito del agente.

Reemplaza el uso de variables globales (df_master) por una dataclass
inyectable que contiene todos los datos que el agente necesita.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AgentContext:
    """Contenedor inmutable con los datos que el agente consume.

    Se construye una sola vez al inicio y se pasa explicitamente
    a tools, nodos y funciones. Ningun modulo debe acceder a
    variables globales de datos.
    """

    df: pd.DataFrame
