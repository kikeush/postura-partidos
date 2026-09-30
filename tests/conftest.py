# -*- coding: utf-8 -*-
"""Configuración común de las pruebas: ruta del paquete y datos compartidos.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

Las etiquetas que usan las pruebas son SINTÉTICAS (tests/sinteticas.py): sirven para ejercitar
el código, no para medir nada.
"""
import os
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for ruta in (RAIZ, os.path.join(RAIZ, "tests")):
    if ruta not in sys.path:
        sys.path.insert(0, ruta)

from postura.particion import RUTA_MENCIONES, RUTA_MUESTRA, leer_muestra, unir_etiquetas  # noqa: E402
from sinteticas import etiquetas_sinteticas  # noqa: E402


def pytest_configure(config):
    """Registra la marca lento también sin pytest.ini (por ejemplo, en el ZIP solo con código)."""
    config.addinivalue_line("markers", "lento: prueba de integración que ejecuta scripts completos")
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning")


@pytest.fixture(scope="session")
def raiz():
    return RAIZ


@pytest.fixture(scope="session")
def muestra():
    if not os.path.exists(RUTA_MUESTRA):
        pytest.skip("No está datos/muestra_anotacion.csv")
    return leer_muestra()


@pytest.fixture(scope="session")
def menciones():
    if not os.path.exists(RUTA_MENCIONES):
        pytest.skip("No está datos/menciones.parquet")
    import pandas as pd
    return pd.read_parquet(RUTA_MENCIONES)


@pytest.fixture(scope="session")
def etiquetas(muestra):
    return etiquetas_sinteticas(muestra, semilla=11)


@pytest.fixture(scope="session")
def tabla_etiquetada(muestra, etiquetas):
    tabla, _ = unir_etiquetas(muestra, etiquetas)
    return tabla
