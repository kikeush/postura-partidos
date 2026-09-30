# -*- coding: utf-8 -*-
"""Los nombres legibles de modelos coinciden entre el paquete, la aplicación y los resultados.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-28. Licencia: MIT.
"""
import json
import os
import sys

from postura.nombres import NOMBRE_MODELO, nombre_modelo

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "app"))
import utilidades as u  # noqa: E402


def test_mismos_nombres_en_paquete_y_aplicacion():
    assert u.NOMBRE_MODELO == NOMBRE_MODELO


def test_todo_modelo_de_los_resultados_tiene_nombre():
    for archivo in ("fase2_resultados.json", "fase3_resultados.json"):
        ruta = os.path.join(RAIZ, "resultados", archivo)
        if not os.path.exists(ruta):
            continue
        tareas = json.load(open(ruta, encoding="utf-8"))["tareas"]
        for t in tareas.values():
            for clave in t.get("modelos", {}):
                assert clave in NOMBRE_MODELO, clave


def test_clave_desconocida_conserva_el_texto():
    assert nombre_modelo("otro_modelo") == "otro modelo"
