# -*- coding: utf-8 -*-
"""Pruebas de postura.pipeline.Clasificador con modelos diminutos entrenados con etiquetas
sintéticas en una carpeta temporal.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.
"""
import json
import os
import sys

import joblib
import numpy as np
import pytest

from postura.menciones import buscar_menciones, oraciones
from postura.modelos import (ajustar_con_probabilidad, estado_pbc4cip, lexico, logistica,
                             svm_lineal, catalogo)
from postura.particion import pliegues_por_fecha
from postura.pipeline import ARCHIVOS, POSTURAS_PREDICHAS, Clasificador, decidir_postura

PARRAFO = ("El PRIAN saqueó al país durante décadas. ¿Y el Partido Verde? Votó con Morena "
           "en la Cámara. Compramos pan en la esquina. Aquí no hay partidos.")


@pytest.fixture(scope="module")
def dir_modelos(tmp_path_factory, tabla_etiquetada):
    """Entrena tres modelos diminutos (T1 léxico, T2a logística, T2b SVM calibrado)."""
    destino = tmp_path_factory.mktemp("modelos")
    sub = tabla_etiquetada.sample(n=400, random_state=0).reset_index(drop=True)
    y1 = sub["es_partido"].astype(int)
    partido = sub[sub["es_partido"] == 1].reset_index(drop=True)
    y2a = (partido["postura"] == "desfavorable").astype(int)
    y2b = (partido["postura"] == "favorable").astype(int)
    modelos = {
        "T1": lexico(rapido=True).estimador.fit(sub, y1),
        "T2a": logistica(rapido=True).estimador.fit(partido, y2a),
        "T2b": ajustar_con_probabilidad(svm_lineal(rapido=True).estimador, partido, y2b,
                                        pliegues_por_fecha(partido["fecha"], 3)),
    }
    for tarea, modelo in modelos.items():
        joblib.dump(modelo, destino / ARCHIVOS[tarea])
    umbrales = {t: {"umbral": 0.5, "modelo": "diminuto"} for t in modelos}
    (destino / "umbrales.json").write_text(json.dumps(umbrales), encoding="utf-8")
    return destino


def test_analizar_una_entrada_por_mencion(dir_modelos):
    clf = Clasificador(dir_modelos)
    salida = clf.analizar(PARRAFO, pregunta_prensa="¿Qué opina de la oposición?")
    esperadas = [(o, m[2]) for o in oraciones(PARRAFO) for m in buscar_menciones(o)]
    assert [(r["oracion"], r["mencion"]) for r in salida] == esperadas
    assert [r["mencion"] for r in salida] == ["PRIAN", "Partido Verde", "Morena", "pan"]
    claves = {"oracion", "mencion", "grupo", "objetivo", "p_partido", "p_desfavorable",
              "p_favorable", "postura_predicha"}
    for r in salida:
        assert set(r) == claves
        for p in ("p_partido", "p_desfavorable", "p_favorable"):
            assert 0.0 <= r[p] <= 1.0 and isinstance(r[p], float)
        assert r["postura_predicha"] in POSTURAS_PREDICHAS
    assert salida[1]["grupo"] == "PVEM" and salida[3]["grupo"] == "ambiguo_pan"


def test_analizar_sin_menciones_devuelve_lista_vacia(dir_modelos):
    assert Clasificador(dir_modelos).analizar("Buenos días a todos. Hoy hace calor.") == []
    assert Clasificador(dir_modelos).analizar("") == []


def test_contexto_previo_se_acumula(dir_modelos):
    clf = Clasificador(dir_modelos)
    tabla = clf.menciones_de_parrafo("Primera oración sin nada. El PRI votó.", contexto_previo="Antes.")
    assert len(tabla) == 1
    assert tabla.loc[0, "contexto_previo"] == "Antes. Primera oración sin nada."


def test_analizar_tabla_vectorizado(dir_modelos, menciones):
    clf = Clasificador(dir_modelos)
    sub = menciones.sample(n=300, random_state=1)
    res = clf.analizar_tabla(sub)
    assert len(res) == len(sub) and (res.index == sub.index).all()
    for p in ("p_partido", "p_desfavorable", "p_favorable"):
        assert res[p].between(0, 1).all()
    assert set(res["postura_predicha"]) <= set(POSTURAS_PREDICHAS)
    # Sin oracion_marcada (se reconstruye con oracion, inicio y fin) da lo mismo.
    res2 = clf.analizar_tabla(sub.drop(columns=["oracion_marcada"]))
    assert np.allclose(res["p_partido"], res2["p_partido"])
    # Tabla vacía.
    vacia = clf.analizar_tabla(sub.head(0))
    assert len(vacia) == 0 and "postura_predicha" in vacia.columns


def test_analizar_coincide_con_analizar_tabla(dir_modelos):
    clf = Clasificador(dir_modelos)
    lista = clf.analizar(PARRAFO)
    tabla = clf.analizar_tabla(clf.menciones_de_parrafo(PARRAFO))
    assert np.allclose([r["p_desfavorable"] for r in lista], tabla["p_desfavorable"])


def test_decidir_postura():
    r = decidir_postura([0.2, 0.9, 0.9, 0.9, 0.9, 0.9],
                        [0.9, 0.8, 0.1, 0.8, 0.6, 0.1],
                        [0.9, 0.1, 0.8, 0.9, 0.9, 0.1],
                        0.5, 0.5, 0.5)
    assert list(r) == ["no_partido", "desfavorable", "favorable", "favorable", "favorable", "neutral"]
    # Margen relativo: con umbral de favorable alto, 0.9 sobre 0.8 pesa menos que 0.8 sobre 0.5.
    assert decidir_postura([1], [0.8], [0.9], 0.5, 0.5, 0.8)[0] == "desfavorable"


def test_contrato_con_la_app(dir_modelos, raiz):
    """app/app.py dibuja la salida de analizar con utilidades.tabla_resultados_html: las claves
    que lee deben existir y no_partido debe mostrarse como no es partido."""
    if os.path.join(raiz, "app") not in sys.path:
        sys.path.insert(0, os.path.join(raiz, "app"))
    import utilidades as u
    salida = Clasificador(dir_modelos).analizar(PARRAFO)
    for r in salida:
        assert {c for c, _, _ in u.COLUMNAS_CLASICO} <= set(r)
        assert {"oracion", "mencion", "objetivo", "postura_predicha"} <= set(r)
    html = u.tabla_resultados_html(salida, u.COLUMNAS_CLASICO)
    assert html.count("<mark>") == len(salida) and "<mark>Partido Verde</mark>" in html
    assert "no es partido" in u.chip_postura("no_partido")


def test_faltan_modelos(tmp_path):
    with pytest.raises(FileNotFoundError, match="04_entrenar_clasicos"):
        Clasificador(tmp_path)


def test_pbc4cip_no_falla():
    estado = estado_pbc4cip()
    assert set(estado) >= {"disponible", "motivo"}
    nombres = set(catalogo(rapido=True))
    assert {"base_grupo", "base_longitud", "logistica", "svm_lineal", "bosque_aleatorio"} <= nombres
    assert ("pbc4cip" in nombres) == estado["disponible"]
