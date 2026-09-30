# -*- coding: utf-8 -*-
"""Pruebas unitarias de app/utilidades.py (funciones sin Streamlit) y del contrato entre la
aplicación y los archivos que escriben scripts/04_entrenar_clasicos.py y 07_fase3_evaluar.py.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(RAIZ, "app") not in sys.path:
    sys.path.insert(0, os.path.join(RAIZ, "app"))

import utilidades as u  # noqa: E402


# ---------------------------------------------------------------------------------------
# Resaltado y textos
# ---------------------------------------------------------------------------------------

def test_resaltar_marcas_y_escape():
    h = u.resaltar_html("Los del [[Prian]] <b>x</b> $5", "Prian")
    assert "<mark>Prian</mark>" in h and "&lt;b&gt;" in h and "&#36;5" in h and "[[" not in h


def test_resaltar_ocurrencia_repetida():
    o = "El PAN, que surge contra Cárdenas, se convirtió en el PRIAN, y fueron el PRIAN."
    filas = [dict(oracion=o, mencion="PRIAN", p_partido=0.9), dict(oracion=o, mencion="PRIAN", p_partido=0.9)]
    html = u.tabla_resultados_html(filas)
    primera, segunda = html.split("<tr><td>2</td>")
    assert primera.index("<mark>PRIAN</mark>") < primera.index(", y fueron")
    assert "y fueron el <mark>PRIAN</mark>" in segunda


def test_resaltar_no_confunde_pan_dentro_de_prian():
    assert u.resaltar_html("El PRIAN y el PAN", "PAN") == "El PRIAN y el <mark>PAN</mark>"


def test_formato_miles():
    # Sin separador hasta cuatro cifras; espacio fino sin salto (U+202F) desde cinco.
    assert u.formato_miles(1945) == "1945"
    assert u.formato_miles(11714) == "11\u202f714"
    assert u.formato_miles(9999) == "9999" and u.formato_miles(10000) == "10\u202f000"
    assert u.formato_miles(1234567.4) == "1\u202f234\u202f567"
    assert u.formato_miles("texto") == "texto"
    assert "," not in u.formato_miles(11714)


def test_normalizar_postura():
    entradas = ("Desfavorable", "FAVORABLE", "neutral", "no_aplica", "no_partido", None, float("nan"), "D")
    assert [u.normalizar_postura(x) for x in entradas] == [
        "desfavorable", "favorable", "neutral", "no_aplica", "no_aplica", "no_aplica", "no_aplica",
        "desfavorable"]


def test_menciones_del_parrafo_contexto():
    items = u.menciones_del_parrafo("Primera oración sin nada. El PRI votó y Morena también.",
                                    "contexto previo", "¿pregunta?")
    assert [i["mencion"] for i in items] == ["PRI", "Morena"]
    assert items[0]["contexto_previo"].endswith("Primera oración sin nada.")
    assert items[1]["oracion_marcada"].endswith("[[Morena]] también.")
    filas = u.qwen_a_filas(items, [{"X": 0.1, "D": 0.6, "F": 0.2, "N": 0.1},
                                   {"X": 0.7, "D": 0.1, "F": 0.1, "N": 0.1}])
    assert [f["postura_predicha"] for f in filas] == ["desfavorable", "no_aplica"]
    assert abs(filas[0]["p_partido"] - 0.9) < 1e-9


# ---------------------------------------------------------------------------------------
# Resultados de ROC-AUC: formatos de 04 y 07
# ---------------------------------------------------------------------------------------

def _prueba(auc, inf, sup):
    """Nodo prueba como lo escribe scripts/04_entrenar_clasicos.py (metricas_prueba)."""
    return {"n": 100, "positivos": 40, "auc": auc,
            "ic_bootstrap_conferencias": {"estimacion": auc, "ic_inferior": inf, "ic_superior": sup},
            "ic_delong": {"auc": auc, "ic_inferior": inf - 0.01, "ic_superior": sup + 0.01},
            "pr_auc": 0.5}


FASE2 = {
    "etiquetas_sinteticas": True,
    "tiempos": {"busqueda_paralela": {"auc_n_jobs_1": 0.8}},
    "tareas": {
        "T1": {"mejor_modelo": "logistica",
               "modelos": {"base_grupo": {"auc_validacion": 0.7, "prueba": _prueba(0.95, 0.9, 0.99)},
                           "logistica": {"auc_validacion": 0.9, "prueba": _prueba(0.9, 0.85, 0.94)}},
               "comparaciones_del_mejor": {"base_grupo": {"delong": {"auc_1": 0.9, "auc_2": 0.95}}},
               "ablacion_enmascarada": {"prueba": _prueba(0.97, 0.9, 0.99)}},
        "T2a": {"mejor_modelo": "svm_lineal",
                "modelos": {"svm_lineal": {"prueba": _prueba(0.8, 0.7, 0.88)}}},
    },
}


def test_resultados_auc_formato_de_04():
    t = u.resultados_auc(FASE2, "Fase 2")
    assert sorted(zip(t.tarea, t.modelo)) == [
        ("T1", "Regresión logística"), ("T1", "Regresión logística sin nombre del partido (ablación)"),
        ("T1", "Solo nombre del partido"), ("T2a", "SVM lineal")]
    fila = t[(t.tarea == "T1") & (t.modelo == "Regresión logística")].iloc[0]
    # El intervalo sale del bootstrap por conferencias, no de DeLong.
    assert (fila.ic_inf, fila.ic_sup) == (0.85, 0.94) and bool(fila.elegido)
    assert t["ic_inf"].notna().all() and int(t["elegido"].sum()) == 2
    m = u.mejores_por_tarea(t)
    # Se muestra el elegido en validación aunque la base y la ablación tengan más AUC en prueba.
    assert list(zip(m.tarea, m.modelo)) == [("T1", "Regresión logística"), ("T2a", "SVM lineal")]
    assert u.son_sinteticos(FASE2) and not u.son_sinteticos({"etiquetas_sinteticas": False})


def test_resultados_auc_formato_de_07():
    fase3 = {"mejor_fase2": {"T1": "logistica"},
             "tareas": {"T1": {"modelos": {"qwen_cero": {"auc": 0.91, "ic_inferior": 0.86, "ic_superior": 0.95}},
                               "comparacion_con_mejor_fase2": {"qwen_cero": {"auc_fase3": 0.91}},
                               "embeddings": {"emb_logistica": {"auc_validacion": 0.8}}}}}
    t = u.resultados_auc(fase3, "Fase 3")
    assert list(zip(t.tarea, t.modelo, t.ic_inf, t.ic_sup)) == [("T1", "LLM sin ejemplos", 0.86, 0.95)]
    assert not t["elegido"].any()
    todo = pd.concat([u.resultados_auc(FASE2, "Fase 2"), t], ignore_index=True)
    assert u.mejores_por_tarea(todo).iloc[0]["modelo"] == "Regresión logística"


def test_nombre_modelo_desconocido_conserva_la_clave():
    assert u.nombre_modelo("otro_modelo") == "otro modelo"
    assert u.nombre_modelo("svm_lineal") == "SVM lineal"


@pytest.mark.parametrize("clave", ["fase2", "fase3"])
def test_nombres_de_modelo_cubren_los_resultados(clave, raiz):
    """Ningún nombre crudo del JSON (con guion bajo o sin acento) llega a la interfaz."""
    ruta = u.rutas(raiz)[clave]
    if not ruta.exists():
        pytest.skip(f"No está {ruta}")
    datos, aviso = u.leer_json(ruta)
    assert aviso is None
    tabla = u.resultados_auc(datos, clave)
    assert not tabla.empty
    crudos = {"logistica", "lexico", "svm lineal", "bosque aleatorio", "qwen cero", "qwen cot", "qwen pocos",
              "emb logistica", "emb svm", "emb bosque", "bert ajustado", "base grupo", "base longitud"}
    for nombre in tabla["modelo"]:
        assert "_" not in nombre and nombre not in crudos, nombre
        assert nombre.split(" sin nombre")[0] in u.NOMBRE_MODELO.values(), nombre


def test_resultados_auc_otras_orientaciones():
    a = u.resultados_auc({"T1": {"svm": {"auc": 0.9, "ic_inf": 0.85, "ic_sup": 0.94}},
                          "t2a": {"rl": {"auc": 0.8}}, "fecha": "x"}, "Fase 2")
    b = u.resultados_auc({"tareas": {"svm": {"T1": {"auc": 0.9}}, "rf": {"T2b": {"auc": 0.7}}}}, "Fase 3")
    assert set(a.tarea) == {"T1", "T2a"} and a.loc[a.tarea == "T2a", "ic_inf"].isna().all()
    assert sorted(zip(b.tarea, b.modelo)) == [("T1", "svm"), ("T2b", "rf")]
    assert u.resultados_auc({"otro": 1}, "F").empty
    assert list(u.mejores_por_tarea(pd.concat([a, b])).tarea) == ["T1", "T2a", "T2b"]


# ---------------------------------------------------------------------------------------
# Resumen, agregados, rutas y ejemplos
# ---------------------------------------------------------------------------------------

def test_menciones_por_grupo_ambas_orientaciones():
    x = u.menciones_por_grupo({"por_grupo_y_presidente": {"AMLO": {"PRI": 3, "ambiguo_pan": 1},
                                                          "Sheinbaum": {"PRI": 2}}})
    y = u.menciones_por_grupo({"por_grupo_y_presidente": {"PRI": {"AMLO": 3, "Sheinbaum": 2},
                                                          "ambiguo_pan": {"AMLO": 1}}})
    ordenar = lambda t: t.sort_values(["presidente", "grupo"]).reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(ordenar(x), ordenar(y))
    assert "pan (ambiguo)" in set(x.etiqueta)
    assert u.menciones_por_grupo({}) is None


def test_agregados_y_porcentajes():
    t = pd.DataFrame({"presidente": ["AMLO", "AMLO"], "anio": [2019, 2020], "objetivo": ["PRI", "PRI"],
                      "desfavorables": [3, 0], "favorables": [1, 0], "neutrales": [1, 0]})
    p = u.preparar_agregados(t)
    assert abs(p.loc[0, "pct_desfavorable"] - 60) < 1e-9 and np.isnan(p.loc[1, "pct_desfavorable"])
    g = u.agregar(p, ["presidente", "objetivo"])
    assert int(g.loc[0, "menciones_partido_pred"]) == 5
    assert abs(u.formato_largo_postura(g).porcentaje.sum() - 100) < 1e-9


def test_insumo_disponible(tmp_path):
    (tmp_path / "modelos").mkdir()
    (tmp_path / "figuras").mkdir()
    assert not u.insumo_disponible("modelos", tmp_path / "modelos")
    assert not u.insumo_disponible("figuras", tmp_path / "figuras")
    # Como en una copia de git: umbrales.json sin los tres modelos entrenados.
    (tmp_path / "modelos" / "umbrales.json").write_text("{}", encoding="utf-8")
    assert not u.insumo_disponible("modelos", tmp_path / "modelos")
    for tarea in ("T1", "T2a"):
        (tmp_path / "modelos" / f"modelo_{tarea}.joblib").write_bytes(b"")
    assert not u.insumo_disponible("modelos", tmp_path / "modelos")
    (tmp_path / "modelos" / "modelo_T2b.joblib").write_bytes(b"")
    (tmp_path / "figuras" / "fase2_roc_T1.png").write_bytes(b"")
    assert u.insumo_disponible("modelos", tmp_path / "modelos")
    assert u.insumo_disponible("figuras", tmp_path / "figuras")
    assert not u.insumo_disponible("fase2", tmp_path / "no_existe.json")


def test_rutas_siguen_a_los_scripts(tmp_path):
    r = u.rutas(tmp_path)
    assert r["fase2"] == tmp_path / "resultados" / "fase2_resultados.json"
    assert r["agregados_corpus"] == tmp_path / "resultados" / "agregados_corpus.csv"
    assert r["agregados_presidente"] == tmp_path / "resultados" / "agregados_por_presidente.csv"
    assert r["figuras"] == tmp_path / "resultados" / "figuras"


def test_ejemplos_existen_en_la_muestra(muestra):
    for clave, ej in u.EJEMPLOS.items():
        e = u.ejemplo_de_muestra(muestra, clave)
        assert e["id"] == ej["id"] and e["particion"] == "prueba" and "[[" not in e["parrafo"]
    assert u.ejemplo_al_azar(muestra, np.random.default_rng(1))["particion"] == "prueba"
    morena = u.ejemplo_de_muestra(muestra, "Morena")
    assert morena["id"] == "2025-07-10_490_1_65" and morena["presidente"] == "Sheinbaum"
    assert u.EJEMPLOS["Morena"]["etiqueta"] == "Morena (Sheinbaum, 10 jul 2025)"
    # Respaldo: si el id fijo no existe, toma otro ítem del mismo grupo en prueba.
    e = u.ejemplo_de_muestra(muestra[muestra.id != u.EJEMPLOS["PRIAN"]["id"]], "PRIAN")
    assert e["grupo"] == "PRIAN" and e["particion"] == "prueba"


def test_etiqueta_referencia_de_los_ejemplos(raiz):
    """Los dos ejemplos precargados tienen etiqueta de referencia (anotación asistida por modelo
    de lenguaje) y la de Morena es favorable, para que la demo muestre un acierto."""
    ruta = u.rutas(raiz)["etiquetas"]
    if not ruta.exists():
        pytest.skip(f"No está {ruta}")
    etiquetas, aviso = u.leer_csv(ruta, ("id", "es_partido", "postura"), keep_default_na=False, dtype=str)
    assert aviso is None
    assert u.etiqueta_referencia(etiquetas, u.EJEMPLOS["Morena"]["id"]) == {"es_partido": 1, "postura": "favorable"}
    assert u.etiqueta_referencia(etiquetas, u.EJEMPLOS["PRIAN"]["id"])["es_partido"] == 1
    assert u.etiqueta_referencia(etiquetas, "no_existe") is None
    assert u.etiqueta_referencia(None, "x") is None


@pytest.mark.parametrize("clave", ["resumen", "muestra"])
def test_insumos_de_datos_en_el_proyecto(clave, raiz):
    ruta = u.rutas(raiz)[clave]
    if not ruta.exists():
        pytest.skip(f"No está {ruta}")
    if clave == "resumen":
        datos, aviso = u.leer_json(ruta)
        assert aviso is None and u.menciones_por_grupo(datos) is not None
    else:
        tabla, aviso = u.leer_csv(ruta, ("id", "oracion_marcada", "contexto_previo", "pregunta_prensa"))
        assert aviso is None and len(tabla) > 0


def test_nombre_figura_legible():
    assert u.nombre_figura("fase2_roc_T1") == "Curva ROC de prueba, Fase 2, T1. Es partido"
    assert u.nombre_figura("fase3_roc_T2b") == "Curva ROC de prueba, Fase 3, T2b. Favorable"
    assert u.nombre_figura("otra_figura") == "otra figura"
