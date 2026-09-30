# -*- coding: utf-8 -*-
"""Prueba de integración: scripts 04 y 05 de principio a fin con etiquetas sintéticas en una
carpeta temporal (nunca en datos/, resultados/ ni modelos/ del proyecto).

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.
"""
import importlib.util
import json
import os
import sys

import pandas as pd
import pytest


RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(RAIZ, "app") not in sys.path:
    sys.path.insert(0, os.path.join(RAIZ, "app"))
import utilidades as u  # noqa: E402


def _cargar(raiz, nombre):
    ruta = os.path.join(raiz, "scripts", nombre)
    spec = importlib.util.spec_from_file_location(nombre.replace(".py", ""), ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@pytest.mark.lento
def test_entrenar_y_aplicar(raiz, etiquetas, menciones, tmp_path):
    ruta_et = tmp_path / "etiquetas_sinteticas.csv"
    etiquetas.to_csv(ruta_et, index=False)
    ruta_men = tmp_path / "menciones_sub.parquet"
    menciones.sample(n=1500, random_state=2).to_parquet(ruta_men, index=False)
    salida, modelos = tmp_path / "resultados", tmp_path / "modelos"

    entrenar = _cargar(raiz, "04_entrenar_clasicos.py")
    res = entrenar.main(["--etiquetas", str(ruta_et), "--salida", str(salida), "--modelos", str(modelos),
                         "--menciones", str(ruta_men), "--rapido", "--n-boot", "50", "--sin-tiempos",
                         "--familias", "base_longitud", "lexico", "logistica", "--n-jobs", "2"])
    assert res["etiquetas_sinteticas"] is True
    datos = json.loads((salida / "fase2_resultados.json").read_text(encoding="utf-8"))
    for tarea in ("T1", "T2a", "T2b"):
        t = datos["tareas"][tarea]
        assert t["mejor_modelo"] in ("lexico", "logistica")
        assert 0 <= t["modelos"][t["mejor_modelo"]]["prueba"]["auc"] <= 1
        assert "delong" in t["comparaciones_del_mejor"]["base_grupo"]
        assert "ablacion_enmascarada" in t
        assert (salida / "figuras" / f"fase2_roc_{tarea}.png").exists()
        assert (modelos / f"modelo_{tarea}.joblib").exists()
    assert "inferencia" in datos["tiempos"]
    # Contrato con la aplicación: cada tarea trae su modelo elegido y todos los AUC su intervalo.
    tabla_auc = u.resultados_auc(datos, "Fase 2")
    assert set(tabla_auc["tarea"]) == {"T1", "T2a", "T2b"}
    assert tabla_auc["ic_inf"].notna().all() and tabla_auc["ic_sup"].notna().all()
    assert tabla_auc.groupby("tarea")["elegido"].sum().eq(1).all()
    assert not tabla_auc["modelo"].str.contains("prueba").any()
    pred = pd.read_csv(salida / "fase2_predicciones_prueba.csv")
    assert {"id", "particion", "tarea", "y_real", "puntaje_base_grupo"} <= set(pred.columns)
    assert set(pred["particion"]) == {"prueba"}

    aplicar = _cargar(raiz, "05_aplicar_corpus.py")
    resumen = aplicar.main(["--menciones", str(ruta_men), "--modelos", str(modelos), "--salida", str(salida)])
    assert resumen["menciones"] == 1500
    pc = pd.read_parquet(salida / "predicciones_corpus.parquet")
    assert len(pc) == 1500 and pc["p_partido"].between(0, 1).all()
    agr = pd.read_csv(salida / "agregados_corpus.csv")
    assert {"presidente", "anio", "objetivo", "menciones", "menciones_partido_pred", "desfavorables",
            "favorables", "neutrales", "pct_partido", "pct_desfavorable"} <= set(agr.columns)
    assert agr["menciones"].sum() == 1500
    por_pres = pd.read_csv(salida / "agregados_por_presidente.csv")
    assert "anio" not in por_pres.columns and por_pres["menciones"].sum() == 1500
    assert set(u.REQUERIDAS_CORPUS) <= set(agr.columns)
    assert set(u.REQUERIDAS_PRESIDENTE) <= set(por_pres.columns)
    assert set(agr["objetivo"]) <= set(u.ORDEN_OBJETIVOS)
