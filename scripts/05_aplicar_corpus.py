# -*- coding: utf-8 -*-
"""Aplica el clasificador entrenado a todas las menciones candidatas del corpus y agrega la
postura por presidente, año y partido objetivo.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

Entradas: datos/menciones.parquet (11 714 menciones) y los modelos de modelos/ (ver
postura.pipeline.Clasificador).

Salidas en resultados/:
* predicciones_corpus.parquet: una fila por mención con p_partido, p_desfavorable,
  p_favorable y postura_predicha (no_partido, desfavorable, favorable o neutral). La columna
  en_muestra_anotada marca los ítems que se usaron para entrenar o evaluar.
* agregados_corpus.csv: por presidente, año y objetivo: menciones, menciones_partido_pred,
  desfavorables, favorables, neutrales y porcentajes. pct_partido se calcula sobre todas las
  menciones; pct_desfavorable, pct_favorable y pct_neutral sobre las menciones que el modelo
  considera partido.
* agregados_por_presidente.csv: lo mismo sin el año.

Uso
---
    python scripts/05_aplicar_corpus.py [--menciones ruta] [--modelos carpeta] [--salida carpeta]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from postura.particion import RUTA_MENCIONES, RUTA_MUESTRA  # noqa: E402
from postura.pipeline import Clasificador  # noqa: E402

COLUMNAS = ["id", "fecha", "anio", "presidente", "grupo", "objetivo", "mencion", "particion",
            "p_partido", "p_desfavorable", "p_favorable", "postura_predicha"]
POSTURAS = ["desfavorable", "favorable", "neutral"]


def agregar(pred, claves):
    """Conteos y porcentajes de postura predicha por las claves dadas."""
    conteo = (pd.crosstab([pred[c] for c in claves], pred["postura_predicha"])
              .reindex(columns=["no_partido"] + POSTURAS, fill_value=0))
    salida = pd.DataFrame({
        "menciones": conteo.sum(axis=1),
        "menciones_partido_pred": conteo[POSTURAS].sum(axis=1),
        "desfavorables": conteo["desfavorable"],
        "favorables": conteo["favorable"],
        "neutrales": conteo["neutral"],
    })
    salida["pct_partido"] = 100 * salida["menciones_partido_pred"] / salida["menciones"]
    base = salida["menciones_partido_pred"].replace(0, np.nan)
    salida["pct_desfavorable"] = 100 * salida["desfavorables"] / base
    salida["pct_favorable"] = 100 * salida["favorables"] / base
    salida["pct_neutral"] = 100 * salida["neutrales"] / base
    for col in ("pct_partido", "pct_desfavorable", "pct_favorable", "pct_neutral"):
        salida[col] = salida[col].round(2)
    return salida.reset_index()


def argumentos(argv=None):
    """Lee de la línea de comandos las rutas de menciones, muestra, modelos y salida."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--menciones", default=RUTA_MENCIONES)
    p.add_argument("--muestra", default=RUTA_MUESTRA)
    p.add_argument("--modelos", default=os.path.join(RAIZ, "modelos"))
    p.add_argument("--salida", default=os.path.join(RAIZ, "resultados"))
    return p.parse_args(argv)


def main(argv=None):
    """Clasifica todas las menciones del corpus y escribe las predicciones y los agregados por presidente, año y partido objetivo."""
    args = argumentos(argv)
    os.makedirs(args.salida, exist_ok=True)
    t0 = time.perf_counter()
    menciones = pd.read_parquet(args.menciones)
    clf = Clasificador(args.modelos)
    t1 = time.perf_counter()
    pred = clf.analizar_tabla(menciones)
    t2 = time.perf_counter()
    salida = pred[[c for c in COLUMNAS if c in pred.columns]].copy()
    if os.path.exists(args.muestra):
        ids = set(pd.read_csv(args.muestra, usecols=["id"], dtype=str)["id"])
        salida["en_muestra_anotada"] = salida["id"].isin(ids)
    salida.to_parquet(os.path.join(args.salida, "predicciones_corpus.parquet"), index=False)
    agregar(salida, ["presidente", "anio", "objetivo"]).to_csv(
        os.path.join(args.salida, "agregados_corpus.csv"), index=False)
    agregar(salida, ["presidente", "objetivo"]).to_csv(
        os.path.join(args.salida, "agregados_por_presidente.csv"), index=False)
    resumen = {"menciones": len(salida), "segundos_carga": round(t1 - t0, 2),
               "segundos_clasificacion": round(t2 - t1, 2),
               "menciones_por_segundo": round(len(salida) / max(t2 - t1, 1e-9), 1),
               "postura_predicha": salida["postura_predicha"].value_counts().to_dict(),
               "etiquetas_sinteticas": bool(clf.metadatos.get("etiquetas_sinteticas", False))}
    print(json.dumps(resumen, ensure_ascii=False))
    return resumen


if __name__ == "__main__":
    main()
