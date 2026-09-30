# -*- coding: utf-8 -*-
"""Lectura de la muestra anotada, unión con las etiquetas y control de la partición por
conferencia.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

La partición (entrenamiento, validación, prueba) la asigna scripts/01_construir_menciones.py
con un hash estable de la fecha, así que todas las menciones de una conferencia caen en la
misma partición. Este módulo lo verifica y ofrece pliegues agrupados por fecha (GroupKFold)
para la búsqueda de hiperparámetros dentro de entrenamiento: una conferencia nunca aparece a
la vez en el pliegue de ajuste y en el de evaluación.

Formato de datos/etiquetas.csv (lo produce el anotador):
    id, es_partido (0/1), postura (desfavorable, favorable, neutral o no_aplica),
    acuerdo_es_partido (1 a 3), acuerdo_postura (1 a 3), adjudicado (0/1).
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MUESTRA = os.path.join(RAIZ, "datos", "muestra_anotacion.csv")
RUTA_ETIQUETAS = os.path.join(RAIZ, "datos", "etiquetas.csv")
RUTA_MENCIONES = os.path.join(RAIZ, "datos", "menciones.parquet")

PARTICIONES = ("entrenamiento", "validacion", "prueba")
POSTURAS = ("desfavorable", "favorable", "neutral", "no_aplica")
COLUMNAS_ETIQUETAS = ("id", "es_partido", "postura", "acuerdo_es_partido", "acuerdo_postura",
                      "adjudicado")


def leer_muestra(ruta=RUTA_MUESTRA):
    """Muestra de anotación (CSV) con cadenas vacías en lugar de NaN."""
    return pd.read_csv(ruta, dtype=str, keep_default_na=False)


def leer_menciones(ruta=RUTA_MENCIONES):
    """Tabla completa de menciones candidatas."""
    return pd.read_parquet(ruta)


def leer_etiquetas(ruta=RUTA_ETIQUETAS):
    """Lee y valida etiquetas.csv. Lanza ValueError con la lista de problemas encontrados."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No existe {ruta}. Anota datos/muestra_anotacion.csv siguiendo "
            "docs/guia_anotacion.md y guarda las etiquetas con las columnas "
            + ", ".join(COLUMNAS_ETIQUETAS) + ".")
    et = pd.read_csv(ruta, dtype={"id": str}, keep_default_na=False)
    faltan = [c for c in COLUMNAS_ETIQUETAS if c not in et.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en {ruta}: {faltan}")
    problemas = []
    for col in ("es_partido", "acuerdo_es_partido", "acuerdo_postura", "adjudicado"):
        et[col] = pd.to_numeric(et[col], errors="coerce")
    if et["id"].duplicated().any():
        problemas.append(f"ids repetidos: {et.loc[et['id'].duplicated(), 'id'].head().tolist()}")
    if not et["es_partido"].isin([0, 1]).all():
        problemas.append("es_partido debe ser 0 o 1")
    et["postura"] = et["postura"].astype(str).str.strip().str.lower()
    if not et["postura"].isin(POSTURAS).all():
        malas = sorted(set(et.loc[~et["postura"].isin(POSTURAS), "postura"]))
        problemas.append(f"posturas no válidas: {malas}")
    incoherentes = ((et["es_partido"] == 0) & (et["postura"] != "no_aplica")) | \
                   ((et["es_partido"] == 1) & (et["postura"] == "no_aplica"))
    if incoherentes.any():
        problemas.append(f"{int(incoherentes.sum())} filas con es_partido y postura incoherentes "
                         "(es_partido = 0 exige no_aplica y viceversa)")
    if problemas:
        raise ValueError("Etiquetas no válidas: " + "; ".join(problemas))
    for col in ("es_partido", "acuerdo_es_partido", "acuerdo_postura", "adjudicado"):
        et[col] = et[col].astype("Int64")
    return et


def unir_etiquetas(muestra, etiquetas):
    """Une muestra y etiquetas por id (solo ítems etiquetados). Devuelve (tabla, resumen)."""
    tabla = muestra.merge(etiquetas, on="id", how="inner", validate="one_to_one")
    resumen = {"muestra": len(muestra), "etiquetas": len(etiquetas), "unidas": len(tabla),
               "sin_etiqueta": int(len(muestra) - len(tabla)),
               "etiquetas_sin_item": int(len(set(etiquetas["id"]) - set(muestra["id"])))}
    return tabla.reset_index(drop=True), resumen


def verificar_particion(df, col_fecha="fecha", col_particion="particion"):
    """Comprueba que cada fecha (conferencia) pertenezca a una sola partición válida.

    Lanza ValueError si alguna fecha aparece en dos particiones o si hay nombres de partición
    desconocidos. Devuelve un resumen con ítems y conferencias por partición."""
    invalidas = sorted(set(df[col_particion]) - set(PARTICIONES))
    if invalidas:
        raise ValueError(f"Particiones desconocidas: {invalidas}")
    por_fecha = df.groupby(col_fecha)[col_particion].nunique()
    repetidas = por_fecha[por_fecha > 1].index.tolist()
    if repetidas:
        raise ValueError(f"{len(repetidas)} fechas aparecen en más de una partición, "
                         f"por ejemplo {repetidas[:5]}")
    return {"items": df[col_particion].value_counts().reindex(PARTICIONES, fill_value=0).to_dict(),
            "conferencias": df.drop_duplicates(col_fecha)[col_particion].value_counts()
                              .reindex(PARTICIONES, fill_value=0).to_dict()}


def fechas_compartidas(a, b, col_fecha="fecha"):
    """Conjunto de fechas presentes en ambas tablas (debe ser vacío entre particiones)."""
    return set(a[col_fecha]) & set(b[col_fecha])


def separar(df, col_particion="particion"):
    """Diccionario partición -> subtabla (índice reiniciado)."""
    return {p: df[df[col_particion] == p].reset_index(drop=True) for p in PARTICIONES}


def pliegues_por_fecha(fechas, n_pliegues=5, y=None, estratificar=False, semilla=20260927):
    """Lista de (índices de ajuste, índices de evaluación) posicionales agrupados por fecha.

    Por omisión usa GroupKFold (determinista). Con estratificar=True y y dado usa
    StratifiedGroupKFold barajado con semilla, útil si la clase positiva es rara. El número
    de pliegues se recorta al número de fechas distintas."""
    fechas = np.asarray(fechas).astype(str)
    n = int(min(n_pliegues, len(np.unique(fechas))))
    if n < 2:
        raise ValueError("Se necesitan al menos dos fechas distintas para los pliegues.")
    X = np.zeros((len(fechas), 1))
    if estratificar and y is not None:
        cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=semilla)
        return [(tr, te) for tr, te in cv.split(X, np.asarray(y), fechas)]
    return [(tr, te) for tr, te in GroupKFold(n_splits=n).split(X, None, fechas)]
