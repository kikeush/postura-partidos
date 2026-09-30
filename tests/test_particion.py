# -*- coding: utf-8 -*-
"""Pruebas de postura.particion: ninguna conferencia en dos particiones y pliegues por fecha.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.
"""
import numpy as np
import pandas as pd
import pytest

from postura.particion import (PARTICIONES, fechas_compartidas, leer_etiquetas, pliegues_por_fecha,
                               separar, unir_etiquetas, verificar_particion)


def test_ninguna_fecha_en_dos_particiones_en_la_muestra_real(muestra):
    resumen = verificar_particion(muestra)
    assert sum(resumen["items"].values()) == len(muestra)
    partes = separar(muestra)
    for i, a in enumerate(PARTICIONES):
        for b in PARTICIONES[i + 1:]:
            assert fechas_compartidas(partes[a], partes[b]) == set()


def test_ninguna_fecha_en_dos_particiones_en_todas_las_menciones(menciones):
    verificar_particion(menciones)
    muestra_ids = set(menciones["id"])
    assert len(muestra_ids) == len(menciones)  # ids únicos


def test_muestra_y_menciones_coinciden_en_particion(muestra, menciones):
    unida = muestra[["id", "particion"]].merge(menciones[["id", "particion"]], on="id",
                                               suffixes=("_m", "_c"))
    assert len(unida) == len(muestra)
    assert (unida["particion_m"] == unida["particion_c"]).all()


def test_verificar_detecta_fecha_repetida():
    df = pd.DataFrame({"fecha": ["2020-01-01", "2020-01-01", "2020-01-02"],
                       "particion": ["entrenamiento", "prueba", "prueba"]})
    with pytest.raises(ValueError, match="más de una partición"):
        verificar_particion(df)
    with pytest.raises(ValueError, match="desconocidas"):
        verificar_particion(df.assign(particion="otra"))


def test_pliegues_agrupados_por_fecha(muestra):
    tr = separar(muestra)["entrenamiento"]
    pliegues = pliegues_por_fecha(tr["fecha"], 5)
    assert len(pliegues) == 5
    cubiertos = np.concatenate([te for _, te in pliegues])
    assert sorted(cubiertos) == list(range(len(tr)))  # cada ítem se evalúa una vez
    fechas = tr["fecha"].to_numpy()
    for ajuste, evaluacion in pliegues:
        assert set(fechas[ajuste]).isdisjoint(fechas[evaluacion])


def test_pliegues_estratificados_y_recorte():
    fechas = np.repeat(["a", "b", "c"], 4)
    y = np.tile([0, 1], 6)
    assert len(pliegues_por_fecha(fechas, 5)) == 3  # se recorta al número de fechas
    for ajuste, evaluacion in pliegues_por_fecha(fechas, 3, y=y, estratificar=True):
        assert set(fechas[ajuste]).isdisjoint(fechas[evaluacion])
    with pytest.raises(ValueError):
        pliegues_por_fecha(["a", "a"], 5)


def test_leer_y_unir_etiquetas(tmp_path, muestra, etiquetas):
    ruta = tmp_path / "etiquetas.csv"
    etiquetas.to_csv(ruta, index=False)
    et = leer_etiquetas(str(ruta))
    tabla, resumen = unir_etiquetas(muestra, et)
    assert resumen["unidas"] == len(muestra) and resumen["sin_etiqueta"] == 0
    assert set(tabla["postura"]) <= {"desfavorable", "favorable", "neutral", "no_aplica"}


def test_leer_etiquetas_rechaza_incoherencias(tmp_path):
    malo = pd.DataFrame([{"id": "x", "es_partido": 0, "postura": "favorable", "acuerdo_es_partido": 3,
                          "acuerdo_postura": 3, "adjudicado": 0}])
    ruta = tmp_path / "malo.csv"
    malo.to_csv(ruta, index=False)
    with pytest.raises(ValueError, match="incoherentes"):
        leer_etiquetas(str(ruta))
    with pytest.raises(FileNotFoundError):
        leer_etiquetas(str(tmp_path / "no_existe.csv"))
