# -*- coding: utf-8 -*-
"""Etiquetas SINTÉTICAS para probar el código de la Fase 2 antes de tener la anotación real.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

Las etiquetas se generan al azar con una semilla, con probabilidades que dependen del grupo de
la mención y de los conteos del léxico, para que los modelos tengan algo que aprender y las
pruebas ejerciten todo el flujo. NO son datos: ningún resultado obtenido con ellas dice nada
sobre las conferencias. La columna sintetica = 1 permite que scripts/04 lo registre en su
salida. Nunca se escriben en datos/etiquetas.csv.
"""
import numpy as np
import pandas as pd

from postura import lexico as lx

PROB_PARTIDO = {"ambiguo_pan": 0.08, "ambiguo_verde": 0.35, "ambiguo_morena": 0.3,
                "conservadores": 0.75, "oposicion": 0.7}
# (sesgo desfavorable, sesgo favorable) por partido objetivo.
SESGOS = {"Morena": (-1.0, 0.8), "PRIAN": (1.3, -1.5), "PRI": (0.7, -1.0), "PAN": (0.7, -1.0),
          "PRD": (0.4, -0.8), "bloque opositor": (1.0, -1.3)}


def etiquetas_sinteticas(muestra, semilla=7):
    """Devuelve un DataFrame con el formato de datos/etiquetas.csv más la columna sintetica."""
    rng = np.random.default_rng(semilla)
    filas = []
    for r in muestra.itertuples(index=False):
        es = int(rng.random() < PROB_PARTIDO.get(r.grupo, 0.96))
        if es:
            desf = lx.contar_en_ventana(r.oracion_marcada, lx.DESFAVORABLE, 10)
            fav = lx.contar_en_ventana(r.oracion_marcada, lx.FAVORABLE, 10)
            sd, sf = SESGOS.get(r.objetivo, (0.0, -0.3))
            logits = np.array([sd + 1.2 * desf, sf + 1.2 * fav, 0.4])
            p = np.exp(logits - logits.max())
            postura = str(rng.choice(["desfavorable", "favorable", "neutral"], p=p / p.sum()))
        else:
            postura = "no_aplica"
        filas.append({"id": r.id, "es_partido": es, "postura": postura,
                      "acuerdo_es_partido": int(rng.choice([1, 2, 3], p=[0.1, 0.3, 0.6])),
                      "acuerdo_postura": int(rng.choice([1, 2, 3], p=[0.15, 0.35, 0.5])),
                      "adjudicado": int(rng.random() < 0.1), "sintetica": 1})
    return pd.DataFrame(filas)
