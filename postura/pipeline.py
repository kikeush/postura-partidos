# -*- coding: utf-8 -*-
"""Herramienta de inferencia: detecta menciones a partidos en un párrafo y estima la postura
del titular hacia cada partido mencionado.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

Flujo
-----
1. postura.menciones.oraciones divide el párrafo en oraciones.
2. postura.menciones.buscar_menciones halla las cadenas candidatas en cada oración.
3. Cada mención se marca con [[ ]] y se arma su fila (contexto previo del párrafo, pregunta de
   la prensa, grupo, objetivo).
4. Tres modelos entrenados por scripts/04_entrenar_clasicos.py dan tres probabilidades:
   T1 p_partido (la cadena es un partido), T2a p_desfavorable y T2b p_favorable.
5. Regla de decisión con los umbrales elegidos en validación (modelos/umbrales.json):
   * si p_partido < umbral_T1, la postura es no_partido;
   * si solo p_desfavorable supera su umbral, desfavorable; si solo p_favorable supera el
     suyo, favorable;
   * si ambos lo superan, gana el de mayor margen relativo (p - u) / (1 - u);
   * si ninguno lo supera, neutral.

Archivos esperados en dir_modelos: modelo_T1.joblib, modelo_T2a.joblib, modelo_T2b.joblib y
umbrales.json con {"T1": {"umbral": ...}, "T2a": {...}, "T2b": {...}}.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

from postura.menciones import OBJETIVO, buscar_menciones, marcar, oraciones

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_MODELOS = os.path.join(RAIZ, "modelos")
TAREAS = ("T1", "T2a", "T2b")
ARCHIVOS = {t: f"modelo_{t}.joblib" for t in TAREAS}
POSTURAS_PREDICHAS = ("no_partido", "desfavorable", "favorable", "neutral")
PALABRAS_CONTEXTO = 120
PALABRAS_PREGUNTA = 60


def palabras_finales(texto, n):
    """Últimas n palabras de un texto (como en scripts/01_construir_menciones.py)."""
    return " ".join(str(texto or "").split()[-n:])


def decidir_postura(p_partido, p_desfavorable, p_favorable, u_partido, u_desfavorable, u_favorable):
    """Regla de decisión vectorizada (ver docstring del módulo). Devuelve un arreglo de cadenas."""
    p1 = np.asarray(p_partido, float)
    pd_ = np.asarray(p_desfavorable, float)
    pf = np.asarray(p_favorable, float)
    es_partido = p1 >= u_partido
    d = pd_ >= u_desfavorable
    f = pf >= u_favorable
    margen_d = (pd_ - u_desfavorable) / max(1e-9, 1.0 - u_desfavorable)
    margen_f = (pf - u_favorable) / max(1e-9, 1.0 - u_favorable)
    gana_d = d & (~f | (margen_d >= margen_f))
    postura = np.where(gana_d, "desfavorable", np.where(f, "favorable", "neutral"))
    return np.where(es_partido, postura, "no_partido").astype(object)


class Clasificador:
    """Carga los modelos de T1, T2a y T2b y analiza párrafos o tablas de menciones.

    Parámetros
    ----------
    dir_modelos : str
        Carpeta con modelo_T1.joblib, modelo_T2a.joblib, modelo_T2b.joblib y umbrales.json.

    Ejemplo
    -------
    >>> clf = Clasificador("modelos")
    >>> clf.analizar("El PRIAN saqueó al país. Morena ganó la votación.")
    [{'oracion': 'El PRIAN saqueó al país.', 'mencion': 'PRIAN', ...}, ...]
    """

    def __init__(self, dir_modelos=DIR_MODELOS):
        self.dir_modelos = str(dir_modelos)
        ruta_umbrales = os.path.join(self.dir_modelos, "umbrales.json")
        faltan = [ARCHIVOS[t] for t in TAREAS
                  if not os.path.exists(os.path.join(self.dir_modelos, ARCHIVOS[t]))]
        if faltan or not os.path.exists(ruta_umbrales):
            raise FileNotFoundError(
                f"Faltan archivos en {self.dir_modelos}: {faltan + ([] if os.path.exists(ruta_umbrales) else ['umbrales.json'])}. "
                "Entrena primero con scripts/04_entrenar_clasicos.py.")
        self.modelos = {t: joblib.load(os.path.join(self.dir_modelos, ARCHIVOS[t])) for t in TAREAS}
        with open(ruta_umbrales, encoding="utf-8") as fh:
            self.metadatos = json.load(fh)
        self.umbrales = {t: float(self.metadatos[t]["umbral"]) for t in TAREAS}

    # -----------------------------------------------------------------------------------
    def _probabilidad(self, tarea, tabla):
        modelo = self.modelos[tarea]
        if hasattr(modelo, "predict_proba"):
            return np.asarray(modelo.predict_proba(tabla))[:, 1]
        # Respaldo por si se guardó un modelo sin probabilidades: sigmoide del margen.
        return 1.0 / (1.0 + np.exp(-np.asarray(modelo.decision_function(tabla), float)))

    def analizar_tabla(self, df):
        """Aplica los tres modelos a muchas menciones ya extraídas, en bloque.

        df necesita oracion_marcada (o bien oracion, inicio y fin) y grupo; contexto_previo y
        pregunta_prensa son opcionales. Devuelve una copia de df con p_partido,
        p_desfavorable, p_favorable y postura_predicha."""
        tabla = df.copy()
        if "oracion_marcada" not in tabla.columns:
            if not {"oracion", "inicio", "fin"} <= set(tabla.columns):
                raise KeyError("Se necesita oracion_marcada, o bien oracion, inicio y fin.")
            tabla["oracion_marcada"] = [marcar(str(o), int(i), int(f)) for o, i, f in
                                        zip(tabla["oracion"], tabla["inicio"], tabla["fin"])]
        if "grupo" not in tabla.columns:
            raise KeyError("Se necesita la columna grupo (ver postura.menciones.PATRONES).")
        for col in ("contexto_previo", "pregunta_prensa"):
            if col not in tabla.columns:
                tabla[col] = ""
        if tabla.empty:
            for col in ("p_partido", "p_desfavorable", "p_favorable"):
                tabla[col] = pd.Series(dtype=float)
            tabla["postura_predicha"] = pd.Series(dtype=object)
            return tabla
        tabla["p_partido"] = self._probabilidad("T1", tabla)
        tabla["p_desfavorable"] = self._probabilidad("T2a", tabla)
        tabla["p_favorable"] = self._probabilidad("T2b", tabla)
        tabla["postura_predicha"] = decidir_postura(
            tabla["p_partido"], tabla["p_desfavorable"], tabla["p_favorable"],
            self.umbrales["T1"], self.umbrales["T2a"], self.umbrales["T2b"])
        return tabla

    def menciones_de_parrafo(self, parrafo, contexto_previo="", pregunta_prensa=""):
        """Tabla de menciones candidatas de un párrafo, con el mismo formato que
        datos/menciones.parquet (sin las columnas de conferencia)."""
        filas, previas = [], ([str(contexto_previo)] if contexto_previo else [])
        pregunta = palabras_finales(pregunta_prensa, PALABRAS_PREGUNTA)
        for o in oraciones(parrafo):
            contexto = palabras_finales(" ".join(previas), PALABRAS_CONTEXTO)
            for (ini, fin, cadena, grupo) in buscar_menciones(o):
                filas.append({"oracion": o, "oracion_marcada": marcar(o, ini, fin), "mencion": cadena,
                              "grupo": grupo, "objetivo": OBJETIVO[grupo], "inicio": ini, "fin": fin,
                              "contexto_previo": contexto, "pregunta_prensa": pregunta,
                              "palabras_oracion": len(o.split())})
            previas.append(o)
        return pd.DataFrame(filas)

    def analizar(self, parrafo, contexto_previo="", pregunta_prensa=""):
        """Analiza un párrafo y devuelve una lista de dicts, uno por mención detectada:
        {oracion, mencion, grupo, objetivo, p_partido, p_desfavorable, p_favorable,
        postura_predicha}. Lista vacía si no hay menciones."""
        tabla = self.menciones_de_parrafo(parrafo, contexto_previo, pregunta_prensa)
        if tabla.empty:
            return []
        tabla = self.analizar_tabla(tabla)
        salida = []
        for r in tabla.itertuples(index=False):
            salida.append({"oracion": r.oracion, "mencion": r.mencion, "grupo": r.grupo,
                           "objetivo": r.objetivo, "p_partido": float(r.p_partido),
                           "p_desfavorable": float(r.p_desfavorable),
                           "p_favorable": float(r.p_favorable),
                           "postura_predicha": str(r.postura_predicha)})
        return salida
