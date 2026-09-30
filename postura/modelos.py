# -*- coding: utf-8 -*-
"""Fábricas de modelos clásicos con sus rejillas de hiperparámetros.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

Cada fábrica devuelve un EspecModelo con un Pipeline sin ajustar (PrepararEntrada, rasgos,
clasificador) y la rejilla que explora GridSearchCV. Los nombres de los parámetros siguen la
convención de scikit-learn: rasgos__palabras__ngram_range, clf__C, etcétera.

Modelos
-------
* logistica: regresión logística L2 sobre TF-IDF de palabras y caracteres, rasgos del dominio
  y grupo; class_weight balanced. Línea base fuerte e interpretable (coeficientes).
* svm_lineal: LinearSVC sobre los mismos rasgos; el AUC se calcula con decision_function.
  Para el producto final se calibra con ajustar_con_probabilidad (sigmoide de Platt con
  pliegues agrupados por conferencia), que conserva el orden de los puntajes y, por lo tanto,
  el AUC.
* bosque_aleatorio: Random Forest sobre rasgos del dominio, grupo y TruncatedSVD de 200
  componentes del TF-IDF; class_weight balanced_subsample, n_jobs = -1.
* lexico: regresión logística solo con rasgos del dominio y grupo (sin TF-IDF). Sirve para
  medir cuánto aporta el vocabulario frente al léxico curado.
* base_grupo y base_longitud: líneas base obligatorias (solo el grupo de la mención en
  one-hot; solo la longitud de la oración).
* pbc4cip: clasificador basado en patrones contrastantes. No está instalado en el entorno del
  curso; estado_pbc4cip lo intenta importar y, si no existe, lo reporta como no disponible.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from postura.rasgos import (Longitud, PrepararEntrada, construir_rasgos,
                            construir_rasgos_densos)

SEMILLA = 20260927
LINEAS_BASE = ("base_grupo", "base_longitud")


@dataclass
class EspecModelo:
    """Especificación de un modelo: estimador sin ajustar, rejilla y metadatos."""
    nombre: str
    estimador: object
    rejilla: dict
    linea_base: bool = False
    descripcion: str = ""
    extra: dict = field(default_factory=dict)


def _logistica(C=1.0, semilla=SEMILLA):
    """Regresión logística con clases balanceadas, solver liblinear y semilla fija."""
    return LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=5000,
                              random_state=semilla)


# ---------------------------------------------------------------------------------------
# Modelos principales
# ---------------------------------------------------------------------------------------

def logistica(enmascarar=False, rapido=False, semilla=SEMILLA):
    """Regresión logística sobre todos los rasgos dispersos."""
    est = Pipeline([("preparar", PrepararEntrada(enmascarar=enmascarar)),
                    ("rasgos", construir_rasgos(enmascarar=enmascarar)),
                    ("clf", _logistica(semilla=semilla))])
    rejilla = ({"clf__C": [0.3, 3.0]} if rapido else
               {"clf__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "rasgos__palabras__ngram_range": [(1, 1), (1, 2)]})
    return EspecModelo("logistica", est, rejilla, descripcion=(
        "Regresión logística L2 (liblinear, class_weight balanced) sobre TF-IDF de palabras 1-2 "
        "y caracteres char_wb 2-5, rasgos del dominio y grupo one-hot."))


def svm_lineal(enmascarar=False, rapido=False, semilla=SEMILLA):
    """SVM lineal; el puntaje para AUC es decision_function."""
    est = Pipeline([("preparar", PrepararEntrada(enmascarar=enmascarar)),
                    ("rasgos", construir_rasgos(enmascarar=enmascarar)),
                    ("clf", LinearSVC(C=0.1, class_weight="balanced", dual="auto",
                                      max_iter=20000, random_state=semilla))])
    rejilla = ({"clf__C": [0.03, 0.3]} if rapido else
               {"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]})
    return EspecModelo("svm_lineal", est, rejilla, descripcion=(
        "LinearSVC (class_weight balanced) sobre los mismos rasgos que la logística; puntaje "
        "decision_function, calibrado con sigmoide para el producto final."))


def bosque_aleatorio(enmascarar=False, rapido=False, semilla=SEMILLA, n_componentes=200):
    """Random Forest sobre rasgos densos (SVD del TF-IDF, dominio y grupo)."""
    n_arboles = 100 if rapido else 500
    est = Pipeline([("preparar", PrepararEntrada(enmascarar=enmascarar)),
                    ("rasgos", construir_rasgos_densos(enmascarar=enmascarar,
                                                       n_componentes=n_componentes,
                                                       semilla=semilla)),
                    ("clf", RandomForestClassifier(n_estimators=n_arboles,
                                                   class_weight="balanced_subsample",
                                                   n_jobs=-1, random_state=semilla))])
    rejilla = ({"clf__max_features": ["sqrt"], "clf__min_samples_leaf": [1]} if rapido else
               {"clf__max_features": ["sqrt", 0.3], "clf__min_samples_leaf": [1, 3]})
    return EspecModelo("bosque_aleatorio", est, rejilla, descripcion=(
        f"Random Forest ({n_arboles} árboles, class_weight balanced_subsample) sobre "
        f"TruncatedSVD de {n_componentes} componentes del TF-IDF, rasgos del dominio y grupo."))


def lexico(enmascarar=False, rapido=False, semilla=SEMILLA):
    """Regresión logística solo con rasgos del dominio (y grupo si no se enmascara)."""
    est = Pipeline([("preparar", PrepararEntrada(enmascarar=enmascarar)),
                    ("rasgos", construir_rasgos(enmascarar=enmascarar, palabras=False,
                                                caracteres=False)),
                    ("clf", _logistica(semilla=semilla))])
    rejilla = {"clf__C": [1.0]} if rapido else {"clf__C": [0.1, 1.0, 10.0]}
    return EspecModelo("lexico", est, rejilla, descripcion=(
        "Regresión logística solo con rasgos del dominio (léxico, negación, posición, signos, "
        "pronombres, contexto) y grupo one-hot."))


# ---------------------------------------------------------------------------------------
# Líneas base obligatorias
# ---------------------------------------------------------------------------------------

def base_grupo(enmascarar=False, rapido=False, semilla=SEMILLA):
    """Solo el grupo de la mención (one-hot) con regresión logística."""
    est = Pipeline([("preparar", PrepararEntrada()),
                    ("rasgos", ColumnTransformer([("grupo", OneHotEncoder(handle_unknown="ignore"),
                                                   ["grupo"])])),
                    ("clf", _logistica(semilla=semilla))])
    return EspecModelo("base_grupo", est, {"clf__C": [1.0]}, linea_base=True,
                       descripcion="Línea base: solo el grupo de la mención (one-hot).")


def base_longitud(enmascarar=False, rapido=False, semilla=SEMILLA):
    """Solo log(1 + palabras de la oración) con regresión logística."""
    est = Pipeline([("preparar", PrepararEntrada()),
                    ("rasgos", ColumnTransformer([("longitud", Longitud(), ["oracion_marcada"])])),
                    ("clf", _logistica(semilla=semilla))])
    return EspecModelo("base_longitud", est, {"clf__C": [1.0]}, linea_base=True,
                       descripcion="Línea base: solo la longitud de la oración en palabras.")


# ---------------------------------------------------------------------------------------
# PBC4cip (opcional)
# ---------------------------------------------------------------------------------------

def estado_pbc4cip():
    """Intenta importar PBC4cip. Devuelve {'disponible': bool, 'motivo': str}. Nunca falla."""
    for modulo in ("PBC4cip", "pbc4cip"):
        try:
            mod = __import__(modulo)
            return {"disponible": True, "modulo": modulo,
                    "version": getattr(mod, "__version__", "desconocida"), "motivo": ""}
        except Exception as error:  # ImportError u otros fallos de carga
            ultimo = f"{type(error).__name__}: {error}"
    return {"disponible": False, "modulo": None, "version": None,
            "motivo": f"PBC4cip no está instalado en este entorno ({ultimo})."}


class AdaptadorPBC4cip(BaseEstimator, ClassifierMixin):
    """Adaptador mínimo de PBC4cip a la interfaz de scikit-learn (fit, predict_proba).

    No se ha podido verificar en el entorno del curso porque la biblioteca no está instalada;
    solo se construye si estado_pbc4cip() la encuentra."""

    def __init__(self, tree_count=100):
        self.tree_count = tree_count

    def fit(self, X, y):
        from PBC4cip import PBC4cip  # noqa: import local para no fallar si no existe
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        self.columnas_ = [f"x{i}" for i in range(X.shape[1])]
        self.modelo_ = PBC4cip(tree_count=self.tree_count)
        self.modelo_.fit(pd.DataFrame(X, columns=self.columnas_),
                         pd.DataFrame({"clase": np.asarray(y).astype(str)}))
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(np.asarray(X, dtype=float), columns=self.columnas_)
        puntajes = np.asarray(self.modelo_.score_samples(X), dtype=float)
        if puntajes.ndim == 1:
            puntajes = np.column_stack([1 - puntajes, puntajes])
        suma = puntajes.sum(axis=1, keepdims=True)
        return puntajes / np.where(suma > 0, suma, 1.0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def pbc4cip(enmascarar=False, rapido=False, semilla=SEMILLA):
    """EspecModelo de PBC4cip sobre rasgos densos, o None si la biblioteca no está."""
    if not estado_pbc4cip()["disponible"]:
        return None
    est = Pipeline([("preparar", PrepararEntrada(enmascarar=enmascarar)),
                    ("rasgos", construir_rasgos_densos(enmascarar=enmascarar, n_componentes=50,
                                                       semilla=semilla)),
                    ("clf", AdaptadorPBC4cip())])
    return EspecModelo("pbc4cip", est, {"clf__tree_count": [100]}, descripcion=(
        "PBC4cip (patrones contrastantes) sobre SVD de 50 componentes, dominio y grupo."))


# ---------------------------------------------------------------------------------------
# Catálogo y utilidades
# ---------------------------------------------------------------------------------------

FABRICAS = {
    "base_grupo": base_grupo, "base_longitud": base_longitud, "lexico": lexico,
    "logistica": logistica, "svm_lineal": svm_lineal, "bosque_aleatorio": bosque_aleatorio,
}


def catalogo(enmascarar=False, rapido=False, semilla=SEMILLA, incluir_pbc4cip=True, nombres=None):
    """Diccionario ordenado nombre -> EspecModelo.

    Parámetros
    ----------
    enmascarar : bool
        Construye las variantes con la mención sustituida por [PARTIDO].
    rapido : bool
        Rejillas reducidas (pruebas unitarias y corridas de humo).
    incluir_pbc4cip : bool
        Añade PBC4cip si está instalado; si no, se omite sin error.
    nombres : lista o None
        Restringe el catálogo a esos modelos.
    """
    salida = {}
    for nombre, fabrica in FABRICAS.items():
        if nombres is None or nombre in nombres:
            salida[nombre] = fabrica(enmascarar=enmascarar, rapido=rapido, semilla=semilla)
    if incluir_pbc4cip and (nombres is None or "pbc4cip" in nombres):
        espec = pbc4cip(enmascarar=enmascarar, rapido=rapido, semilla=semilla)
        if espec is not None:
            salida["pbc4cip"] = espec
    return salida


def tiene_probabilidad(estimador):
    """True si el estimador (o el último paso del Pipeline) ofrece predict_proba."""
    return hasattr(estimador, "predict_proba")


def puntaje(estimador, X):
    """Puntaje continuo de la clase positiva: predict_proba[:, 1] o decision_function."""
    if tiene_probabilidad(estimador):
        return np.asarray(estimador.predict_proba(X))[:, 1]
    return np.asarray(estimador.decision_function(X), dtype=float).ravel()


def fijar_n_jobs(estimador, n_jobs):
    """Fija n_jobs en todos los pasos que lo tengan (por ejemplo el Random Forest).

    Durante GridSearchCV con n_jobs = -1 conviene dejar el bosque en 1 para no sobresuscribir
    los núcleos (procesos por fuera, hilos por dentro)."""
    claves = {k: n_jobs for k in estimador.get_params() if k.endswith("n_jobs")}
    if claves:
        estimador.set_params(**claves)
    return estimador


def ajustar_con_probabilidad(estimador, X, y, pliegues=None):
    """Ajusta una copia del estimador y garantiza que el resultado tenga predict_proba.

    Si el estimador ya da probabilidades, solo se ajusta. Si no (LinearSVC), se envuelve en
    CalibratedClassifierCV con sigmoide y ensemble=False: los puntajes fuera de pliegue
    (pliegues agrupados por conferencia) ajustan la sigmoide y luego se reentrena el SVM con
    todos los datos. La transformación es monótona, así que el AUC no cambia."""
    est = clone(estimador)
    if tiene_probabilidad(est):
        return est.fit(X, y)
    cv = pliegues if pliegues is not None else 5
    return CalibratedClassifierCV(est, method="sigmoid", cv=cv, ensemble=False).fit(X, y)
