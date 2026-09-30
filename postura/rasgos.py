# -*- coding: utf-8 -*-
"""Construcción de rasgos con scikit-learn para las tareas T1 (¿es partido?) y T2 (postura).

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

Entradas
--------
Una tabla (pandas.DataFrame) con las columnas de datos/menciones.parquet o de
datos/muestra_anotacion.csv. Se usan:

* oracion_marcada: la oración con la mención entre [[ ]];
* contexto_previo: texto previo del mismo turno del titular (hasta 120 palabras);
* pregunta_prensa: última pregunta de la prensa (hasta 60 palabras);
* grupo: grupo de la mención según postura.menciones.PATRONES.

Si falta contexto_previo o pregunta_prensa se toma como cadena vacía. Si falta
oracion_marcada pero hay oracion, inicio y fin, la oración se marca con postura.menciones.marcar.

Bloques de rasgos
-----------------
(a) TF-IDF de palabras (unigramas y bigramas) y de caracteres (char_wb de 2 a 5) sobre la
    oración, con sublinear_tf y min_df = 2. Las marcas [[ ]] se quitan antes de vectorizar.
(b) Rasgos del dominio (clase RasgosDominio): conteos del léxico de postura.lexico en
    ventanas de 5 y 10 palabras, negación cercana, posición relativa de la mención, longitud,
    signos de exclamación e interrogación, comillas, cifras, pronombres de primera persona,
    presencia y contenido de la pregunta de prensa y del contexto previo.
(c) Grupo de la mención en codificación one-hot.

Con enmascarar=True la mención se sustituye por [PARTIDO] (postura.menciones.enmascarar) y,
por omisión, se retiran el one-hot del grupo y los rasgos de forma de la mención (mayúsculas,
plural), para que el modelo no conozca la identidad del partido. Es la ablación que mide
cuánto depende el clasificador del nombre del partido.

Salidas
-------
Matrices dispersas (scipy.sparse.csr_matrix) desde un ColumnTransformer dentro de un
Pipeline. Todas las clases son importables desde este módulo y, por lo tanto, serializables
con joblib. La variante para bosques aleatorios (construir_rasgos_densos) reduce el TF-IDF
con TruncatedSVD y devuelve una matriz densa.
"""
import contextlib
import math
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from threadpoolctl import threadpool_limits

from postura import lexico as lx
from postura.menciones import buscar_menciones, enmascarar as _enmascarar, marcar

TOKEN_MASCARA = "[PARTIDO]"
# Patrón de palabras: el token de máscara (ya en minúsculas) o palabras de dos o más caracteres.
PATRON_PALABRAS = r"(?u)\[partido\]|\b\w\w+\b"
COLUMNAS_TEXTO = ["oracion_marcada", "contexto_previo", "pregunta_prensa"]
COLUMNAS_SALIDA = ["texto", "oracion_marcada", "contexto_previo", "pregunta_prensa", "grupo"]
GRUPOS_AMBIGUOS = {"ambiguo_verde", "ambiguo_pan", "ambiguo_morena"}
_COMILLAS = "\"" + "".join(chr(c) for c in (0x201C, 0x201D, 0x00AB, 0x00BB, 0x2018, 0x2019))


# ---------------------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------------------

def posicion_mencion(oracion_marcada):
    """(inicio, fin) del tramo [[...]] dentro de la oración marcada, o None si no hay marcas.

    fin apunta justo después de ]], de modo que oracion_marcada[inicio:fin] == "[[...]]"."""
    texto = str(oracion_marcada or "")
    ini = texto.find("[[")
    if ini < 0:
        return None
    fin = texto.find("]]", ini + 2)
    return None if fin < 0 else (ini, fin + 2)


def texto_para_modelo(oracion_marcada, enmascarar=False):
    """Texto que ven los vectorizadores TF-IDF.

    Sin máscara: la oración original (se quitan las marcas [[ ]]). Con máscara: la mención
    marcada se sustituye por [PARTIDO] con postura.menciones.enmascarar."""
    texto = str(oracion_marcada or "")
    pos = posicion_mencion(texto)
    if pos is None:
        return texto
    if enmascarar:
        return _enmascarar(texto, pos[0], pos[1], token=TOKEN_MASCARA)
    ini, fin = pos
    return texto[:ini] + texto[ini + 2:fin - 2] + texto[fin:]


def _como_tabla(X):
    """Convierte la entrada (DataFrame, Series, diccionario o lista de registros) en DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, pd.Series):
        return X.to_frame().T if X.name is None else X.to_frame()
    if isinstance(X, dict):
        return pd.DataFrame([X])
    return pd.DataFrame(list(X))


# ---------------------------------------------------------------------------------------
# Transformadores
# ---------------------------------------------------------------------------------------

class PrepararEntrada(BaseEstimator, TransformerMixin):
    """Normaliza la tabla de entrada y añade la columna texto para los vectorizadores.

    No aprende nada (fit no hace nada). Devuelve un DataFrame con las columnas
    texto, oracion_marcada, contexto_previo, pregunta_prensa y grupo, en ese orden.

    Parámetros
    ----------
    enmascarar : bool
        Si es True, la columna texto lleva [PARTIDO] en lugar de la mención marcada.
    """

    def __init__(self, enmascarar=False):
        self.enmascarar = enmascarar

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = _como_tabla(X)
        salida = pd.DataFrame(index=df.index)
        if "oracion_marcada" in df.columns:
            marcada = df["oracion_marcada"].fillna("").astype(str)
        elif {"oracion", "inicio", "fin"} <= set(df.columns):
            marcada = pd.Series([marcar(str(o), int(i), int(f)) for o, i, f in
                                 zip(df["oracion"], df["inicio"], df["fin"])], index=df.index)
        else:
            raise KeyError("La tabla necesita oracion_marcada, o bien oracion, inicio y fin.")
        salida["texto"] = [texto_para_modelo(o, self.enmascarar) for o in marcada]
        salida["oracion_marcada"] = marcada
        for col in ("contexto_previo", "pregunta_prensa", "grupo"):
            salida[col] = df[col].fillna("").astype(str) if col in df.columns else ""
        return salida

    def get_feature_names_out(self, input_features=None):
        return np.array(COLUMNAS_SALIDA, dtype=object)


class RasgosDominio(BaseEstimator, TransformerMixin):
    """Rasgos del dominio a partir de oracion_marcada, contexto_previo y pregunta_prensa.

    Es un transformador sin estado: calcula para cada fila un vector de conteos y
    proporciones y lo devuelve como matriz dispersa. El escalado lo hace un MaxAbsScaler
    posterior dentro del Pipeline.

    Parámetros
    ----------
    ventanas : tupla de int
        Tamaños de ventana (en palabras a cada lado de la mención) para los conteos del léxico.
    con_identidad : bool
        Si es True añade rasgos de forma de la mención (todo en mayúsculas, inicial mayúscula,
        plural). Se apagan en la ablación con máscara porque delatan al partido.
    alcance_negacion : int
        Palabras hacia atrás en las que una negación afecta a un término del léxico.
    """

    def __init__(self, ventanas=(5, 10), con_identidad=True, alcance_negacion=3):
        self.ventanas = ventanas
        self.con_identidad = con_identidad
        self.alcance_negacion = alcance_negacion

    def fit(self, X, y=None):
        self.n_features_out_ = len(self._nombres())
        return self

    def _nombres(self):
        nombres = []
        for k in self.ventanas:
            nombres += [f"desf_v{k}", f"fav_v{k}", f"info_v{k}", f"neg_v{k}", f"intens_v{k}"]
        kmax = max(self.ventanas)
        nombres += [
            "desf_oracion", "fav_oracion", "deslinde_oracion",
            f"desf_negado_v{kmax}", f"fav_negado_v{kmax}", "negacion_antes_3",
            "posicion_relativa", "log_longitud", "exclamaciones", "interrogaciones", "comillas",
            "cifras", "pron_1s", "pron_1p", "menciones_en_oracion",
            "hay_pregunta", "pregunta_menciona_partido", "desf_pregunta", "fav_pregunta",
            "hay_contexto", "contexto_menciona_partido", "desf_contexto", "fav_contexto",
        ]
        if self.con_identidad:
            nombres += ["mencion_mayusculas", "mencion_inicial_mayuscula", "mencion_plural",
                        "mencion_varias_palabras"]
        return nombres

    def get_feature_names_out(self, input_features=None):
        return np.array(self._nombres(), dtype=object)

    @staticmethod
    def _menciona_partido(texto):
        return any(g not in GRUPOS_AMBIGUOS for (_, _, _, g) in buscar_menciones(str(texto or "")))

    def _fila(self, oracion_marcada, contexto, pregunta):
        antes_txt, mencion, despues_txt = lx.separar_marcada(oracion_marcada)
        antes, despues = lx.tokenizar(antes_txt), lx.tokenizar(despues_txt)
        plana = antes_txt + mencion + despues_txt
        v = []
        for k in self.ventanas:
            izq, der = (antes[-k:] if k > 0 else []), despues[:k]
            v += [lx.DESFAVORABLE.contar(izq) + lx.DESFAVORABLE.contar(der),
                  lx.FAVORABLE.contar(izq) + lx.FAVORABLE.contar(der),
                  lx.INFORMATIVO.contar(izq) + lx.INFORMATIVO.contar(der),
                  lx.NEGACIONES.contar(izq) + lx.NEGACIONES.contar(der),
                  lx.INTENSIFICADORES.contar(izq) + lx.INTENSIFICADORES.contar(der)]
        kmax = max(self.ventanas)
        izq, der = antes[-kmax:], despues[:kmax]
        todos = antes + despues
        v += [lx.DESFAVORABLE.contar(antes) + lx.DESFAVORABLE.contar(despues),
              lx.FAVORABLE.contar(antes) + lx.FAVORABLE.contar(despues),
              lx.DESLINDE.contar(antes) + lx.DESLINDE.contar(despues),
              lx.contar_negados(izq, lx.DESFAVORABLE, self.alcance_negacion)
              + lx.contar_negados(der, lx.DESFAVORABLE, self.alcance_negacion),
              lx.contar_negados(izq, lx.FAVORABLE, self.alcance_negacion)
              + lx.contar_negados(der, lx.FAVORABLE, self.alcance_negacion),
              float(any(lx.NEGACIONES.coincide(t) for t in antes[-3:]))]
        n_antes, n_despues = len(antes), len(despues)
        v += [n_antes / max(1, n_antes + n_despues),
              math.log1p(len(plana.split())),
              plana.count("!") + plana.count("¡"),
              plana.count("?") + plana.count("¿"),
              sum(plana.count(c) for c in _COMILLAS),
              sum(1 for t in todos if any(ch.isdigit() for ch in t)),
              lx.PRIMERA_SINGULAR.contar(todos),
              lx.PRIMERA_PLURAL.contar(todos),
              len(buscar_menciones(plana))]
        tok_preg = lx.tokenizar(pregunta)
        tok_ctx = lx.tokenizar(contexto)
        v += [float(bool(tok_preg)), float(self._menciona_partido(pregunta)),
              math.log1p(lx.DESFAVORABLE.contar(tok_preg)), math.log1p(lx.FAVORABLE.contar(tok_preg)),
              float(bool(tok_ctx)), float(self._menciona_partido(contexto)),
              math.log1p(lx.DESFAVORABLE.contar(tok_ctx)), math.log1p(lx.FAVORABLE.contar(tok_ctx))]
        if self.con_identidad:
            m = mencion.strip()
            letras = [c for c in m if c.isalpha()]
            v += [float(len(letras) >= 2 and all(c.isupper() for c in letras)),
                  float(bool(letras) and letras[0].isupper()),
                  float(m.lower().endswith("s")),
                  float(len(m.split()) > 1)]
        return v

    def transform(self, X):
        df = _como_tabla(X)
        cols = [df[c].fillna("").astype(str).tolist() if c in df.columns else [""] * len(df)
                for c in COLUMNAS_TEXTO]
        config = (tuple(self.ventanas), bool(self.con_identidad), int(self.alcance_negacion))
        filas = [_fila_memorizada(o, c, p, config) for o, c, p in zip(*cols)]
        matriz = np.asarray(filas, dtype=np.float64).reshape(len(filas), len(self._nombres()))
        return sparse.csr_matrix(matriz)


@lru_cache(maxsize=65536)
def _fila_memorizada(oracion_marcada, contexto, pregunta, config):
    """Memoización de RasgosDominio._fila. En la búsqueda de hiperparámetros las mismas filas
    se transforman una vez por pliegue y por configuración; con la caché se calculan una sola
    vez por proceso. config = (ventanas, con_identidad, alcance_negacion)."""
    ventanas, con_identidad, alcance = config
    calc = RasgosDominio(ventanas=ventanas, con_identidad=con_identidad, alcance_negacion=alcance)
    return tuple(calc._fila(oracion_marcada, contexto, pregunta))


class Longitud(BaseEstimator, TransformerMixin):
    """Un solo rasgo: log(1 + número de palabras de la oración). Para la línea base."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = _como_tabla(X)
        col = df["oracion_marcada"] if "oracion_marcada" in df.columns else df.iloc[:, 0]
        valores = [math.log1p(len(texto_para_modelo(o).split())) for o in col.fillna("").astype(str)]
        return sparse.csr_matrix(np.asarray(valores, dtype=np.float64).reshape(-1, 1))

    def get_feature_names_out(self, input_features=None):
        return np.array(["log_longitud"], dtype=object)


class SVDSeguro(BaseEstimator, TransformerMixin):
    """TruncatedSVD que recorta el número de componentes a lo que la matriz permite.

    Con muestras pequeñas (pruebas, pliegues chicos) puede haber menos rasgos que los 200
    componentes pedidos; en ese caso usa min(n_components, rasgos - 1, filas - 1).

    hilos_blas limita los hilos de la biblioteca BLAS durante el ajuste. En la máquina de
    desarrollo (14 núcleos, OpenBLAS) el SVD aleatorizado con 14 hilos tarda cerca de cuatro
    veces más que con 2 a 4 hilos por sobresuscripción; None deja el valor del sistema."""

    def __init__(self, n_components=200, random_state=None, hilos_blas=4):
        self.n_components = n_components
        self.random_state = random_state
        self.hilos_blas = hilos_blas

    def _limite(self):
        if self.hilos_blas is None:
            return contextlib.nullcontext()
        return threadpool_limits(limits=self.hilos_blas, user_api="blas")

    def fit(self, X, y=None):
        k = max(1, min(self.n_components, X.shape[1] - 1, X.shape[0] - 1))
        with self._limite():
            self.svd_ = TruncatedSVD(n_components=k, random_state=self.random_state).fit(X)
        # La matriz de componentes (k x vocabulario) domina el tamaño del modelo guardado;
        # en float32 ocupa la mitad y la proyección cambia en menos de 1e-6.
        self.svd_.components_ = self.svd_.components_.astype(np.float32)
        self.n_componentes_ = k
        return self

    def transform(self, X):
        with self._limite():
            return self.svd_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"svd{i}" for i in range(self.n_componentes_)], dtype=object)


# ---------------------------------------------------------------------------------------
# Fábricas de rasgos
# ---------------------------------------------------------------------------------------

def vectorizador_palabras(ngramas=(1, 2), min_df=2):
    """TF-IDF de palabras; conserva el token [partido] de la máscara."""
    return TfidfVectorizer(lowercase=True, strip_accents="unicode", token_pattern=PATRON_PALABRAS,
                           ngram_range=ngramas, min_df=min_df, sublinear_tf=True)


def vectorizador_caracteres(ngramas=(2, 5), min_df=2):
    """TF-IDF de n-gramas de caracteres dentro de los límites de palabra (char_wb)."""
    return TfidfVectorizer(lowercase=True, strip_accents="unicode", analyzer="char_wb",
                           ngram_range=ngramas, min_df=min_df, sublinear_tf=True)


def rasgos_dominio(enmascarar=False, ventanas=(5, 10)):
    """Rasgos del dominio escalados con MaxAbsScaler (conserva la dispersión)."""
    return Pipeline([("calculo", RasgosDominio(ventanas=ventanas, con_identidad=not enmascarar)),
                     ("escala", MaxAbsScaler())])


def construir_rasgos(enmascarar=False, incluir_grupo=None, palabras=True, caracteres=True,
                     dominio=True, min_df=2):
    """ColumnTransformer disperso que se coloca después de PrepararEntrada.

    Parámetros
    ----------
    enmascarar : bool
        Ablación sin nombre del partido (ver docstring del módulo).
    incluir_grupo : bool o None
        One-hot del grupo de la mención. Por omisión, se incluye salvo que se enmascare.
    palabras, caracteres, dominio : bool
        Permiten apagar bloques para ablaciones.
    """
    if incluir_grupo is None:
        incluir_grupo = not enmascarar
    bloques = []
    if palabras:
        bloques.append(("palabras", vectorizador_palabras(min_df=min_df), "texto"))
    if caracteres:
        bloques.append(("caracteres", vectorizador_caracteres(min_df=min_df), "texto"))
    if dominio:
        bloques.append(("dominio", rasgos_dominio(enmascarar), COLUMNAS_TEXTO))
    if incluir_grupo:
        bloques.append(("grupo", OneHotEncoder(handle_unknown="ignore"), ["grupo"]))
    if not bloques:
        raise ValueError("Se necesita al menos un bloque de rasgos.")
    return ColumnTransformer(bloques, sparse_threshold=1.0)


def construir_rasgos_densos(enmascarar=False, incluir_grupo=None, n_componentes=200,
                            semilla=20260927, min_df=2):
    """Rasgos para bosques aleatorios: TruncatedSVD del TF-IDF (palabras y caracteres) más
    los rasgos del dominio y el one-hot del grupo. Devuelve matriz densa."""
    if incluir_grupo is None:
        incluir_grupo = not enmascarar
    tfidf = FeatureUnion([("palabras", vectorizador_palabras(min_df=min_df)),
                          ("caracteres", vectorizador_caracteres(min_df=min_df))])
    bloques = [
        ("tfidf_svd", Pipeline([("tfidf", tfidf),
                                ("svd", SVDSeguro(n_components=n_componentes, random_state=semilla))]),
         "texto"),
        ("dominio", RasgosDominio(con_identidad=not enmascarar), COLUMNAS_TEXTO),
    ]
    if incluir_grupo:
        bloques.append(("grupo", OneHotEncoder(handle_unknown="ignore"), ["grupo"]))
    return ColumnTransformer(bloques, sparse_threshold=0.0)


def construir_vectorizador(enmascarar=False, **kwargs):
    """Pipeline completo tabla -> matriz dispersa (PrepararEntrada + construir_rasgos).

    Útil para medir tiempos de vectorización y para inspeccionar rasgos fuera de un modelo."""
    return Pipeline([("preparar", PrepararEntrada(enmascarar=enmascarar)),
                     ("rasgos", construir_rasgos(enmascarar=enmascarar, **kwargs))])
