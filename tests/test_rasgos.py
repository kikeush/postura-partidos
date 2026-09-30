# -*- coding: utf-8 -*-
"""Pruebas de postura.lexico y postura.rasgos: dimensiones, enmascarado y conteos en ventana.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.
"""
import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from postura import lexico as lx
from postura.rasgos import (PrepararEntrada, RasgosDominio, SVDSeguro, construir_rasgos_densos,
                            construir_vectorizador, posicion_mencion, texto_para_modelo)
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------------ léxico
def test_normalizar_quita_acentos_y_minusculas():
    assert lx.normalizar("Corrupción ÑOÑO") == "corrupcion nono"
    assert "corrupción" in lx.DESFAVORABLE and "CORRUPTOS" in lx.DESFAVORABLE
    assert "honestidad" in lx.FAVORABLE
    assert "robusto" not in lx.DESFAVORABLE and "decenas" not in lx.FAVORABLE


def test_lexico_cuenta_en_ventana():
    om = "Los del [[PRI]] fueron corruptos y ladrones, uno dos tres cuatro cinco seis saqueo"
    # a la derecha: fueron corruptos y ladrones uno | dos tres cuatro cinco seis saqueo
    assert lx.contar_en_ventana(om, lx.DESFAVORABLE, 5) == 2
    assert lx.contar_en_ventana(om, lx.DESFAVORABLE, 10) == 2
    assert lx.contar_en_ventana(om, lx.DESFAVORABLE, 11) == 3
    assert lx.contar_en_ventana(om, "favorable", 10) == 0


def test_ventana_excluye_la_mencion_y_respeta_lados():
    om = "corrupto uno dos [[PRIAN]] tres honestos"
    izq, der = lx.ventana(om, 2)
    assert izq == ["uno", "dos"] and der == ["tres", "honestos"]
    assert lx.contar_en_ventana(om, lx.DESFAVORABLE, 2) == 0
    assert lx.contar_en_ventana(om, lx.DESFAVORABLE, 3) == 1
    assert lx.contar_en_ventana(om, lx.FAVORABLE, 2) == 1


def test_expresiones_de_varias_palabras():
    assert lx.contar("están moralmente derrotados", lx.DESFAVORABLE) == 2  # derrotados + expresión
    assert lx.contar("la guerra sucia", lx.DESFAVORABLE) == 1


def test_negacion_cercana_y_negados():
    # "no" está a cuatro palabras de la mención: entra con k = 4 y no con k = 3.
    assert lx.negacion_cercana("Yo no creo que el [[PRI]] cambie", 4)
    assert not lx.negacion_cercana("Yo no creo que el [[PRI]] cambie", 3)
    assert lx.negacion_cercana("Yo no digo el [[PRI]]", 3)
    assert not lx.negacion_cercana("No es cierto, uno dos tres cuatro, el [[PRI]] cambió", 3)
    assert lx.contar_negados(lx.tokenizar("no son corruptos ni honestos"), lx.DESFAVORABLE) == 1
    assert lx.contar_negados(lx.tokenizar("no son corruptos ni honestos"), lx.FAVORABLE) == 1
    assert lx.contar_negados(lx.tokenizar("son corruptos"), lx.DESFAVORABLE) == 0


def test_separar_marcada_sin_marcas():
    assert lx.separar_marcada("sin marcas") == ("sin marcas", "", "")


# ------------------------------------------------------------------ texto y máscara
def test_texto_para_modelo_quita_marcas_o_enmascara():
    om = "El [[PRI]] y el PAN votaron."
    assert posicion_mencion(om) == (3, 10)
    assert texto_para_modelo(om) == "El PRI y el PAN votaron."
    assert texto_para_modelo(om, enmascarar=True) == "El [PARTIDO] y el PAN votaron."


def _tabla_prian(n=30):
    filas = []
    for i in range(n):
        filas.append({"oracion_marcada": f"El [[PRIAN]] saqueó al país número {i % 5} y mintió.",
                      "contexto_previo": "Ya lo dije antes.", "pregunta_prensa": "¿Qué opina?",
                      "grupo": "PRIAN"})
        filas.append({"oracion_marcada": f"Morena ganó la votación {i % 5} en la [[oposición]] honesta.",
                      "contexto_previo": "", "pregunta_prensa": "", "grupo": "oposicion"})
    return pd.DataFrame(filas)


def test_enmascarado_quita_el_nombre_del_partido():
    tabla = _tabla_prian()
    textos = PrepararEntrada(enmascarar=True).transform(tabla)["texto"]
    assert all("[PARTIDO]" in t for t in textos)
    assert not any("PRIAN" in t or "oposición" in t for t in textos)
    vec = construir_vectorizador(enmascarar=True).fit(tabla)
    ct = vec.named_steps["rasgos"]
    vocab = ct.named_transformers_["palabras"].vocabulary_
    assert "[partido]" in vocab
    assert not any("prian" in w or "oposicion" in w for w in vocab)
    assert "grupo" not in ct.named_transformers_  # sin one-hot del grupo
    nombres = ct.get_feature_names_out()
    assert not any("mencion_mayusculas" in n for n in nombres)


def test_sin_mascara_conserva_el_nombre():
    vec = construir_vectorizador(enmascarar=False).fit(_tabla_prian())
    vocab = vec.named_steps["rasgos"].named_transformers_["palabras"].vocabulary_
    assert "prian" in vocab and "[partido]" not in vocab


# ------------------------------------------------------------------ dimensiones
def test_dimensiones_y_nombres(muestra):
    sub = muestra.head(200)
    vec = construir_vectorizador().fit(sub)
    X = vec.transform(sub)
    assert sparse.issparse(X) and X.shape[0] == 200
    nombres = vec.named_steps["rasgos"].get_feature_names_out()
    assert len(nombres) == X.shape[1]
    ct = vec.named_steps["rasgos"]
    n_pal = len(ct.named_transformers_["palabras"].vocabulary_)
    n_car = len(ct.named_transformers_["caracteres"].vocabulary_)
    n_dom = len(RasgosDominio().fit(sub).get_feature_names_out())
    n_grp = len(ct.named_transformers_["grupo"].categories_[0])
    assert X.shape[1] == n_pal + n_car + n_dom + n_grp
    # Filas nuevas con un grupo no visto no rompen la transformación.
    nueva = sub.head(1).assign(grupo="grupo_inexistente")
    assert vec.transform(nueva).shape == (1, X.shape[1])


def test_rasgos_dominio_valores(muestra):
    calc = RasgosDominio().fit(muestra.head(5))
    fila = pd.DataFrame([{"oracion_marcada": "¡No! Yo creo que el [[PRI]] fue corrupto y nosotros honestos, ¿verdad?",
                          "contexto_previo": "Hablemos del PAN.", "pregunta_prensa": ""}])
    x = dict(zip(calc.get_feature_names_out(), calc.transform(fila).toarray()[0]))
    assert x["desf_v5"] == 1 and x["fav_v5"] == 1
    assert x["pron_1s"] == 1 and x["pron_1p"] == 1
    assert x["exclamaciones"] == 2 and x["interrogaciones"] == 2
    assert x["hay_pregunta"] == 0 and x["contexto_menciona_partido"] == 1
    assert 0 < x["posicion_relativa"] < 1
    assert x["mencion_mayusculas"] == 1


def test_rasgos_dominio_con_columnas_faltantes():
    calc = RasgosDominio().fit(None)
    X = calc.transform(pd.DataFrame({"oracion_marcada": ["El [[PAN]] votó."]}))
    assert X.shape == (1, len(calc.get_feature_names_out()))


def test_preparar_entrada_desde_oracion_inicio_fin():
    tabla = pd.DataFrame([{"oracion": "El PRI votó.", "inicio": 3, "fin": 6, "grupo": "PRI"}])
    salida = PrepararEntrada().transform(tabla)
    assert salida.loc[0, "oracion_marcada"] == "El [[PRI]] votó."
    assert salida.loc[0, "contexto_previo"] == ""


def test_densos_y_svd_seguro(muestra):
    sub = muestra.head(60)
    tub = Pipeline([("preparar", PrepararEntrada()), ("rasgos", construir_rasgos_densos(n_componentes=200))])
    X = tub.fit_transform(sub)
    assert isinstance(X, np.ndarray) and X.shape[0] == 60
    svd = tub.named_steps["rasgos"].named_transformers_["tfidf_svd"].named_steps["svd"]
    assert svd.n_componentes_ <= 59  # recortado por el número de filas
    assert SVDSeguro(5).fit(sparse.random(10, 3, density=0.5, random_state=0)).n_componentes_ == 2


def test_serializable_con_joblib(muestra, tmp_path):
    sub = muestra.head(150)
    vec = construir_vectorizador().fit(sub)
    ruta = tmp_path / "vec.joblib"
    joblib.dump(vec, ruta)
    otro = joblib.load(ruta)
    assert abs(otro.transform(sub) - vec.transform(sub)).max() == pytest.approx(0.0)
