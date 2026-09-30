# -*- coding: utf-8 -*-
"""Entrena, selecciona y evalúa los clasificadores clásicos de la Fase 2 (T1, T2a y T2b).

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

Protocolo
---------
Para cada tarea binaria (T1 es_partido sobre todos los ítems; T2a desfavorable frente al resto
y T2b favorable frente al resto sobre los ítems con es_partido = 1):

1. Búsqueda de hiperparámetros en entrenamiento con GridSearchCV, 5 pliegues agrupados por
   fecha (GroupKFold), puntuación roc_auc y n_jobs = -1.
2. Cada modelo, con sus mejores hiperparámetros, se ajusta en entrenamiento y se puntúa en
   validación. El umbral que maximiza F1 en validación queda fijo.
3. El mejor modelo (sin contar líneas base) es el de mayor AUC en validación.
4. Todos los modelos se reentrenan en entrenamiento más validación y se evalúan en prueba:
   AUC, intervalo de confianza por remuestreo de conferencias (1000 réplicas, semilla fija),
   intervalo de DeLong, PR-AUC y F1, precisión y exhaustividad al umbral de validación.
5. Prueba de DeLong del mejor modelo frente a la línea base de solo grupo (y frente a los
   demás), más la diferencia de AUC por bootstrap agrupado.
6. Ablación: el mejor tipo de modelo con la mención sustituida por [PARTIDO] y sin grupo.
7. Importancia de rasgos: coeficientes de la regresión logística final, 20 por clase.
8. Tiempos para el reporte de optimización: carga, vectorización, memoización de rasgos,
   búsqueda con n_jobs = 1 frente a n_jobs = -1, bosque con 1 frente a todos los núcleos,
   SVD con y sin límite de hilos BLAS e inferencia por cada mil menciones.

Salidas (por omisión)
---------------------
resultados/fase2_resultados.json, resultados/fase2_predicciones_prueba.csv,
resultados/figuras/fase2_roc_T1.png, _T2a.png, _T2b.png,
modelos/modelo_T1.joblib, modelo_T2a.joblib, modelo_T2b.joblib y modelos/umbrales.json.

Uso
---
    python scripts/04_entrenar_clasicos.py
    python scripts/04_entrenar_clasicos.py --etiquetas otra/ruta.csv --salida otra/carpeta \
        --modelos otra/carpeta_modelos [--rapido] [--n-boot 1000] [--tareas T1 T2a] \
        [--familias logistica svm_lineal] [--sin-tiempos]
"""
import argparse
import json
import os
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from postura.nombres import NOMBRE_TAREA, nombre_modelo  # noqa: E402
from postura import evaluacion as ev  # noqa: E402
from postura import rasgos  # noqa: E402
from postura.modelos import (SEMILLA, ajustar_con_probabilidad,  # noqa: E402
                             bosque_aleatorio, catalogo, estado_pbc4cip, fijar_n_jobs,
                             logistica, puntaje)
from postura.particion import (PARTICIONES, RUTA_ETIQUETAS, RUTA_MENCIONES,  # noqa: E402
                               RUTA_MUESTRA, leer_etiquetas, leer_muestra, pliegues_por_fecha,
                               separar, unir_etiquetas, verificar_particion)
from postura.pipeline import ARCHIVOS, Clasificador  # noqa: E402

TAREAS = {
    "T1": {"descripcion": "La cadena marcada se refiere a un partido (1) o no (0); todos los ítems.",
           "positiva": "es_partido", "negativa": "no_partido", "acuerdo": "acuerdo_es_partido"},
    "T2a": {"descripcion": "Postura desfavorable (1) frente a favorable o neutral (0); ítems con es_partido = 1.",
            "positiva": "desfavorable", "negativa": "favorable_o_neutral", "acuerdo": "acuerdo_postura"},
    "T2b": {"descripcion": "Postura favorable (1) frente a desfavorable o neutral (0); ítems con es_partido = 1.",
            "positiva": "favorable", "negativa": "desfavorable_o_neutral", "acuerdo": "acuerdo_postura"},
}


# ---------------------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------------------

def a_json(obj):
    """Convierte recursivamente tipos de numpy, tuplas y NaN a tipos JSON estrictos."""
    if isinstance(obj, dict):
        return {str(k): a_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [a_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return a_json(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return repr(obj)


def relativa(ruta):
    """Devuelve la ruta relativa a la raíz del proyecto si está dentro de ella; si no, la ruta absoluta."""
    ruta = os.path.abspath(ruta)
    return os.path.relpath(ruta, RAIZ) if ruta.startswith(RAIZ + os.sep) else ruta


def datos_tarea(tabla, tarea):
    """Subconjunto y variable objetivo y de una tarea."""
    if tarea == "T1":
        d = tabla.copy()
        d["y"] = d["es_partido"].astype(int)
    else:
        d = tabla[(tabla["es_partido"] == 1)
                  & tabla["postura"].isin(["desfavorable", "favorable", "neutral"])].copy()
        clase = "desfavorable" if tarea == "T2a" else "favorable"
        d["y"] = (d["postura"] == clase).astype(int)
    return d.reset_index(drop=True)


def _ordenable(valor):
    """Convierte None y los valores no finitos en menos infinito para ordenar por AUC."""
    return -np.inf if valor is None or not np.isfinite(valor) else valor


def buscar(espec, X, y, pliegues, n_jobs):
    """GridSearchCV con AUC; devuelve (estimador sin ajustar con los mejores parámetros, info)."""
    est = clone(espec.estimador)
    if n_jobs != 1:
        fijar_n_jobs(est, 1)  # procesos por fuera, un hilo por dentro
    gs = GridSearchCV(est, espec.rejilla, scoring="roc_auc", cv=pliegues, n_jobs=n_jobs,
                      refit=False, error_score=np.nan)
    t0 = time.perf_counter()
    gs.fit(X, y)
    segundos = time.perf_counter() - t0
    mejor = clone(espec.estimador).set_params(**gs.best_params_)
    i = gs.best_index_
    info = {"auc_cv_media": gs.best_score_, "auc_cv_de": gs.cv_results_["std_test_score"][i],
            "mejores_parametros": {k: (list(v) if isinstance(v, tuple) else v)
                                   for k, v in gs.best_params_.items()},
            "configuraciones": len(gs.cv_results_["params"]),
            "ajustes": len(gs.cv_results_["params"]) * len(pliegues),
            "segundos_busqueda": segundos}
    return mejor, info


def metricas_prueba(y, s, fechas, umbral, n_boot, semilla):
    """AUC con IC por conferencias y de DeLong, PR-AUC y métricas al umbral de validación."""
    y = np.asarray(y)
    salida = {"n": int(len(y)), "positivos": int(y.sum()), "prevalencia": float(y.mean()) if len(y) else None}
    if y.min() == y.max():
        salida["nota"] = "una sola clase en prueba; AUC no definida"
        return salida
    salida["auc"] = ev.roc_auc(y, s)
    salida["ic_bootstrap_conferencias"] = ev.bootstrap_agrupado(y, s, fechas, n_replicas=n_boot, semilla=semilla)
    try:
        salida["ic_delong"] = ev.ic_delong(y, s)
    except ValueError as error:
        salida["ic_delong"] = {"error": str(error)}
    salida["pr_auc"] = ev.pr_auc(y, s)
    salida["al_umbral_de_validacion"] = ev.metricas_umbral(y, s, umbral)
    return salida


def importancia_logistica(modelo, n=20, positiva="1", negativa="0"):
    """Top n coeficientes positivos y negativos de un Pipeline con regresión logística."""
    nombres = modelo.named_steps["rasgos"].get_feature_names_out()
    coef = modelo.named_steps["clf"].coef_.ravel()
    orden = np.argsort(coef)
    return {positiva: [{"rasgo": str(nombres[i]), "coeficiente": float(coef[i])} for i in orden[::-1][:n]],
            negativa: [{"rasgo": str(nombres[i]), "coeficiente": float(coef[i])} for i in orden[:n]]}


def _tarea_trivial(i):
    """Devuelve su argumento; sirve para arrancar los procesos de joblib antes de medir tiempos."""
    return i


# ---------------------------------------------------------------------------------------
# Experimentos de tiempo (reporte de optimización)
# ---------------------------------------------------------------------------------------

def medir_tiempos(tr, te, pliegues, args):
    """Tiempos de vectorización, memoización, paralelismo y límite de hilos BLAS."""
    tiempos = {}
    rasgos._fila_memorizada.cache_clear()
    vec = rasgos.construir_vectorizador()
    t0 = time.perf_counter()
    X = vec.fit_transform(tr)
    t_ajuste = time.perf_counter() - t0
    t0 = time.perf_counter()
    Xte = vec.transform(te)
    t_trans = time.perf_counter() - t0
    bytes_disperso = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    tiempos["vectorizacion"] = {
        "filas_ajuste": X.shape[0], "rasgos": X.shape[1], "no_ceros": int(X.nnz),
        "densidad": X.nnz / (X.shape[0] * X.shape[1]),
        "mb_dispersa": bytes_disperso / 2 ** 20, "mb_si_fuera_densa": X.shape[0] * X.shape[1] * 8 / 2 ** 20,
        "segundos_ajuste_y_transformacion": t_ajuste, "segundos_transformar_prueba": t_trans,
        "filas_prueba": Xte.shape[0]}

    rasgos._fila_memorizada.cache_clear()
    calc = rasgos.RasgosDominio().fit(tr)
    t0 = time.perf_counter()
    calc.transform(tr)
    t_frio = time.perf_counter() - t0
    t0 = time.perf_counter()
    calc.transform(tr)
    t_caliente = time.perf_counter() - t0
    tiempos["memoizacion_rasgos_dominio"] = {
        "filas": len(tr), "segundos_sin_cache": t_frio, "segundos_con_cache": t_caliente,
        "aceleracion": t_frio / max(t_caliente, 1e-9)}

    # Búsqueda en rejilla pequeña: 4 valores de C x pliegues, n_jobs = 1 frente a -1.
    Parallel(n_jobs=args.n_jobs)(delayed(_tarea_trivial)(i) for i in range(64))  # arranque de procesos
    espec = logistica(semilla=args.semilla)
    rejilla = {"clf__C": [0.1, 0.3, 1.0, 3.0]}
    resultado = {"rejilla": rejilla, "pliegues": len(pliegues), "ajustes": 4 * len(pliegues)}
    for nj in (1, args.n_jobs):
        rasgos._fila_memorizada.cache_clear()
        gs = GridSearchCV(clone(espec.estimador), rejilla, scoring="roc_auc", cv=pliegues,
                          n_jobs=nj, refit=False)
        t0 = time.perf_counter()
        gs.fit(tr, tr["y"])
        resultado[f"segundos_n_jobs_{nj}"] = time.perf_counter() - t0
        resultado[f"auc_n_jobs_{nj}"] = gs.best_score_
    resultado["aceleracion"] = resultado["segundos_n_jobs_1"] / resultado[f"segundos_n_jobs_{args.n_jobs}"]
    resultado["nucleos"] = os.cpu_count()
    tiempos["busqueda_paralela"] = resultado

    # Bosque aleatorio de 500 árboles sobre rasgos densos precalculados.
    densos = rasgos.construir_rasgos_densos(semilla=args.semilla)
    prep = rasgos.PrepararEntrada()
    Xd = densos.fit_transform(prep.transform(tr))
    bosque = {"arboles": 100 if args.rapido else 500, "filas": Xd.shape[0], "rasgos": Xd.shape[1]}
    for nj in (1, -1):
        rf = clone(bosque_aleatorio(rapido=args.rapido, semilla=args.semilla).estimador.named_steps["clf"])
        rf.set_params(n_jobs=nj)
        t0 = time.perf_counter()
        rf.fit(Xd, tr["y"])
        bosque[f"segundos_n_jobs_{nj}"] = time.perf_counter() - t0
    bosque["aceleracion"] = bosque["segundos_n_jobs_1"] / bosque["segundos_n_jobs_-1"]
    tiempos["bosque_paralelo"] = bosque

    # SVD de 200 componentes con y sin límite de hilos BLAS.
    tfidf = rasgos.FeatureUnion([("palabras", rasgos.vectorizador_palabras()),
                                 ("caracteres", rasgos.vectorizador_caracteres())])
    T = tfidf.fit_transform(prep.transform(tr)["texto"])
    svd = {"filas": T.shape[0], "rasgos": T.shape[1], "componentes": 200}
    for hilos in (None, 4):
        s = rasgos.SVDSeguro(200, random_state=args.semilla, hilos_blas=hilos)
        t0 = time.perf_counter()
        s.fit(T)
        svd["segundos_hilos_" + ("sistema" if hilos is None else str(hilos))] = time.perf_counter() - t0
    svd["aceleracion"] = svd["segundos_hilos_sistema"] / svd["segundos_hilos_4"]
    tiempos["svd_hilos_blas"] = svd
    return tiempos


def medir_inferencia(dir_modelos, ruta_menciones, respaldo, semilla, n=1000, repeticiones=3):
    """Segundos para clasificar mil menciones con Clasificador.analizar_tabla (mediana)."""
    clf = Clasificador(dir_modelos)
    if ruta_menciones and os.path.exists(ruta_menciones):
        tabla = pd.read_parquet(ruta_menciones)
        tabla = tabla.sample(n=min(n, len(tabla)), random_state=semilla)
        origen = relativa(ruta_menciones)
    else:
        tabla = respaldo.sample(n=n, replace=True, random_state=semilla)
        origen = "muestra de prueba con reemplazo"
    tiempos = []
    for _ in range(repeticiones):
        rasgos._fila_memorizada.cache_clear()
        t0 = time.perf_counter()
        clf.analizar_tabla(tabla)
        tiempos.append(time.perf_counter() - t0)
    mediana = float(np.median(tiempos))
    parrafo = ("Los del PRIAN saquearon al país y ahora se dicen demócratas. Morena ganó la "
               "votación en la Cámara. Compramos pan en la esquina.")
    t0 = time.perf_counter()
    ejemplo = clf.analizar(parrafo)
    return {"menciones": len(tabla), "origen": origen, "repeticiones": repeticiones,
            "segundos": tiempos, "segundos_por_mil": mediana * 1000 / len(tabla),
            "menciones_por_segundo": len(tabla) / mediana,
            "segundos_parrafo_ejemplo": time.perf_counter() - t0, "ejemplo": ejemplo}


# ---------------------------------------------------------------------------------------
# Una tarea
# ---------------------------------------------------------------------------------------

def evaluar_tarea(tarea, datos, args, dir_figuras, dir_modelos):
    """Aplica a una tarea los pasos 1 a 7 del protocolo, guarda su figura ROC y su modelo final y devuelve el resumen, el umbral y las predicciones de prueba."""
    info_t = TAREAS[tarea]
    part = separar(datos)
    tr, va, te = part["entrenamiento"], part["validacion"], part["prueba"]
    trva = pd.concat([tr, va], ignore_index=True)
    pl_tr = pliegues_por_fecha(tr["fecha"], args.pliegues)
    pl_trva = pliegues_por_fecha(trva["fecha"], args.pliegues)
    resumen = {"descripcion": info_t["descripcion"],
               "n": {p: int(len(part[p])) for p in PARTICIONES},
               "positivos": {p: int(part[p]["y"].sum()) for p in PARTICIONES},
               "conferencias": {p: int(part[p]["fecha"].nunique()) for p in PARTICIONES},
               "modelos": {}}
    print(f"\n[{tarea}] n = {resumen['n']}, positivos = {resumen['positivos']}", flush=True)

    familias = None if not args.familias else sorted(set(args.familias) | {"base_grupo"})
    especs = catalogo(rapido=args.rapido, semilla=args.semilla, nombres=familias)
    if not [n for n, e in especs.items() if not e.linea_base]:
        raise ValueError("--familias debe incluir al menos un modelo que no sea línea base.")
    puntajes_te, finales, mejores_est = {}, {}, {}
    for nombre, espec in especs.items():
        est, info = buscar(espec, tr, tr["y"], pl_tr, args.n_jobs)
        t0 = time.perf_counter()
        m_tr = ajustar_con_probabilidad(est, tr, tr["y"], pl_tr)
        s_va = puntaje(m_tr, va)
        info["auc_validacion"] = ev.roc_auc(va["y"], s_va)
        info["pr_auc_validacion"] = ev.pr_auc(va["y"], s_va)
        umbral, f1_va = ev.umbral_f1(va["y"], s_va)
        info["umbral_validacion"], info["f1_validacion"] = umbral, f1_va
        t1 = time.perf_counter()
        m_final = ajustar_con_probabilidad(est, trva, trva["y"], pl_trva)
        t2 = time.perf_counter()
        s_te = puntaje(m_final, te)
        t3 = time.perf_counter()
        info["prueba"] = metricas_prueba(te["y"], s_te, te["fecha"], umbral, args.n_boot, args.semilla)
        info["segundos_ajuste_entrenamiento"] = t1 - t0
        info["segundos_ajuste_final"] = t2 - t1
        info["segundos_prediccion_prueba"] = t3 - t2
        info["descripcion"] = espec.descripcion
        info["linea_base"] = espec.linea_base
        resumen["modelos"][nombre] = info
        puntajes_te[nombre], finales[nombre], mejores_est[nombre] = s_te, m_final, est
        print(f"  {nombre:17s} AUC cv {info['auc_cv_media']:.3f}  val {info['auc_validacion']:.3f}  "
              f"prueba {info['prueba'].get('auc', float('nan')):.3f}  ({info['segundos_busqueda']:.1f} s)",
              flush=True)

    candidatos = [n for n, e in especs.items() if not e.linea_base]
    mejor = max(candidatos, key=lambda n: (_ordenable(resumen["modelos"][n]["auc_validacion"]),
                                           _ordenable(resumen["modelos"][n]["auc_cv_media"])))
    resumen["mejor_modelo"] = mejor
    resumen["criterio_seleccion"] = "mayor AUC en validación entre modelos que no son línea base"
    y_te = te["y"].to_numpy()
    s_mejor = puntajes_te[mejor]
    print(f"  mejor: {mejor}", flush=True)

    # Comparaciones pareadas en prueba.
    comparaciones = {}
    if y_te.min() != y_te.max():
        for otro in especs:
            if otro == mejor:
                continue
            try:
                comparaciones[otro] = {"delong": ev.delong(y_te, s_mejor, puntajes_te[otro])}
            except ValueError as error:
                comparaciones[otro] = {"delong": {"error": str(error)}}
        comparaciones["base_grupo"]["bootstrap_conferencias"] = ev.bootstrap_diferencia_agrupado(
            y_te, s_mejor, puntajes_te["base_grupo"], te["fecha"], n_replicas=args.n_boot,
            semilla=args.semilla)
    resumen["comparaciones_del_mejor"] = comparaciones
    resumen["nota_comparaciones"] = ("DeLong supone ítems independientes; el bootstrap por "
                                     "conferencias respeta la dependencia dentro de cada fecha. "
                                     "Varias comparaciones: interpretar con corrección de Holm.")

    # Ablación con la mención enmascarada.
    espec_m = catalogo(enmascarar=True, rapido=args.rapido, semilla=args.semilla, nombres=[mejor])[mejor]
    est_m, info_m = buscar(espec_m, tr, tr["y"], pl_tr, args.n_jobs)
    m_tr_m = ajustar_con_probabilidad(est_m, tr, tr["y"], pl_tr)
    s_va_m = puntaje(m_tr_m, va)
    umbral_m, _ = ev.umbral_f1(va["y"], s_va_m)
    m_final_m = ajustar_con_probabilidad(est_m, trva, trva["y"], pl_trva)
    s_te_m = puntaje(m_final_m, te)
    info_m["auc_validacion"] = ev.roc_auc(va["y"], s_va_m)
    info_m["prueba"] = metricas_prueba(y_te, s_te_m, te["fecha"], umbral_m, args.n_boot, args.semilla)
    if y_te.min() != y_te.max():
        info_m["delong_frente_al_mejor_sin_mascara"] = ev.delong(y_te, s_mejor, s_te_m)
    info_m["descripcion"] = (f"{mejor} con la mención sustituida por [PARTIDO], sin one-hot de grupo "
                             "ni rasgos de forma de la mención.")
    resumen["ablacion_enmascarada"] = info_m
    print(f"  ablación enmascarada: AUC prueba {info_m['prueba'].get('auc', float('nan')):.3f}", flush=True)

    # Subconjunto de acuerdo máximo.
    col_acuerdo = info_t["acuerdo"]
    if col_acuerdo in te.columns:
        mascara = (pd.to_numeric(te[col_acuerdo], errors="coerce") == 3).to_numpy()
        sub = {"criterio": f"{col_acuerdo} = 3", "n": int(mascara.sum()),
               "positivos": int(y_te[mascara].sum()) if mascara.any() else 0}
        if mascara.any() and y_te[mascara].min() != y_te[mascara].max():
            sub["auc_mejor"] = ev.roc_auc(y_te[mascara], s_mejor[mascara])
            sub["auc_base_grupo"] = ev.roc_auc(y_te[mascara], puntajes_te["base_grupo"][mascara])
        resumen["prueba_acuerdo_maximo"] = sub

    # Importancia de rasgos de la regresión logística final.
    if "logistica" in finales:
        resumen["importancia_logistica"] = importancia_logistica(
            finales["logistica"], 20, info_t["positiva"], info_t["negativa"])

    # Figura ROC en prueba.
    if y_te.min() != y_te.max():
        elegido = f"{nombre_modelo(mejor)} (Fase 2)"
        curvas = {(elegido if n == mejor else nombre_modelo(n)): ev.curva_roc(y_te, s) for n, s in puntajes_te.items()}
        curvas[f"{nombre_modelo(mejor)} sin nombre del partido"] = ev.curva_roc(y_te, s_te_m)
        ruta_fig = os.path.join(dir_figuras, f"fase2_roc_{tarea}.png")
        ev.graficar_roc(curvas, ruta_fig, titulo=f"Fase 2, {tarea}. {NOMBRE_TAREA[tarea]} (prueba, n = {len(y_te)})",
                        destacar=elegido)
        resumen["figura_roc"] = relativa(ruta_fig)

    # Modelo final y umbral.
    ruta_modelo = os.path.join(dir_modelos, ARCHIVOS[tarea])
    fijar_n_jobs(finales[mejor], 1)  # en inferencia bastan pocos datos por llamada
    joblib.dump(finales[mejor], ruta_modelo, compress=3)
    resumen["archivo_modelo"] = relativa(ruta_modelo)
    umbral_info = {"umbral": resumen["modelos"][mejor]["umbral_validacion"], "modelo": mejor,
                   "parametros": resumen["modelos"][mejor]["mejores_parametros"],
                   "auc_validacion": resumen["modelos"][mejor]["auc_validacion"],
                   "auc_prueba": resumen["modelos"][mejor]["prueba"].get("auc"),
                   "clase_positiva": info_t["positiva"]}

    pred = pd.DataFrame({"id": te["id"], "particion": te["particion"], "fecha": te["fecha"],
                         "tarea": tarea, "y_real": y_te, "mejor_modelo": mejor})
    for n, s in puntajes_te.items():
        pred[f"puntaje_{n}"] = s
    pred["puntaje_mejor_enmascarado"] = s_te_m
    return resumen, umbral_info, pred


# ---------------------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------------------

def argumentos(argv=None):
    """Lee las opciones de la línea de comandos."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--muestra", default=RUTA_MUESTRA)
    p.add_argument("--etiquetas", default=RUTA_ETIQUETAS)
    p.add_argument("--menciones", default=RUTA_MENCIONES,
                   help="Tabla de menciones para medir la inferencia por mil menciones.")
    p.add_argument("--salida", default=os.path.join(RAIZ, "resultados"))
    p.add_argument("--modelos", default=os.path.join(RAIZ, "modelos"))
    p.add_argument("--tareas", nargs="+", default=list(TAREAS), choices=list(TAREAS))
    p.add_argument("--pliegues", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--semilla", type=int, default=SEMILLA)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--rapido", action="store_true", help="Rejillas reducidas (prueba de humo).")
    p.add_argument("--familias", nargs="+", default=None,
                   choices=["base_grupo", "base_longitud", "lexico", "logistica", "svm_lineal",
                            "bosque_aleatorio", "pbc4cip"],
                   help="Restringe los modelos (base_grupo siempre se incluye). Por omisión, todos.")
    p.add_argument("--sin-tiempos", action="store_true",
                   help="Omite los experimentos de aceleración (útil en pruebas).")
    return p.parse_args(argv)


def main(argv=None):
    """Corre el protocolo en las tareas pedidas, mide los tiempos (salvo con --sin-tiempos) y escribe resultados, predicciones de prueba, modelos, umbrales y figuras."""
    args = argumentos(argv)
    t_inicio = time.perf_counter()
    dir_figuras = os.path.join(args.salida, "figuras")
    for d in (args.salida, dir_figuras, args.modelos):
        os.makedirs(d, exist_ok=True)

    t0 = time.perf_counter()
    muestra = leer_muestra(args.muestra)
    etiquetas = leer_etiquetas(args.etiquetas)
    tabla, resumen_union = unir_etiquetas(muestra, etiquetas)
    t_carga = time.perf_counter() - t0
    sinteticas = bool("sintetica" in etiquetas.columns and (etiquetas["sintetica"] == 1).any())
    if sinteticas:
        print("AVISO: etiquetas sintéticas; los resultados solo prueban el código.", flush=True)
    resumen_particion = verificar_particion(tabla)

    resultados = {
        "descripcion": "Fase 2: clasificadores clásicos para menciones a partidos y postura del titular.",
        "autor": "Enrique Alberto Mendoza Ruiz, Universidad Panamericana",
        "fecha_ejecucion": time.strftime("%Y-%m-%d %H:%M:%S"),
        "etiquetas_sinteticas": sinteticas,
        "configuracion": {"muestra": relativa(args.muestra), "etiquetas": relativa(args.etiquetas),
                          "pliegues": args.pliegues, "n_boot": args.n_boot, "semilla": args.semilla,
                          "n_jobs": args.n_jobs, "rapido": args.rapido, "tareas": args.tareas,
                          "familias": args.familias or "todas"},
        "entorno": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "plataforma": platform.platform(), "nucleos": os.cpu_count()},
        "union_etiquetas": resumen_union, "particion": resumen_particion,
        "pbc4cip": estado_pbc4cip(), "tareas": {}, "tiempos": {"segundos_carga": t_carga},
    }

    if not args.sin_tiempos:
        d1 = datos_tarea(tabla, "T1")
        p1 = separar(d1)
        print("Midiendo tiempos para el reporte de optimización...", flush=True)
        resultados["tiempos"].update(medir_tiempos(
            p1["entrenamiento"], p1["prueba"],
            pliegues_por_fecha(p1["entrenamiento"]["fecha"], args.pliegues), args))

    umbrales = {"fecha": time.strftime("%Y-%m-%d"), "sklearn": sklearn.__version__,
                "etiquetas_sinteticas": sinteticas,
                "regla": "no_partido si p_partido < T1; si no, desfavorable o favorable según sus "
                         "umbrales (gana el mayor margen relativo) y neutral si ninguno los supera."}
    predicciones = []
    for tarea in args.tareas:
        t0 = time.perf_counter()
        res, umb, pred = evaluar_tarea(tarea, datos_tarea(tabla, tarea), args, dir_figuras, args.modelos)
        res["segundos_tarea"] = time.perf_counter() - t0
        resultados["tareas"][tarea] = res
        umbrales[tarea] = umb
        predicciones.append(pred)

    with open(os.path.join(args.modelos, "umbrales.json"), "w", encoding="utf-8") as fh:
        json.dump(a_json(umbrales), fh, ensure_ascii=False, indent=1)
    pd.concat(predicciones, ignore_index=True).to_csv(
        os.path.join(args.salida, "fase2_predicciones_prueba.csv"), index=False)

    if set(args.tareas) == set(TAREAS):
        tabla_prueba = datos_tarea(tabla, "T1")
        resultados["tiempos"]["inferencia"] = medir_inferencia(
            args.modelos, args.menciones, tabla_prueba[tabla_prueba["particion"] == "prueba"],
            args.semilla)
    resultados["segundos_totales"] = time.perf_counter() - t_inicio
    ruta_json = os.path.join(args.salida, "fase2_resultados.json")
    with open(ruta_json, "w", encoding="utf-8") as fh:
        json.dump(a_json(resultados), fh, ensure_ascii=False, indent=1)
    print(f"\nListo en {resultados['segundos_totales']:.1f} s. Resultados en {ruta_json}", flush=True)
    return resultados


if __name__ == "__main__":
    main()
