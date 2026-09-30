# -*- coding: utf-8 -*-
"""Evaluación de la Fase 3 y comparación con la Fase 2 sobre el mismo conjunto de prueba.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

Modelos de la Fase 3:
  qwen_cero      LLM local sin ejemplos, probabilidad de la primera letra (X, D, F, N)
  qwen_pocos     LLM local con ocho ejemplos resueltos
  qwen_cot       LLM local con razonamiento en cadena y JSON validado (subconjunto de prueba)
  emb_logistica  embeddings del LLM (vector medio) + regresión logística
  emb_svm        embeddings del LLM + SVM lineal
  emb_bosque     embeddings del LLM + Random Forest
  bert_ajustado  ajuste fino completo de un transformer de tipo BERT (si existe su archivo)
Para cada tarea se reporta ROC-AUC con intervalo por remuestreo de conferencias, PR-AUC y la
prueba de DeLong contra el mejor modelo de la Fase 2 en los mismos ítems.
Los tres modelos de embeddings necesitan datos/fase3_embeddings_qwen.npy (unos 15 MB, no viaja
en el repositorio; se regenera con python scripts/06_fase3_qwen.py embeddings). Si falta, el
script avisa, lo anota en el JSON de salida y sigue sin ellos. Una fuente con puntajes no
finitos (NaN) se omite con aviso. El mejor modelo de la Fase 3 que se usa para listar errores es
el de mayor AUC en prueba: es descriptivo, no una selección por validación como en la Fase 2.
Salidas: resultados/fase3_resultados.json, resultados/fase3_predicciones_prueba.csv,
resultados/figuras/fase3_roc_T*.png, resultados/fase3_errores.csv.
Uso: python scripts/07_fase3_evaluar.py (sin argumentos; --help muestra esta ayuda).
"""
import json, os, sys, time
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__.strip()); sys.exit(0)  # ayuda sin leer ni escribir nada
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from postura.nombres import nombre_modelo  # noqa: E402
from postura import evaluacion as ev  # noqa: E402

DATOS, RES = os.path.join(RAIZ, "datos"), os.path.join(RAIZ, "resultados")
FIG = os.path.join(RES, "figuras")
SEMILLA = 20260927
TAREAS = {
    "T1": "¿La mención se refiere a un partido?",
    "T2a": "Postura desfavorable frente al resto",
    "T2b": "Postura favorable frente al resto",
}


def cargar():
    """Une la muestra de anotación con es_partido y postura de datos/etiquetas.csv."""
    m = pd.read_csv(os.path.join(DATOS, "muestra_anotacion.csv"))
    et = pd.read_csv(os.path.join(DATOS, "etiquetas.csv"))
    return m.merge(et[["id", "es_partido", "postura"]], on="id")


def subconjunto(df, tarea):
    """Ítems de una tarea con su variable y: todos en T1 y solo los que son partido en T2a y T2b."""
    if tarea == "T1":
        d = df.copy(); d["y"] = d.es_partido.astype(int)
    else:
        d = df[df.es_partido == 1].copy()
        d["y"] = (d.postura == ("desfavorable" if tarea == "T2a" else "favorable")).astype(int)
    return d


def puntajes_letras(ruta):
    """Convierte las probabilidades de letra en puntajes T1 = 1 - X, T2a = D / (D + F + N) y T2b = F / (D + F + N); None si falta el archivo."""
    if not os.path.exists(ruta):
        return None
    q = pd.read_csv(ruta)
    s = q[["D", "F", "N"]].sum(axis=1).clip(lower=1e-9)
    return pd.DataFrame({"id": q.id, "T1": 1 - q.X, "T2a": q.D / s, "T2b": q.F / s})


def puntajes_cot(ruta):
    """Puntajes de las respuestas JSON válidas (es_partido para T1, probabilidad desfavorable y favorable para T2a y T2b) y conteo de válidas."""
    if not os.path.exists(ruta):
        return None, {}
    regs = [json.loads(l) for l in open(ruta, encoding="utf-8")]
    val = [r for r in regs if r.get("valido")]
    df = pd.DataFrame([{"id": r["id"],
                        "T1": 1.0 if r["es_partido"] else 0.0,
                        "T2a": float(r["probabilidad_desfavorable"]),
                        "T2b": float(r["probabilidad_favorable"]),
                        "postura_cot": r["postura"], "razonamiento": r["razonamiento"],
                        "evidencia": r["evidencia"]} for r in val])
    return df, {"items": len(regs), "json_validos": len(val),
                "pct_validos": round(100 * len(val) / max(1, len(regs)), 1)}


def modelos_embeddings(df, tarea):
    """Ajusta logística, SVM lineal y Random Forest sobre los embeddings con búsqueda agrupada por fecha, los reajusta con entrenamiento y validación y puntúa la prueba."""
    ruta = os.path.join(DATOS, "fase3_embeddings_qwen.npy")
    if not os.path.exists(ruta):
        return {}, {}
    X = np.load(ruta).astype(np.float32)
    ids = json.load(open(os.path.join(DATOS, "fase3_embeddings_ids.json")))
    pos = {i: k for k, i in enumerate(ids)}
    d = subconjunto(df, tarea)
    d = d[d.id.isin(pos)]
    Xd = X[[pos[i] for i in d.id]]
    ent = (d.particion == "entrenamiento").values
    val = (d.particion == "validacion").values
    pru = (d.particion == "prueba").values
    cv = GroupKFold(n_splits=5)
    grids = {
        "emb_logistica": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced")),
                          {"logisticregression__C": [0.001, 0.01, 0.1]}),
        "emb_svm": (make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", max_iter=20000)),
                    {"linearsvc__C": [0.0001, 0.001, 0.01]}),
        "emb_bosque": (RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", n_jobs=-1,
                                              random_state=SEMILLA), {"max_features": ["sqrt", 0.05]}),
    }
    salida, tiempos = {}, {}
    for nombre, (est, grid) in grids.items():
        t = time.perf_counter()
        gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        gs.fit(Xd[ent], d.y.values[ent], groups=d.fecha.values[ent])
        mejor = gs.best_estimator_
        auc_val = ev.roc_auc(d.y.values[val], _puntuar(mejor, Xd[val]))
        mejor.fit(Xd[ent | val], d.y.values[ent | val])
        salida[nombre] = pd.DataFrame({"id": d.id.values[pru], "s": _puntuar(mejor, Xd[pru])})
        tiempos[nombre] = {"segundos": round(time.perf_counter() - t, 1), "mejores_parametros": gs.best_params_,
                           "auc_validacion": round(float(auc_val), 4)}
    return salida, tiempos


def _puntuar(est, X):
    """Probabilidad de la clase positiva o, si el estimador no la da, su decision_function."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def fase2_mejores():
    """Lee las predicciones de prueba de la Fase 2 y el mejor modelo por tarea."""
    ruta_p = os.path.join(RES, "fase2_predicciones_prueba.csv")
    ruta_r = os.path.join(RES, "fase2_resultados.json")
    if not (os.path.exists(ruta_p) and os.path.exists(ruta_r)):
        return None, {}
    pred = pd.read_csv(ruta_p)
    res = json.load(open(ruta_r))
    mejores = {}
    for t in TAREAS:
        info = res.get("tareas", {}).get(t, {})
        mejores[t] = info.get("mejor_modelo") or info.get("mejor")
    return pred, mejores


def aviso(avisos, texto):
    """Añade un aviso a la lista y lo imprime en la salida de error."""
    avisos.append(texto)
    print("AVISO:", texto, file=sys.stderr, flush=True)


def main():
    """Evalúa la Fase 3 en prueba, la compara con la Fase 2 mediante DeLong y escribe resultados, predicciones, figuras y errores."""
    t0 = time.time()
    avisos = []
    if not os.path.exists(os.path.join(DATOS, "fase3_embeddings_qwen.npy")):
        aviso(avisos, "falta datos/fase3_embeddings_qwen.npy: se omiten emb_logistica, emb_svm y emb_bosque. "
                      "Se regenera con: python scripts/06_fase3_qwen.py embeddings")
    df = cargar()
    prueba = df[df.particion == "prueba"]
    cero = puntajes_letras(os.path.join(RES, "fase3_qwen_cero.csv"))
    pocos = puntajes_letras(os.path.join(RES, "fase3_qwen_pocos.csv"))
    cot, info_cot = puntajes_cot(os.path.join(RES, "fase3_qwen_cot.jsonl"))
    pred2, mejores2 = fase2_mejores()
    resultados = {"fecha": time.strftime("%Y-%m-%d"), "avisos": avisos, "tareas": {}, "cot": info_cot,
                  "tiempos_llm": json.load(open(os.path.join(RES, "fase3_qwen_tiempos.json")))
                  if os.path.exists(os.path.join(RES, "fase3_qwen_tiempos.json")) else {},
                  "mejor_fase2": mejores2}
    filas_pred, errores = [], []
    for tarea in TAREAS:
        d = subconjunto(prueba, tarea)
        base = d[["id", "fecha", "y", "grupo", "presidente", "oracion_marcada"]].copy()
        fuentes = {}
        if cero is not None:
            fuentes["qwen_cero"] = cero[["id", tarea]].rename(columns={tarea: "s"})
        if pocos is not None:
            fuentes["qwen_pocos"] = pocos[["id", tarea]].rename(columns={tarea: "s"})
        if cot is not None and len(cot):
            fuentes["qwen_cot"] = cot[["id", tarea]].rename(columns={tarea: "s"})
        emb, t_emb = modelos_embeddings(df, tarea)
        fuentes.update(emb)
        ruta_bert = os.path.join(RES, f"fase3_bert_{tarea}.csv")
        if os.path.exists(ruta_bert):
            fuentes["bert_ajustado"] = pd.read_csv(ruta_bert)[["id", "s"]]
        for nombre in [n for n, f in fuentes.items() if not np.isfinite(f.s.values.astype(float)).all()]:
            aviso(avisos, f"{tarea}: {nombre} tiene puntajes no finitos (NaN) y se omite")
            del fuentes[nombre]
        modelos = {}
        for nombre, f in fuentes.items():
            x = base.merge(f, on="id")
            if x.y.nunique() < 2:
                continue
            bs = ev.bootstrap_agrupado(x.y.values, x.s.values, x.fecha.values, n_replicas=1000)
            modelos[nombre] = {"auc": bs["estimacion"], "ic_inferior": bs["ic_inferior"], "ic_superior": bs["ic_superior"],
                               "pr_auc": ev.pr_auc(x.y.values, x.s.values), "n": int(len(x)), "positivos": int(x.y.sum())}
            for r in x.itertuples():
                filas_pred.append({"id": r.id, "tarea": tarea, "modelo": nombre, "y_real": r.y, "puntaje": r.s})
        # Comparación con el mejor de la Fase 2 en los mismos ítems
        comp = {}
        if pred2 is not None and mejores2.get(tarea):
            p2 = pred2[pred2.tarea == tarea] if "tarea" in pred2.columns else pred2
            col = mejores2[tarea]
            if col not in p2.columns:
                cands = [c for c in p2.columns if col in c]
                col = cands[0] if cands else None
            if col:
                p2 = p2[["id", col]].rename(columns={col: "s2"})
                for nombre, f in fuentes.items():
                    x = base.merge(f, on="id").merge(p2, on="id")
                    if x.y.nunique() < 2 or len(x) < 20:
                        continue
                    dl = ev.delong(x.y.values, x.s.values, x.s2.values)
                    comp[nombre] = {"n": int(len(x)), "auc_fase3": dl["auc_1"], "auc_fase2": dl["auc_2"],
                                    "diferencia": dl["diferencia"], "p_delong": dl["p_valor"]}
        resultados["tareas"][tarea] = {"descripcion": TAREAS[tarea], "n_prueba": int(len(d)),
                                       "positivos_prueba": int(d.y.sum()), "modelos": modelos,
                                       "comparacion_con_mejor_fase2": comp, "embeddings": t_emb}
        # Curvas ROC
        curvas = {}
        for nombre, f in fuentes.items():
            x = base.merge(f, on="id")
            if x.y.nunique() == 2 and nombre != "qwen_cot":
                curvas[nombre_modelo(nombre)] = ev.curva_roc(x.y.values, x.s.values)
        if pred2 is not None and mejores2.get(tarea) and "p2" in dir():
            pass
        try:
            ev.graficar_roc(curvas, os.path.join(FIG, f"fase3_roc_{tarea}.png"), titulo=f"Fase 3, {tarea}. {TAREAS[tarea]} (prueba, n = {len(d)})")
        except Exception as e:  # noqa: BLE001
            print("no se pudo graficar", tarea, e)
        # Errores del modelo de la Fase 3 con mayor AUC en prueba (para el análisis cualitativo).
        # Es descriptivo: no es una selección por validación como en la Fase 2, porque cero, pocos
        # y cot solo se puntuaron en prueba.
        if modelos:
            mejor = max(modelos, key=lambda k: modelos[k]["auc"] if k != "qwen_cot" else -1)
            x = base.merge(fuentes[mejor], on="id")
            x["error"] = np.abs(x.y - x.s)
            for r in x.sort_values("error", ascending=False).head(15).itertuples():
                errores.append({"tarea": tarea, "modelo": mejor, "id": r.id, "grupo": r.grupo, "presidente": r.presidente,
                                "y_real": r.y, "puntaje": round(float(r.s), 3), "oracion_marcada": r.oracion_marcada})
    pd.DataFrame(filas_pred).to_csv(os.path.join(RES, "fase3_predicciones_prueba.csv"), index=False)
    pd.DataFrame(errores).to_csv(os.path.join(RES, "fase3_errores.csv"), index=False)
    resultados["segundos"] = round(time.time() - t0, 1)
    json.dump(resultados, open(os.path.join(RES, "fase3_resultados.json"), "w"), ensure_ascii=False, indent=1, default=float)
    for t, info in resultados["tareas"].items():
        print(t, {k: round(v["auc"], 3) for k, v in info["modelos"].items()})


if __name__ == "__main__":
    main()
