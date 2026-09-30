# -*- coding: utf-8 -*-
"""Consolida las tres anotaciones independientes de cada ítem en una etiqueta de referencia.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

Entrada: datos/anotacion/anotador_{A,B,C}_lote_NN.json y, si existe, adjudicacion.json.
Regla: etiqueta combinada = no_aplica si es_partido = 0; si no, la postura. Gana la mayoría
(2 o 3 de 3); sin mayoría, decide la adjudicación. Salidas: datos/etiquetas.csv y
datos/etiquetas_resumen.json con acuerdo entre anotadores (kappa de Fleiss y alfa de
Krippendorff nominal) y distribución de etiquetas por grupo y presidente.
Uso: python scripts/02_consolidar_etiquetas.py (sin argumentos; --help muestra esta ayuda).
"""
import collections, glob, json, os, sys, time
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__.strip()); sys.exit(0)  # ayuda sin leer ni escribir nada
import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATOS = os.path.join(RAIZ, "datos")
ANOT = os.path.join(DATOS, "anotacion")


def etiqueta(v):
    """Etiqueta de cuatro clases de una anotación: no_aplica si es_partido = 0; si no, la postura."""
    return "no_aplica" if int(v["es_partido"]) == 0 else v["postura"]


def fleiss_kappa(tabla):
    """tabla: matriz ítems x categorías con el número de anotadores que eligió cada una."""
    tabla = np.asarray(tabla, dtype=float)
    n = tabla.sum(axis=1)[0]
    p_j = tabla.sum(axis=0) / tabla.sum()
    P_i = ((tabla ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e))


def krippendorff_nominal(unidades):
    """unidades: lista de listas de valores (uno por anotador presente). Alfa nominal."""
    coinc = collections.Counter(); total = 0
    for vals in unidades:
        m = len(vals)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a != b:
                    coinc[(vals[a], vals[b])] += 1 / (m - 1)
        total += m
    cats = sorted({c for par in coinc for c in par})
    n_c = {c: sum(v for (x, _), v in coinc.items() if x == c) for c in cats}
    n = sum(n_c.values())
    D_o = sum(v for (x, y), v in coinc.items() if x != y) / n
    D_e = sum(n_c[x] * n_c[y] for x in cats for y in cats if x != y) / (n * (n - 1))
    return float(1 - D_o / D_e)


def main():
    """Reúne las anotaciones de cada ítem, vota por mayoría o adjudica, calcula el acuerdo y escribe datos/etiquetas.csv y datos/etiquetas_resumen.json."""
    votos = collections.defaultdict(dict)
    archivos = sorted(glob.glob(os.path.join(ANOT, "anotador_*_lote_*.json")))
    for f in archivos:
        anot = os.path.basename(f).split("_")[1]
        for it in json.load(open(f, encoding="utf-8"))["items"]:
            votos[it["id"]][anot] = it
    adj = {}
    ruta_adj = os.path.join(ANOT, "adjudicacion.json")
    if os.path.exists(ruta_adj):
        adj = {it["id"]: it for it in json.load(open(ruta_adj, encoding="utf-8"))["items"]}
    muestra = pd.read_csv(os.path.join(DATOS, "muestra_anotacion.csv"))
    filas, sin_votos, cat4, bin2, unidades4, unidades2 = [], [], [], [], [], []
    CATS = ["no_aplica", "desfavorable", "favorable", "neutral"]
    for id_ in muestra.id:
        vs = list(votos.get(id_, {}).values())
        if not vs:
            sin_votos.append(id_); continue
        etqs = [etiqueta(v) for v in vs]
        cuenta = collections.Counter(etqs)
        top, n_top = cuenta.most_common(1)[0]
        adjudicado = 0
        if n_top < 2:
            if id_ in adj:
                top = etiqueta(adj[id_]); adjudicado = 1
            else:
                top = "neutral" if sum(e != "no_aplica" for e in etqs) >= 2 else "no_aplica"; adjudicado = 1
        es_p = [int(v["es_partido"]) for v in vs]
        acuerdo_ep = max(es_p.count(0), es_p.count(1))
        filas.append({"id": id_, "es_partido": int(top != "no_aplica"), "postura": top,
                      "acuerdo_es_partido": acuerdo_ep, "acuerdo_postura": n_top, "adjudicado": adjudicado,
                      "n_anotadores": len(vs), "confianza_media": round(float(np.mean([v["confianza"] for v in vs])), 2),
                      "evidencia": vs[0].get("evidencia", "")})
        if len(vs) == 3:
            cat4.append([etqs.count(c) for c in CATS]); bin2.append([es_p.count(0), es_p.count(1)])
        unidades4.append(etqs); unidades2.append(es_p)
    et = pd.DataFrame(filas)
    et.to_csv(os.path.join(DATOS, "etiquetas.csv"), index=False)
    m = muestra.merge(et, on="id")
    pos = m[m.es_partido == 1]
    resumen = {
        "fecha": time.strftime("%Y-%m-%d"), "archivos_anotacion": len(archivos), "items_muestra": len(muestra),
        "items_etiquetados": len(et), "items_sin_votos": len(sin_votos), "ids_sin_votos": sin_votos[:50],
        "items_con_tres_anotaciones": len(cat4),
        "acuerdo": {
            "fleiss_kappa_es_partido": round(fleiss_kappa(bin2), 3) if bin2 else None,
            "fleiss_kappa_etiqueta_4": round(fleiss_kappa(cat4), 3) if cat4 else None,
            "krippendorff_alfa_es_partido": round(krippendorff_nominal(unidades2), 3),
            "krippendorff_alfa_etiqueta_4": round(krippendorff_nominal(unidades4), 3),
            "unanimidad_etiqueta_4_pct": round(100 * float((et.acuerdo_postura == 3).mean()), 1),
            "sin_mayoria_adjudicados": int(et.adjudicado.sum()),
        },
        "distribucion_etiqueta": et.postura.value_counts().to_dict(),
        "es_partido_por_grupo": m.groupby("grupo").es_partido.mean().round(3).to_dict(),
        "postura_por_objetivo_y_presidente": pos.groupby(["objetivo", "presidente"]).postura.value_counts()
            .unstack(fill_value=0).reset_index().to_dict("records"),
        "por_particion": m.groupby("particion").agg(n=("id", "size"), partido=("es_partido", "sum"),
            desfavorable=("postura", lambda s: int((s == "desfavorable").sum())),
            favorable=("postura", lambda s: int((s == "favorable").sum()))).to_dict("index"),
    }
    json.dump(resumen, open(os.path.join(DATOS, "etiquetas_resumen.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: resumen[k] for k in ("archivos_anotacion", "items_etiquetados", "items_sin_votos", "acuerdo", "distribucion_etiqueta")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
