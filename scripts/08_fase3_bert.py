# -*- coding: utf-8 -*-
"""Fase 3, ajuste fino completo de BETO (BERT preentrenado en español).

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-29. Licencia: MIT.

Ajusta todos los parámetros de BETO (dccuchile/bert-base-spanish-wwm-cased; Cañete et al., 2020),
que sustituye al bert-base-uncased en inglés de la versión 1.0, para cada tarea
binaria, con tasa de aprendizaje 2e-5, lote 16, hasta 4 épocas y selección de la época por
ROC-AUC en validación.
Entrada: la oración con la mención marcada entre [[ ]]. Salidas: resultados/fase3_bert_T*.csv
(id, s) con el puntaje de prueba de la mejor época y resultados/fase3_bert_tiempos.json con el
estado, la mejor época y el AUC de validación por época de cada tarea.
Si en una tarea la pérdida o la salida dejan de ser finitas (NaN), el script no escribe su CSV,
anota el estado en el JSON (por ejemplo "salida no finita (NaN) en la época N") y sigue con la
siguiente tarea; el JSON se escribe tras cada tarea. El script 07 solo reporta BERT en las tareas
que tienen CSV.
En MPS el modelo se carga con attn_implementation='eager', porque la atención rápida no admite
dropout durante el entrenamiento.
Uso: python scripts/08_fase3_bert.py (sin argumentos; --help muestra esta ayuda).
"""
import json, os, sys, time
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__.strip()); sys.exit(0)  # ayuda sin cargar PyTorch ni escribir nada
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATOS, RES = os.path.join(RAIZ, "datos"), os.path.join(RAIZ, "resultados")
MODELO, SEMILLA, LR, LOTE, EPOCAS, MAXLEN = os.environ.get("MODELO_BERT", "dccuchile/bert-base-spanish-wwm-cased"), 20260927, 2e-5, 16, 4, 128
dispositivo = "mps" if torch.backends.mps.is_available() else "cpu"


def datos_tarea(tarea):
    """Une la muestra con las etiquetas y devuelve los ítems de la tarea con su variable y."""
    m = pd.read_csv(os.path.join(DATOS, "muestra_anotacion.csv")).merge(
        pd.read_csv(os.path.join(DATOS, "etiquetas.csv"))[["id", "es_partido", "postura"]], on="id")
    if tarea == "T1":
        m["y"] = m.es_partido
    else:
        m = m[m.es_partido == 1].copy()
        m["y"] = (m.postura == ("desfavorable" if tarea == "T2a" else "favorable")).astype(int)
    return m


def lotes(tok, textos, ys, barajar):
    """Genera lotes tokenizados de LOTE textos, truncados a MAXLEN tokens y barajados con semilla fija si se pide."""
    idx = np.arange(len(textos))
    if barajar:
        np.random.default_rng(SEMILLA).shuffle(idx)
    for k in range(0, len(idx), LOTE):
        j = idx[k:k + LOTE]
        enc = tok([textos[i] for i in j], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        yield {k2: v.to(dispositivo) for k2, v in enc.items()}, torch.tensor([ys[i] for i in j]).to(dispositivo)


@torch.no_grad()
def puntuar(modelo, tok, textos):
    """Probabilidad de la clase positiva de cada texto, en modo de evaluación y sin gradientes."""
    modelo.eval(); out = []
    for enc, _ in lotes(tok, textos, [0] * len(textos), False):
        out.append(torch.softmax(modelo(**enc).logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def entrenar_tarea(tarea):
    """Ajusta el modelo para una tarea. Devuelve (puntajes de prueba o None, registro para el JSON).
    Los puntajes son None cuando la pérdida o la salida dejaron de ser finitas."""
    t0 = time.time()
    d = datos_tarea(tarea)
    ent, val, pru = (d[d.particion == p] for p in ("entrenamiento", "validacion", "prueba"))
    tok = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForSequenceClassification.from_pretrained(MODELO, num_labels=2, attn_implementation="eager").to(dispositivo)
    opt = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=0.01)
    pesos = torch.tensor([1.0, float((ent.y == 0).sum() / max(1, (ent.y == 1).sum()))], device=dispositivo)
    perdida = torch.nn.CrossEntropyLoss(weight=pesos.float())
    mejor, historial, estado = (-1.0, None, 0), [], "ok"
    for ep in range(EPOCAS):
        modelo.train()
        for enc, y in lotes(tok, ent.oracion_marcada.tolist(), ent.y.tolist(), True):
            opt.zero_grad()
            loss = perdida(modelo(**enc).logits.float(), y)
            if not torch.isfinite(loss):
                estado = f"pérdida no finita (NaN) en la época {ep + 1}"
                break
            loss.backward(); opt.step()
        if estado != "ok":
            break
        s_val = puntuar(modelo, tok, val.oracion_marcada.tolist())
        if not np.isfinite(s_val).all():
            estado = f"salida no finita (NaN) en la época {ep + 1}"
            break
        auc_val = float(roc_auc_score(val.y, s_val))
        historial.append(round(auc_val, 4))
        print(f"{tarea} época {ep + 1}: AUC validación {auc_val:.3f}", flush=True)
        if auc_val > mejor[0]:
            mejor = (auc_val, puntuar(modelo, tok, pru.oracion_marcada.tolist()), ep + 1)
    if estado == "ok" and (mejor[1] is None or not np.isfinite(mejor[1]).all()):
        estado = "salida no finita (NaN) en los puntajes de prueba"
    if estado != "ok":
        print(f"{tarea}: {estado}; no se escribe fase3_bert_{tarea}.csv", flush=True)
    registro = {"estado": estado, "segundos": round(time.time() - t0, 1),
                "mejor_epoca": mejor[2] if estado == "ok" else None,
                "auc_validacion": round(mejor[0], 4) if estado == "ok" else None,
                "auc_validacion_por_epoca": historial, "n_entrenamiento": int(len(ent)),
                "dispositivo": dispositivo, "modelo": MODELO}
    puntajes = pd.DataFrame({"id": pru.id, "s": mejor[1]}) if estado == "ok" else None
    del modelo
    if dispositivo == "mps":
        torch.mps.empty_cache()
    return puntajes, registro


def main():
    """Ajusta BETO en T1, T2a y T2b y escribe los puntajes de prueba de cada tarea sin NaN y el registro en resultados/fase3_bert_tiempos.json."""
    torch.manual_seed(SEMILLA)
    os.makedirs(RES, exist_ok=True)
    tiempos = {}
    for tarea in ["T1", "T2a", "T2b"]:
        puntajes, tiempos[tarea] = entrenar_tarea(tarea)
        if puntajes is not None:
            puntajes.to_csv(os.path.join(RES, f"fase3_bert_{tarea}.csv"), index=False)
        # El JSON se escribe tras cada tarea para conservar el registro aunque una posterior falle.
        json.dump(tiempos, open(os.path.join(RES, "fase3_bert_tiempos.json"), "w"), indent=1)
    print(json.dumps(tiempos, indent=1))
    return tiempos


if __name__ == "__main__":
    main()
