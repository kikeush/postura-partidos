# -*- coding: utf-8 -*-
"""Fase 3 con el modelo de lenguaje grande local (Qwen3-32B, 4 bits, MLX).

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

Modos (se pueden encadenar en una sola corrida para cargar el modelo una vez; sin argumentos
corren los cuatro en este orden):
  cero         clasificación sin ejemplos (probabilidad de la primera letra X, D, F, N)
               sobre validación y prueba -> resultados/fase3_qwen_cero.csv
  embeddings   vector medio del último estado oculto de la oración marcada, para toda la
               muestra -> datos/fase3_embeddings_qwen.npy y datos/fase3_embeddings_ids.json
  pocos        clasificación con ocho ejemplos resueltos del conjunto de entrenamiento
               (requiere datos/etiquetas.csv) -> resultados/fase3_qwen_pocos.csv
  cot          razonamiento en cadena con salida JSON validada con Pydantic sobre un
               subconjunto estratificado de prueba -> resultados/fase3_qwen_cot.jsonl
Uso: python scripts/06_fase3_qwen.py cero embeddings pocos cot
Un modo desconocido o --help muestra esta ayuda y termina sin cargar el modelo.
"""
import json, os, re, sys, time
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__.strip()); sys.exit(0)  # ayuda sin cargar el modelo ni escribir nada
from typing import Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
from qwen_local import Qwen, INSTRUCCIONES, item_texto  # noqa: E402

DATOS, RES = os.path.join(RAIZ, "datos"), os.path.join(RAIZ, "resultados")
SEMILLA = 20260927
LETRA = {"no_aplica": "X", "desfavorable": "D", "favorable": "F", "neutral": "N"}


class Juicio(BaseModel):
    """Esquema Pydantic de la respuesta JSON del razonamiento en cadena."""
    razonamiento: str
    es_partido: bool
    postura: Literal["desfavorable", "favorable", "neutral", "no_aplica"]
    probabilidad_desfavorable: float = Field(ge=0, le=1)
    probabilidad_favorable: float = Field(ge=0, le=1)
    evidencia: str


PROMPT_COT = (
    INSTRUCCIONES.split("Responde con UNA sola letra")[0]
    + "Piensa paso a paso y responde SOLO con un objeto JSON con estas claves, en este orden: "
    "razonamiento (máximo 40 palabras: qué es la cadena marcada, a quién se dirige la valoración y por qué), "
    "es_partido (true o false), postura (desfavorable, favorable, neutral o no_aplica si no es partido), "
    "probabilidad_desfavorable (0 a 1), probabilidad_favorable (0 a 1), "
    "evidencia (hasta 12 palabras copiadas del texto). La postura se juzga hacia el partido, "
    "no hacia personas ni políticas, con el mismo criterio para todos los partidos."
)


def muestra():
    """Lee la muestra de anotación de datos/muestra_anotacion.jsonl como lista de diccionarios."""
    return [json.loads(l) for l in open(os.path.join(DATOS, "muestra_anotacion.jsonl"), encoding="utf-8")]


def log(msg):
    """Imprime un mensaje precedido de la hora."""
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def modo_cero(q, items):
    """Puntúa sin ejemplos los ítems de validación y prueba, en lotes de 25, y escribe resultados/fase3_qwen_cero.csv."""
    sel = [it for it in items if it["particion"] in ("validacion", "prueba")]
    t = time.time(); salida = []
    for k in range(0, len(sel), 25):
        salida += q.puntuar(sel[k:k + 25])
        log(f"cero: {min(k + 25, len(sel))}/{len(sel)} ({(time.time()-t)/min(k+25,len(sel)):.2f} s por ítem)")
    df = pd.DataFrame([{"id": it["id"], "particion": it["particion"], **p} for it, p in zip(sel, salida)])
    df.to_csv(os.path.join(RES, "fase3_qwen_cero.csv"), index=False)
    return {"items": len(sel), "segundos": round(time.time() - t, 1)}


def modo_embeddings(q, items):
    """Calcula el embedding de la oración marcada de cada ítem (hasta 160 tokens) y lo guarda en datos/ en float16, con sus ids."""
    t = time.time(); vecs = []
    for k, it in enumerate(items):
        vecs.append(q.embedding(it["oracion_marcada"], max_tokens=160).astype(np.float16))
        if (k + 1) % 100 == 0:
            log(f"embeddings: {k+1}/{len(items)} ({(time.time()-t)/(k+1):.2f} s por ítem)")
    np.save(os.path.join(DATOS, "fase3_embeddings_qwen.npy"), np.stack(vecs))
    json.dump([it["id"] for it in items], open(os.path.join(DATOS, "fase3_embeddings_ids.json"), "w"))
    return {"items": len(items), "dim": int(vecs[0].shape[0]), "segundos": round(time.time() - t, 1)}


def ejemplos_pocos(items):
    """Elige al azar, con semilla fija, un ejemplo de entrenamiento con acuerdo unánime por cada letra y presidente."""
    et = pd.read_csv(os.path.join(DATOS, "etiquetas.csv"))
    df = pd.DataFrame(items).merge(et, on="id")
    df = df[(df.particion == "entrenamiento") & (df.acuerdo_postura == 3)]
    df["letra"] = df.apply(lambda r: "X" if r.es_partido == 0 else LETRA[r.postura], axis=1)
    elegidos = []
    for letra in ["X", "D", "F", "N"]:
        sub = df[df.letra == letra]
        for pres in ["AMLO", "Sheinbaum"]:
            s2 = sub[sub.presidente == pres]
            if len(s2):
                elegidos.append(s2.sample(1, random_state=SEMILLA).iloc[0].to_dict())
    return elegidos


def modo_pocos(q, items):
    """Puntúa la prueba con los ejemplos elegidos, que se excluyen de ella, y escribe fase3_qwen_pocos.csv y fase3_qwen_pocos_ejemplos.json en resultados/."""
    ej = ejemplos_pocos(items)
    json.dump([{k: e[k] for k in ("id", "letra", "oracion_marcada")} for e in ej],
              open(os.path.join(RES, "fase3_qwen_pocos_ejemplos.json"), "w"), ensure_ascii=False, indent=1)
    ids_ej = {e["id"] for e in ej}
    sel = [it for it in items if it["particion"] == "prueba" and it["id"] not in ids_ej]
    t = time.time(); salida = []
    for k in range(0, len(sel), 25):
        salida += q.puntuar(sel[k:k + 25], ejemplos=ej)
        log(f"pocos: {min(k + 25, len(sel))}/{len(sel)}")
    df = pd.DataFrame([{"id": it["id"], "particion": it["particion"], **p} for it, p in zip(sel, salida)])
    df.to_csv(os.path.join(RES, "fase3_qwen_pocos.csv"), index=False)
    return {"items": len(sel), "ejemplos": len(ej), "segundos": round(time.time() - t, 1)}


def modo_cot(q, items, n=150):
    """Clasifica con razonamiento en cadena y JSON validado hasta n ítems de prueba estratificados por grupo, reintenta una vez cada salida inválida y escribe resultados/fase3_qwen_cot.jsonl."""
    from mlx_lm import generate
    prueba = pd.DataFrame([it for it in items if it["particion"] == "prueba"])
    sel = (prueba.groupby("grupo", group_keys=False)
           .apply(lambda g: g.sample(min(len(g), max(4, round(n * len(g) / len(prueba)))), random_state=SEMILLA)))
    sel = sel.sample(min(n, len(sel)), random_state=SEMILLA).to_dict("records")
    ruta = os.path.join(RES, "fase3_qwen_cot.jsonl")
    t = time.time(); validos = 0
    with open(ruta, "w", encoding="utf-8") as fh:
        for k, it in enumerate(sel):
            prompt = q._chat(PROMPT_COT, item_texto(it) + "\nJSON:")
            reg = {"id": it["id"]}
            for intento in range(2):
                txt = generate(q.model, q.tok, prompt=prompt, max_tokens=260, verbose=False)
                m = re.search(r"\{.*\}", txt, re.S)
                try:
                    j = Juicio.model_validate(json.loads(m.group(0))) if m else None
                    if j:
                        reg.update(j.model_dump()); reg["valido"] = True; validos += 1
                        break
                except (json.JSONDecodeError, ValidationError) as e:
                    reg["error"] = str(e)[:200]
            reg.setdefault("valido", False); reg["salida_cruda"] = txt[:600]
            fh.write(json.dumps(reg, ensure_ascii=False) + "\n"); fh.flush()
            if (k + 1) % 10 == 0:
                log(f"cot: {k+1}/{len(sel)} ({(time.time()-t)/(k+1):.1f} s por ítem, válidos {validos})")
    return {"items": len(sel), "json_validos": validos, "segundos": round(time.time() - t, 1)}


MODOS = {"cero": modo_cero, "embeddings": modo_embeddings, "pocos": modo_pocos, "cot": modo_cot}


if __name__ == "__main__":
    modos = sys.argv[1:] or list(MODOS)
    desconocidos = [m for m in modos if m not in MODOS]
    if desconocidos:  # se valida antes de cargar el modelo
        print(__doc__.strip())
        print(f"\nModo desconocido: {', '.join(desconocidos)}. Modos válidos: {', '.join(MODOS)}.", file=sys.stderr)
        sys.exit(2)
    q = Qwen(); items = muestra(); resumen_ruta = os.path.join(RES, "fase3_qwen_tiempos.json")
    resumen = json.load(open(resumen_ruta)) if os.path.exists(resumen_ruta) else {}
    for m in modos:
        log(f"inicio modo {m}")
        resumen[m] = MODOS[m](q, items)
        json.dump(resumen, open(resumen_ruta, "w"), indent=1)
        log(f"fin modo {m}: {resumen[m]}")
