# -*- coding: utf-8 -*-
"""Modelo de lenguaje grande local (Qwen3-32B cuantizado a 4 bits, MLX) para la Fase 3:
representaciones vectoriales (embeddings) y clasificación sin ejemplos y con pocos ejemplos.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

La clasificación usa la probabilidad que el modelo asigna a la primera letra de la respuesta
(X: no es partido; D: desfavorable; F: favorable; N: neutral), lo que da puntajes continuos
para ROC-AUC con una sola pasada por ítem. El prefijo fijo (instrucciones y ejemplos) se
procesa una vez y su caché se reutiliza, recortándola después de cada ítem.
Uso: python scripts/qwen_local.py (prueba rápida: carga el modelo, calcula un embedding y puntúa
cuatro ítems de la muestra sin escribir nada; --help muestra esta ayuda).
"""
import json, os, sys, time
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__.strip()); sys.exit(0)  # ayuda sin leer ni escribir nada
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODELO = "mlx-community/Qwen3-32B-4bit"
LETRAS = ["X", "D", "F", "N"]

INSTRUCCIONES = (
    "Eres un anotador experto en discurso político mexicano. Recibirás una oración dicha por el "
    "titular del Ejecutivo en una conferencia matutina, con una cadena marcada entre [[ ]] y su contexto. "
    "Decide qué es la cadena marcada y, si es un partido, qué postura expresa el titular hacia ese partido.\n"
    "Responde con UNA sola letra:\n"
    "X = la cadena NO se refiere a un partido político, coalición o bloque de partidos (por ejemplo pan como "
    "alimento, verde como color, conservadores del siglo XIX, oposición en sentido genérico).\n"
    "D = sí es un partido o bloque y la postura del titular hacia él es desfavorable (lo descalifica, le atribuye "
    "corrupción, fracaso, mentira o malas intenciones, lo ridiculiza o lo presenta como adversario).\n"
    "F = sí es un partido o bloque y la postura es favorable (lo elogia, lo defiende, le atribuye logros).\n"
    "N = sí es un partido o bloque y la mención es neutral o informativa (votaciones, candidaturas, encuestas, "
    "historia sin juicio, deslinde).\n"
    "La postura se juzga hacia el partido, no hacia personas ni políticas. Aplica el mismo criterio a todos los partidos."
)


def item_texto(it):
    """Texto del ítem para el prompt: pregunta de la prensa y contexto previo si existen, y la oración marcada."""
    partes = []
    if it.get("pregunta_prensa"):
        partes.append(f"Pregunta de la prensa: {it['pregunta_prensa']}")
    if it.get("contexto_previo"):
        partes.append(f"Contexto previo del titular: {it['contexto_previo']}")
    partes.append(f"Oración: {it['oracion_marcada']}")
    return "\n".join(partes)


class Qwen:
    """Qwen3-32B de 4 bits sobre MLX: embeddings y probabilidades de la primera letra de la respuesta."""
    def __init__(self):
        """Carga el modelo y el tokenizador y guarda los identificadores de las letras X, D, F y N."""
        t = time.time()
        self.model, self.tok = load(MODELO)
        self.ids_letras = [self.tok.encode(l, add_special_tokens=False)[0] for l in LETRAS]
        print(f"modelo cargado en {time.time()-t:.1f} s; ids letras {self.ids_letras}", flush=True)

    def _chat(self, sistema, usuario, prefijo_respuesta=""):
        """Arma el prompt con la plantilla de chat, sin modo de pensamiento, y le añade el prefijo de respuesta."""
        msgs = [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}]
        txt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return txt + prefijo_respuesta

    def embedding(self, texto, max_tokens=256):
        """Promedio del último estado oculto, ya normalizado, sobre los primeros max_tokens tokens del texto."""
        ids = self.tok.encode(texto)[:max_tokens]
        h = self.model.model(mx.array(ids)[None])  # (1, T, d), ya normalizado
        v = mx.mean(h[0].astype(mx.float32), axis=0)
        mx.eval(v)
        return np.array(v)

    def puntuar(self, items, ejemplos=None):
        """Devuelve para cada ítem las probabilidades normalizadas de X, D, F y N."""
        sistema = INSTRUCCIONES
        if ejemplos:
            sistema += "\n\nEjemplos resueltos:\n" + "\n\n".join(
                f"{item_texto(e)}\nRespuesta: {e['letra']}" for e in ejemplos)
        marcador = "<<ITEM>>"
        plantilla = self._chat(sistema, marcador + "\nRespuesta (una letra):")
        pre, post = plantilla.split(marcador)
        ids_pre = self.tok.encode(pre)
        cache = make_prompt_cache(self.model)
        self.model(mx.array(ids_pre)[None], cache=cache)
        mx.eval([c.state for c in cache])
        salida = []
        for it in items:
            ids = self.tok.encode(item_texto(it) + post, add_special_tokens=False)
            logits = self.model(mx.array(ids)[None], cache=cache)[0, -1]
            sel = logits[mx.array(self.ids_letras)].astype(mx.float32)
            p = mx.softmax(sel)
            mx.eval(p)
            for c in cache:
                c.trim(len(ids))
            salida.append(dict(zip(LETRAS, [float(x) for x in np.array(p)])))
        return salida


if __name__ == "__main__":
    q = Qwen()
    datos = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datos")
    items = [json.loads(l) for l in open(os.path.join(datos, "muestra_anotacion.jsonl"), encoding="utf-8")][:4]
    t = time.time(); e = q.embedding(items[0]["oracion_marcada"]); print("embedding", e.shape, f"{time.time()-t:.2f} s")
    t = time.time(); r = q.puntuar(items); print(f"4 ítems en {time.time()-t:.1f} s")
    for it, p in zip(items, r):
        print(it["grupo"], it["mencion"], {k: round(v, 3) for k, v in p.items()}, "|", it["oracion_marcada"][:110])
