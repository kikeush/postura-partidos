# -*- coding: utf-8 -*-
"""Construye la tabla de menciones candidatas a partidos en el habla en vivo del titular y la
muestra estratificada para anotación, con partición por conferencia.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

Entrada: corpus de versiones estenográficas (manifest.csv y un CSV por conferencia).
Salidas en datos/: menciones.parquet (todas), muestra_anotacion.jsonl y .csv (muestra),
resumen_menciones.json. Uso: python scripts/01_construir_menciones.py (sin argumentos; --help
muestra esta ayuda). La partición (entrenamiento, validación, prueba) se asigna por
conferencia con un hash estable de la fecha, así ninguna conferencia cae en dos particiones.
"""
import hashlib, json, os, sys, time
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__.strip()); sys.exit(0)  # ayuda sin leer ni escribir nada
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from postura.menciones import oraciones, buscar_menciones, marcar, OBJETIVO  # noqa: E402

# Carpeta del corpus: variable CORPUS_MANANERAS o, por omisión, la ubicación relativa a este
# repositorio en la máquina del autor (tres niveles arriba del proyecto).
CORPUS = os.environ.get("CORPUS_MANANERAS") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(RAIZ))), "0000 INVESTIGACIÓN", "00Articulos",
    "06_Datos", "Mananeras")
DATOS = os.path.join(RAIZ, "datos")
SEMILLA = 20260927


def particion(fecha):
    """Asigna la partición de una conferencia por el hash SHA-256 de su fecha módulo 100: 0 a 59 entrenamiento, 60 a 79 validación y 80 a 99 prueba."""
    h = int(hashlib.sha256(fecha.encode()).hexdigest(), 16) % 100
    return "entrenamiento" if h < 60 else ("validacion" if h < 80 else "prueba")


def palabras_finales(texto, n):
    """Devuelve las últimas n palabras de un texto, separadas por espacios."""
    t = str(texto).split()
    return " ".join(t[-n:])


def main():
    """Lee el manifiesto del corpus, extrae las menciones del habla en vivo del titular, arma la muestra estratificada y escribe las salidas en datos/."""
    t0 = time.time()
    idx = pd.read_csv(os.path.join(CORPUS, "manifest.csv"), keep_default_na=False)
    filas = []
    for _, r in idx.iterrows():
        df = pd.read_csv(os.path.join(CORPUS, r["path"]), usecols=["Rol", "Texto", "Turno", "En_video"],
                         dtype=str, keep_default_na=False)
        df = df[df["Rol"] != "acotacion"].reset_index(drop=True)
        ultima_pregunta = ""
        previo_turno, turno_actual = [], None
        for i, row in df.iterrows():
            rol, texto = row["Rol"], row["Texto"]
            if rol == "prensa":
                ultima_pregunta = texto
                previo_turno, turno_actual = [], None
                continue
            if rol not in ("presidente", "presidenta") or row["En_video"] == "inferido":
                if rol not in ("presidente", "presidenta"):
                    previo_turno, turno_actual = [], None
                continue
            if row["Turno"] != turno_actual:
                previo_turno, turno_actual = [], row["Turno"]
            for j, o in enumerate(oraciones(texto)):
                for (ini, fin, cadena, grupo) in buscar_menciones(o):
                    filas.append({
                        "id": f"{r['date']}_{i}_{j}_{ini}", "fecha": r["date"], "anio": int(r["year"]),
                        "presidente": r["presidente"], "grupo": grupo, "objetivo": OBJETIVO[grupo],
                        "mencion": cadena, "oracion": o, "oracion_marcada": marcar(o, ini, fin),
                        "inicio": ini, "fin": fin,
                        "contexto_previo": palabras_finales(" ".join(previo_turno), 120),
                        "pregunta_prensa": palabras_finales(ultima_pregunta, 60),
                        "palabras_oracion": len(o.split()), "particion": particion(r["date"]),
                    })
                previo_turno.append(o)
    men = pd.DataFrame(filas)
    men.to_parquet(os.path.join(DATOS, "menciones.parquet"), index=False)

    # Muestra estratificada por grupo y presidente
    cuotas = {"Morena": (150, 150), "PRI": (130, 130), "PAN": (130, 130), "PRD": (30, 20), "PVEM": (10, 20),
              "PT": (25, 30), "MC": (25, 30), "PRIAN": (60, 70), "conservadores": (110, 60),
              "oposicion": (70, 50), "ambiguo_verde": (25, 30), "ambiguo_pan": (40, 2), "ambiguo_morena": (4, 1)}
    partes = []
    for g, (qa, qs) in cuotas.items():
        for pres, q in (("AMLO", qa), ("Sheinbaum", qs)):
            sub = men[(men.grupo == g) & (men.presidente == pres)]
            partes.append(sub.sample(n=min(q, len(sub)), random_state=SEMILLA))
    muestra = pd.concat(partes).sample(frac=1, random_state=SEMILLA).reset_index(drop=True)
    cols = ["id", "fecha", "presidente", "grupo", "objetivo", "mencion", "oracion_marcada",
            "contexto_previo", "pregunta_prensa", "particion"]
    muestra[cols].to_csv(os.path.join(DATOS, "muestra_anotacion.csv"), index=False)
    with open(os.path.join(DATOS, "muestra_anotacion.jsonl"), "w", encoding="utf-8") as fh:
        for rec in muestra[cols].to_dict("records"):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    resumen = {
        "fecha": time.strftime("%Y-%m-%d"), "segundos": round(time.time() - t0, 1), "menciones_totales": len(men),
        "por_grupo_y_presidente": men.groupby(["grupo", "presidente"]).size().unstack(fill_value=0).to_dict(),
        "muestra": len(muestra),
        "muestra_por_particion": muestra.particion.value_counts().to_dict(),
        "muestra_por_grupo": muestra.groupby(["grupo", "presidente"]).size().unstack(fill_value=0).to_dict(),
        "conferencias_por_particion": men.drop_duplicates("fecha").particion.value_counts().to_dict(),
    }
    json.dump(resumen, open(os.path.join(DATOS, "resumen_menciones.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: resumen[k] for k in ("segundos", "menciones_totales", "muestra", "muestra_por_particion", "conferencias_por_particion")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
