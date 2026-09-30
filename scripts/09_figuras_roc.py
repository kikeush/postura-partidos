# -*- coding: utf-8 -*-
"""Redibuja las curvas ROC de prueba de las Fases 2 y 3 a partir de las predicciones guardadas.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-28. Licencia: MIT.

No entrena ni puntúa nada: lee resultados/fase2_predicciones_prueba.csv y
resultados/fase3_predicciones_prueba.csv y escribe resultados/figuras/fase2_roc_T*.png y
fase3_roc_T*.png con los nombres legibles de postura/nombres.py. Las figuras son las mismas
que dibujan los scripts 04 y 07; sirve para rehacerlas sin volver a correr esos pasos.
Fase 2: todos los modelos, el elegido en validación destacado y su versión sin nombre del
partido. Fase 3: los enfoques evaluados en todas las menciones de prueba más el modelo elegido
de la Fase 2 como referencia (el razonamiento en cadena se omite porque se evaluó en un
subconjunto).
Uso: python scripts/09_figuras_roc.py (sin argumentos; --help muestra esta ayuda).
"""
import os, sys
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(__doc__.strip()); sys.exit(0)  # ayuda sin leer ni escribir nada
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from postura import evaluacion as ev  # noqa: E402
from postura.nombres import NOMBRE_TAREA, nombre_modelo  # noqa: E402

RES = os.path.join(RAIZ, "resultados")
FIG = os.path.join(RES, "figuras")
FASE2 = ["base_grupo", "base_longitud", "lexico", "logistica", "svm_lineal", "bosque_aleatorio"]


def main():
    """Redibuja las curvas ROC de prueba de las Fases 2 y 3 para cada tarea a partir de las predicciones guardadas."""
    os.makedirs(FIG, exist_ok=True)
    p2 = pd.read_csv(os.path.join(RES, "fase2_predicciones_prueba.csv"))
    p3 = pd.read_csv(os.path.join(RES, "fase3_predicciones_prueba.csv"))
    for tarea in ["T1", "T2a", "T2b"]:
        d2 = p2[p2.tarea == tarea]
        mejor = d2.mejor_modelo.iloc[0]
        elegido = f"{nombre_modelo(mejor)} (Fase 2)"
        curvas = {}
        for m in FASE2:
            nombre = elegido if m == mejor else nombre_modelo(m)
            curvas[nombre] = ev.curva_roc(d2.y_real.values, d2[f"puntaje_{m}"].values)
        curvas[f"{nombre_modelo(mejor)} sin nombre del partido"] = ev.curva_roc(
            d2.y_real.values, d2.puntaje_mejor_enmascarado.values)
        ev.graficar_roc(curvas, os.path.join(FIG, f"fase2_roc_{tarea}.png"),
                        titulo=f"Fase 2, {tarea}. {NOMBRE_TAREA[tarea]} (prueba, n = {len(d2)})",
                        destacar=elegido)
        d3 = p3[p3.tarea == tarea]
        curvas = {elegido: ev.curva_roc(d2.y_real.values, d2[f"puntaje_{mejor}"].values)}
        for m, g in d3.groupby("modelo", sort=False):
            if m == "qwen_cot" or len(g) != len(d2) or g.y_real.nunique() < 2:
                continue
            curvas[nombre_modelo(m)] = ev.curva_roc(g.y_real.values, g.puntaje.values)
        ev.graficar_roc(curvas, os.path.join(FIG, f"fase3_roc_{tarea}.png"),
                        titulo=f"Fase 3, {tarea}. {NOMBRE_TAREA[tarea]} (prueba, n = {len(d2)})",
                        destacar=elegido)
        print(tarea, "listo:", ", ".join(curvas))


if __name__ == "__main__":
    main()
