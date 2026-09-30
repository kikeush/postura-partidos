# -*- coding: utf-8 -*-
"""Métricas y pruebas estadísticas para el protocolo de evaluación de la Fase 2.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

Contenido
---------
* roc_auc: área bajo la curva ROC por la estadística de Mann-Whitney con rangos medios (los
  empates cuentan 1/2), idéntica a sklearn.metrics.roc_auc_score.
* pr_auc: área bajo la curva precisión-exhaustividad (precisión promedio).
* umbral_f1 y metricas_umbral: umbral que maximiza F1 en validación y métricas con ese umbral
  aplicadas a otra partición.
* bootstrap_agrupado: intervalo de confianza por remuestreo de conferencias (se remuestrean
  fechas completas con reemplazo, 1000 réplicas y semilla fija). Respeta la dependencia entre
  menciones de la misma conferencia.
* bootstrap_diferencia_agrupado: lo mismo para la diferencia de AUC entre dos modelos
  evaluados sobre los mismos ítems.
* delong: prueba pareada para dos AUC correlacionadas con la varianza de DeLong, DeLong y
  Clarke-Pearson (1988, Biometrics 44: 837-845), calculada con el algoritmo de rangos medios
  de Sun y Xu (2014, IEEE Signal Processing Letters 21: 1389-1393).
* curva_roc y graficar_roc: puntos de la curva y figura en escala de grises.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_curve)

SEMILLA = 20260927


def _arreglos(y, s):
    """Convierte etiquetas y puntajes en arreglos planos y comprueba que tengan la misma forma."""
    y = np.asarray(y).astype(int).ravel()
    s = np.asarray(s, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y y puntajes tienen tamaños distintos: {y.shape} y {s.shape}")
    return y, s


def roc_auc(y, s):
    """AUC ROC = P(puntaje de un positivo > puntaje de un negativo) + 1/2 P(empate).

    Devuelve nan si solo hay una clase."""
    y, s = _arreglos(y, s)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    rangos = stats.rankdata(s)
    return float((rangos[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pr_auc(y, s):
    """Precisión promedio (área bajo la curva precisión-exhaustividad)."""
    y, s = _arreglos(y, s)
    if y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, s))


def umbral_f1(y, s):
    """Umbral que maximiza F1 (se predice positivo si puntaje >= umbral).

    Devuelve (umbral, f1). Se elige en validación y se aplica sin cambios en prueba."""
    y, s = _arreglos(y, s)
    if y.min() == y.max():
        return 0.5, float("nan")
    precision, exhaustividad, umbrales = precision_recall_curve(y, s)
    p, r = precision[:-1], exhaustividad[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    i = int(np.argmax(f1))
    return float(umbrales[i]), float(f1[i])


def metricas_umbral(y, s, umbral):
    """F1, precisión, exhaustividad y exactitud al umbral dado."""
    y, s = _arreglos(y, s)
    pred = (s >= umbral).astype(int)
    return {"umbral": float(umbral),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "exhaustividad": float(recall_score(y, pred, zero_division=0)),
            "exactitud": float((pred == y).mean()) if y.size else float("nan"),
            "predichos_positivos": int(pred.sum())}


# ---------------------------------------------------------------------------------------
# Bootstrap agrupado por conferencia
# ---------------------------------------------------------------------------------------

def _indices_por_grupo(grupos):
    """Índices de las filas de cada grupo, para remuestrear conferencias completas."""
    _, inversa = np.unique(np.asarray(grupos).astype(str), return_inverse=True)
    orden = np.argsort(inversa, kind="stable")
    cortes = np.flatnonzero(np.diff(inversa[orden])) + 1
    return np.split(orden, cortes)


def replicas_agrupadas(grupos, n_replicas=1000, semilla=SEMILLA):
    """Generador de arreglos de índices: cada réplica remuestrea grupos completos (fechas)
    con reemplazo y concatena sus ítems."""
    por_grupo = _indices_por_grupo(grupos)
    g = len(por_grupo)
    rng = np.random.default_rng(semilla)
    for _ in range(n_replicas):
        elegidos = rng.integers(0, g, size=g)
        yield np.concatenate([por_grupo[j] for j in elegidos])


def bootstrap_agrupado(y, s, grupos, metrica=roc_auc, n_replicas=1000, semilla=SEMILLA, alfa=0.05):
    """Intervalo percentil (1 - alfa) de una métrica remuestreando conferencias.

    Las réplicas con una sola clase se descartan y se reporta cuántas quedaron válidas."""
    y, s = _arreglos(y, s)
    valores = []
    for idx in replicas_agrupadas(grupos, n_replicas, semilla):
        yb = y[idx]
        if yb.min() != yb.max():
            valores.append(metrica(yb, s[idx]))
    valores = np.asarray(valores, dtype=float)
    estimacion = metrica(y, s) if y.min() != y.max() else float("nan")
    if valores.size == 0:
        return {"estimacion": estimacion, "ic_inferior": float("nan"), "ic_superior": float("nan"),
                "error_estandar": float("nan"), "replicas_validas": 0,
                "n_grupos": len(_indices_por_grupo(grupos)), "semilla": semilla}
    inf, sup = np.percentile(valores, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    return {"estimacion": float(estimacion), "ic_inferior": float(inf), "ic_superior": float(sup),
            "error_estandar": float(valores.std(ddof=1)) if valores.size > 1 else float("nan"),
            "replicas_validas": int(valores.size), "n_grupos": len(_indices_por_grupo(grupos)),
            "semilla": semilla}


def bootstrap_diferencia_agrupado(y, s1, s2, grupos, metrica=roc_auc, n_replicas=1000,
                                  semilla=SEMILLA, alfa=0.05):
    """IC de metrica(s1) - metrica(s2) con las mismas réplicas por conferencia.

    p_valor es la proporción bilateral de réplicas en las que la diferencia cambia de signo
    (aproximación de bootstrap; complementa a DeLong, que supone ítems independientes)."""
    y, s1 = _arreglos(y, s1)
    _, s2 = _arreglos(y, s2)
    difs = []
    for idx in replicas_agrupadas(grupos, n_replicas, semilla):
        yb = y[idx]
        if yb.min() != yb.max():
            difs.append(metrica(yb, s1[idx]) - metrica(yb, s2[idx]))
    difs = np.asarray(difs, dtype=float)
    diferencia = metrica(y, s1) - metrica(y, s2)
    if difs.size == 0:
        return {"diferencia": float(diferencia), "ic_inferior": float("nan"),
                "ic_superior": float("nan"), "p_valor": float("nan"), "replicas_validas": 0}
    inf, sup = np.percentile(difs, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    p = 2 * min((difs <= 0).mean(), (difs >= 0).mean())
    return {"diferencia": float(diferencia), "ic_inferior": float(inf), "ic_superior": float(sup),
            "p_valor": float(min(1.0, p)), "replicas_validas": int(difs.size), "semilla": semilla}


# ---------------------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------------------

def componentes_delong(y, puntajes):
    """AUC y matriz de covarianza de DeLong para k modelos evaluados sobre los mismos ítems.

    Parámetros
    ----------
    y : arreglo (n,) con 0 y 1.
    puntajes : arreglo (k, n).

    Para cada positivo i, V10_i = (1/n0) sum_j psi(X_i, Y_j); para cada negativo j,
    V01_j = (1/n1) sum_i psi(X_i, Y_j), con psi = 1 si X > Y, 1/2 si empatan, 0 si no.
    Con rangos medios: V10 = (rango en el conjunto total - rango entre positivos) / n0 y
    V01 = 1 - (rango en el conjunto total - rango entre negativos) / n1.
    La covarianza es S10 / n1 + S01 / n0 (DeLong et al., 1988)."""
    y = np.asarray(y).astype(int).ravel()
    puntajes = np.atleast_2d(np.asarray(puntajes, dtype=float))
    pos, neg = puntajes[:, y == 1], puntajes[:, y == 0]
    n1, n0 = pos.shape[1], neg.shape[1]
    if n1 < 2 or n0 < 2:
        raise ValueError("DeLong necesita al menos dos positivos y dos negativos.")
    tx = np.vstack([stats.rankdata(fila) for fila in pos])
    ty = np.vstack([stats.rankdata(fila) for fila in neg])
    tz = np.vstack([stats.rankdata(np.concatenate([p, q])) for p, q in zip(pos, neg)])
    auc = (tz[:, :n1].sum(axis=1) / n1 - (n1 + 1) / 2.0) / n0
    v10 = (tz[:, :n1] - tx) / n0
    v01 = 1.0 - (tz[:, n1:] - ty) / n1
    s10 = np.atleast_2d(np.cov(v10))
    s01 = np.atleast_2d(np.cov(v01))
    return auc, s10 / n1 + s01 / n0


def delong(y, s1, s2, alfa=0.05):
    """Prueba de DeLong para H0: AUC(s1) = AUC(s2), con s1 y s2 evaluados sobre los mismos ítems.

    Devuelve AUC de cada modelo, diferencia, error estándar, z, p bilateral e intervalo
    (1 - alfa) de la diferencia. Si la varianza de la diferencia es cero (por ejemplo, un
    modelo contra sí mismo) y la diferencia también, p = 1."""
    auc, cov = componentes_delong(y, np.vstack([np.asarray(s1, float), np.asarray(s2, float)]))
    dif = float(auc[0] - auc[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    ee = float(np.sqrt(max(var, 0.0)))
    if ee < 1e-12:
        z = 0.0 if abs(dif) < 1e-12 else float(np.sign(dif) * np.inf)
        p = 1.0 if abs(dif) < 1e-12 else 0.0
    else:
        z = dif / ee
        p = float(2 * stats.norm.sf(abs(z)))
    q = stats.norm.ppf(1 - alfa / 2)
    return {"auc_1": float(auc[0]), "auc_2": float(auc[1]), "diferencia": dif,
            "error_estandar": ee, "z": float(z), "p_valor": p,
            "ic_inferior": dif - q * ee, "ic_superior": dif + q * ee}


def ic_delong(y, s, alfa=0.05):
    """Intervalo (1 - alfa) de un AUC con la varianza de DeLong (aproximación normal)."""
    auc, cov = componentes_delong(y, np.asarray(s, float)[None, :])
    ee = float(np.sqrt(max(cov[0, 0], 0.0)))
    q = stats.norm.ppf(1 - alfa / 2)
    return {"auc": float(auc[0]), "error_estandar": ee,
            "ic_inferior": float(max(0.0, auc[0] - q * ee)),
            "ic_superior": float(min(1.0, auc[0] + q * ee))}


# ---------------------------------------------------------------------------------------
# Curvas ROC
# ---------------------------------------------------------------------------------------

def curva_roc(y, s):
    """Puntos de la curva ROC y su AUC: {'fpr', 'tpr', 'umbrales', 'auc'}."""
    y, s = _arreglos(y, s)
    fpr, tpr, umbrales = roc_curve(y, s)
    return {"fpr": fpr, "tpr": tpr, "umbrales": umbrales, "auc": roc_auc(y, s)}


_ESTILOS = [("0.0", "-", 2.0), ("0.0", "--", 1.6), ("0.35", "-", 1.6), ("0.35", "-.", 1.6),
            ("0.55", ":", 2.0), ("0.55", "--", 1.4), ("0.7", "-", 1.4), ("0.2", ":", 1.4)]


def graficar_roc(curvas, ruta, titulo="Curva ROC", destacar=None):
    """Guarda una figura con varias curvas ROC en escala de grises.

    Parámetros
    ----------
    curvas : dict nombre -> (y, puntajes) o nombre -> resultado de curva_roc.
    ruta : archivo de salida (PNG).
    destacar : nombre de la curva que se dibuja en negro sólido y más gruesa.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 5.6), dpi=200)
    ax.plot([0, 1], [0, 1], color="0.75", lw=1, ls=(0, (2, 2)), label="Azar (AUC = 0.500)")
    nombres = list(curvas)
    if destacar in nombres:
        nombres.remove(destacar)
        nombres.insert(0, destacar)
    for i, nombre in enumerate(nombres):
        c = curvas[nombre]
        if not isinstance(c, dict):
            c = curva_roc(*c)
        color, estilo, grosor = _ESTILOS[i % len(_ESTILOS)]
        if nombre == destacar:
            color, estilo, grosor = "0.0", "-", 2.6
        ax.plot(c["fpr"], c["tpr"], color=color, ls=estilo, lw=grosor,
                label=f"{nombre} (AUC = {c['auc']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(titulo, fontsize=11)
    ax.grid(color="0.9", lw=0.6)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    return ruta
