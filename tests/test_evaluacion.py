# -*- coding: utf-8 -*-
"""Pruebas de postura.evaluacion: AUC, umbrales, bootstrap agrupado y prueba de DeLong.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.
"""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from postura import evaluacion as ev


def _datos(n=400, semilla=0, senal=1.0):
    rng = np.random.default_rng(semilla)
    y = rng.integers(0, 2, n)
    s = senal * y + rng.normal(size=n)
    return y, s


def test_auc_igual_a_sklearn_con_y_sin_empates():
    for semilla in range(5):
        y, s = _datos(semilla=semilla)
        assert ev.roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
        s_emp = np.round(s)  # muchos empates
        assert ev.roc_auc(y, s_emp) == pytest.approx(roc_auc_score(y, s_emp), abs=1e-12)


def test_auc_casos_limite():
    assert ev.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert ev.roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert ev.roc_auc([0, 1], [0.5, 0.5]) == 0.5
    assert np.isnan(ev.roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_pr_auc_igual_a_precision_promedio():
    y, s = _datos(semilla=3)
    assert ev.pr_auc(y, s) == pytest.approx(average_precision_score(y, s))


def test_umbral_f1_maximiza_f1():
    y, s = _datos(semilla=4)
    umbral, f1 = ev.umbral_f1(y, s)
    assert f1 == pytest.approx(f1_score(y, (s >= umbral).astype(int)))
    for u in np.quantile(s, np.linspace(0.05, 0.95, 19)):
        assert f1_score(y, (s >= u).astype(int)) <= f1 + 1e-12
    m = ev.metricas_umbral(y, s, umbral)
    assert m["f1"] == pytest.approx(f1)


def test_bootstrap_reproducible_con_semilla():
    y, s = _datos(n=300, semilla=5)
    fechas = np.repeat(np.arange(60), 5)
    a = ev.bootstrap_agrupado(y, s, fechas, n_replicas=300, semilla=123)
    b = ev.bootstrap_agrupado(y, s, fechas, n_replicas=300, semilla=123)
    c = ev.bootstrap_agrupado(y, s, fechas, n_replicas=300, semilla=124)
    assert a == b
    assert a["ic_inferior"] != c["ic_inferior"] or a["ic_superior"] != c["ic_superior"]
    assert a["ic_inferior"] <= a["estimacion"] <= a["ic_superior"]
    assert a["n_grupos"] == 60 and a["replicas_validas"] == 300


def test_bootstrap_remuestrea_conferencias_completas():
    fechas = np.array(["a", "a", "b", "b", "b", "c"])
    for idx in ev.replicas_agrupadas(fechas, n_replicas=50, semilla=1):
        elegidas = fechas[idx]
        for f in set(elegidas):
            # cada fecha elegida aparece con todos sus ítems (múltiplos del tamaño del grupo)
            assert (elegidas == f).sum() % (fechas == f).sum() == 0


def test_bootstrap_diferencia_consigo_mismo_es_cero():
    y, s = _datos(n=200, semilla=6)
    fechas = np.repeat(np.arange(40), 5)
    r = ev.bootstrap_diferencia_agrupado(y, s, s, fechas, n_replicas=100)
    assert r["diferencia"] == 0 and r["ic_inferior"] == 0 and r["ic_superior"] == 0


def test_delong_modelo_consigo_mismo_p_cercano_a_uno():
    y, s = _datos(n=500, semilla=7)
    r = ev.delong(y, s, s)
    assert r["p_valor"] >= 0.99 and r["diferencia"] == 0
    # Una transformación monótona tampoco cambia el AUC.
    r2 = ev.delong(y, s, np.exp(s))
    assert r2["p_valor"] >= 0.99


def test_delong_perfecto_frente_a_aleatorio_p_pequeno():
    rng = np.random.default_rng(8)
    y = rng.integers(0, 2, 2000)
    perfecto = y + rng.uniform(0, 0.5, y.size)
    aleatorio = rng.normal(size=y.size)
    r = ev.delong(y, perfecto, aleatorio)
    assert r["auc_1"] == 1.0 and 0.4 < r["auc_2"] < 0.6
    assert r["p_valor"] < 1e-10


def test_delong_auc_coincide_y_varianza_igual_a_formula_directa():
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, 80)
    s1 = y + rng.normal(size=80)
    s2 = np.round(0.5 * y + rng.normal(size=80), 1)  # con empates
    r = ev.delong(y, s1, s2)
    assert r["auc_1"] == pytest.approx(roc_auc_score(y, s1))
    assert r["auc_2"] == pytest.approx(roc_auc_score(y, s2))

    # Fórmula directa de DeLong et al. (1988) con la matriz psi completa.
    def componentes(s):
        pos, neg = s[y == 1], s[y == 0]
        psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        return psi.mean(axis=1), psi.mean(axis=0)
    v10_1, v01_1 = componentes(s1)
    v10_2, v01_2 = componentes(s2)
    m, n = v10_1.size, v01_1.size
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    assert r["error_estandar"] == pytest.approx(np.sqrt(var), rel=1e-10)


def test_ic_delong_contiene_el_auc():
    y, s = _datos(n=300, semilla=10)
    r = ev.ic_delong(y, s)
    assert r["ic_inferior"] < r["auc"] < r["ic_superior"]


def test_delong_exige_dos_clases():
    with pytest.raises(ValueError):
        ev.delong([1, 1, 1, 0], [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1])


def test_curva_roc_y_figura(tmp_path):
    y, s = _datos(n=200, semilla=11)
    c = ev.curva_roc(y, s)
    assert c["fpr"][0] == 0 and c["tpr"][-1] == 1
    assert c["auc"] == pytest.approx(roc_auc_score(y, s))
    ruta = ev.graficar_roc({"modelo": (y, s), "otro": c}, str(tmp_path / "roc.png"), destacar="modelo")
    assert (tmp_path / "roc.png").stat().st_size > 1000 and ruta.endswith("roc.png")
