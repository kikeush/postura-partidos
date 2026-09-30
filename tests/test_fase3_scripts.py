# -*- coding: utf-8 -*-
"""Robustez de los scripts 07 y 08 de la Fase 3 con datos sintéticos en una carpeta temporal.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-28. Licencia: MIT.

07: avisa cuando falta datos/fase3_embeddings_qwen.npy, omite una fuente con puntajes no finitos
y escribe la fecha de ejecución. 08: con un modelo falso diminuto en lugar de BERT, una tarea cuya
pérdida o salida deja de ser finita no produce CSV, queda anotada en fase3_bert_tiempos.json y no
detiene a las demás. Nunca se toca datos/, modelos/ ni resultados/ del proyecto.
"""
import importlib.util
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cargar(nombre):
    ruta = os.path.join(RAIZ, "scripts", nombre)
    spec = importlib.util.spec_from_file_location(nombre.replace(".py", ""), ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _datos_sinteticos(carpeta, n=90):
    """Muestra y etiquetas con las columnas que leen 07 y 08; cada partición tiene ambas clases
    en las tres tareas."""
    partes = ["entrenamiento", "validacion", "prueba"]
    posturas = ["desfavorable", "favorable", "neutral"]
    filas, etiquetas = [], []
    for k in range(n):  # partición, postura y es_partido siguen ciclos distintos para mezclar clases
        es_partido = 0 if k % 7 == 6 else 1
        postura = posturas[(k // 3) % 3] if es_partido else "no_aplica"
        filas.append({"id": f"s{k}", "fecha": f"2024-01-{1 + k % 20:02d}", "particion": partes[k % 3],
                      "grupo": "PRI", "presidente": "AMLO",
                      "oracion_marcada": f"Oración {k} {'mala ' * (k % 4)}sobre el [[PRI]]{' buena' * (k % 5)}."})
        etiquetas.append({"id": f"s{k}", "es_partido": es_partido, "postura": postura})
    os.makedirs(carpeta, exist_ok=True)
    pd.DataFrame(filas).to_csv(os.path.join(carpeta, "muestra_anotacion.csv"), index=False)
    pd.DataFrame(etiquetas).to_csv(os.path.join(carpeta, "etiquetas.csv"), index=False)
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ 07
def test_07_avisa_sin_embeddings_y_omite_puntajes_no_finitos(tmp_path, capsys):
    ev = _cargar("07_fase3_evaluar.py")
    muestra = _datos_sinteticos(tmp_path / "datos")
    res = tmp_path / "resultados"
    (res / "figuras").mkdir(parents=True)
    ev.DATOS, ev.RES, ev.FIG = str(tmp_path / "datos"), str(res), str(res / "figuras")
    resultados_proyecto = os.path.join(RAIZ, "resultados")
    antes = {f: os.path.getmtime(os.path.join(resultados_proyecto, f)) for f in os.listdir(resultados_proyecto)}
    prueba = muestra[muestra.particion == "prueba"]
    # BERT en T1 con un NaN: se omite con aviso. En T2a con puntajes finitos: se evalúa.
    pd.DataFrame({"id": prueba.id, "s": [np.nan] + [0.5] * (len(prueba) - 1)}).to_csv(res / "fase3_bert_T1.csv", index=False)
    rng = np.random.default_rng(3)
    pd.DataFrame({"id": prueba.id, "s": rng.random(len(prueba))}).to_csv(res / "fase3_bert_T2a.csv", index=False)

    ev.main()

    err = capsys.readouterr().err
    assert "falta datos/fase3_embeddings_qwen.npy" in err and "06_fase3_qwen.py embeddings" in err
    assert "T1: bert_ajustado tiene puntajes no finitos" in err
    salida = json.loads((res / "fase3_resultados.json").read_text(encoding="utf-8"))
    assert salida["fecha"] == time.strftime("%Y-%m-%d")
    assert any("fase3_embeddings_qwen.npy" in a for a in salida["avisos"])
    assert "bert_ajustado" not in salida["tareas"]["T1"]["modelos"]
    assert 0 <= salida["tareas"]["T2a"]["modelos"]["bert_ajustado"]["auc"] <= 1
    assert salida["tareas"]["T1"]["embeddings"] == {}
    despues = {f: os.path.getmtime(os.path.join(resultados_proyecto, f)) for f in os.listdir(resultados_proyecto)}
    assert despues == antes  # nada se escribió en resultados/ del proyecto


# ------------------------------------------------------------------ 08
class _TokFalso:
    def __call__(self, textos, **kw):
        import torch
        x = [[len(t) % 7 / 7, t.count("mala") / 4, t.count("buena") / 5, 1.0] for t in textos]
        return {"input_ids": torch.tensor(x, dtype=torch.float32)}


def _modelo_falso(nan_en_train_desde=None, nan_en_eval_desde=None):
    import torch

    class ModeloFalso(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 2)
            self.epocas = 0

        def train(self, mode=True):
            if mode:
                self.epocas += 1
            return super().train(mode)

        def forward(self, input_ids=None, **kw):
            logits = self.lin(input_ids)
            if self.training and nan_en_train_desde and self.epocas >= nan_en_train_desde:
                logits = logits * float("nan")
            if not self.training and nan_en_eval_desde and self.epocas >= nan_en_eval_desde:
                logits = logits * float("nan")
            return SimpleNamespace(logits=logits)

    return ModeloFalso()


@pytest.mark.lento
def test_08_tarea_con_salida_no_finita_no_detiene_a_las_demas(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    bert = _cargar("08_fase3_bert.py")
    _datos_sinteticos(tmp_path / "datos")
    bert.DATOS, bert.RES = str(tmp_path / "datos"), str(tmp_path / "resultados")
    bert.EPOCAS, bert.dispositivo = 3, "cpu"
    # T1 entrena bien; en T2a la pérdida se vuelve NaN en la época 2; en T2b la salida de
    # validación se vuelve NaN en la época 3.
    modelos = iter([_modelo_falso(), _modelo_falso(nan_en_train_desde=2), _modelo_falso(nan_en_eval_desde=3)])
    bert.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: _TokFalso())
    bert.AutoModelForSequenceClassification = SimpleNamespace(from_pretrained=lambda *a, **k: next(modelos))

    tiempos = bert.main()

    res = tmp_path / "resultados"
    registro = json.loads((res / "fase3_bert_tiempos.json").read_text(encoding="utf-8"))
    assert registro == tiempos and set(registro) == {"T1", "T2a", "T2b"}
    assert registro["T1"]["estado"] == "ok" and registro["T1"]["mejor_epoca"] in (1, 2, 3)
    assert len(registro["T1"]["auc_validacion_por_epoca"]) == 3
    assert registro["T2a"]["estado"] == "pérdida no finita (NaN) en la época 2"
    assert registro["T2b"]["estado"] == "salida no finita (NaN) en la época 3"
    assert len(registro["T2b"]["auc_validacion_por_epoca"]) == 2
    for tarea in ("T2a", "T2b"):
        assert registro[tarea]["mejor_epoca"] is None and registro[tarea]["auc_validacion"] is None
        assert not (res / f"fase3_bert_{tarea}.csv").exists()
    csv = pd.read_csv(res / "fase3_bert_T1.csv")
    assert list(csv.columns) == ["id", "s"] and len(csv) == 30 and np.isfinite(csv.s).all()
