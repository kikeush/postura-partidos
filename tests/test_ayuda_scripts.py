# -*- coding: utf-8 -*-
"""Los scripts responden a -h y --help con su docstring y terminan sin leer ni escribir nada.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-28. Licencia: MIT.

Cada script se ejecuta en una copia mínima del repositorio (scripts/ y postura/ con carpetas
datos/, modelos/ y resultados/ vacías), así que cualquier lectura o escritura fallaría o
dejaría rastro. Los scripts de la Fase 3 imprimen la ayuda antes de importar PyTorch o MLX.
"""
import os
import shutil
import subprocess
import sys

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.lento  # arranca un subproceso por script y bandera

SCRIPTS = ["01_construir_menciones.py", "02_consolidar_etiquetas.py", "04_entrenar_clasicos.py",
           "05_aplicar_corpus.py", "06_fase3_qwen.py", "07_fase3_evaluar.py", "08_fase3_bert.py",
           "09_figuras_roc.py", "qwen_local.py"]


@pytest.fixture(scope="module")
def esqueleto(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("esqueleto")
    for carpeta in ("scripts", "postura"):
        shutil.copytree(os.path.join(RAIZ, carpeta), raiz / carpeta, ignore=shutil.ignore_patterns("__pycache__"))
    for carpeta in ("datos", "modelos", "resultados"):
        (raiz / carpeta).mkdir()
    return raiz


def _archivos(raiz):
    return sorted(str(p.relative_to(raiz)) for c in ("datos", "modelos", "resultados")
                  for p in (raiz / c).rglob("*") if p.is_file())


def _correr(raiz, script, *args):
    return subprocess.run([sys.executable, str(raiz / "scripts" / script), *args],
                          capture_output=True, text=True, cwd=str(raiz), timeout=120)


@pytest.mark.parametrize("bandera", ["--help", "-h"])
@pytest.mark.parametrize("script", SCRIPTS)
def test_ayuda_sin_efectos(esqueleto, script, bandera):
    r = _correr(esqueleto, script, bandera)
    assert r.returncode == 0, r.stderr
    assert "Autor: Enrique Alberto Mendoza Ruiz" in r.stdout or "usage" in r.stdout
    assert _archivos(esqueleto) == []


def test_06_rechaza_modo_desconocido_sin_cargar_el_modelo(esqueleto):
    pytest.importorskip("mlx_lm")
    r = _correr(esqueleto, "06_fase3_qwen.py", "cero", "inventado")
    assert r.returncode == 2
    assert "Modo desconocido: inventado" in r.stderr
    assert "modelo cargado" not in r.stdout
    assert _archivos(esqueleto) == []
