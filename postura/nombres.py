# -*- coding: utf-8 -*-
"""Nombres legibles de modelos y tareas para figuras y tablas.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-28. Licencia: MIT.

Son los mismos nombres que usan el informe (tablas 2 y 3) y la aplicación de demostración
(app/utilidades.py); una prueba comprueba que ambos diccionarios coinciden.
"""

NOMBRE_MODELO = {
    "base_grupo": "Solo nombre del partido",
    "base_longitud": "Solo longitud",
    "lexico": "Léxico y nombre del partido",
    "logistica": "Regresión logística",
    "svm_lineal": "SVM lineal",
    "bosque_aleatorio": "Random Forest",
    "qwen_cero": "LLM sin ejemplos",
    "qwen_pocos": "LLM con ocho ejemplos",
    "qwen_cot": "LLM con razonamiento en cadena",
    "emb_logistica": "Embeddings + regresión logística",
    "emb_svm": "Embeddings + SVM",
    "emb_bosque": "Embeddings + Random Forest",
    "bert_ajustado": "BERT con ajuste fino",
}

NOMBRE_TAREA = {
    "T1": "¿La mención se refiere a un partido?",
    "T2a": "Postura desfavorable frente al resto",
    "T2b": "Postura favorable frente al resto",
}


def nombre_modelo(clave):
    """Nombre legible de un modelo; una clave desconocida se devuelve con espacios."""
    clave = str(clave)
    return NOMBRE_MODELO.get(clave, clave.replace("_", " "))
