# -*- coding: utf-8 -*-
"""Pruebas de postura.menciones: segmentación en oraciones y extracción de menciones.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.
"""
import pytest

from postura.menciones import OBJETIVO, PATRONES, buscar_menciones, enmascarar, marcar, oraciones


def grupos(texto):
    return [(m[2], m[3]) for m in buscar_menciones(texto)]


# ------------------------------------------------------------------ segmentación
def test_oraciones_con_signos_de_apertura():
    texto = "¿Qué pasó con el PRI? ¡No lo sé! El PAN votó en contra."
    assert oraciones(texto) == ["¿Qué pasó con el PRI?", "¡No lo sé!", "El PAN votó en contra."]


def test_oraciones_que_empiezan_con_cifra():
    texto = "Se contaron los votos. 30 millones votaron por Morena. Así fue."
    assert oraciones(texto) == ["Se contaron los votos.", "30 millones votaron por Morena.", "Así fue."]


def test_decimales_no_parten_la_oracion():
    assert oraciones("Fueron 3.5 millones de votos para el PRI.") == ["Fueron 3.5 millones de votos para el PRI."]


def test_minuscula_tras_punto_no_parte():
    assert len(oraciones("Lo dijo el lic. pérez ayer.")) == 1


def test_oraciones_vacias_y_espacios():
    assert oraciones("") == []
    assert oraciones(None) == []
    assert oraciones("  Hola.   \n  Adiós.  ") == ["Hola.", "Adiós."]


def test_puntos_suspensivos_y_comillas():
    texto = "Eso dijeron… “Nosotros no fuimos”, afirmaron los del PRD."
    assert oraciones(texto) == ["Eso dijeron…", "“Nosotros no fuimos”, afirmaron los del PRD."]


# ------------------------------------------------------------------ traslapes y formas
def test_partido_verde_gana_a_verde():
    assert grupos("El Partido Verde votó con Morena.") == [("Partido Verde", "PVEM"), ("Morena", "Morena")]


def test_partido_verde_nombre_completo():
    assert grupos("el Partido Verde Ecologista de México") == [("Partido Verde Ecologista de México", "PVEM")]


def test_verde_suelto_es_ambiguo():
    assert grupos("La coalición Morena-Verde-PT ganó.") == [("Morena", "Morena"), ("Verde", "ambiguo_verde"), ("PT", "PT")]


@pytest.mark.parametrize("forma", ["PRIAN", "Prian", "PRIANismo", "prianistas", "PRIANISTAS"])
def test_prian_y_variantes(forma):
    assert grupos(f"Eso lo hizo el {forma} siempre.") == [(forma, "PRIAN")]


def test_prian_no_se_parte_en_pri_y_pan():
    resultado = grupos("El PRIAN, el PRI y el PAN.")
    assert resultado == [("PRIAN", "PRIAN"), ("PRI", "PRI"), ("PAN", "PAN")]


def test_pan_en_minuscula_es_ambiguo_y_en_mayuscula_es_partido():
    assert grupos("Compramos pan en la esquina.") == [("pan", "ambiguo_pan")]
    assert grupos("El PAN votó en contra.") == [("PAN", "PAN")]
    assert grupos("Los panistas y Acción Nacional.") == [("panistas", "PAN"), ("Acción Nacional", "PAN")]


def test_morena_en_minuscula_es_ambiguo():
    assert grupos("una mujer morena") == [("morena", "ambiguo_morena")]
    assert grupos("los morenistas") == [("morenistas", "Morena")]


def test_gentilicios_y_nombres_largos():
    assert grupos("los priistas y los priístas") == [("priistas", "PRI"), ("priístas", "PRI")]
    assert grupos("Movimiento Ciudadano y el MC") == [("Movimiento Ciudadano", "MC"), ("MC", "MC")]
    assert grupos("el Partido del Trabajo") == [("Partido del Trabajo", "PT")]


def test_palabras_que_contienen_siglas_no_cuentan():
    assert grupos("PANTALLA, PRIMERO, panorama, oposicionista") == []


def test_menciones_en_orden_y_sin_traslape():
    texto = "Los conservadores del PRIAN y la oposición del Partido Verde."
    halladas = buscar_menciones(texto)
    inicios = [h[0] for h in halladas]
    assert inicios == sorted(inicios)
    for a, b in zip(halladas, halladas[1:]):
        assert a[1] <= b[0]
    assert [h[3] for h in halladas] == ["conservadores", "PRIAN", "oposicion", "PVEM"]


def test_todo_grupo_tiene_objetivo():
    assert {g for g, _ in PATRONES} == set(OBJETIVO)


# ------------------------------------------------------------------ marcar y enmascarar
def test_marcar_y_enmascarar():
    o = "El PRI votó en contra."
    (ini, fin, cadena, _), = buscar_menciones(o)
    assert marcar(o, ini, fin) == "El [[PRI]] votó en contra."
    assert enmascarar(o, ini, fin) == "El [PARTIDO] votó en contra."
