# -*- coding: utf-8 -*-
"""Diccionario de partidos, segmentación en oraciones y extracción de menciones candidatas.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

Una mención candidata es una cadena que puede referirse a un partido político o a un bloque
de partidos: nombres y siglas (Morena, PRI, PAN, PRD, PVEM, PT, MC), gentilicios (priista,
panista), etiquetas de bloque (PRIAN, conservadores, oposición) y formas ambiguas que a veces
son otra cosa (pan, Verde suelto, morena en minúscula). La decisión de si de verdad es un
partido la toma el clasificador de la Tarea 1.

Límite conocido: los gentilicios y las etiquetas de bloque (panistas, priistas, morenistas,
conservadores, opositor) solo se buscan en minúscula, así que sus formas con inicial mayúscula,
por ejemplo al inicio de una oración (Conservadores, Panistas, Priistas, Opositor), no se
capturan. En el corpus son 19 cadenas frente a las 11 714 capturadas (0.16 %). Los patrones se
conservan así porque cambiarlos alteraría la muestra anotada y los resultados publicados; una
versión futura puede admitir la inicial mayúscula con [Cc]onservador, [Pp]anistas, etcétera.
"""
import re

# (grupo, patrón). Orden: primero las formas largas para que ganen en caso de traslape.
PATRONES = [
    ("PVEM", r"\bPartido Verde(?: Ecologista(?: de México)?)?\b|\bVerde Ecologista\b|\bPVEM\b"),
    ("PT", r"\bPartido del Trabajo\b|\bPT\b|\bpetistas?\b"),
    ("MC", r"\bMovimiento Ciudadano\b|\bMC\b"),
    ("PAN", r"\bAcción Nacional\b|\bPAN\b|\bpanistas?\b"),
    ("PRIAN", r"\bPRIAN\w*|\bPrian\w*|\bpriani\w+"),
    ("PRI", r"\bPRI\b|\bpri[ií]stas?\b|\bpri[ií]smo\b"),
    ("PRD", r"\bPRD\b|\bperredistas?\b"),
    ("Morena", r"\bMorena\b|\bMORENA\b|\bmorenistas?\b"),
    ("conservadores", r"\bconservador(?:es|a|as)?\b"),
    ("oposicion", r"\boposici[oó]n\b|\bopositor(?:es|a|as)?\b"),
    ("ambiguo_verde", r"\bVerde\b"),
    ("ambiguo_pan", r"\bpan\b"),
    ("ambiguo_morena", r"\bmorenas?\b"),
]
_RX = [(g, re.compile(p)) for g, p in PATRONES]

# Partido objetivo al que apunta cada grupo cuando la mención sí es a un partido.
OBJETIVO = {
    "PVEM": "PVEM", "PT": "PT", "MC": "MC", "PAN": "PAN", "PRIAN": "PRIAN", "PRI": "PRI",
    "PRD": "PRD", "Morena": "Morena", "conservadores": "bloque opositor",
    "oposicion": "bloque opositor", "ambiguo_verde": "PVEM", "ambiguo_pan": "PAN",
    "ambiguo_morena": "Morena",
}

SEGMENTADOR = re.compile(r"(?<=[.!?…])\s+(?=[¿¡\"“‘A-ZÁÉÍÓÚÑ0-9])")


def oraciones(texto):
    """Divide un párrafo en oraciones tras un signo de cierre seguido de mayúscula,
    signo de apertura, comilla o dígito. Devuelve la lista de oraciones sin vacías."""
    texto = re.sub(r"\s+", " ", str(texto or "")).strip()
    if not texto:
        return []
    return [o.strip() for o in SEGMENTADOR.split(texto) if o.strip()]


def buscar_menciones(texto):
    """Devuelve una lista de (inicio, fin, cadena, grupo) sin traslapes, en orden de aparición.
    Ante traslape gana la coincidencia que empieza antes y, si empiezan igual, la más larga."""
    halladas = []
    for grupo, rx in _RX:
        for m in rx.finditer(texto):
            halladas.append((m.start(), m.end(), m.group(0), grupo))
    halladas.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    resultado, fin_prev = [], -1
    for h in halladas:
        if h[0] >= fin_prev:
            resultado.append(h)
            fin_prev = h[1]
    return resultado


def marcar(oracion, inicio, fin):
    """Devuelve la oración con la mención envuelta en [[ ]] para que el modelo sepa cuál es."""
    return oracion[:inicio] + "[[" + oracion[inicio:fin] + "]]" + oracion[fin:]


def enmascarar(oracion, inicio, fin, token="[PARTIDO]"):
    """Sustituye la mención por un token neutro (ablación sin nombre del partido)."""
    return oracion[:inicio] + token + oracion[fin:]
