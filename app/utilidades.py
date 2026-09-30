# -*- coding: utf-8 -*-
"""Utilidades de la aplicación de demostración: rutas, lectura tolerante de archivos de
resultados, transformación de datos para las gráficas y tablas HTML de la sección de análisis.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Curso: Machine Learning II, Prof. Dr. Miguel González Mendoza. Assignment 3.
Fecha: 2026-09-27. Licencia: MIT.

Ninguna función de este módulo importa Streamlit, así que todas se pueden probar con pytest
sin levantar la aplicación. Las funciones de lectura no lanzan excepciones cuando un archivo
falta o está incompleto: devuelven (None, aviso) y la aplicación muestra el aviso.

La raíz de datos y resultados se toma de la variable de entorno POSTURA_RAIZ; si no existe,
es la carpeta del proyecto. El código (paquete postura y scripts) siempre se toma del proyecto.
"""
from __future__ import annotations

import html
import json
import math
import os
import sys
import unicodedata
from pathlib import Path

import pandas as pd

RAIZ_CODIGO = Path(__file__).resolve().parents[1]
if str(RAIZ_CODIGO) not in sys.path:
    sys.path.insert(0, str(RAIZ_CODIGO))

from postura.menciones import OBJETIVO, buscar_menciones, marcar, oraciones  # noqa: E402

# Cifras fijas del corpus (manifiesto de versiones estenográficas, 2018-2026).
CONFERENCIAS = {"total": 1945, "AMLO": 1462, "Sheinbaum": 483}
MENCIONES_TOTALES = 11714

# Colores. Rojo bermellón y verde azulado se separan también con daltonismo rojo verde;
# el gris es el punto medio neutro. Los colores de presidente y de fase son pares validados.
COLOR_POSTURA = {"desfavorable": "#cc3d2f", "neutral": "#c9c8c2", "favorable": "#0f8a64",
                 "no_aplica": "#ffffff"}
NOMBRE_POSTURA = {"desfavorable": "desfavorable", "favorable": "favorable", "neutral": "neutral",
                  "no_aplica": "no es partido"}
ORDEN_POSTURA = ["desfavorable", "neutral", "favorable"]
COLOR_PRESIDENTE = {"AMLO": "#2a78d6", "Sheinbaum": "#eb6834"}
COLOR_FASE = {"Fase 2": "#2a78d6", "Fase 3": "#d55181"}
COLOR_TINTA = "#52514e"

ETIQUETA_GRUPO = {
    "Morena": "Morena", "PRI": "PRI", "PAN": "PAN", "PRD": "PRD", "PVEM": "PVEM", "PT": "PT",
    "MC": "MC", "PRIAN": "PRIAN", "conservadores": "conservadores", "oposicion": "oposición",
    "ambiguo_verde": "Verde (ambiguo)", "ambiguo_pan": "pan (ambiguo)",
    "ambiguo_morena": "morena (ambiguo)",
}
ORDEN_OBJETIVOS = ["Morena", "PT", "PVEM", "MC", "PRI", "PAN", "PRD", "PRIAN", "bloque opositor"]

TAREAS = {
    "T1": "T1. ¿Es un partido?",
    "T2a": "T2a. Desfavorable frente al resto",
    "T2b": "T2b. Favorable frente al resto",
}
TAREAS_CORTAS = {"T1": "T1. Es partido", "T2a": "T2a. Desfavorable", "T2b": "T2b. Favorable"}

# Nombres legibles de los modelos, los mismos que usan las tablas 2 y 3 del informe. Las claves
# son las de resultados/fase2_resultados.json y fase3_resultados.json.
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


def nombre_figura(stem) -> str:
    """Leyenda legible de una figura: 'fase2_roc_T1' da 'Curva ROC de prueba, Fase 2, T1. Es partido'."""
    partes = str(stem).split("_")
    if len(partes) == 3 and partes[0].startswith("fase") and partes[1] == "roc" and partes[2] in TAREAS_CORTAS:
        return f"Curva ROC de prueba, Fase {partes[0][4:]}, {TAREAS_CORTAS[partes[2]]}"
    return str(stem).replace("_", " ")


def nombre_modelo(clave) -> str:
    """Nombre legible de un modelo; una clave desconocida se devuelve con espacios en vez de guiones bajos."""
    clave = str(clave)
    return NOMBRE_MODELO.get(clave, clave.replace("_", " "))


DESCRIPCION_TAREAS = {
    "T1": "Sobre todas las menciones candidatas: 1 si la cadena designa a un partido o bloque.",
    "T2a": "Solo menciones a partidos: 1 si la postura es desfavorable; 0 si es favorable o neutral.",
    "T2b": "Solo menciones a partidos: 1 si la postura es favorable; 0 si es desfavorable o neutral.",
}

# Ejemplos reales de la muestra de anotación, ambos de la partición de prueba.
EJEMPLOS = {
    "PRIAN": {"id": "2023-10-02_196_3_98", "grupo": "PRIAN",
              "etiqueta": "PRIAN (López Obrador, 2 oct 2023)"},
    "Morena": {"id": "2025-07-10_490_1_65", "grupo": "Morena",
               "etiqueta": "Morena (Sheinbaum, 10 jul 2025)"},
}

# Columnas de probabilidad en las tablas: (clave, encabezado, color de la barra).
COLUMNAS_CLASICO = [
    ("p_partido", "P(partido)", COLOR_TINTA),
    ("p_desfavorable", "P(desfavorable)", COLOR_POSTURA["desfavorable"]),
    ("p_favorable", "P(favorable)", COLOR_POSTURA["favorable"]),
]
COLUMNAS_LLM = [
    ("p_partido", "P(partido) = 1 - P(X)", COLOR_TINTA),
    ("p_D", "P(D)", COLOR_POSTURA["desfavorable"]),
    ("p_F", "P(F)", COLOR_POSTURA["favorable"]),
    ("p_N", "P(N)", "#8f8e89"),
]
LETRA_POSTURA = {"X": "no_aplica", "D": "desfavorable", "F": "favorable", "N": "neutral"}

ESTILO_CSS = (
    "<style>"
    ".pp-envoltura{overflow-x:auto;margin:0.25rem 0 0.75rem 0;}"
    "table.pp-tabla{border-collapse:collapse;width:100%;font-size:0.92rem;}"
    "table.pp-tabla th{text-align:left;font-weight:600;padding:6px 8px;"
    "border-bottom:1px solid rgba(128,128,128,0.45);white-space:nowrap;}"
    "table.pp-tabla td{padding:8px;border-bottom:1px solid rgba(128,128,128,0.2);vertical-align:top;}"
    "table.pp-tabla td.pp-oracion{width:40%;min-width:260px;line-height:1.45;}"
    "table.pp-tabla mark{background:#ffe58a;color:#1f1f1d;padding:0 3px;border-radius:3px;font-weight:600;}"
    ".pp-barra{display:flex;align-items:center;gap:6px;min-width:90px;}"
    ".pp-pista{flex:1;height:8px;background:rgba(128,128,128,0.18);border-radius:4px;overflow:hidden;}"
    ".pp-relleno{height:100%;border-radius:4px;}"
    ".pp-valor{font-variant-numeric:tabular-nums;font-size:0.85rem;min-width:2.6em;text-align:right;}"
    ".pp-chip{display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.85rem;"
    "font-weight:600;white-space:nowrap;border:1px solid transparent;}"
    ".pp-desfavorable{background:#cc3d2f;color:#ffffff;}"
    ".pp-favorable{background:#0f8a64;color:#ffffff;}"
    ".pp-neutral{background:#c9c8c2;color:#1f1f1d;}"
    ".pp-no_aplica{background:#ffffff;color:#1f1f1d;border-color:rgba(128,128,128,0.6);}"
    ".pp-na{opacity:0.7;font-size:0.85rem;}"
    ".pp-sub{opacity:0.75;font-size:0.8rem;}"
    ".pp-leyenda{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0.25rem 0 0.5rem 0;}"
    "</style>"
)


# ---------------------------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------------------------

def raiz_proyecto() -> Path:
    """Raíz de datos y resultados: POSTURA_RAIZ si está definida; si no, la del proyecto."""
    return Path(os.environ.get("POSTURA_RAIZ") or RAIZ_CODIGO).expanduser().resolve()


def rutas(raiz: Path | str | None = None) -> dict[str, Path]:
    """Diccionario con las rutas de todos los insumos que usa la aplicación."""
    r = Path(raiz) if raiz else raiz_proyecto()
    return {
        "resumen": r / "datos" / "resumen_menciones.json",
        "muestra": r / "datos" / "muestra_anotacion.csv",
        "etiquetas": r / "datos" / "etiquetas.csv",
        "fase2": r / "resultados" / "fase2_resultados.json",
        "fase3": r / "resultados" / "fase3_resultados.json",
        "agregados_corpus": r / "resultados" / "agregados_corpus.csv",
        "agregados_presidente": r / "resultados" / "agregados_por_presidente.csv",
        "figuras": r / "resultados" / "figuras",
        "modelos": Path(os.environ.get("POSTURA_MODELOS") or (r / "modelos")),
    }


# Archivos que escribe scripts/04_entrenar_clasicos.py y que carga postura.pipeline.Clasificador.
ARCHIVOS_MODELOS = ("umbrales.json","modelo_T1.joblib", "modelo_T2a.joblib", "modelo_T2b.joblib")


def insumo_disponible(clave: str, ruta: Path | str) -> bool:
    """Estado de la barra lateral. Las carpetas de modelos y figuras existen desde el inicio
    del proyecto. La de modelos cuenta como disponible solo con umbrales.json y los tres
    modelos entrenados (una copia de git trae umbrales.json sin los .joblib); la de figuras,
    con alguna imagen."""
    ruta = Path(ruta)
    if clave == "modelos":
        return all((ruta / nombre).is_file() for nombre in ARCHIVOS_MODELOS)
    if clave == "figuras":
        return bool(listar_figuras(ruta))
    return ruta.exists()


def nombre_corto(ruta: Path | str) -> str:
    """Carpeta y nombre del archivo, para los avisos (por ejemplo resultados/fase2_resultados.json)."""
    p = Path(ruta)
    return f"{p.parent.name}/{p.name}" if p.parent.name else p.name


# ---------------------------------------------------------------------------------------------
# Lectura tolerante
# ---------------------------------------------------------------------------------------------

def leer_json(ruta: Path | str):
    """Lee un JSON con un objeto en la raíz. Devuelve (datos, None) o (None, aviso)."""
    ruta = Path(ruta)
    if not ruta.is_file():
        return None, f"Pendiente: todavía no existe {nombre_corto(ruta)}."
    try:
        with open(ruta, encoding="utf-8") as fh:
            datos = json.load(fh)
    except (OSError, ValueError) as error:
        return None, f"No se pudo leer {nombre_corto(ruta)}: {error}"
    if not isinstance(datos, dict):
        return None, f"{nombre_corto(ruta)} no tiene la estructura esperada (un objeto JSON)."
    return datos, None


def leer_csv(ruta: Path | str, requeridas=(), **opciones):
    """Lee un CSV y verifica columnas. Devuelve (tabla, None) o (None, aviso)."""
    ruta = Path(ruta)
    if not ruta.is_file():
        return None, f"Pendiente: todavía no existe {nombre_corto(ruta)}."
    try:
        tabla = pd.read_csv(ruta, **opciones)
    except (OSError, ValueError) as error:
        return None, f"No se pudo leer {nombre_corto(ruta)}: {error}"
    faltan = [c for c in requeridas if c not in tabla.columns]
    if faltan:
        return None, f"A {nombre_corto(ruta)} le faltan columnas: {', '.join(faltan)}."
    return tabla, None


def listar_figuras(carpeta: Path | str) -> list[Path]:
    """Imágenes PNG de la carpeta de figuras; primero las que tienen roc en el nombre."""
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return []
    pngs = sorted(p for p in carpeta.glob("*.png") if p.is_file())
    return sorted(pngs, key=lambda p: (0 if "roc" in p.stem.lower() else 1, p.name))


# ---------------------------------------------------------------------------------------------
# Números y textos
# ---------------------------------------------------------------------------------------------

ESPACIO_FINO = "\u202f"  # espacio fino sin salto, separador de millares


def formato_miles(n) -> str:
    """Entero sin separador hasta cuatro cifras (1945) y con espacio fino de millares desde
    cinco cifras (11\u202f714), como en el informe."""
    try:
        entero = int(round(float(n)))
    except (TypeError, ValueError):
        return str(n)
    if abs(entero) < 10000:
        return str(entero)
    return f"{entero:,}".replace(",", ESPACIO_FINO)


def a_probabilidad(valor):
    """Convierte a float en [0, 1]; devuelve None si el valor falta o no es numérico."""
    try:
        x = float(valor)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return min(1.0, max(0.0, x))


NOMBRE_PARTICION = {"entrenamiento": "entrenamiento", "validacion": "validación", "prueba": "prueba"}


def nombre_particion(clave) -> str:
    """Nombre legible de la partición (las claves de los archivos van sin acento)."""
    return NOMBRE_PARTICION.get(str(clave), str(clave))


def ordenar_particiones(claves) -> list[str]:
    """Entrenamiento, validación y prueba en ese orden; otras claves al final."""
    claves = list(claves)
    return [c for c in NOMBRE_PARTICION if c in claves] + [c for c in claves if c not in NOMBRE_PARTICION]


def _sin_acentos(texto: str) -> str:
    """Texto sin diacríticos, en ASCII."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def normalizar_postura(valor) -> str:
    """Lleva la postura predicha a desfavorable, favorable, neutral o no_aplica."""
    if valor is None:
        return "no_aplica"
    t = _sin_acentos(str(valor)).strip().lower()
    if t.startswith("desfav") or t == "d":
        return "desfavorable"
    if t.startswith("fav") or t == "f":
        return "favorable"
    if t.startswith("neutr") or t == "n":
        return "neutral"
    return "no_aplica"


def limpiar_marcas(texto) -> str:
    """Quita los corchetes dobles con que la muestra marca la mención."""
    return str(texto or "").replace("[[", "").replace("]]", "")


def palabras_finales(texto, n: int) -> str:
    """Últimas n palabras de un texto (mismo recorte que al construir la muestra)."""
    return " ".join(str(texto or "").split()[-n:])


def _escapar(texto) -> str:
    """Escapa HTML y el signo de pesos para mostrar texto seguro en Markdown."""
    # html.escape evita inyectar etiquetas; el signo de pesos se escapa para que el
    # visor de Markdown no lo lea como fórmula.
    return html.escape(str(texto)).replace("$", "&#36;")


# ---------------------------------------------------------------------------------------------
# Resumen del corpus
# ---------------------------------------------------------------------------------------------

def menciones_por_grupo(resumen) -> pd.DataFrame | None:
    """Tabla larga (presidente, grupo, etiqueta, menciones) a partir de resumen_menciones.json.

    Acepta las dos orientaciones posibles del diccionario anidado: presidente y luego grupo
    (la que escribe scripts/01_construir_menciones.py) o grupo y luego presidente."""
    tabla = resumen.get("por_grupo_y_presidente") if isinstance(resumen, dict) else None
    if not isinstance(tabla, dict) or not tabla:
        return None
    externas_son_grupos = all(k in OBJETIVO for k in tabla)
    filas = []
    for externa, interno in tabla.items():
        if not isinstance(interno, dict):
            return None
        for interna, n in interno.items():
            grupo, presidente = (externa, interna) if externas_son_grupos else (interna, externa)
            filas.append({"presidente": str(presidente), "grupo": str(grupo), "menciones": n})
    df = pd.DataFrame(filas)
    df["menciones"] = pd.to_numeric(df["menciones"], errors="coerce").fillna(0).astype(int)
    df["etiqueta"] = df["grupo"].map(lambda g: ETIQUETA_GRUPO.get(g, g))
    return df


# ---------------------------------------------------------------------------------------------
# Muestra de anotación y ejemplos
# ---------------------------------------------------------------------------------------------

def ejemplo_de_fila(fila) -> dict:
    """Convierte una fila de la muestra en los campos del formulario de análisis."""
    return {
        "id": str(fila.get("id", "")),
        "parrafo": limpiar_marcas(fila.get("oracion_marcada", "")),
        "contexto": str(fila.get("contexto_previo", "") or ""),
        "pregunta": str(fila.get("pregunta_prensa", "") or ""),
        "presidente": str(fila.get("presidente", "")),
        "fecha": str(fila.get("fecha", "")),
        "mencion": str(fila.get("mencion", "")),
        "grupo": str(fila.get("grupo", "")),
        "particion": str(fila.get("particion", "")),
    }


def ejemplo_de_muestra(muestra: pd.DataFrame | None, clave: str) -> dict | None:
    """Ejemplo precargado: el ítem fijo de EJEMPLOS o, si no está, el primero del mismo grupo
    en la partición de prueba."""
    if muestra is None or muestra.empty or clave not in EJEMPLOS:
        return None
    ej = EJEMPLOS[clave]
    fila = muestra[muestra["id"].astype(str) == ej["id"]] if "id" in muestra else muestra.iloc[0:0]
    if fila.empty and "grupo" in muestra:
        candidatos = muestra[muestra["grupo"] == ej["grupo"]]
        if "particion" in muestra:
            prueba = candidatos[candidatos["particion"] == "prueba"]
            candidatos = prueba if not prueba.empty else candidatos
        fila = candidatos.head(1)
    if fila.empty:
        return None
    return ejemplo_de_fila(fila.iloc[0].to_dict())


def ejemplo_al_azar(muestra: pd.DataFrame | None, generador) -> dict | None:
    """Ítem al azar de la partición de prueba (o de toda la muestra si no hay partición)."""
    if muestra is None or muestra.empty:
        return None
    base = muestra
    if "particion" in muestra:
        prueba = muestra[muestra["particion"] == "prueba"]
        base = prueba if not prueba.empty else muestra
    i = int(generador.integers(0, len(base)))
    return ejemplo_de_fila(base.iloc[i].to_dict())


def etiqueta_referencia(etiquetas: pd.DataFrame | None, id_item: str) -> dict | None:
    """Etiqueta de referencia de un ítem si datos/etiquetas.csv ya existe y lo contiene."""
    if etiquetas is None or not id_item or "id" not in etiquetas:
        return None
    fila = etiquetas[etiquetas["id"].astype(str) == str(id_item)]
    if fila.empty:
        return None
    f = fila.iloc[0]
    try:
        es_partido = int(float(f.get("es_partido")))
    except (TypeError, ValueError):
        return None
    return {"es_partido": es_partido, "postura": normalizar_postura(f.get("postura"))}


# ---------------------------------------------------------------------------------------------
# Menciones de un párrafo (para mostrar candidatas y para el modelo de lenguaje)
# ---------------------------------------------------------------------------------------------

def menciones_del_parrafo(parrafo, contexto_previo="", pregunta_prensa="") -> list[dict]:
    """Segmenta el párrafo, busca menciones candidatas y arma un ítem por mención con el mismo
    formato de la muestra: oración marcada, contexto previo (120 palabras) y pregunta (60)."""
    base = " ".join(str(contexto_previo or "").split())
    pregunta = palabras_finales(pregunta_prensa, 60)
    items, previas = [], []
    for oracion in oraciones(parrafo):
        contexto = palabras_finales(" ".join([base] + previas), 120)
        for inicio, fin, cadena, grupo in buscar_menciones(oracion):
            items.append({
                "oracion": oracion, "oracion_marcada": marcar(oracion, inicio, fin),
                "mencion": cadena, "grupo": grupo, "objetivo": OBJETIVO.get(grupo, grupo),
                "inicio": inicio, "fin": fin, "contexto_previo": contexto,
                "pregunta_prensa": pregunta,
            })
        previas.append(oracion)
    return items


def qwen_a_filas(items: list[dict], probabilidades: list[dict]) -> list[dict]:
    """Une los ítems con las probabilidades X, D, F y N del modelo de lenguaje."""
    filas = []
    for item, p in zip(items, probabilidades):
        valores = {letra: a_probabilidad(p.get(letra)) or 0.0 for letra in "XDFN"}
        letra = max("XDFN", key=lambda l: valores[l])
        filas.append({
            **item, "p_partido": 1.0 - valores["X"], "p_D": valores["D"], "p_F": valores["F"],
            "p_N": valores["N"], "postura_predicha": LETRA_POSTURA[letra],
        })
    return filas


# ---------------------------------------------------------------------------------------------
# HTML de la tabla de resultados
# ---------------------------------------------------------------------------------------------

def resaltar_html(oracion, mencion, ocurrencia: int = 0, inicio=None, fin=None) -> str:
    """Oración escapada con la mención dentro de <mark>.

    Prioridad: marcas [[ ]] en la oración; luego las posiciones inicio y fin; luego la
    ocurrencia número k de la mención según buscar_menciones; por último una búsqueda literal."""
    texto = str(oracion or "")
    mencion = str(mencion or "")
    a = texto.find("[[")
    b = texto.find("]]", a + 2) if a >= 0 else -1
    if a >= 0 and b > a:
        dentro = texto[a + 2:b]
        return (_escapar(limpiar_marcas(texto[:a])) + "<mark>" + _escapar(dentro) + "</mark>"
                + _escapar(limpiar_marcas(texto[b + 2:])))
    try:
        i, j = int(inicio), int(fin)
        if 0 <= i < j <= len(texto) and (not mencion or texto[i:j] == mencion):
            return _escapar(texto[:i]) + "<mark>" + _escapar(texto[i:j]) + "</mark>" + _escapar(texto[j:])
    except (TypeError, ValueError):
        pass
    if mencion:
        posiciones = [(ini, fin_) for ini, fin_, cadena, _ in buscar_menciones(texto) if cadena == mencion]
        if not posiciones:
            k, desde = 0, 0
            while (k := texto.find(mencion, desde)) >= 0:
                posiciones.append((k, k + len(mencion)))
                desde = k + 1
        if posiciones:
            i, j = posiciones[min(ocurrencia, len(posiciones) - 1)]
            return _escapar(texto[:i]) + "<mark>" + _escapar(texto[i:j]) + "</mark>" + _escapar(texto[j:])
    return _escapar(texto)


def barra_html(p, color: str) -> str:
    """Barra horizontal con el valor numérico a la derecha."""
    p = a_probabilidad(p)
    if p is None:
        return '<span class="pp-na">no aplica</span>'
    return (f'<div class="pp-barra"><div class="pp-pista"><div class="pp-relleno" '
            f'style="width:{100 * p:.1f}%;background:{color}"></div></div>'
            f'<span class="pp-valor">{p:.2f}</span></div>')


def chip_postura(valor) -> str:
    """Etiqueta HTML de color para una postura."""
    clave = normalizar_postura(valor)
    return f'<span class="pp-chip pp-{clave}">{NOMBRE_POSTURA[clave]}</span>'


def leyenda_html() -> str:
    """Leyenda HTML con las cuatro posturas."""
    return '<div class="pp-leyenda">' + "".join(
        chip_postura(c) for c in ("desfavorable", "favorable", "neutral", "no_aplica")) + "</div>"


def tabla_resultados_html(filas: list[dict], columnas=COLUMNAS_CLASICO, con_postura: bool = True) -> str:
    """Tabla HTML: oración con la mención resaltada, mención y objetivo, barras de
    probabilidad y postura predicha con color. Todo el texto se escapa."""
    enc = "".join(f"<th>{_escapar(titulo)}</th>" for _, titulo, _ in columnas)
    enc = f"<tr><th>#</th><th>Oración con la mención resaltada</th><th>Mención</th>{enc}"
    enc += "<th>Postura predicha</th></tr>" if con_postura else "</tr>"
    cuerpo, vistas = [], {}
    for n, fila in enumerate(filas, start=1):
        oracion = str(fila.get("oracion_marcada") or fila.get("oracion") or "")
        mencion = str(fila.get("mencion", ""))
        clave = (limpiar_marcas(oracion), mencion)
        k = vistas.get(clave, 0)
        vistas[clave] = k + 1
        marcada = resaltar_html(oracion, mencion, k, fila.get("inicio"), fila.get("fin"))
        objetivo = fila.get("objetivo") or OBJETIVO.get(str(fila.get("grupo", "")), "")
        celdas = "".join(f"<td>{barra_html(fila.get(c), color)}</td>" for c, _, color in columnas)
        postura = f"<td>{chip_postura(fila.get('postura_predicha'))}</td>" if con_postura else ""
        cuerpo.append(
            f'<tr><td>{n}</td><td class="pp-oracion">{marcada}</td>'
            f'<td><b>{_escapar(mencion)}</b><br><span class="pp-sub">{_escapar(objetivo)}</span></td>'
            f"{celdas}{postura}</tr>")
    return (f'<div class="pp-envoltura"><table class="pp-tabla"><thead>{enc}</thead>'
            f'<tbody>{"".join(cuerpo)}</tbody></table></div>')


# ---------------------------------------------------------------------------------------------
# Resultados de ROC-AUC
# ---------------------------------------------------------------------------------------------

_TAREAS_MINUSCULAS = {t.lower(): t for t in TAREAS}


def _numero(valor):
    """Número finito o None si el valor no es numérico."""
    try:
        x = float(valor)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _hojas_auc(nodo, ruta):
    """Recorre el JSON y entrega (ruta, hoja) por cada diccionario que tiene una clave auc."""
    if isinstance(nodo, dict):
        if _numero(nodo.get("auc")) is not None:
            yield ruta, nodo
            return
        for clave, valor in nodo.items():
            yield from _hojas_auc(valor, ruta + [str(clave)])


def _tarea_de_ruta(ruta) -> str | None:
    """Tarea (T1, T2a o T2b) que aparece en una ruta del JSON de resultados."""
    tareas = [_TAREAS_MINUSCULAS[e.lower()] for e in ruta if e.lower() in _TAREAS_MINUSCULAS]
    return tareas[0] if tareas else None


def _elegidos(nodo, ruta=()):
    """Entrega (tarea, modelo) por cada tarea con clave mejor_modelo: el modelo que
    scripts/04_entrenar_clasicos.py eligió por su AUC en validación."""
    if isinstance(nodo, dict):
        tarea = _tarea_de_ruta(ruta)
        if tarea and isinstance(nodo.get("mejor_modelo"), str):
            yield tarea, nodo["mejor_modelo"]
        for clave, valor in nodo.items():
            yield from _elegidos(valor, ruta + (str(clave),))


def _intervalo(hoja):
    """Intervalo de confianza de una hoja. Acepta ic_inf e ic_sup, ic_inferior e ic_superior
    (scripts/07_fase3_evaluar.py) o los diccionarios ic_bootstrap_conferencias e ic_delong que
    escribe scripts/04_entrenar_clasicos.py; se prefiere el bootstrap por conferencias."""
    for fuente in (hoja, hoja.get("ic_bootstrap_conferencias"), hoja.get("ic_delong")):
        if not isinstance(fuente, dict):
            continue
        for a, b in (("ic_inf", "ic_sup"), ("ic_inferior", "ic_superior")):
            inf, sup = _numero(fuente.get(a)), _numero(fuente.get(b))
            if inf is not None and sup is not None:
                return inf, sup
    return None, None


# Claves contenedoras que no forman parte del nombre del modelo. prueba es el nivel en que
# scripts/04_entrenar_clasicos.py guarda las métricas de la partición de prueba.
_CONTENEDORAS = ("tareas", "resultados", "modelos", "prueba")


def resultados_auc(datos, fase: str) -> pd.DataFrame:
    """Tabla (fase, tarea, modelo, auc, ic_inf, ic_sup, elegido) a partir de un JSON de resultados.

    La tarea es el elemento de la ruta que coincide con T1, T2a o T2b (sin distinguir
    mayúsculas) y el modelo son los demás elementos, así se aceptan tanto tarea y luego modelo
    como modelo y luego tarea, con o sin una clave contenedora. El nombre del modelo se traduce
    con NOMBRE_MODELO; elegido se decide con la clave cruda y marca el modelo que el JSON declara
    como mejor_modelo de la tarea (la ablación enmascarada nunca lo es)."""
    columnas = ["fase", "tarea", "modelo", "auc", "ic_inf", "ic_sup", "elegido"]
    datos = datos if isinstance(datos, dict) else {}
    elegidos = dict(_elegidos(datos))
    filas = []
    for ruta, hoja in _hojas_auc(datos, []):
        tarea = _tarea_de_ruta(ruta)
        if not tarea:
            continue
        resto = [e for e in ruta if e.lower() not in _TAREAS_MINUSCULAS
                 and e.lower() not in _CONTENEDORAS]
        crudo = "/".join(resto)
        if crudo == "ablacion_enmascarada":
            base = elegidos.get(tarea)
            nombre = (f"{nombre_modelo(base)} sin nombre del partido (ablación)" if base
                      else "Elegido, sin nombre del partido (ablación)")
        else:
            nombre = " / ".join(nombre_modelo(e) for e in resto) if resto else "modelo"
        ic_inf, ic_sup = _intervalo(hoja)
        filas.append({
            "fase": fase, "tarea": tarea, "modelo": nombre,
            "auc": _numero(hoja.get("auc")), "ic_inf": ic_inf, "ic_sup": ic_sup,
            "elegido": bool(resto) and crudo == elegidos.get(tarea),
        })
    tabla = pd.DataFrame(filas, columns=columnas)
    # Tipos fijos: un intervalo ausente es NaN (no None), así concatenar fases no cambia dtypes.
    return tabla.astype({"auc": float, "ic_inf": float, "ic_sup": float, "elegido": bool})


def mejores_por_tarea(tabla: pd.DataFrame) -> pd.DataFrame:
    """Una fila por tarea, en el orden T1, T2a, T2b.

    Si el JSON declara el modelo elegido en validación (columna elegido), se muestra ese: elegir
    por el AUC de prueba sería seleccionar con los datos de evaluación. Si no, se toma el ROC-AUC
    más alto sin contar las ablaciones."""
    if tabla is None or tabla.empty:
        return pd.DataFrame(columns=getattr(tabla, "columns", []))
    tabla = tabla.reset_index(drop=True)  # idxmax devuelve etiquetas; deben ser únicas
    if "elegido" not in tabla:
        tabla["elegido"] = False
    filas = []
    for tarea, grupo in tabla.groupby("tarea"):
        elegidos = grupo[grupo["elegido"].fillna(False).astype(bool)]
        if not elegidos.empty:
            filas.append(elegidos.index[0])
            continue
        sin_ablacion = grupo[~grupo["modelo"].str.contains("ablación", regex=False)]
        candidatos = sin_ablacion if not sin_ablacion.empty else grupo
        filas.append(candidatos["auc"].idxmax())
    mejores = tabla.loc[filas].copy()
    mejores["orden"] = mejores["tarea"].map({t: i for i, t in enumerate(TAREAS)})
    return mejores.sort_values("orden").drop(columns="orden")


def son_sinteticos(datos) -> bool:
    """True si un JSON de resultados o de umbrales declara etiquetas_sinteticas = true
    (scripts/04_entrenar_clasicos.py lo marca cuando las etiquetas traen la columna sintetica)."""
    return isinstance(datos, dict) and datos.get("etiquetas_sinteticas") is True


# ---------------------------------------------------------------------------------------------
# Agregados sobre el corpus (predicciones del modelo)
# ---------------------------------------------------------------------------------------------

CONTEOS = ["desfavorables", "favorables", "neutrales"]
REQUERIDAS_PRESIDENTE = ["presidente", "objetivo"] + CONTEOS
REQUERIDAS_CORPUS = ["presidente", "anio", "objetivo"] + CONTEOS


def preparar_agregados(tabla: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos y recalcula los porcentajes sobre desfavorables + favorables + neutrales,
    para que las barras apiladas sumen 100 aunque el archivo traiga otro denominador."""
    d = tabla.copy()
    d["presidente"] = d["presidente"].astype(str)
    d["objetivo"] = d["objetivo"].astype(str)
    for c in CONTEOS + ["menciones", "menciones_partido_pred"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    total = d[CONTEOS].sum(axis=1)
    if "menciones_partido_pred" not in d:
        d["menciones_partido_pred"] = total
    if "menciones" not in d:
        d["menciones"] = d["menciones_partido_pred"]
    base = total.where(total > 0)
    d["pct_desfavorable"] = 100 * d["desfavorables"] / base
    d["pct_favorable"] = 100 * d["favorables"] / base
    d["pct_neutral"] = 100 * d["neutrales"] / base
    if "anio" in d:
        d["anio"] = pd.to_numeric(d["anio"], errors="coerce")
        d = d.dropna(subset=["anio"])
        d["anio"] = d["anio"].astype(int)
    return d


def agregar(tabla: pd.DataFrame, por: list[str]) -> pd.DataFrame:
    """Suma conteos por las columnas indicadas y recalcula porcentajes."""
    sumas = ["menciones", "menciones_partido_pred"] + CONTEOS
    g = tabla.groupby(por, as_index=False)[sumas].sum()
    return preparar_agregados(g) if {"presidente", "objetivo"} <= set(g.columns) else _pcts(g)


def _pcts(g: pd.DataFrame) -> pd.DataFrame:
    """Añade los porcentajes de cada postura sobre las menciones clasificadas como partido."""
    total = g[CONTEOS].sum(axis=1)
    base = total.where(total > 0)
    g = g.copy()
    g["pct_desfavorable"] = 100 * g["desfavorables"] / base
    g["pct_favorable"] = 100 * g["favorables"] / base
    g["pct_neutral"] = 100 * g["neutrales"] / base
    return g


def ordenar_objetivos(valores) -> list[str]:
    """Orden fijo de partidos y bloques; los desconocidos van al final en orden alfabético."""
    presentes = set(map(str, valores))
    return [o for o in ORDEN_OBJETIVOS if o in presentes] + sorted(presentes - set(ORDEN_OBJETIVOS))


def formato_largo_postura(tabla: pd.DataFrame) -> pd.DataFrame:
    """Pasa pct_desfavorable, pct_neutral y pct_favorable a formato largo para barras apiladas."""
    filas = []
    for _, f in tabla.iterrows():
        for postura, conteo in (("desfavorable", "desfavorables"), ("neutral", "neutrales"),
                                ("favorable", "favorables")):
            filas.append({"presidente": f["presidente"], "objetivo": f["objetivo"], "postura": postura,
                          "porcentaje": f[f"pct_{postura}"], "conteo": int(f[conteo]),
                          "total": int(f[CONTEOS].sum())})
    return pd.DataFrame(filas)
