# -*- coding: utf-8 -*-
"""Aplicación de demostración: menciones a partidos políticos en las conferencias matutinas
de la Presidencia de México (2018-2026) y postura del titular hacia el partido mencionado.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Curso: Machine Learning II, Prof. Dr. Miguel González Mendoza. Assignment 3, Fases 2 y 3.
Fecha: 2026-09-27. Licencia: MIT.

Ejecución, desde la carpeta del proyecto:
    streamlit run app/app.py

Secciones (pestañas):
  1. El problema: qué se mide, cifras del corpus y menciones candidatas por grupo.
  2. Analizar un párrafo: llama a postura.pipeline.Clasificador.analizar y, solo si el usuario
     lo pide, al modelo de lenguaje local de scripts/qwen_local.py.
  3. Cómo habla la Presidencia de los partidos: agregados de las predicciones sobre el corpus.
  4. Resultados de los modelos: ROC-AUC con intervalos por tarea y modelo, y curvas ROC.

Cada sección es tolerante: si un archivo de resultados falta o le faltan columnas, muestra un
aviso de pendiente y la aplicación sigue. Variables de entorno opcionales:
  POSTURA_RAIZ     raíz de datos/, resultados/ y modelos/ (por omisión, la carpeta del proyecto);
  POSTURA_MODELOS  carpeta de modelos entrenados (por omisión, <raíz>/modelos);
  POSTURA_QWEN     ruta del módulo del modelo de lenguaje (por omisión, scripts/qwen_local.py).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DIR_APP = Path(__file__).resolve().parent
if str(DIR_APP) not in sys.path:
    sys.path.insert(0, str(DIR_APP))

import utilidades as u  # noqa: E402

st.set_page_config(page_title="Postura hacia los partidos", layout="wide")
st.markdown(u.ESTILO_CSS, unsafe_allow_html=True)

RUTAS = u.rutas()
FUENTE_GRAFICAS = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13)


# ---------------------------------------------------------------------------------------------
# Carga con caché. La marca de tiempo del archivo entra en la llave, así un resultado nuevo se
# refleja sin reiniciar la aplicación.
# ---------------------------------------------------------------------------------------------

def _marca(ruta: Path) -> float:
    """Fecha de modificación del archivo, o -1 si no existe; entra en la llave de la caché."""
    try:
        return ruta.stat().st_mtime
    except OSError:
        return -1.0


@st.cache_data(show_spinner=False)
def _json(ruta: str, marca: float):
    """Lee un JSON con caché; devuelve (datos, aviso)."""
    return u.leer_json(ruta)


@st.cache_data(show_spinner=False)
def _csv(ruta: str, marca: float, requeridas: tuple, texto: bool):
    """Lee un CSV con caché, como texto si se pide, y verifica las columnas requeridas; devuelve (tabla, aviso)."""
    opciones = {"keep_default_na": False, "dtype": str} if texto else {}
    return u.leer_csv(ruta, requeridas, **opciones)


def cargar_json(clave: str):
    """Carga el JSON de un insumo de RUTAS; su fecha de modificación renueva la caché."""
    return _json(str(RUTAS[clave]), _marca(RUTAS[clave]))


def cargar_csv(clave: str, requeridas=(), texto: bool = False):
    """Carga el CSV de un insumo de RUTAS con sus columnas requeridas; su fecha de modificación renueva la caché."""
    return _csv(str(RUTAS[clave]), _marca(RUTAS[clave]), tuple(requeridas), texto)


def cargar_muestra():
    """Carga la muestra de anotación como texto, con las columnas que usan los ejemplos."""
    return cargar_csv("muestra", ("id", "oracion_marcada", "contexto_previo", "pregunta_prensa"), texto=True)


# ---------------------------------------------------------------------------------------------
# Modelos: el clasificador clásico y, solo a petición, el modelo de lenguaje local.
# ---------------------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def cargar_clasificador(dir_modelos: str):
    """Clasificador de la Fase 2. Una excepción no se guarda en caché: si faltan los modelos,
    el siguiente intento vuelve a buscarlos."""
    from postura.pipeline import Clasificador
    return Clasificador(dir_modelos)


def _importar_qwen():
    """Importa el módulo del modelo de lenguaje desde POSTURA_QWEN o scripts/qwen_local.py."""
    ruta = Path(os.environ.get("POSTURA_QWEN") or (u.RAIZ_CODIGO / "scripts" / "qwen_local.py"))
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe {ruta}")
    spec = importlib.util.spec_from_file_location("qwen_local", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@st.cache_resource(show_spinner=False)
def motor_qwen():
    """Carga el modelo de lenguaje en un hilo propio y persistente. MLX asocia sus flujos de
    cómputo al hilo que los crea, y Streamlit ejecuta cada interacción en otro hilo, así que
    la carga y todas las consultas pasan por este único hilo."""
    ejecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen")
    try:
        modelo = ejecutor.submit(lambda: _importar_qwen().Qwen()).result()
    except BaseException:
        ejecutor.shutdown(wait=False)
        raise
    return ejecutor, modelo


def puntuar_qwen(items: list[dict]) -> list[dict]:
    """Puntúa los ítems con el modelo de lenguaje, sin ejemplos y en su hilo propio; devuelve las probabilidades de X, D, F y N."""
    ejecutor, modelo = motor_qwen()
    return ejecutor.submit(modelo.puntuar, items).result()


# ---------------------------------------------------------------------------------------------
# Auxiliares de presentación
# ---------------------------------------------------------------------------------------------

def detalle_tecnico(texto: str | None):
    """Muestra un texto técnico, como una traza de error, en un desplegable si no está vacío."""
    if texto:
        with st.expander("Detalle técnico"):
            st.code(texto, language=None)


def ejecutar_seccion(funcion):
    """Aísla cada pestaña: un error inesperado se informa sin tumbar las demás."""
    try:
        funcion()
    except Exception:  # noqa: BLE001
        st.error("Esta sección no pudo mostrarse con los archivos actuales.")
        detalle_tecnico(traceback.format_exc())


def estilo_figura(fig, alto: int, leyenda: bool = True, categorias: int | None = None):
    """Estilo común. La leyenda va anclada al borde superior del contenedor para no chocar con
    los títulos de las facetas. Con ejes de categorías se fija el rango de forma explícita:
    Streamlit dibuja las pestañas ocultas con tamaño cero y ahí Plotly calcula mal el
    autorango de un eje de categorías, dejando fuera parte de ellas."""
    fig.update_layout(
        height=alto, margin=dict(l=10, r=10, t=64 if leyenda else 30, b=10), font=FUENTE_GRAFICAS,
        showlegend=leyenda, hoverlabel=dict(font_size=13),
    )
    if leyenda:
        fig.update_layout(legend=dict(orientation="h", yref="container", y=0.995, yanchor="top",
                                      xanchor="left", x=0, title_text=""))
    if categorias:
        fig.update_yaxes(range=[-0.5, categorias - 0.5], autorange=False)
    fig.update_xaxes(showgrid=True, gridwidth=1, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


# ---------------------------------------------------------------------------------------------
# 1. El problema
# ---------------------------------------------------------------------------------------------

def seccion_problema():
    """Pestaña 1: el problema, las cifras del corpus y las menciones candidatas por grupo y presidente."""
    st.header("El problema")
    st.markdown(
        "Se mide cómo nombra el titular del Ejecutivo a los partidos políticos en sus conferencias "
        "matutinas: cada vez que aparece una mención candidata (siglas, nombres, gentilicios o "
        "etiquetas de bloque como PRIAN, conservadores u oposición).  \n"
        "Hay dos tareas: la **Tarea 1** decide si la mención se refiere de verdad a un partido "
        "(pan, Verde u oposición también tienen otros sentidos) y la **Tarea 2** clasifica la postura "
        "hacia ese partido, con un clasificador para desfavorable y otro para favorable, cada uno "
        "frente al resto.  \n"
        "Todo se evalúa con **ROC-AUC** sobre una partición de prueba separada por conferencia, "
        "así ninguna conferencia aparece a la vez en entrenamiento y en prueba."
    )
    resumen, aviso = cargar_json("resumen")
    menciones = u.MENCIONES_TOTALES
    if resumen:
        menciones = resumen.get("menciones_totales", menciones)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conferencias en el corpus", u.formato_miles(u.CONFERENCIAS["total"]))
    c1.caption(f"López Obrador {u.formato_miles(u.CONFERENCIAS['AMLO'])}, "
               f"Sheinbaum {u.formato_miles(u.CONFERENCIAS['Sheinbaum'])}")
    c2.metric("Menciones candidatas", u.formato_miles(menciones))
    c2.caption("Habla en vivo del titular, 2018-2026")
    if resumen:
        c3.metric("Ítems en la muestra de anotación", u.formato_miles(resumen.get("muestra", 0)))
        part = resumen.get("muestra_por_particion", {}) or {}
        c3.caption(", ".join(f"{u.nombre_particion(k)} {u.formato_miles(part[k])}"
                             for k in u.ordenar_particiones(part)))
        conf = resumen.get("conferencias_por_particion", {}) or {}
        if conf:
            c4.metric("Conferencias con menciones", u.formato_miles(sum(conf.values())))
            c4.caption("Base del reparto en entrenamiento, validación y prueba")

    st.subheader("Menciones candidatas por grupo y presidente")
    if aviso:
        st.warning(f"{aviso} La gráfica de menciones por grupo queda pendiente.")
        return
    tabla = u.menciones_por_grupo(resumen)
    if tabla is None or tabla.empty:
        st.warning("resumen_menciones.json no trae la clave por_grupo_y_presidente; la gráfica queda pendiente.")
        return
    orden = tabla.groupby("etiqueta")["menciones"].sum().sort_values().index.tolist()
    presidentes = [p for p in u.COLOR_PRESIDENTE if p in set(tabla["presidente"])] + \
        sorted(set(tabla["presidente"]) - set(u.COLOR_PRESIDENTE))
    tabla = tabla.assign(menciones_texto=tabla["menciones"].map(u.formato_miles))
    fig = px.bar(
        tabla, x="menciones", y="etiqueta", color="presidente", orientation="h", barmode="group",
        category_orders={"etiqueta": orden[::-1], "presidente": presidentes[::-1]},
        color_discrete_map=u.COLOR_PRESIDENTE,
        labels={"menciones": "Menciones candidatas", "etiqueta": "", "presidente": "Presidente"},
        custom_data=["presidente", "menciones_texto"],
    )
    fig.update_traces(hovertemplate="%{y}, %{customdata[0]}: %{customdata[1]} menciones<extra></extra>",
                      marker_line_width=0)
    fig.update_layout(bargap=0.25, bargroupgap=0.08, legend_traceorder="reversed")
    st.plotly_chart(estilo_figura(fig, 110 + 34 * len(orden), categorias=len(orden)),
                    use_container_width=True, key="fig_grupos")
    st.caption("Los grupos ambiguos (pan, Verde y morena en minúscula) y las etiquetas de bloque son "
               "justo los casos que la Tarea 1 debe separar. Pase el cursor sobre una barra para ver el conteo.")
    with st.expander("Ver la tabla"):
        ancha = tabla.pivot_table(index="etiqueta", columns="presidente", values="menciones",
                                  aggfunc="sum", fill_value=0)
        ancha["Total"] = ancha.sum(axis=1)
        st.dataframe(ancha.sort_values("Total", ascending=False).rename_axis("Grupo"),
                     use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(str(c), format="%d") for c in ancha.columns})


# ---------------------------------------------------------------------------------------------
# 2. Analizar un párrafo
# ---------------------------------------------------------------------------------------------

OPCIONES_EJEMPLO = list(u.EJEMPLOS) + ["azar", "libre"]
NOMBRE_OPCION = {**{k: v["etiqueta"] for k, v in u.EJEMPLOS.items()},
                 "azar": "Ítem al azar de la partición de prueba", "libre": "Texto libre"}


def _poner_en_estado(ejemplo: dict | None):
    """Copia el párrafo, el contexto y la pregunta de un ejemplo al estado de la sesión, o los vacía si no hay ejemplo."""
    ejemplo = ejemplo or {}
    st.session_state["parrafo"] = ejemplo.get("parrafo", "")
    st.session_state["contexto"] = ejemplo.get("contexto", "")
    st.session_state["pregunta"] = ejemplo.get("pregunta", "")
    st.session_state["ejemplo_cargado"] = ejemplo or None


def al_cambiar_ejemplo(muestra):
    """Carga en la sesión la opción elegida: un ejemplo precargado, un ítem al azar o texto libre vacío."""
    opcion = st.session_state.get("ejemplo")
    if opcion in u.EJEMPLOS:
        _poner_en_estado(u.ejemplo_de_muestra(muestra, opcion))
    elif opcion == "azar":
        al_pedir_azar(muestra)
    else:
        _poner_en_estado(None)


def al_pedir_azar(muestra):
    """Carga en la sesión un ítem al azar de la partición de prueba."""
    generador = np.random.default_rng()
    _poner_en_estado(u.ejemplo_al_azar(muestra, generador))


def seccion_analizar():
    """Pestaña 2: análisis de un párrafo con el clasificador clásico y, si se pide, con el modelo de lenguaje."""
    st.header("Analizar un párrafo")
    st.markdown("Escriba o pegue un párrafo dicho por el titular del Ejecutivo. La herramienta localiza "
                "las menciones candidatas, estima si cada una se refiere a un partido y la postura hacia él.")
    muestra, aviso = cargar_muestra()
    if aviso:
        st.warning(f"{aviso} Los ejemplos precargados no están disponibles; puede escribir un texto libre.")
    if "parrafo" not in st.session_state:
        inicial = "PRIAN" if muestra is not None else "libre"
        st.session_state["ejemplo"] = inicial
        _poner_en_estado(u.ejemplo_de_muestra(muestra, inicial) if muestra is not None else None)

    opciones = OPCIONES_EJEMPLO if muestra is not None else ["libre"]
    if st.session_state.get("ejemplo") not in opciones:
        st.session_state["ejemplo"] = "libre"
    st.radio("Ejemplo", opciones, key="ejemplo", horizontal=True, format_func=NOMBRE_OPCION.get,
             on_change=al_cambiar_ejemplo, args=(muestra,))
    if st.session_state.get("ejemplo") == "azar":
        st.button("Otro ítem al azar", on_click=al_pedir_azar, args=(muestra,), key="otro_azar")
    cargado = st.session_state.get("ejemplo_cargado")
    if cargado:
        st.caption(f"Ítem {cargado['id']} de la muestra de anotación: {cargado['presidente']}, "
                   f"{cargado['fecha']}, partición de {u.nombre_particion(cargado['particion'])}. Mención marcada en la "
                   f"muestra: {cargado['mencion']}.")

    st.text_area("Párrafo", key="parrafo", height=130)
    c1, c2 = st.columns(2)
    c1.text_area("Contexto previo del titular (opcional)", key="contexto", height=110)
    c2.text_area("Pregunta de la prensa (opcional)", key="pregunta", height=110)
    comparar = st.checkbox("Comparar con el modelo de lenguaje local (Qwen3-32B cuantizado a 4 bits)",
                           key="comparar_llm")
    if comparar:
        st.warning("El modelo de lenguaje se carga solo al pulsar Analizar. Con el modelo ya descargado "
                   "(unos 17 GB en disco), la carga tarda unos segundos y la primera consulta alrededor "
                   "de medio minuto; las siguientes, unos segundos por mención. Requiere una Mac con chip "
                   "Apple y la biblioteca MLX.")
    if st.button("Analizar", type="primary", key="analizar"):
        analizar(st.session_state["parrafo"], st.session_state["contexto"],
                 st.session_state["pregunta"], comparar)
    mostrar_resultado()


def analizar(parrafo: str, contexto: str, pregunta: str, comparar: bool):
    """Ejecuta los modelos y guarda el resultado en la sesión, para que sobreviva a los
    reinicios del guion que provocan otros controles."""
    if not str(parrafo).strip():
        st.session_state["resultado"] = None
        st.warning("Escriba un párrafo para analizar.")
        return
    res = {"entrada": (parrafo, contexto, pregunta),
           "candidatas": u.menciones_del_parrafo(parrafo, contexto, pregunta),
           "clasico": None, "aviso_clasico": None, "detalle_clasico": None,
           "qwen": None, "aviso_qwen": None, "detalle_qwen": None, "comparar": comparar,
           "ejemplo": st.session_state.get("ejemplo_cargado")}
    with st.spinner("Analizando con el modelo clásico..."):
        try:
            clasificador = cargar_clasificador(str(RUTAS["modelos"]))
        except ModuleNotFoundError as error:
            if error.name in ("postura.pipeline", "postura"):
                res["aviso_clasico"] = ("Pendiente: el módulo postura/pipeline.py todavía no está "
                                        "disponible, así que no hay predicciones del modelo clásico.")
            else:
                res["aviso_clasico"] = f"Falta una dependencia del clasificador: {error.name}."
            res["detalle_clasico"] = traceback.format_exc()
        except Exception:  # noqa: BLE001
            res["aviso_clasico"] = (f"Pendiente: los modelos entrenados de la Fase 2 no están disponibles "
                                    f"en {u.nombre_corto(RUTAS['modelos'])}.")
            res["detalle_clasico"] = traceback.format_exc()
        else:
            res["sinteticas"] = u.son_sinteticos(getattr(clasificador, "metadatos", None))
            try:
                res["clasico"] = list(clasificador.analizar(parrafo, contexto_previo=contexto,
                                                            pregunta_prensa=pregunta))
            except Exception:  # noqa: BLE001
                res["aviso_clasico"] = "El clasificador no pudo analizar este párrafo."
                res["detalle_clasico"] = traceback.format_exc()
    if comparar:
        if not res["candidatas"]:
            res["aviso_qwen"] = "No hay menciones candidatas que enviar al modelo de lenguaje."
        else:
            with st.spinner("Cargando y consultando el modelo de lenguaje local. La primera consulta tarda "
                            "alrededor de medio minuto; las siguientes, unos segundos por mención..."):
                try:
                    probabilidades = puntuar_qwen(res["candidatas"])
                    res["qwen"] = u.qwen_a_filas(res["candidatas"], probabilidades)
                except Exception:  # noqa: BLE001
                    res["aviso_qwen"] = "No se pudo cargar o consultar el modelo de lenguaje local."
                    res["detalle_qwen"] = traceback.format_exc()
    st.session_state["resultado"] = res


def mostrar_resultado():
    """Muestra el último análisis guardado en la sesión: modelo clásico, etiqueta de referencia y, si se pidió, modelo de lenguaje."""
    res = st.session_state.get("resultado")
    if not res:
        st.caption("Pulse Analizar para ver las menciones detectadas y las probabilidades.")
        return
    actual = (st.session_state.get("parrafo"), st.session_state.get("contexto"), st.session_state.get("pregunta"))
    if tuple(res["entrada"]) != actual:
        st.info("El texto cambió desde el último análisis; pulse Analizar para actualizar los resultados.")

    st.subheader("Modelo clásico (Fase 2)")
    if res.get("sinteticas"):
        st.warning("Estos modelos se entrenaron con etiquetas sintéticas: las probabilidades solo "
                   "prueban el código.")
    if res["clasico"] is not None:
        if not res["clasico"]:
            st.info("No se encontraron menciones candidatas a partidos en el párrafo.")
        else:
            st.markdown(u.leyenda_html(), unsafe_allow_html=True)
            st.markdown(u.tabla_resultados_html(res["clasico"], u.COLUMNAS_CLASICO), unsafe_allow_html=True)
            st.caption("P(partido) viene del clasificador de la Tarea 1. P(desfavorable) y P(favorable) "
                       "vienen de los dos clasificadores de la Tarea 2 y solo se interpretan cuando la "
                       "mención sí es a un partido; por eso no tienen que sumar 1.")
    else:
        st.warning(res["aviso_clasico"])
        detalle_tecnico(res["detalle_clasico"])
        if res["candidatas"]:
            st.markdown("Menciones candidatas que localizan las reglas, todavía sin clasificar:")
            st.markdown(u.tabla_resultados_html(res["candidatas"], [], con_postura=False),
                        unsafe_allow_html=True)
        else:
            st.info("Las reglas no encontraron menciones candidatas a partidos en el párrafo.")

    mostrar_etiqueta_referencia(res)

    if res["comparar"]:
        st.subheader("Modelo de lenguaje local (Fase 3, sin ejemplos)")
        if res["qwen"]:
            st.markdown(u.tabla_resultados_html(res["qwen"], u.COLUMNAS_LLM), unsafe_allow_html=True)
            st.caption("Probabilidad que el modelo asigna a la primera letra de su respuesta: X no es "
                       "partido, D desfavorable, F favorable, N neutral. La postura predicha es la letra "
                       "más probable.")
        else:
            st.warning(res["aviso_qwen"])
            detalle_tecnico(res["detalle_qwen"])


def mostrar_etiqueta_referencia(res):
    """Si el párrafo es un ítem intacto de la muestra y ya existe datos/etiquetas.csv, muestra la
    etiqueta de referencia de la mención marcada, para contrastarla con la predicción."""
    ejemplo = res.get("ejemplo")
    if not ejemplo or res["entrada"][0] != ejemplo.get("parrafo"):
        return
    etiquetas, _ = cargar_csv("etiquetas", ("id", "es_partido", "postura"), texto=True)
    referencia = u.etiqueta_referencia(etiquetas, ejemplo.get("id"))
    if referencia:
        postura = u.NOMBRE_POSTURA[referencia["postura"]] if referencia["es_partido"] else "no aplica"
        st.markdown(f"Etiqueta de referencia de la mención marcada en la muestra ({ejemplo.get('mencion')}): "
                    f"es partido = {'sí' if referencia['es_partido'] else 'no'}; postura = {postura}.")


# ---------------------------------------------------------------------------------------------
# 3. Cómo habla la Presidencia de los partidos
# ---------------------------------------------------------------------------------------------

def seccion_corpus():
    """Pestaña 3: postura predicha sobre el corpus por partido y presidente, evolución anual y tabla."""
    st.header("Cómo habla la Presidencia de los partidos")
    st.info("Son predicciones del modelo sobre todo el corpus, no conteos de la muestra anotada. Los "
            "porcentajes se calculan sobre las menciones que el modelo clasifica como partido.")
    corpus, aviso_corpus = cargar_csv("agregados_corpus", u.REQUERIDAS_CORPUS)
    por_pres, aviso_pres = cargar_csv("agregados_presidente", u.REQUERIDAS_PRESIDENTE)
    corpus = u.preparar_agregados(corpus) if corpus is not None else None
    if por_pres is not None:
        por_pres = u.preparar_agregados(por_pres)
    elif corpus is not None:
        por_pres = u.agregar(corpus, ["presidente", "objetivo"])
        st.caption("agregados_por_presidente.csv no está disponible; los totales por presidente se "
                   "suman desde agregados_corpus.csv.")
    if por_pres is None:
        st.warning(f"{aviso_pres} {aviso_corpus} Esta sección queda pendiente hasta que se clasifique el corpus.")
        return

    objetivos = u.ordenar_objetivos(por_pres["objetivo"])
    presidentes = [p for p in u.COLOR_PRESIDENTE if p in set(por_pres["presidente"])] + \
        sorted(set(por_pres["presidente"]) - set(u.COLOR_PRESIDENTE))
    c1, c2 = st.columns([3, 2])
    sel_obj = c1.multiselect("Partidos o bloques", objetivos, default=objetivos, key="sel_objetivos")
    sel_pres = c2.multiselect("Presidente", presidentes, default=presidentes, key="sel_presidentes")
    if not sel_obj or not sel_pres:
        st.info("Elija al menos un partido y un presidente.")
        return

    filtro = por_pres[por_pres["objetivo"].isin(sel_obj) & por_pres["presidente"].isin(sel_pres)]
    st.subheader("Postura predicha por partido y presidente")
    largo = u.formato_largo_postura(filtro.dropna(subset=["pct_desfavorable"]))
    if not largo.empty:
        largo = largo.assign(conteo_texto=largo["conteo"].map(u.formato_miles),
                             total_texto=largo["total"].map(u.formato_miles))
    if largo.empty:
        st.info("No hay menciones clasificadas como partido para esa selección.")
    else:
        orden_obj = [o for o in objetivos if o in sel_obj]
        fig = px.bar(
            largo, x="porcentaje", y="objetivo", color="postura", orientation="h",
            facet_col="presidente", facet_col_spacing=0.06, barmode="stack",
            category_orders={"postura": u.ORDEN_POSTURA, "objetivo": orden_obj,
                             "presidente": [p for p in presidentes if p in sel_pres]},
            color_discrete_map={k: u.COLOR_POSTURA[k] for k in u.ORDEN_POSTURA},
            labels={"porcentaje": "% de las menciones a partidos", "objetivo": "", "postura": "Postura"},
            custom_data=["postura", "conteo_texto", "total_texto", "presidente"],
        )
        # Plotly Express da a cada color su propio offsetgroup y eso impide apilar.
        fig.update_traces(offsetgroup="postura", marker_line_width=0, hovertemplate=(
            "%{y}, %{customdata[3]}<br>%{customdata[0]}: %{x:.1f} %"
            "<br>%{customdata[1]} de %{customdata[2]} menciones a partidos<extra></extra>"))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_xaxes(range=[0, 100], ticksuffix=" %")
        fig.update_layout(bargap=0.3)
        st.plotly_chart(estilo_figura(fig, 130 + 38 * len(orden_obj), categorias=len(orden_obj)),
                        use_container_width=True, key="fig_postura")

    st.subheader("Evolución por año")
    if corpus is None:
        st.warning(f"{aviso_corpus} La serie anual queda pendiente.")
    else:
        anual = corpus[corpus["objetivo"].isin(sel_obj) & corpus["presidente"].isin(sel_pres)]
        if anual.empty:
            st.info("No hay datos anuales para esa selección.")
        else:
            anual = u.agregar(anual, ["presidente", "anio"])
            anual = anual.assign(menciones_texto=anual["menciones_partido_pred"].map(u.formato_miles),
                                 desfavorables_texto=anual["desfavorables"].map(u.formato_miles))
            nombre = sel_obj[0] if len(sel_obj) == 1 else f"{len(sel_obj)} partidos o bloques seleccionados"
            st.caption(f"Suma de {nombre}.")
            g1, g2 = st.columns(2)
            orden_pres = {"presidente": [p for p in presidentes if p in sel_pres]}
            f1 = px.line(anual, x="anio", y="menciones_partido_pred", color="presidente", markers=True,
                         category_orders=orden_pres, color_discrete_map=u.COLOR_PRESIDENTE,
                         labels={"anio": "Año", "menciones_partido_pred": "Menciones a partidos",
                                 "presidente": "Presidente"},
                         custom_data=["menciones_texto"])
            f1.update_traces(line_width=2, marker_size=8,
                             hovertemplate="%{x}: %{customdata[0]} menciones<extra>%{fullData.name}</extra>")
            f1.update_xaxes(dtick=1)
            f1.update_yaxes(rangemode="tozero")
            g1.markdown("**Menciones clasificadas como partido**")
            g1.plotly_chart(estilo_figura(f1, 340), use_container_width=True, key="fig_anual_menciones")
            f2 = px.line(anual, x="anio", y="pct_desfavorable", color="presidente", markers=True,
                         category_orders=orden_pres, color_discrete_map=u.COLOR_PRESIDENTE,
                         labels={"anio": "Año", "pct_desfavorable": "% desfavorable",
                                 "presidente": "Presidente"},
                         custom_data=["desfavorables_texto", "menciones_texto"])
            f2.update_traces(line_width=2, marker_size=8, hovertemplate=(
                "%{x}: %{y:.1f} %<br>%{customdata[0]} de %{customdata[1]} menciones<extra>%{fullData.name}</extra>"))
            f2.update_xaxes(dtick=1)
            f2.update_yaxes(rangemode="tozero", ticksuffix=" %")
            g2.markdown("**Porcentaje desfavorable**")
            g2.plotly_chart(estilo_figura(f2, 340), use_container_width=True, key="fig_anual_pct")

    st.subheader("Tabla por presidente")
    tabla = filtro.copy()
    tabla["_orden"] = tabla["objetivo"].map({o: i for i, o in enumerate(objetivos)})
    tabla = tabla.sort_values(["presidente", "_orden"])
    tabla = tabla[["presidente", "objetivo", "menciones", "menciones_partido_pred", "desfavorables",
                   "favorables", "neutrales", "pct_desfavorable", "pct_favorable", "pct_neutral"]]
    st.dataframe(
        tabla, hide_index=True, use_container_width=True,
        column_config={
            "presidente": "Presidente", "objetivo": "Partido o bloque",
            "menciones": st.column_config.NumberColumn("Menciones candidatas", format="%d"),
            "menciones_partido_pred": st.column_config.NumberColumn("Clasificadas como partido", format="%d"),
            "desfavorables": st.column_config.NumberColumn("Desfavorables", format="%d"),
            "favorables": st.column_config.NumberColumn("Favorables", format="%d"),
            "neutrales": st.column_config.NumberColumn("Neutrales", format="%d"),
            "pct_desfavorable": st.column_config.NumberColumn("% desfavorable", format="%.1f"),
            "pct_favorable": st.column_config.NumberColumn("% favorable", format="%.1f"),
            "pct_neutral": st.column_config.NumberColumn("% neutral", format="%.1f"),
        },
    )


# ---------------------------------------------------------------------------------------------
# 4. Resultados de los modelos
# ---------------------------------------------------------------------------------------------

def seccion_resultados():
    """Pestaña 4: ROC-AUC con intervalos por tarea y modelo de las Fases 2 y 3, curvas ROC y los JSON completos."""
    st.header("Resultados de los modelos")
    st.markdown("ROC-AUC por tarea y modelo con su intervalo de confianza. Un valor de 0.5 equivale a "
                "clasificar al azar y 1 a separar perfectamente las dos clases.")
    st.markdown("  \n".join(f"**{t}.** {d}" for t, d in u.DESCRIPCION_TAREAS.items()))

    tablas, crudos = [], {}
    for fase, clave, severo in (("Fase 2", "fase2", True), ("Fase 3", "fase3", False)):
        datos, aviso = cargar_json(clave)
        if datos is None:
            texto = f"{aviso} Los resultados de la {fase} están pendientes."
            if severo:
                st.warning(texto)
            else:
                st.info(texto)
            continue
        crudos[fase] = datos
        if u.son_sinteticos(datos):
            st.warning(f"{u.nombre_corto(RUTAS[clave])} se obtuvo con etiquetas sintéticas: estas cifras "
                       "solo prueban el código y no dicen nada sobre las conferencias.")
        tabla = u.resultados_auc(datos, fase)
        if tabla.empty:
            st.warning(f"{u.nombre_corto(RUTAS[clave])} no contiene valores de ROC-AUC por tarea "
                       "(claves T1, T2a o T2b con auc).")
        else:
            tablas.append(tabla)

    if tablas:
        todo = pd.concat(tablas, ignore_index=True)
        mejores = u.mejores_por_tarea(todo)
        columnas = st.columns(max(1, len(mejores)))
        for col, (_, f) in zip(columnas, mejores.iterrows()):
            elegido = bool(f.get("elegido", False))
            titulo = u.TAREAS.get(f["tarea"], f["tarea"])
            col.metric(titulo if elegido else f"Mejor en {titulo}", f"{f['auc']:.3f}")
            intervalo = (f"IC {f['ic_inf']:.3f} a {f['ic_sup']:.3f}"
                         if pd.notna(f["ic_inf"]) and pd.notna(f["ic_sup"]) else "sin intervalo")
            origen = "elegido por su AUC en validación" if elegido else "mayor AUC en prueba"
            col.caption(f"{f['modelo']} ({f['fase']}, {origen}); {intervalo}")

        grafica = todo.copy()
        grafica["error_sup"] = (grafica["ic_sup"] - grafica["auc"]).clip(lower=0).fillna(0)
        grafica["error_inf"] = (grafica["auc"] - grafica["ic_inf"]).clip(lower=0).fillna(0)
        grafica["etiqueta_tarea"] = grafica["tarea"].map(u.TAREAS_CORTAS)
        orden_modelos = (grafica.groupby("modelo")["auc"].mean().sort_values().index.tolist())[::-1]
        minimo = float(np.nanmin(grafica[["auc", "ic_inf"]].to_numpy(dtype=float)))
        fig = px.scatter(
            grafica, x="auc", y="modelo", color="fase", facet_col="etiqueta_tarea",
            error_x="error_sup", error_x_minus="error_inf",
            category_orders={"etiqueta_tarea": [u.TAREAS_CORTAS[t] for t in u.TAREAS],
                             "modelo": orden_modelos, "fase": list(u.COLOR_FASE)},
            color_discrete_map=u.COLOR_FASE, labels={"auc": "ROC-AUC", "modelo": "", "fase": "Fase"},
            facet_col_spacing=0.04,
            custom_data=["ic_inf", "ic_sup", "fase"],
        )
        fig.update_traces(marker_size=10, hovertemplate=(
            "%{y} (%{customdata[2]})<br>ROC-AUC %{x:.3f}<br>IC %{customdata[0]:.3f} a %{customdata[1]:.3f}<extra></extra>"))
        fig.add_vline(x=0.5, line_width=1, line_dash="dot", line_color="#898781")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_xaxes(range=[min(0.45, minimo - 0.03), 1.0])
        st.plotly_chart(estilo_figura(fig, 140 + 42 * len(orden_modelos), categorias=len(orden_modelos)),
                        use_container_width=True, key="fig_auc")
        st.caption("Cada punto es un modelo y la barra horizontal su intervalo de confianza. La línea "
                   "punteada marca 0.5, el nivel del azar.")
        orden_tarea = {t: i for i, t in enumerate(u.TAREAS)}
        vista = todo.assign(_o=todo["tarea"].map(orden_tarea)).sort_values(["_o", "auc"], ascending=[True, False])
        st.dataframe(
            vista[["fase", "tarea", "modelo", "auc", "ic_inf", "ic_sup"]], hide_index=True,
            use_container_width=True,
            column_config={
                "fase": "Fase", "tarea": "Tarea", "modelo": "Modelo",
                "auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
                "ic_inf": st.column_config.NumberColumn("IC inferior", format="%.3f"),
                "ic_sup": st.column_config.NumberColumn("IC superior", format="%.3f"),
            },
        )

    st.subheader("Curvas ROC y otras figuras")
    figuras = u.listar_figuras(RUTAS["figuras"])
    if not figuras:
        st.info(f"Pendiente: no hay imágenes PNG en {u.nombre_corto(RUTAS['figuras'])}.")
    else:
        columnas = st.columns(2)
        for i, ruta in enumerate(figuras):
            columnas[i % 2].image(str(ruta), caption=u.nombre_figura(ruta.stem), use_container_width=True)

    for fase, datos in crudos.items():
        with st.expander(f"Archivo completo de la {fase}"):
            st.json(datos, expanded=False)


# ---------------------------------------------------------------------------------------------
# Barra lateral y armado de la página
# ---------------------------------------------------------------------------------------------

def barra_lateral():
    """Barra lateral con los datos del trabajo y el estado de los insumos."""
    with st.sidebar:
        st.markdown("**Postura hacia los partidos en las conferencias matutinas**")
        st.caption("Machine Learning II, Universidad Panamericana. Assignment 3: herramienta clásica "
                   "(Fase 2) y comparación con un modelo de lenguaje local (Fase 3).")
        st.caption("Enrique Alberto Mendoza Ruiz")
        st.markdown("**Estado de los insumos**")
        insumos = [("Resumen de menciones", "resumen"), ("Muestra de anotación", "muestra"),
                   ("Etiquetas de referencia", "etiquetas"), ("Modelos entrenados", "modelos"),
                   ("Resultados Fase 2", "fase2"), ("Resultados Fase 3", "fase3"),
                   ("Agregados del corpus", "agregados_corpus"),
                   ("Agregados por presidente", "agregados_presidente"), ("Figuras", "figuras")]
        lineas = [f"{nombre}: {'disponible' if u.insumo_disponible(clave, RUTAS[clave]) else 'pendiente'}"
                  for nombre, clave in insumos]
        st.markdown("  \n".join(lineas))
        if os.environ.get("POSTURA_RAIZ"):
            st.caption(f"Raíz de datos: {u.raiz_proyecto()}")


def principal():
    """Arma la página: barra lateral, título y las cuatro pestañas, cada una aislada ante errores."""
    barra_lateral()
    st.title("Menciones a partidos y postura de la Presidencia en las conferencias matutinas")
    st.caption("Conferencias matutinas de la Presidencia de México, 2018-2026. Demostración de las "
               "Fases 2 y 3 del Assignment 3 de Machine Learning II.")
    pestanas = st.tabs(["El problema", "Analizar un párrafo", "Cómo habla la Presidencia de los partidos",
                        "Resultados de los modelos"])
    for pestana, seccion in zip(pestanas, (seccion_problema, seccion_analizar, seccion_corpus,
                                           seccion_resultados)):
        with pestana:
            ejecutar_seccion(seccion)


principal()
