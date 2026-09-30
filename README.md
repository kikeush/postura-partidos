# postura-partidos

Herramienta para detectar menciones a partidos políticos mexicanos en transcripciones de discurso oral en español y estimar la postura de quien habla hacia el partido mencionado.

- **Autor:** Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
- **Estado:** herramienta en desarrollo. Este repositorio publica solo el código; los datos, las anotaciones, los resultados y los análisis forman parte de una investigación en curso y no se incluyen.

## Qué hace

Cada mención candidata es una oración con un nombre, sigla, gentilicio o etiqueta de partido (Morena, PRI, PAN, PRD, PVEM, PT, MC, PRIAN, conservadores, oposición). El paquete resuelve tres tareas binarias:

| Tarea | Pregunta |
|---|---|
| T1 | ¿La mención se refiere a un partido o bloque como actor político? |
| T2a | ¿La postura hacia el partido es desfavorable? |
| T2b | ¿La postura hacia el partido es favorable? |

Incluye extracción de menciones, rasgos léxicos, clasificadores clásicos (regresión logística, SVM lineal, Random Forest), ajuste fino de BETO (BERT en español, `dccuchile/bert-base-spanish-wwm-cased`), un cliente para un modelo de lenguaje local sobre MLX, evaluación con ROC-AUC (remuestreo por conferencia y prueba de DeLong) y una aplicación de demostración en Streamlit.

## Estructura

```
postura/   paquete: menciones, léxico, rasgos, modelos, evaluación, partición, pipeline
scripts/   pasos numerados del flujo (cada uno acepta --help)
app/       aplicación de demostración en Streamlit
tests/     pruebas unitarias y de integración (pytest)
modelos/   umbrales de decisión (los clasificadores entrenados se publican como adjuntos de la versión)
datos/     vacía: se llena con tus propios datos
```

## Instalación

Probado con Python 3.13 en macOS con Apple Silicon.

```bash
git clone https://github.com/kikeush/postura-partidos
cd postura-partidos
pip install -r requirements.txt            # herramienta y fase clásica
pip install -r requirements-fase3.txt      # BETO y modelo local (opcional)
```

## Uso con tus datos

El paso 1 lee un corpus propio desde la carpeta que indique la variable `CORPUS_MANANERAS`: un `manifest.csv` con una fila por conferencia y un CSV por conferencia con las columnas `Rol`, `Texto`, `Turno` y `En_video`. Los pasos de entrenamiento necesitan una muestra anotada en `datos/` con el formato que documentan `scripts/02_consolidar_etiquetas.py` y `postura/particion.py`.

| Paso | Comando | Qué hace |
|---|---|---|
| 1 | `python scripts/01_construir_menciones.py` | Extrae menciones y arma la muestra estratificada |
| 2 | `python scripts/02_consolidar_etiquetas.py` | Consolida anotaciones por voto mayoritario |
| 3 | `python scripts/04_entrenar_clasicos.py` | Entrena y selecciona los clasificadores clásicos |
| 4 | `python scripts/05_aplicar_corpus.py` | Aplica los modelos a todas las menciones |
| 5 | `python scripts/06_fase3_qwen.py` | Modelo de lenguaje local: cero y pocos ejemplos, embeddings |
| 6 | `python scripts/08_fase3_bert.py` | Ajuste fino de BETO (otro modelo con `MODELO_BERT=...`) |
| 7 | `python scripts/07_fase3_evaluar.py` | Evaluación y comparación con DeLong |
| 8 | `python scripts/09_figuras_roc.py` | Curvas ROC |

## Pruebas

```bash
pytest
```

Sin datos locales, las pruebas que dependen de la muestra anotada se omiten; las demás usan datos sintéticos.

## Licencia

Código bajo licencia MIT (ver `LICENSE`). El repositorio no contiene transcripciones ni fragmentos de texto de terceros.

## Cita

Si usas este código, cita el repositorio. La publicación asociada se indicará aquí cuando esté disponible.
