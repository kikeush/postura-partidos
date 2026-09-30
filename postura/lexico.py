# -*- coding: utf-8 -*-
"""Léxicos de valoración para el discurso político mexicano y conteos en ventanas alrededor
de la mención marcada.

Autor: Enrique Alberto Mendoza Ruiz, Universidad Panamericana (eamendoza@up.edu.mx)
Fecha: 2026-09-27. Licencia: MIT.

Propósito
---------
Dar al clasificador clásico señales interpretables de postura: cuántos términos de valoración
desfavorable o favorable aparecen cerca de la cadena marcada entre [[ ]], si hay una negación
justo antes, cuántos pronombres de primera persona usa el titular y cuántos términos
informativos (votaciones, candidaturas, encuestas) rodean la mención.

Diseño
------
Todo se compara en forma normalizada: minúsculas y sin diacríticos (la eñe se vuelve ene), de
modo que "corrupción" y "corrupcion" cuentan igual. Cada léxico tiene tres partes:

* formas exactas (conjunto de palabras completas);
* raíces (prefijos seguros que cubren la flexión: "corrup" cubre corrupto, corrupción,
  corruptela). Solo se usan raíces que no chocan con palabras frecuentes ajenas; por ejemplo
  no se usa "rob" (robusto, Roberto) ni "decen" (decenas);
* expresiones de varias palabras ("moralmente derrotados", "guerra sucia").

Las listas son deliberadamente pequeñas y revisables a mano. No pretenden ser un léxico de
sentimiento general: capturan cómo se descalifica o se elogia a un partido en las
conferencias matutinas. Se excluyen muletillas muy frecuentes que no valoran (bueno al
inicio de turno, por ejemplo, a lo mejor).
"""
import re
import unicodedata

__all__ = [
    "normalizar", "tokenizar", "separar_marcada", "tokens_marcada", "ventana",
    "Lexico", "DESFAVORABLE", "FAVORABLE", "INFORMATIVO", "DESLINDE", "NEGACIONES",
    "PRIMERA_SINGULAR", "PRIMERA_PLURAL", "INTENSIFICADORES", "LEXICOS",
    "contar", "contar_en_ventana", "negacion_cercana", "contar_negados",
]

_RX_PALABRA = re.compile(r"\w+")


def normalizar(texto):
    """Minúsculas y sin diacríticos (NFKD sin marcas combinantes)."""
    texto = unicodedata.normalize("NFKD", str(texto or "").lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def tokenizar(texto):
    """Lista de palabras normalizadas (secuencias alfanuméricas)."""
    return _RX_PALABRA.findall(normalizar(texto))


def separar_marcada(oracion_marcada):
    """Divide una oración con la mención entre [[ ]] en (antes, mención, después).

    Si la oración no trae marcas, devuelve (oración, "", "")."""
    texto = str(oracion_marcada or "")
    ini = texto.find("[[")
    fin = texto.find("]]", ini + 2) if ini >= 0 else -1
    if ini < 0 or fin < 0:
        return texto, "", ""
    return texto[:ini], texto[ini + 2:fin], texto[fin + 2:]


def tokens_marcada(oracion_marcada):
    """Tokens normalizados de (antes, mención, después) de una oración marcada."""
    antes, mencion, despues = separar_marcada(oracion_marcada)
    return tokenizar(antes), tokenizar(mencion), tokenizar(despues)


def ventana(oracion_marcada, k):
    """Tokens a k palabras a la izquierda y a la derecha de la mención (sin la mención)."""
    antes, _, despues = tokens_marcada(oracion_marcada)
    return (antes[-k:] if k > 0 else []), despues[:k]


class Lexico:
    """Léxico con formas exactas, raíces y expresiones de varias palabras.

    Las entradas se escriben con acentos para que sean legibles y se normalizan al construir
    el objeto."""

    def __init__(self, nombre, formas=(), raices=(), expresiones=()):
        self.nombre = nombre
        self.formas = frozenset(normalizar(f) for f in formas)
        self.raices = tuple(sorted({normalizar(r) for r in raices}))
        self.expresiones = tuple(tuple(tokenizar(e)) for e in expresiones if tokenizar(e))
        # Índice por primera palabra: solo se comparan expresiones cuyo inicio coincide.
        self._por_inicio = {}
        for exp in self.expresiones:
            self._por_inicio.setdefault(exp[0], []).append(exp)

    def coincide(self, token):
        """True si el token (ya normalizado) pertenece al léxico por forma o por raíz."""
        return token in self.formas or (bool(self.raices) and token.startswith(self.raices))

    def contar(self, tokens):
        """Número de tokens del léxico más número de expresiones encontradas."""
        formas, raices, por_inicio = self.formas, self.raices, self._por_inicio
        n = 0
        for i, t in enumerate(tokens):
            if t in formas or (raices and t.startswith(raices)):
                n += 1
            if t in por_inicio:
                for exp in por_inicio[t]:
                    if tuple(tokens[i:i + len(exp)]) == exp:
                        n += 1
        return n

    def posiciones(self, tokens):
        """Índices de los tokens que pertenecen al léxico (sin expresiones)."""
        return [i for i, t in enumerate(tokens) if self.coincide(t)]

    def __contains__(self, palabra):
        return self.coincide(normalizar(palabra))

    def __repr__(self):
        return (f"Lexico({self.nombre!r}, formas={len(self.formas)}, raices={len(self.raices)}, "
                f"expresiones={len(self.expresiones)})")


# Valoración desfavorable: corrupción, robo, traición, mentira, fracaso, autoritarismo y
# epítetos frecuentes contra adversarios en el discurso político mexicano.
DESFAVORABLE = Lexico(
    "desfavorable",
    formas=[
        "robo", "robos", "robar", "robaron", "roban", "robaban", "robando", "robado", "robada",
        "robados", "robó", "ladrona", "ladronas", "mintieron", "miente", "mienten", "mintió",
        "mintiendo", "cínico", "cínicos", "cínica", "cínicas", "cinismo", "engaño", "engaños",
        "engañar", "engañaron", "engañan", "engañado", "engañados", "cómplice", "cómplices",
        "complicidad", "impunidad", "represión", "represivo", "represivos", "represores",
        "reprimieron", "reprimir", "moches", "pillos", "pillaje", "rapiña", "atraco", "atracos",
        "farsa", "cloaca", "ruin", "ruines", "ruindad", "ruina", "decadente", "decadentes",
        "decadencia", "inepto", "ineptos", "ineptitud", "vulgar", "vulgares", "grotesco",
        "odio", "odian", "rencor", "rencorosos", "canalla", "canallas", "cretinos",
        "abusivo", "abusivos", "abuso", "abusos", "abusaron", "retroceso", "retrocesos",
        "desastre", "desastres", "desastroso", "desastrosa", "fobaproa", "desfalco",
        "despilfarro", "derroche", "derrocharon", "rateros", "ratero", "ratería", "hampa",
        "maleantes", "bandidos", "bandido", "malandrines", "malandros", "criminal",
        "criminales", "hipocresía", "desvergonzados", "sinvergüenza", "sinvergüenzas",
        "vergüenza", "vergonzoso", "irresponsable", "irresponsables", "irresponsabilidad",
        "codicia", "avaricia", "voraces", "voracidad", "ambiciosos", "zopilotes", "buitres",
        "carroñeros", "calumnia", "calumnias", "calumniadores", "golpistas", "golpeteo",
        "tramposos", "trampa", "trampas", "farsantes", "simulación", "retrógrados",
        "retrógradas", "antipatriotas", "antipatriótico", "clasistas", "clasismo", "racistas",
        "racismo", "autoritario", "autoritarios", "autoritarismo", "nefasto", "nefastos",
        "nefasta", "perverso", "perversos", "perversidad", "mezquino", "mezquinos",
        "mezquindad", "demagogia", "demagogos", "entreguistas", "entreguismo", "vendepatrias",
        "adversarios", "adversario", "derrotados", "hundieron", "arruinaron", "destruyeron",
        "destrucción", "desmantelaron", "chanchullo", "chanchullos", "cochupo", "cochupos",
        "huachicol", "huachicoleo", "chayote", "chayoteros", "privilegios", "privilegiados",
        "oligarcas", "oligarquía", "narcogobierno", "narcoestado",
    ],
    raices=[
        "corrup", "corrompi", "saque", "traicion", "traidor", "hipocr", "mentir", "fraud",
        "farsant", "simulad", "mafi", "deshonest", "fracas", "reaccionari", "privatiz",
        "entreguis", "vendepatri", "neoliberal", "ladron", "calumni", "difam", "manipul",
        "estaf", "oligarq", "oligarc", "golpist", "autoritari", "desfalc", "despilfarr",
        "endeud", "delincu", "podrid", "podredumbre", "nefast", "perver", "demagog",
        "desvergonz", "sinverguenz", "decaden", "retrogr", "clasist", "racist", "chayot",
        "huachicol", "narco", "vergonzos", "desastros", "ridicul", "hipocres",
    ],
    expresiones=[
        "moralmente derrotados", "guerra sucia", "campaña sucia", "cuello blanco",
        "mafia del poder", "no tienen llenadera", "se robaron", "se sirvieron con la cuchara grande",
        "saquearon al país", "los de arriba", "doble moral", "en contra del pueblo",
        "en contra de México", "al servicio de", "a espaldas del pueblo", "le dieron la espalda",
    ],
)

# Valoración favorable: honestidad, respaldo, reconocimiento, logros y virtudes.
FAVORABLE = Lexico(
    "favorable",
    formas=[
        "respaldo", "respaldamos", "respaldar", "respaldan", "apoyo", "apoyamos", "apoyar",
        "apoyaron", "apoyan", "felicidades", "reconozco", "reconocemos", "reconocer",
        "reconoció", "ejemplar", "ejemplares", "excelente", "excelentes",
        "integridad", "íntegro", "íntegros", "decente", "decentes", "decencia", "digno", "digna",
        "dignos", "dignas", "dignidad", "leal", "leales", "lealtad", "valiente", "valientes",
        "valentía", "justicia", "justo", "justa", "confianza", "confío", "confiamos",
        "enhorabuena", "principios", "ideales", "convicción", "convicciones", "esperanza",
        "unidad", "hermandad", "fraternidad", "bien", "buen", "buena", "buenos",
        "buenas", "mejores", "ganamos", "transformación", "cuarta", "humanismo",
        "humanista", "humanistas", "patriota", "patriotas", "patriótico", "patriotismo",
        "noble", "nobles", "generoso", "generosos", "generosidad", "compañeros", "compañeras",
        "compañero", "compañera", "aliado", "aliados", "aliada", "aliadas",
    ],
    raices=[
        "honest", "honrad", "respald", "felicit", "reconocim", "logr", "triunf", "avanc",
        "avanz", "orgull", "agradez", "agradec", "extraordinari", "admir", "congruen",
        "coheren", "patriot", "humanis", "solidari", "exito", "exitos", "eficien", "eficaz",
        "limpi", "democr", "respet", "aplau", "celebr", "bienvenid", "incorruptib",
        "fortalec", "fraterni", "transformador",
    ],
    expresiones=[
        "cuarta transformación", "estamos muy contentos", "hicieron un buen trabajo",
        "muy buen trabajo", "gente buena", "mujeres y hombres honestos", "al servicio del pueblo",
        "con el pueblo", "por el bien de todos", "primero los pobres",
    ],
)

# Términos informativos: votaciones, candidaturas, encuestas, estructura legislativa. Suelen
# acompañar a menciones neutrales.
INFORMATIVO = Lexico(
    "informativo",
    formas=["ine", "pri", "ciento", "porcentaje", "curul", "curules", "escaños", "escaño"],
    raices=[
        "vot", "encuest", "eleccion", "elector", "candidat", "diputad", "senad", "legisl",
        "camara", "congres", "bancad", "coalicion", "alianz", "registr", "dirigen",
        "asamble", "iniciativ", "reform", "dictamen", "porcentaj", "distrit", "gubernatur",
        "alcald", "municip", "plurinomin", "tribunal", "consejer", "militan", "afiliad",
        "padron", "prerrogativ", "financiamient", "consult", "estatut",
    ],
    expresiones=["por ciento", "mayoría calificada", "mayoría simple", "primera minoría"],
)

# Deslinde: el titular dice que algo no le corresponde o que no interviene. Según la guía,
# el deslinde es neutral.
DESLINDE = Lexico(
    "deslinde",
    formas=["corresponde", "corresponden", "compete", "competen", "intervengo", "intervenimos",
            "intervenir", "meto", "metemos", "injerencia", "incumbe", "decidan", "deciden",
            "resuelvan", "resuelven"],
    raices=[],
    expresiones=["no me corresponde", "no nos corresponde", "no me meto", "no nos metemos",
                 "no voy a opinar", "no opino", "es asunto de", "es un asunto de",
                 "le corresponde al partido", "eso le toca"],
)

NEGACIONES = Lexico(
    "negacion",
    formas=["no", "ni", "nunca", "jamás", "tampoco", "sin", "nadie", "nada", "ningún",
            "ninguno", "ninguna", "ningunos", "ningunas"],
)

PRIMERA_SINGULAR = Lexico(
    "primera_singular",
    formas=["yo", "me", "mi", "mí", "mis", "mío", "mía", "míos", "mías", "conmigo"],
)

PRIMERA_PLURAL = Lexico(
    "primera_plural",
    formas=["nosotros", "nosotras", "nos", "nuestro", "nuestra", "nuestros", "nuestras"],
)

INTENSIFICADORES = Lexico(
    "intensificador",
    formas=["muy", "tan", "tanto", "tanta", "tantos", "tantas", "mucho", "mucha", "muchos",
            "muchas", "demasiado", "demasiada", "sumamente", "totalmente", "completamente",
            "absolutamente", "extremadamente", "bastante", "siempre"],
)

LEXICOS = {
    "desfavorable": DESFAVORABLE, "favorable": FAVORABLE, "informativo": INFORMATIVO,
    "deslinde": DESLINDE, "negacion": NEGACIONES, "primera_singular": PRIMERA_SINGULAR,
    "primera_plural": PRIMERA_PLURAL, "intensificador": INTENSIFICADORES,
}


def _como_lexico(lexico):
    """Devuelve el léxico registrado con ese nombre o el objeto Lexico recibido."""
    return LEXICOS[lexico] if isinstance(lexico, str) else lexico


def contar(texto_o_tokens, lexico):
    """Ocurrencias del léxico en un texto (se tokeniza) o en una lista de tokens."""
    tokens = tokenizar(texto_o_tokens) if isinstance(texto_o_tokens, str) else list(texto_o_tokens)
    return _como_lexico(lexico).contar(tokens)


def contar_en_ventana(oracion_marcada, lexico, k):
    """Ocurrencias del léxico a k palabras de cada lado de la mención marcada.

    La mención misma no cuenta. Las expresiones se buscan por separado a la izquierda y a la
    derecha para no unir palabras a través de la mención."""
    lex = _como_lexico(lexico)
    izq, der = ventana(oracion_marcada, k)
    return lex.contar(izq) + lex.contar(der)


def negacion_cercana(oracion_marcada, k=3):
    """True si hay una negación entre las k palabras inmediatamente anteriores a la mención."""
    izq, _ = ventana(oracion_marcada, k)
    return any(NEGACIONES.coincide(t) for t in izq)


def contar_negados(tokens, lexico, alcance=3):
    """Ocurrencias del léxico precedidas por una negación a no más de `alcance` palabras.

    Ejemplo: en "no son corruptos" el término corruptos cuenta como desfavorable negado."""
    lex = _como_lexico(lexico)
    tokens = list(tokens)
    n = 0
    for i in lex.posiciones(tokens):
        if any(NEGACIONES.coincide(t) for t in tokens[max(0, i - alcance):i]):
            n += 1
    return n
