"""NLP de tickets: sarcasmo, sentimiento real, intención, frustración, recurrencia y tema precio.

Enfoque: reglas transparentes sobre léxicos genéricos en español (no frases exactas).
Por qué reglas y no un modelo grande:
- Solo hay 38 textos distintos de cliente (plantillas sintéticas): no alcanza para entrenar.
- Es explicable y editable en vivo (sustentación), gratis y reproducible en Colab.
- Se valida contra (1) un set de referencia etiquetado a mano, (2) frases nuevas de
  robustez escritas aparte, (3) un modelo preentrenado independiente de ironía
  (pysentimiento) y (4) su relación con el churn real.

Regla central de sarcasmo: un marcador de polaridad POSITIVA (o risa) junto a un EVENTO
NEGATIVO del servicio, sin señal de que el problema ya se resolvió.
  "Qué eficiencia la suya, me volvieron a cobrar doble"        -> sarcasmo  (ejemplo propio, no de los datos)
  "Excelente atención, me resolvieron el cobro doble"          -> sincero
Límites: no detecta ironía sin evento negativo explícito ni sarcasmo por contexto externo;
léxicos ajustados al dominio (suscripción/app/facturación).
"""
import hashlib
import re
import unicodedata

import pandas as pd


def normalizar(texto) -> str:
    s = unicodedata.normalize("NFKD", str(texto or "")).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def clave_texto(texto) -> str:
    """Huella del texto normalizado (para cruzar etiquetas sin guardar el texto en el repo)."""
    return hashlib.sha1(normalizar(texto).encode()).hexdigest()[:16]


def _rx(*patrones):
    return re.compile("|".join(f"(?:{p})" for p in patrones))


# Marcadores de polaridad positiva / elogio (lo que un léxico de sentimiento leería como "bueno")
POSITIVO = _rx(r"\bexcelente", r"\bgenial", r"\bfelicitaciones", r"^super\b|\bsuper,", r"\bme encanta",
               r"\bmaravill", r"\bimpresionante", r"\bque bonit", r"\binnovador", r"\binolvidable",
               r"\bconsentir", r"\bbuena estrategia", r"\bgracias (de verdad|por todo)",
               r"justo lo que necesitaba", r"\bque (velocidad|rapidez|eficiencia)", r"\bbravo\b", r"\bperfecto\b",
               r"que (amable|detalle)|bonito detalle", r"mil gracias")
RISA = _rx(r"\b(ja){2,}", r"\b(je){2,}", r"\bxd\b", r":\)")
# Eventos negativos del servicio (hechos, sin carga emocional)
EVENTO_NEG = _rx(r"se (cierra|cierre|cae|cayo)", r"congel", r"no (abre|carga|sirve|funciona|llega)", r"nunca llega",
                 r"cobr\w* (doble|dos veces)", r"(volvieron|vuelven) a cobrar", r"dos cobros", r"otro cobro",
                 r"pagar dos veces", r"duplicad", r"lent", r"tarda", r"minutos para cargar", r"aumento", r"subieron",
                 r"sin avis", r"cancel", r"dar de baja", r"\bfalla", r"\berror", r"sin poder", r"\bmulta",
                 r"pag\w* mas", r"funcionaba mejor", r"sigue igual", r"no dice nada", r"esperando",
                 r"necesito el reembolso", r"cobrarme")
RESUELTO = _rx(r"resolv", r"solucion", r"arreglaron", r"despues del arreglo", r"ya quedo", r"ya funciona",
               r"me devolvieron", r"ya me llego", r"llego el reembolso")
# Negativo emocional (queja con carga afectiva)
NEG_EMOCION = _rx(r"no sirve", r"pesim", r"me canse", r"cansad", r"no puede ser", r"no voy a seguir",
                  r"pagando por nada", r"\bnadie\b", r"sin avis", r"\bcaro\b", r"ojala", r"ninguna oferta",
                  r"pensando seriamente", r"no me parece justo", r"\bya$|\bya\.")
# Intensidad de frustración alta
FRUSTRACION_ALTA = _rx(r"pesim", r"me canse", r"no voy a seguir", r"pagando por nada", r"no sirve",
                       r"no puede ser", r"ninguna oferta", r"si no lo arreglan", r"nadie me responde",
                       r"pensando seriamente", r"\bya\.$|cancelar ya", r"\bya$", r"!!")
RECURRENTE = _rx(r"otra vez", r"segunda vez", r"de nuevo", r"ya van", r"ya son", r"\bllevo\b", r"otro cobro",
                 r"cada vez", r"\bsiempre\b", r"tres (semanas|meses)")
PRECIO = _rx(r"\bcaro\b", r"economic", r"barat", r"tarifa", r"aumento", r"pago mas", r"valor diferente",
             r"subieron")
# Intención (orden de prioridad)
AMENAZA = _rx(r"si no \w+.* cancel", r"pensando (seriamente )?en irme", r"me voy a ir", r"me cambio a (la )?competencia")
CANCELAR = _rx(r"cancel", r"dar de baja", r"darme de baja")
BAJAR = _rx(r"bajar de plan", r"plan mas (economico|barato)", r"plan mas bajo")
SUBIR = _rx(r"mas beneficios", r"subir de plan", r"plan (superior|mas completo)")


def analizar(texto) -> dict:
    t = normalizar(texto)
    original = str(texto or "")
    pos, risa, ev = bool(POSITIVO.search(t)), bool(RISA.search(t)), bool(EVENTO_NEG.search(t))
    sarcasmo = bool((pos or risa) and ev and not RESUELTO.search(t))
    resuelto = bool(RESUELTO.search(t))
    emocion = bool(NEG_EMOCION.search(t))
    # pedir cancelar/bajar de plan con cortesía es neutro; la intención va en otra columna
    evento_factual = bool(EVENTO_NEG.search(re.sub(r"cancel\w*|dar(me)? de baja", "", t)))
    if sarcasmo or emocion or (evento_factual and not resuelto):
        sentimiento = "negativo"
    elif pos or resuelto:
        sentimiento = "positivo"
    else:
        sentimiento = "neutro"
    if AMENAZA.search(t):
        intencion = "amenaza_irse"
    elif CANCELAR.search(t):
        intencion = "cancelar"
    elif BAJAR.search(t):
        intencion = "bajar_plan"
    elif SUBIR.search(t):
        intencion = "subir_plan"
    else:
        intencion = "soporte"
    gritos = bool(re.search(r"\b[A-ZÁÉÍÓÚ]{2,}\b", original))
    recurrente = bool(RECURRENTE.search(t))
    # 2 = enojo explícito · 1 = sarcasmo, carga emocional o problema recurrente · 0 = reporte factual
    frustracion = 2 if (FRUSTRACION_ALTA.search(t) or gritos) else (1 if (sarcasmo or emocion or recurrente) else 0)
    return {
        "sarcasmo": int(sarcasmo),
        # lo que diría un léxico de sentimiento ingenuo (sin sarcasmo): línea base de comparación
        "sentimiento_lexico": "positivo" if pos else ("negativo" if (emocion or ev) else "neutro"),
        "sentimiento": sentimiento,
        "intencion": intencion,
        "frustracion": frustracion,
        "recurrente": int(recurrente),
        "tema_precio": int(bool(PRECIO.search(t))),
    }


def enriquecer(df: pd.DataFrame, col="texto_cliente", prefijo="nlp_") -> pd.DataFrame:
    """Aplica analizar() a cada texto único (rápido) y une las columnas con prefijo."""
    unicos = df[col].dropna().unique()
    tabla = pd.DataFrame([{col: u, **analizar(u)} for u in unicos])
    tabla = tabla.rename(columns={c: prefijo + c for c in tabla.columns if c != col})
    out = df.merge(tabla, on=col, how="left")
    out[prefijo + "clave_texto"] = out[col].map(clave_texto)
    return out
