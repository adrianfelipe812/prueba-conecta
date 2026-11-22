"""Validación del NLP de tickets: referencia, robustez, línea base ingenua y modelo externo."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from . import io, nlp

RECURSOS = Path(__file__).parent / "recursos"
CAMPOS = ["sarcasmo", "sentimiento", "intencion", "frustracion", "recurrente", "tema_precio"]


def referencia(lake_dir) -> pd.DataFrame:
    """Textos únicos de silver + etiquetas manuales + predicción de reglas + frecuencia."""
    tk = io.read(lake_dir, "silver", "tickets")
    u = tk.groupby(["categoria", "texto_cliente"]).size().rename("n_tickets").reset_index()
    u["clave_texto"] = u["texto_cliente"].map(nlp.clave_texto)
    ref = pd.read_csv(RECURSOS / "referencia_nlp.csv")
    pred = pd.DataFrame([nlp.analizar(t) for t in u["texto_cliente"]]).add_prefix("pred_")
    return u.merge(ref, on="clave_texto", how="left").join(pred)


def robustez() -> pd.DataFrame:
    r = pd.read_csv(RECURSOS / "robustez_nlp.csv")
    return r.join(pd.DataFrame([nlp.analizar(t) for t in r["texto"]]).add_prefix("pred_"))


def metricas(df, campos, peso=None) -> pd.DataFrame:
    filas = []
    for c in campos:
        y, p = df[c].astype(str), df["pred_" + c].astype(str)
        w = df[peso] if peso else None
        filas.append({"campo": c, "exactitud": accuracy_score(y, p, sample_weight=w),
                      "f1_macro": f1_score(y, p, average="macro", sample_weight=w), "n": len(df)})
    return pd.DataFrame(filas).round(3)


def linea_base_ingenua(df) -> float:
    """Exactitud de un léxico de sentimiento SIN tratar el sarcasmo, sobre los textos sarcásticos."""
    s = df[df["sarcasmo"] == 1]
    return (s["pred_sentimiento_lexico"] == s["sentimiento"]).mean()


def modelo_externo(textos, lote=16):
    """Ironía y sentimiento con pysentimiento (RoBERTuito, español). Devuelve None si no está instalado."""
    try:
        from pysentimiento import create_analyzer
    except ImportError:
        return None
    ironia = create_analyzer(task="irony", lang="es")
    senti = create_analyzer(task="sentiment", lang="es")
    textos = list(textos)
    ir, se = ironia.predict(textos), senti.predict(textos)
    m = {"POS": "positivo", "NEU": "neutro", "NEG": "negativo"}
    return pd.DataFrame({"ext_ironia": [int(x.output == "ironic") for x in ir],
                         "ext_p_ironia": [x.probas.get("ironic", 0) for x in ir],
                         "ext_sentimiento": [m[x.output] for x in se]})
