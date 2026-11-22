"""Valor de negocio de la decisión de contacto ("costo de la decisión").

Supuestos del enunciado (config.py):
    contactar cuesta 25.000; la oferta retiene al 40 % de quienes iban a cancelar;
    retener vale 12 meses de la mensualidad vigente; contactar a quien no iba a cancelar
    solo pierde el contacto; no contactar vale 0 (línea de referencia).

Valor de contactar a un cliente:
    si iba a cancelar:     0,40 × 12 × precio − 25.000
    si no iba a cancelar:  −25.000
Valor esperado = p × 0,40 × 12 × precio − 25.000  → contactar si p > umbral(precio).
"""
import numpy as np
import pandas as pd

from . import config

GANANCIA = config.TASA_EXITO_OFERTA * config.MESES_VALOR_RETENCION   # 4,8


def umbral(precio):
    return config.COSTO_CONTACTO / (GANANCIA * np.asarray(precio, dtype=float))


def valor_esperado(p, precio):
    return np.asarray(p) * GANANCIA * np.asarray(precio, dtype=float) - config.COSTO_CONTACTO


def decidir(p, precio, excluir=None):
    """1 = contactar (valor esperado positivo). `excluir`: máscara de clientes que no se contactan
    por política de negocio (p. ej. ya recibieron una oferta de retención)."""
    k = (valor_esperado(p, precio) > 0).astype(int)
    if excluir is not None:
        k = k * (~np.asarray(excluir, dtype=bool))
    return k


def valor_realizado(y, contactar, precio):
    """Valor neto real de una decisión dada la realidad y (1 = canceló)."""
    y, k, precio = map(np.asarray, (y, contactar, precio))
    return float(np.sum(k * (y * GANANCIA * precio - config.COSTO_CONTACTO)))


def resumen(y, p, precio) -> dict:
    """Métricas de negocio de la regla de valor esperado sobre probabilidades p."""
    y, p, precio = np.asarray(y), np.asarray(p), np.asarray(precio, dtype=float)
    k = decidir(p, precio)
    v = valor_realizado(y, k, precio)
    oraculo = valor_realizado(y, y, precio)                       # contactar solo a quienes cancelan
    todos = valor_realizado(y, np.ones_like(y), precio)
    return {
        "valor_neto": v,
        "pct_del_maximo": v / oraculo if oraculo else np.nan,
        "contactados": int(k.sum()),
        "pct_contactados": k.mean(),
        "churners_contactados": int((k * y).sum()),
        "recall_churn_contactado": (k * y).sum() / max(y.sum(), 1),
        "precision_contacto": (k * y).sum() / max(k.sum(), 1),
        "valor_contactar_todos": todos,
        "valor_maximo_oraculo": oraculo,
    }


def curva_valor(y, p, precio, pasos=101) -> pd.DataFrame:
    """Valor neto si se contacta al top-x% por valor esperado (para ver la forma de la curva)."""
    ve = valor_esperado(p, precio)
    orden = np.argsort(-ve)
    y, precio = np.asarray(y)[orden], np.asarray(precio, dtype=float)[orden]
    gan = y * GANANCIA * precio - config.COSTO_CONTACTO
    acum = np.concatenate([[0], np.cumsum(gan)])
    idx = np.linspace(0, len(y), pasos).astype(int)
    return pd.DataFrame({"pct_contactados": idx / len(y), "valor_neto": acum[idx]})
