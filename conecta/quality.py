"""Checks de calidad por capa.

- bronze: MIDE los problemas tal como llegan (n_afectados > 0 es esperado).
- silver: VERIFICA que quedaron resueltos o marcados (esperado = 0 salvo 'info').
- integridad: llaves y reglas de negocio entre tablas (en silver).

Salida: DataFrame con columnas
    capa, tabla, codigo, regla, n_afectados, total, pct, severidad, estado, accion
estado: 'detectado' (bronze) | 'ok' | 'FALLA' | 'info'
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, io

_rows = []


def _add(capa, tabla, codigo, regla, n, total, severidad, accion, esperado_cero=None):
    if esperado_cero is None:
        estado = "detectado" if capa == "bronze" else "info"
    else:
        estado = "ok" if n == 0 else "FALLA"
    _rows.append(dict(capa=capa, tabla=tabla, codigo=codigo, regla=regla, n_afectados=int(n),
                      total=int(total), pct=round(100 * n / total, 2) if total else 0.0,
                      severidad=severidad, estado=estado, accion=accion))


def _bronze(lake):
    r = lambda t: io.read(lake, "bronze", t)
    cl, su, fa = r("clientes"), r("suscripciones"), r("facturas")
    tk, ev = r("tickets"), r("eventos")
    N = len(cl)
    canon = cl["ciudad"].str.lower().str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii").map(config.CIUDADES)
    _add("bronze", "clientes", "P01", "ciudad escrita con variantes (17 valores para 6 ciudades)",
         (cl["ciudad"] != canon).sum(), N, "media", "Normalizar a nombre canónico")
    lo, hi = config.EDAD_VALIDA
    _add("bronze", "clientes", "P02", f"edad fuera de [{lo},{hi}] (centinelas 0/99/999)",
         (~cl["edad"].between(lo, hi)).sum(), N, "media", "Reemplazar por NaN + flag")
    fc, fa_ = pd.to_datetime(cl["fecha_contacto_retencion"]), pd.to_datetime(cl["fecha_alta"])
    _add("bronze", "clientes", "P03", "contacto de retención anterior a la fecha de alta",
         (fc < fa_).sum(), N, "baja", "Marcar flag; no usar esa fecha")
    _add("bronze", "clientes", "P04", "estado_actual refleja HOY (fuga para cortes pasados)",
         N, N, "alta", "Excluir de features")
    sub_sig = su.sort_values(["cliente_id", "fecha_inicio"]).groupby("cliente_id")["suscripcion_id"].shift(-1)
    _add("bronze", "suscripciones", "P05", "suscripciones cerradas por cambio de plan (no son churn)",
         (su.sort_values(["cliente_id", "fecha_inicio"])["fecha_fin"].notna() & sub_sig.notna()).sum(),
         len(su), "alta", "Churn = fin de la ÚLTIMA suscripción")
    _add("bronze", "facturas", "P06", "factura_id duplicado (recarga día 5, 07:00)",
         fa["factura_id"].duplicated().sum(), len(fa), "alta", "Deduplicar, conservar primera carga")
    _add("bronze", "facturas", "P07", "facturas sin fecha_pago (pago se actualiza sin fecha de carga)",
         fa["fecha_pago"].isna().sum(), len(fa), "alta", "Pago conocido solo si fecha_pago <= corte")
    _add("bronze", "facturas", "P08", "monto del Básico = 19.900 (precio anterior a abr-2025)",
         (fa["monto"] == 19900).sum(), len(fa), "info", "No es error: subida de precio (hallazgo)")
    esquemas = tk["esquema"].value_counts()
    _add("bronze", "tickets", "P09", f"dos esquemas JSON distintos ({len(esquemas)} combinaciones de claves)",
         (~tk["esquema"].str.contains("ticket_id")).sum(), len(tk), "alta", "Unificar A y B en un esquema")
    parsed = tk["raw_json"].map(json.loads)
    _add("bronze", "tickets", "P10", "ID de cliente con formato texto 'CL-0xxxxx' (esquema A)",
         parsed.map(lambda d: str(d.get("customer", "")).startswith("CL-")).sum(), len(tk), "alta", "Convertir a entero")
    _add("bronze", "tickets", "P11", "csat ausente",
         parsed.map(lambda d: d.get("csat") is None and (d.get("satisfaccion") or {}).get("csat") is None).sum(),
         len(tk), "media", "Mantener nulo + usar estado; no imputar")
    _add("bronze", "tickets", "P12", "fechas con zona -05:00 (B) mezcladas con UTC 'Z' (A)",
         tk["raw_json"].str.contains('-05:00"').sum(), len(tk), "media", "Convertir todo a hora local Bogotá")
    _add("bronze", "tickets", "P14", "líneas JSON inválidas", tk["error_parseo"].notna().sum(), len(tk), "alta", "Descartar y reportar")
    _add("bronze", "eventos", "P13", "eventos duplicados exactos", tk.shape[0] * 0 + ev["raw_json"].duplicated().sum(),
         len(ev), "media", "Eliminar duplicados")


def _silver(lake):
    r = lambda t: io.read(lake, "silver", t)
    cl, su, fa, tk, ev = r("clientes"), r("suscripciones"), r("facturas"), r("tickets"), r("eventos")
    et, cp = r("etiquetas_entrenamiento"), r("clientes_a_puntuar")
    ids = set(cl["cliente_id"])
    S = "silver"
    _add(S, "clientes", "P01", "ciudad no canónica", (cl["ciudad"] == "DESCONOCIDA").sum(), len(cl), "media", "-", True)
    _add(S, "clientes", "P02", "edad inválida remanente", (~cl["edad"].dropna().between(*config.EDAD_VALIDA)).sum(), len(cl), "media", "-", True)
    _add(S, "clientes", "P02", "edad nula tras limpieza (a imputar en gold)", cl["edad"].isna().sum(), len(cl), "media", "Imputar en gold")
    _add(S, "clientes", "PK", "cliente_id duplicado", cl["cliente_id"].duplicated().sum(), len(cl), "alta", "-", True)
    _add(S, "suscripciones", "PK", "suscripcion_id duplicado", su["suscripcion_id"].duplicated().sum(), len(su), "alta", "-", True)
    _add(S, "suscripciones", "R01", "cliente con >1 suscripción activa", (su[su["activa"]].groupby("cliente_id").size() > 1).sum(), len(cl), "alta", "-", True)
    _add(S, "suscripciones", "R02", "fecha_fin < fecha_inicio", (su["fecha_fin"] < su["fecha_inicio"]).sum(), len(su), "alta", "-", True)
    _add(S, "facturas", "P06", "factura_id duplicado", fa["factura_id"].duplicated().sum(), len(fa), "alta", "-", True)
    _add(S, "facturas", "FK", "factura sin suscripción", fa["plan"].isna().sum(), len(fa), "alta", "-", True)
    _add(S, "tickets", "P09", "tickets sin esquema unificado", tk["esquema"].isna().sum(), len(tk), "alta", "-", True)
    _add(S, "tickets", "FK", "ticket de cliente inexistente", (~tk["cliente_id"].isin(ids)).sum(), len(tk), "alta", "-", True)
    _add(S, "tickets", "P12", "creado_en nulo (fecha no parseable)", tk["creado_en"].isna().sum(), len(tk), "alta", "-", True)
    _add(S, "tickets", "R03", "resuelto antes de creado", (tk["resuelto_en"] < tk["creado_en"]).sum(), len(tk), "media", "-", True)
    _add(S, "tickets", "R04", "categoría sin mapeo a categoría principal", tk["categoria_principal"].isna().sum(), len(tk), "media", "-", True)
    _add(S, "tickets", "R05", "segmento del ticket distinto al CRM (esquema B)",
         tk.merge(cl[["cliente_id", "segmento"]], on="cliente_id").pipe(
             lambda d: (d["segmento_ticket"].notna() & (d["segmento_ticket"] != d["segmento"])).sum()), len(tk), "media", "-", True)
    _add(S, "tickets", "R06", f"tickets posteriores a {config.FECHA_HOY} (hora local)",
         (tk["creado_en"] > pd.Timestamp(config.FECHA_HOY) + pd.Timedelta(days=1)).sum(), len(tk), "media", "-", True)
    # NLP (P15): cobertura y concordancia con el set de referencia etiquetado a mano
    _add(S, "tickets", "P15", "ticket sin columnas NLP", tk["nlp_sentimiento"].isna().sum(), len(tk), "alta", "-", True)
    ref = pd.read_csv(Path(__file__).parent / "recursos" / "referencia_nlp.csv")
    m = tk.merge(ref, left_on="nlp_clave_texto", right_on="clave_texto", how="left")
    _add(S, "tickets", "P15", "texto de cliente fuera del set de referencia (nuevo texto: revisar)",
         m["sarcasmo"].isna().sum(), len(tk), "media", "Etiquetar y validar", True)
    _add(S, "tickets", "P15", "sarcasmo: regla distinta a la etiqueta de referencia",
         (m["sarcasmo"].notna() & (m["sarcasmo"] != m["nlp_sarcasmo"])).sum(), len(tk), "alta", "-", True)
    _add(S, "tickets", "P15", "intención: regla distinta a la etiqueta de referencia",
         (m["intencion"].notna() & (m["intencion"] != m["nlp_intencion"])).sum(), len(tk), "alta", "-", True)
    _add(S, "tickets", "P15", "tickets sarcásticos (leídos como positivos por un léxico ingenuo)",
         tk["nlp_sarcasmo"].sum(), len(tk), "info", "Usar nlp_sentimiento, no el léxico")
    _add(S, "eventos", "P13", "event_id duplicado", ev["event_id"].duplicated().sum(), len(ev), "media", "-", True)
    _add(S, "eventos", "FK", "evento de cliente inexistente", (~ev["cliente_id"].isin(ids)).sum(), len(ev), "alta", "-", True)
    _add(S, "eventos", "R07", "eventos fuera de 2025 (hora local)", (ev["ts"].dt.year != 2025).sum(), len(ev), "media", "-", True)
    _add(S, "etiquetas_entrenamiento", "PK", "(cliente, corte) duplicado", et.duplicated(["cliente_id", "fecha_corte"]).sum(), len(et), "alta", "-", True)
    _add(S, "clientes_a_puntuar", "R08", "cliente a puntuar sin suscripción activa",
         (~cp["cliente_id"].isin(su.loc[su["activa"], "cliente_id"])).sum(), len(cp), "alta", "-", True)
    # R09: la etiqueta se reconstruye con la regla de churn de silver
    fin = su.groupby("cliente_id").apply(lambda s: pd.NaT if s["activa"].any() else s["fecha_fin"].max(), include_groups=False)
    e = et.merge(fin.rename("fin_cliente").reset_index(), on="cliente_id")
    calc = (e["fin_cliente"] > e["fecha_corte"]) & (e["fin_cliente"] <= e["fecha_corte"] + pd.Timedelta(days=config.HORIZONTE_DIAS))
    _add(S, "etiquetas_entrenamiento", "R09", "churn_90d no coincide con la regla 'fin de última suscripción en 90d'",
         (calc.astype(int) != e["churn_90d"]).sum(), len(et), "alta", "-", True)
    activa_corte = e.merge(su[["cliente_id", "fecha_inicio", "fecha_fin"]], on="cliente_id").pipe(
        lambda d: d[(d["fecha_inicio"] <= d["fecha_corte"]) & (d["fecha_fin"].isna() | (d["fecha_fin"] > d["fecha_corte"]))])
    _add(S, "etiquetas_entrenamiento", "R10", "cliente etiquetado no activo en su fecha de corte",
         len(et) - activa_corte[["cliente_id", "fecha_corte"]].drop_duplicates().shape[0], len(et), "alta", "-", True)


def run(lake_dir) -> pd.DataFrame:
    _rows.clear()
    _bronze(lake_dir)
    _silver(lake_dir)
    rep = pd.DataFrame(_rows)
    p = io.path(lake_dir, "_calidad", "reporte_calidad")
    p.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8")
    return rep
