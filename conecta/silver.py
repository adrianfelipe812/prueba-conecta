"""Capa SILVER: entidades limpias, tipadas, deduplicadas y conformadas.

Cada transformación corresponde a un problema documentado en
docs/calidad_datos.md (código Pxx). No se elimina información útil: cuando se
corrige un valor se conserva el original o una bandera (flag_*) para auditar.

Convenciones:
- Marcas de tiempo en hora local America/Bogota, sin zona (naive).
- Fechas puras (alta, inicio, fin, corte) como datetime64 a medianoche.
- Nombres de columnas en español y snake_case; IDs de cliente como int64.
"""
import json
import unicodedata

import numpy as np
import pandas as pd

from . import config, io, nlp

META = ["_fila", "_fuente", "_sha256_fuente", "_ingestado_en"]


def _local(serie: pd.Series) -> pd.Series:
    """Texto ISO con zona (Z o -05:00) -> hora local Bogotá sin zona."""
    ts = pd.to_datetime(serie, utc=True, format="ISO8601", errors="coerce")
    return ts.dt.tz_convert(config.ZONA_HORARIA).dt.tz_localize(None)


def _norm_texto(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


# --------------------------------------------------------------------------- clientes
def clientes(b: pd.DataFrame) -> pd.DataFrame:
    df = b.drop(columns=META).copy()
    df["fecha_alta"] = pd.to_datetime(df["fecha_alta"], format="%Y-%m-%d")
    df["fecha_contacto_retencion"] = pd.to_datetime(df["fecha_contacto_retencion"], format="%Y-%m-%d")
    # P01 ciudades con 17 variantes de 6 ciudades
    df["ciudad_original"] = df["ciudad"]
    df["ciudad"] = df["ciudad"].map(lambda c: config.CIUDADES.get(_norm_texto(c), "DESCONOCIDA"))
    # P02 edades centinela (0, 99, 999)
    lo, hi = config.EDAD_VALIDA
    df["flag_edad_invalida"] = ~df["edad"].between(lo, hi)
    df["edad"] = df["edad"].where(~df["flag_edad_invalida"]).astype("Int64")
    # P03 contacto de retención anterior a la fecha de alta (imposible)
    df["flag_contacto_antes_alta"] = df["fecha_contacto_retencion"] < df["fecha_alta"]
    df["contactado_por_retencion"] = df["contactado_por_retencion"].astype(bool)
    # P04 estado_actual es la foto de HOY -> fuga para cortes pasados. Se conserva
    # solo para validación/análisis; gold NO debe usarlo como feature.
    return df


# --------------------------------------------------------------------- suscripciones
def suscripciones(b: pd.DataFrame) -> pd.DataFrame:
    df = b.drop(columns=META).copy()
    df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], format="%Y-%m-%d")
    df["fecha_fin"] = pd.to_datetime(df["fecha_fin"], format="%Y-%m-%d")
    df["precio_lista_actual"] = df["plan"].map(config.PRECIO_PLAN)
    df = df.sort_values(["cliente_id", "fecha_inicio"]).reset_index(drop=True)
    g = df.groupby("cliente_id")
    df["orden"] = g.cumcount() + 1
    df["siguiente_suscripcion_id"] = g["suscripcion_id"].shift(-1).astype("Int64")
    df["plan_siguiente"] = g["plan"].shift(-1)
    # P05 un cambio de plan cierra una suscripción pero NO es churn.
    # Regla verificada contra etiquetas_entrenamiento (4810/4810 coinciden):
    # churn = termina la ÚLTIMA suscripción del cliente (sin otra que la continúe).
    df["activa"] = df["fecha_fin"].isna()
    df["fin_es_churn"] = df["fecha_fin"].notna() & df["siguiente_suscripcion_id"].isna()
    df["fin_es_cambio_plan"] = df["fecha_fin"].notna() & df["siguiente_suscripcion_id"].notna()
    return df


# -------------------------------------------------------------------------- facturas
def facturas(b: pd.DataFrame, subs: pd.DataFrame) -> pd.DataFrame:
    df = b.drop(columns=META).copy()
    df["fecha_carga"] = pd.to_datetime(df["fecha_carga"])
    # P06 779 facturas cargadas dos veces (recarga día 5 a las 07:00, contenido idéntico).
    df["n_cargas"] = df.groupby("factura_id")["factura_id"].transform("size")
    df = df.sort_values(["factura_id", "fecha_carga"]).drop_duplicates("factura_id", keep="first")
    df["periodo"] = pd.PeriodIndex(df["periodo"], freq="M").to_timestamp()
    df["fecha_vencimiento"] = pd.to_datetime(df["fecha_vencimiento"], format="%Y-%m-%d")
    df["fecha_pago"] = pd.to_datetime(df["fecha_pago"], format="%Y-%m-%d")
    # P07 fecha_pago se actualiza después de la carga y no tiene fecha de carga propia:
    # en gold, un pago solo se conoce si fecha_pago <= corte (si no, "pendiente al corte").
    df["dias_atraso_pago"] = (df["fecha_pago"] - df["fecha_vencimiento"]).dt.days
    df["pagada"] = df["fecha_pago"].notna()
    # P08 precio del Básico subió de 19.900 a 23.900 desde abril 2025 (no es error).
    df = df.merge(subs[["suscripcion_id", "plan"]], on="suscripcion_id", how="left")
    df["precio_lista_actual"] = df["plan"].map(config.PRECIO_PLAN)
    df["monto_bajo_lista_actual"] = df["monto"] < df["precio_lista_actual"]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- tickets
def tickets(b: pd.DataFrame) -> pd.DataFrame:
    """P09 Dos esquemas: A (hasta 2025-06-10, inglés, UTC) y B (desde 2025-06-10,
    español, anidado, -05:00). Se unifican a un solo esquema en hora local."""
    filas = []
    for raw in b["raw_json"]:
        r = json.loads(raw)
        if "ticket_id" in r:                                   # esquema A
            filas.append({
                "ticket_id": r["ticket_id"],
                "cliente_id": int(str(r["customer"]).replace("CL-", "")),   # P10 'CL-010792' -> 10792
                "segmento_ticket": None,
                "creado_en": r["created_at"],
                "canal": r["channel"],
                "categoria": r["category"],
                "texto_cliente": r["text"],
                "respuesta_agente": None,
                "estado": {"closed": "resuelto", "open": "abierto"}.get(r["status"], r["status"]),
                "resuelto_en": r.get("resolved_at"),
                "csat": r.get("csat"),
                "esquema": "A_plataforma_antigua",
            })
        else:                                                  # esquema B
            msgs = r.get("mensajes") or []
            txt = lambda autor: " ".join(m["texto"] for m in msgs if m.get("autor") == autor) or None
            filas.append({
                "ticket_id": r["id"],
                "cliente_id": int(r["cliente"]["id"]),
                "segmento_ticket": r["cliente"].get("segmento"),
                "creado_en": r["fecha"],
                "canal": r["canal"],
                "categoria": r["tipo"]["sub"],
                "texto_cliente": txt("cliente"),
                "respuesta_agente": txt("agente"),
                "estado": r["estado"]["valor"],
                "resuelto_en": r["estado"].get("fecha"),
                "csat": (r.get("satisfaccion") or {}).get("csat"),   # P11 csat ausente en 22 % de resueltos B
                "esquema": "B_plataforma_nueva",
            })
    df = pd.DataFrame(filas)
    df["creado_en"] = _local(df["creado_en"])                 # P12 zonas horarias mezcladas
    df["resuelto_en"] = _local(df["resuelto_en"])
    df["csat"] = pd.to_numeric(df["csat"]).astype("Int64")
    df["categoria_principal"] = df["categoria"].map(config.CATEGORIA_PRINCIPAL)
    df["horas_resolucion"] = (df["resuelto_en"] - df["creado_en"]).dt.total_seconds() / 3600
    # P15 sarcasmo: palabras positivas para quejas (elogio irónico junto a un problema del
    # servicio). NLP por reglas validado (conecta/nlp.py, notebooks/02_nlp_tickets.ipynb):
    # nlp_sarcasmo, nlp_sentimiento (real), nlp_sentimiento_lexico (ingenuo, solo comparación),
    # nlp_intencion, nlp_frustracion (0-2), nlp_recurrente, nlp_tema_precio, nlp_clave_texto.
    df = nlp.enriquecer(df, col="texto_cliente")
    return df.drop_duplicates("ticket_id").sort_values("creado_en").reset_index(drop=True)


# --------------------------------------------------------------------------- eventos
def eventos(b: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame([json.loads(x) for x in b["raw_json"]])
    # P13 1.463 eventos duplicados exactos (mismo event_id y contenido)
    df = df.drop_duplicates()
    assert df["event_id"].is_unique, "event_id duplicado con contenido distinto"
    df["ts"] = _local(df["ts"])
    df["cliente_id"] = df["cliente_id"].astype("int64")
    return df.sort_values("ts").reset_index(drop=True)


# ------------------------------------------------------------------ etiquetas/puntuar
def etiquetas(b: pd.DataFrame) -> pd.DataFrame:
    df = b.drop(columns=META).copy()
    df["fecha_corte"] = pd.to_datetime(df["fecha_corte"], format="%Y-%m-%d")
    return df


def clientes_a_puntuar(b: pd.DataFrame) -> pd.DataFrame:
    return b.drop(columns=META).copy()


def run(lake_dir) -> dict:
    rb = lambda t: io.read(lake_dir, "bronze", t)
    out = {}
    subs = suscripciones(rb("suscripciones"))
    tablas = {
        "clientes": clientes(rb("clientes")),
        "suscripciones": subs,
        "facturas": facturas(rb("facturas"), subs),
        "tickets": tickets(rb("tickets")),
        "eventos": eventos(rb("eventos")),
        "etiquetas_entrenamiento": etiquetas(rb("etiquetas_entrenamiento")),
        "clientes_a_puntuar": clientes_a_puntuar(rb("clientes_a_puntuar")),
    }
    for t, df in tablas.items():
        out[t] = io.write(df, lake_dir, "silver", t)
    return out
