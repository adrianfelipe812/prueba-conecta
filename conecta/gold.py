"""Capa GOLD: tablas listas para modelar.

- spine: cliente x fecha_corte (3 cortes de entrenamiento + 2025-12-31 a puntuar).
- features: variables calculadas AL CORTE (point-in-time) con la MISMA función para
  entrenamiento y puntuación. Reglas anti-fuga:
    * facturas: solo con fecha_carga <= corte; pago conocido solo si fecha_pago <= corte
    * tickets/eventos: marca de tiempo local < corte + 1 día
    * tickets: estado/csat solo si se resolvieron antes del corte
    * nunca: estado_actual, motivo_cancelacion, fecha_fin, contactos posteriores al corte
"""
import numpy as np
import pandas as pd

from . import config, io

# Variables del modelo
NUM = [  # diccionario: docs/features.md
    # perfil y contrato
    "antiguedad_dias", "edad", "flag_edad_invalida", "precio_plan", "n_cambios_plan", "bajo_de_plan_antes",
    "dias_en_plan_actual", "afectado_aumento_basico", "contacto_retencion_previo",
    # facturación
    "facturas_impagas", "facturas_impagas_3m", "atraso_prom_6m", "pagos_tarde_6m", "dias_desde_ultimo_pago",
    # tickets + NLP
    "tickets_30d", "tickets_90d", "tickets_total", "dias_desde_ultimo_ticket", "tickets_cancelar_30d",
    "tickets_amenaza_90d", "tickets_bajar_plan_90d", "tickets_sarcasmo_90d", "frustracion_max_90d",
    "tickets_recurrente_90d", "tickets_precio_90d", "tickets_app_90d", "tickets_cobro_90d", "tickets_abiertos",
    "csat_min_90d", "csat_prom_90d",
    # uso de la app
    "logins_30d", "logins_31_90d", "tendencia_logins", "uso_30d", "duracion_uso_30d", "crash_30d",
    "errores_30d", "dias_desde_ultimo_evento", "sin_eventos_90d", "usa_version_540",
]
CAT = ["plan", "segmento", "ciudad", "canal_adquisicion", "plataforma_principal"]


def spine(lake_dir) -> pd.DataFrame:
    et = io.read(lake_dir, "silver", "etiquetas_entrenamiento").assign(conjunto="entrenamiento")
    cp = io.read(lake_dir, "silver", "clientes_a_puntuar").assign(
        fecha_corte=pd.Timestamp(config.FECHA_HOY), churn_90d=pd.NA, conjunto="puntuar")
    df = pd.concat([et, cp], ignore_index=True)
    df["churn_90d"] = df["churn_90d"].astype("Int64")
    return df[["cliente_id", "fecha_corte", "churn_90d", "conjunto"]]


def cargar_silver(lake_dir) -> dict:
    return {t: io.read(lake_dir, "silver", t) for t in ["clientes", "suscripciones", "facturas", "tickets", "eventos"]}


def _contar(serie_bool, ids):
    return serie_bool.groupby(ids).sum()


def features_corte(S: dict, base: pd.DataFrame) -> pd.DataFrame:
    """base: filas (cliente_id, fecha_corte, ...) de UN solo corte."""
    c = pd.Timestamp(base["fecha_corte"].iloc[0])
    c_fin = c + pd.Timedelta(days=1)
    cl, su, fa, tk, ev = S["clientes"], S["suscripciones"], S["facturas"], S["tickets"], S["eventos"]
    X = base.merge(cl[["cliente_id", "fecha_alta", "segmento", "ciudad", "edad", "flag_edad_invalida",
                       "canal_adquisicion", "plataforma_principal", "fecha_contacto_retencion",
                       "flag_contacto_antes_alta"]],
                   on="cliente_id", how="left")
    ids = X["cliente_id"]
    m = lambda s: ids.map(s)

    # --- contrato al corte
    s_prev = su[su.fecha_inicio <= c]
    vig = s_prev[s_prev.fecha_fin.isna() | (s_prev.fecha_fin > c)].set_index("cliente_id")
    X["plan"] = m(vig["plan"])
    X["precio_plan"] = X["plan"].map(config.PRECIO_PLAN)
    X["dias_en_plan_actual"] = (c - m(vig["fecha_inicio"])).dt.days
    X["antiguedad_dias"] = (c - X["fecha_alta"]).dt.days
    X["n_cambios_plan"] = m(s_prev.groupby("cliente_id").size()).fillna(1) - 1
    s_prev = s_prev.assign(precio=s_prev["plan"].map(config.PRECIO_PLAN))
    prev_precio = s_prev.sort_values("fecha_inicio").groupby("cliente_id")["precio"].apply(
        lambda p: int((p.diff() < 0).any()))
    X["bajo_de_plan_antes"] = m(prev_precio).fillna(0)
    # P03: las 76 fechas de contacto anteriores al alta son errores -> no cuentan como contacto
    X["contacto_retencion_previo"] = ((X["fecha_contacto_retencion"] <= c) &
                                      ~X["flag_contacto_antes_alta"].astype(bool)).astype(int)

    # --- facturación conocida al corte
    f = fa[fa["fecha_carga"] <= c]
    X["afectado_aumento_basico"] = ((X["plan"] == "Basico") &
                                    ids.isin(f.loc[f["monto"] == 19900, "cliente_id"]) &
                                    (c >= pd.Timestamp("2025-04-01"))).astype(int)
    venc = f[f["fecha_vencimiento"] <= c]
    impaga = venc[venc["fecha_pago"].isna() | (venc["fecha_pago"] > c)]
    X["facturas_impagas"] = m(impaga.groupby("cliente_id").size()).fillna(0)
    X["facturas_impagas_3m"] = m(impaga[impaga["periodo"] > c - pd.DateOffset(months=3)]
                                 .groupby("cliente_id").size()).fillna(0)
    pag = venc[venc["fecha_pago"] <= c]
    pag6 = pag[pag["periodo"] > c - pd.DateOffset(months=6)]
    X["atraso_prom_6m"] = m(pag6.groupby("cliente_id")["dias_atraso_pago"].mean())
    X["pagos_tarde_6m"] = m(_contar(pag6["dias_atraso_pago"] > 0, pag6["cliente_id"])).fillna(0)
    X["dias_desde_ultimo_pago"] = (c - m(pag.groupby("cliente_id")["fecha_pago"].max())).dt.days

    # --- tickets + NLP
    t = tk[tk["creado_en"] < c_fin]
    t90 = t[t["creado_en"] >= c_fin - pd.Timedelta(days=90)]
    t30 = t[t["creado_en"] >= c_fin - pd.Timedelta(days=30)]
    g90 = t90.groupby("cliente_id")
    X["tickets_total"] = m(t.groupby("cliente_id").size()).fillna(0)
    X["tickets_90d"] = m(g90.size()).fillna(0)
    X["tickets_30d"] = m(t30.groupby("cliente_id").size()).fillna(0)
    X["dias_desde_ultimo_ticket"] = (c_fin - m(t.groupby("cliente_id")["creado_en"].max())).dt.days
    X["tickets_cancelar_30d"] = m(_contar(t30["nlp_intencion"] == "cancelar", t30["cliente_id"])).fillna(0)
    for col, cond in [("tickets_amenaza_90d", t90["nlp_intencion"] == "amenaza_irse"),
                      ("tickets_bajar_plan_90d", t90["nlp_intencion"] == "bajar_plan"),
                      ("tickets_app_90d", t90["categoria"] == "app_se_cierra"),
                      ("tickets_cobro_90d", t90["categoria"].isin(["cobro_doble", "no_reconoce_cobro"]))]:
        X[col] = m(_contar(cond, t90["cliente_id"])).fillna(0)
    X["tickets_sarcasmo_90d"] = m(g90["nlp_sarcasmo"].sum()).fillna(0)
    X["tickets_recurrente_90d"] = m(g90["nlp_recurrente"].sum()).fillna(0)
    X["tickets_precio_90d"] = m(g90["nlp_tema_precio"].sum()).fillna(0)
    X["frustracion_max_90d"] = m(g90["nlp_frustracion"].max()).fillna(0)
    abiertos = t[t["resuelto_en"].isna() | (t["resuelto_en"] >= c_fin)]
    X["tickets_abiertos"] = m(abiertos.groupby("cliente_id").size()).fillna(0)
    res90 = t90[t90["resuelto_en"] < c_fin]                       # csat conocido al corte
    X["csat_min_90d"] = m(res90.groupby("cliente_id")["csat"].min().astype("float"))
    X["csat_prom_90d"] = m(res90.groupby("cliente_id")["csat"].mean().astype("float"))

    # --- eventos de la app (solo 2025)
    e = ev[ev["ts"] < c_fin]
    e30 = e[e["ts"] >= c_fin - pd.Timedelta(days=30)]
    e90 = e[e["ts"] >= c_fin - pd.Timedelta(days=90)]
    e31_90 = e90[e90["ts"] < c_fin - pd.Timedelta(days=30)]
    X["logins_30d"] = m(_contar(e30["tipo"] == "login", e30["cliente_id"])).fillna(0)
    X["logins_31_90d"] = m(_contar(e31_90["tipo"] == "login", e31_90["cliente_id"])).fillna(0)
    X["tendencia_logins"] = X["logins_30d"] - X["logins_31_90d"] / 2       # >0 sube, <0 cae
    X["uso_30d"] = m(_contar(e30["tipo"] == "uso_funcionalidad", e30["cliente_id"])).fillna(0)
    X["duracion_uso_30d"] = m(e30.groupby("cliente_id")["duracion_seg"].sum()).fillna(0)
    X["crash_30d"] = m(_contar(e30["codigo_error"] == "CRASH_ON_START", e30["cliente_id"])).fillna(0)
    X["errores_30d"] = m(_contar(e30["tipo"] == "error_app", e30["cliente_id"])).fillna(0)
    X["dias_desde_ultimo_evento"] = (c_fin - m(e.groupby("cliente_id")["ts"].max())).dt.days
    X["sin_eventos_90d"] = (~ids.isin(e90["cliente_id"])).astype(int)
    X["usa_version_540"] = ids.isin(e90.loc[e90["version_app"] == "5.4.0", "cliente_id"]).astype(int)

    X["flag_edad_invalida"] = X["flag_edad_invalida"].astype(int)
    X[NUM] = X[NUM].astype("float64")
    return X[["cliente_id", "fecha_corte"] + [c for c in base.columns if c not in ("cliente_id", "fecha_corte")]
             + NUM + CAT]


def features(lake_dir, sp: pd.DataFrame = None) -> pd.DataFrame:
    S = cargar_silver(lake_dir)
    sp = spine(lake_dir) if sp is None else sp
    partes = [features_corte(S, g) for _, g in sp.groupby("fecha_corte")]
    return pd.concat(partes, ignore_index=True)


def run(lake_dir) -> dict:
    sp = spine(lake_dir)
    out = {"spine": io.write(sp, lake_dir, "gold", "spine")}
    out["features"] = io.write(features(lake_dir, sp), lake_dir, "gold", "features")
    return out
