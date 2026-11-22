"""Experimento: ¿generar cortes de entrenamiento adicionales mejora el modelo?

Diseño
- Población de un corte = clientes con suscripción activa en esa fecha (verificado: coincide
  3/3 con etiquetas_entrenamiento). Etiqueta = regla verificada (fin de la última suscripción
  en (corte, corte+90d]).
- Features PROVISIONALES calculadas al corte (point-in-time); las definitivas se construyen en
  gold en la fase de modelado. Aquí solo importa la comparación relativa.
- Prueba fuera de tiempo: corte oficial 2025-09-30 (etiquetas dadas). Ningún corte de
  entrenamiento tiene ventana de etiqueta que se solape con la prueba (último corte: 2025-06-30).
- Comparación pareada con bootstrap sobre el set de prueba.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config, io

H = config.HORIZONTE_DIAS


def _cargar(lake):
    r = lambda t: io.read(lake, "silver", t)
    return {t: r(t) for t in ["clientes", "suscripciones", "facturas", "tickets", "eventos"]}


def poblacion(S, corte):
    c = pd.Timestamp(corte)
    su = S["suscripciones"]
    act = su[(su.fecha_inicio <= c) & (su.fecha_fin.isna() | (su.fecha_fin > c))][["cliente_id", "plan"]]
    fin = su.groupby("cliente_id").apply(lambda s: pd.NaT if s.activa.any() else s.fecha_fin.max(), include_groups=False)
    act = act.merge(fin.rename("fin_cliente").reset_index(), on="cliente_id", how="left")
    act["churn_90d"] = ((act.fin_cliente > c) & (act.fin_cliente <= c + pd.Timedelta(days=H))).astype(int)
    act["fecha_corte"] = c
    return act.drop(columns="fin_cliente")


def features(S, pob):
    """Features al corte. Todo hecho con fecha <= corte (eventos/tickets: < corte + 1 día)."""
    c = pob.fecha_corte.iloc[0]
    c_fin = c + pd.Timedelta(days=1)
    cl, su, fa, tk, ev = S["clientes"], S["suscripciones"], S["facturas"], S["tickets"], S["eventos"]
    X = pob.merge(cl[["cliente_id", "fecha_alta", "segmento", "ciudad", "edad", "canal_adquisicion",
                      "plataforma_principal", "fecha_contacto_retencion"]], on="cliente_id", how="left")
    X["antiguedad_dias"] = (c - X.fecha_alta).dt.days
    X["precio_plan"] = X.plan.map(config.PRECIO_PLAN)
    X["contacto_previo"] = (X.fecha_contacto_retencion <= c).astype(int)
    X["n_suscripciones_previas"] = X.cliente_id.map(su[su.fecha_inicio <= c].groupby("cliente_id").size()) - 1
    # facturas conocidas al corte
    f = fa[fa.fecha_carga <= c]
    venc = f[f.fecha_vencimiento <= c]
    impaga = venc[venc.fecha_pago.isna() | (venc.fecha_pago > c)]
    X["facturas_impagas"] = X.cliente_id.map(impaga.groupby("cliente_id").size()).fillna(0)
    pag = venc[(venc.fecha_pago <= c) & (venc.periodo > c - pd.DateOffset(months=6))]
    X["atraso_prom_6m"] = X.cliente_id.map(pag.groupby("cliente_id").dias_atraso_pago.mean())
    X["pagos_tarde_6m"] = X.cliente_id.map((pag.dias_atraso_pago > 0).groupby(pag.cliente_id).sum()).fillna(0)
    # tickets
    t = tk[tk.creado_en < c_fin]
    t90 = t[t.creado_en >= c_fin - pd.Timedelta(days=90)]
    t30 = t[t.creado_en >= c_fin - pd.Timedelta(days=30)]
    g90 = t90.groupby("cliente_id")
    X["tickets_90d"] = X.cliente_id.map(g90.size()).fillna(0)
    X["tickets_cancelar_30d"] = X.cliente_id.map((t30.nlp_intencion == "cancelar").groupby(t30.cliente_id).sum()).fillna(0)
    X["tickets_amenaza_90d"] = X.cliente_id.map((t90.nlp_intencion == "amenaza_irse").groupby(t90.cliente_id).sum()).fillna(0)
    X["tickets_sarcasmo_90d"] = X.cliente_id.map(g90.nlp_sarcasmo.sum()).fillna(0)
    X["frustracion_max_90d"] = X.cliente_id.map(g90.nlp_frustracion.max()).fillna(0)
    X["tickets_precio_90d"] = X.cliente_id.map(g90.nlp_tema_precio.sum()).fillna(0)
    X["tickets_app_90d"] = X.cliente_id.map((t90.categoria == "app_se_cierra").groupby(t90.cliente_id).sum()).fillna(0)
    abiertos = t[(t.resuelto_en.isna()) | (t.resuelto_en >= c_fin)]
    X["tickets_abiertos"] = X.cliente_id.map(abiertos.groupby("cliente_id").size()).fillna(0)
    # eventos de app (solo existen en 2025)
    e = ev[(ev.ts < c_fin) & (ev.ts >= c_fin - pd.Timedelta(days=30))]
    X["logins_30d"] = X.cliente_id.map((e.tipo == "login").groupby(e.cliente_id).sum()).fillna(0)
    X["uso_30d"] = X.cliente_id.map((e.tipo == "uso_funcionalidad").groupby(e.cliente_id).sum()).fillna(0)
    X["crash_30d"] = X.cliente_id.map((e.codigo_error == "CRASH_ON_START").groupby(e.cliente_id).sum()).fillna(0)
    X[NUM] = X[NUM].astype("float64")
    return X


NUM = ["antiguedad_dias", "precio_plan", "edad", "contacto_previo", "n_suscripciones_previas", "facturas_impagas",
       "atraso_prom_6m", "pagos_tarde_6m", "tickets_90d", "tickets_cancelar_30d", "tickets_amenaza_90d",
       "tickets_sarcasmo_90d", "frustracion_max_90d", "tickets_precio_90d", "tickets_app_90d", "tickets_abiertos",
       "logins_30d", "uso_30d", "crash_30d"]
CAT = ["plan", "segmento", "ciudad", "canal_adquisicion", "plataforma_principal"]


def dataset(S, cortes):
    return pd.concat([features(S, poblacion(S, c)) for c in cortes], ignore_index=True)


def modelos():
    import lightgbm as lgb
    pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    pre_arbol = ColumnTransformer([("num", "passthrough", NUM),
                                   ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return {
        "logistica": make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5)),
        "lightgbm": make_pipeline(pre_arbol, lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                                               min_child_samples=40, subsample=0.8, subsample_freq=1,
                                                               colsample_bytree=0.8, random_state=42, verbose=-1)),
    }


def metricas(y, p):
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p)}


def bootstrap_delta(y, p_ref, p_alt, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p_ref, p_alt = map(np.asarray, (y, p_ref, p_alt))
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p_alt[i]) - roc_auc_score(y[i], p_ref[i]))
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


VARIANTES = {
    "A_oficiales_mar_jun": ["2025-03-31", "2025-06-30"],
    "B_mensual_2025": [str(d.date()) for d in pd.date_range("2025-01-31", "2025-06-30", freq="ME")],
    "C_mensual_desde_2024": [str(d.date()) for d in pd.date_range("2024-01-31", "2025-06-30", freq="ME")],
    "D_mensual_desde_2023": [str(d.date()) for d in pd.date_range("2023-07-31", "2025-06-30", freq="ME")],
}
CORTE_PRUEBA = "2025-09-30"


def run(lake_dir):
    S = _cargar(lake_dir)
    test = dataset(S, [CORTE_PRUEBA])
    filas, preds = [], {}
    for var, cortes in VARIANTES.items():
        train = dataset(S, cortes)
        for nombre, m in modelos().items():
            m.fit(train[NUM + CAT], train.churn_90d)
            p = m.predict_proba(test[NUM + CAT])[:, 1]
            preds[(var, nombre)] = p
            filas.append({"variante": var, "modelo": nombre, "n_cortes": len(cortes), "filas_train": len(train),
                          "tasa_churn_train": train.churn_90d.mean(), **metricas(test.churn_90d, p)})
    res = pd.DataFrame(filas)
    deltas = []
    for (var, nombre), p in preds.items():
        if var.startswith("A_"):
            continue
        m, lo, hi = bootstrap_delta(test.churn_90d, preds[("A_oficiales_mar_jun", nombre)], p)
        deltas.append({"variante": var, "modelo": nombre, "delta_auc_vs_A": m, "ic95_bajo": lo, "ic95_alto": hi})
    return res, pd.DataFrame(deltas), test
