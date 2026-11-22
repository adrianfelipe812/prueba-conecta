"""Comparación de modelos de churn con selección por costo de la decisión.

Protocolo
1. Entrenamiento: cortes 2025-03-31 y 2025-06-30. Validación fuera de tiempo: 2025-09-30.
2. Cada candidato se calibra (Platt/sigmoide) con predicciones fuera de pliegue
   (GroupKFold por cliente: un cliente nunca está en entrenamiento y calibración a la vez),
   y luego se reentrena con todo el entrenamiento (CalibratedClassifierCV ensemble=False).
3. Se aplica la regla de valor esperado (negocio.decidir) y se mide el valor neto real en
   validación. Criterio principal: valor neto; desempate: PR-AUC y Brier.
4. Variantes "peso_valor": entrenamiento sensible al costo (cada churner pesa lo que vale
   retenerlo, 4,8 × precio); se recalibra igual.
5. El ganador se reentrena con los 3 cortes y puntúa al 2025-12-31.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config, gold, negocio

warnings.filterwarnings("ignore")
SEED = 42
CORTES_TRAIN = ["2025-03-31", "2025-06-30"]
CORTE_VALID = "2025-09-30"
X_COLS = gold.NUM + gold.CAT


class TasaBase(BaseEstimator, ClassifierMixin):
    """Baseline sin ML: tasa histórica de churn por plan × segmento."""

    def fit(self, X, y, sample_weight=None):
        d = pd.DataFrame({"k": X["plan"].astype(str) + "|" + X["segmento"].astype(str), "y": np.asarray(y)})
        self.tasas_ = d.groupby("k")["y"].mean().to_dict()
        self.global_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        k = X["plan"].astype(str) + "|" + X["segmento"].astype(str)
        p = k.map(self.tasas_).fillna(self.global_).to_numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def _pre(escalar=True):
    num = [SimpleImputer(strategy="median", add_indicator=True)] + ([StandardScaler()] if escalar else [])
    return ColumnTransformer([("num", make_pipeline(*num), gold.NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), gold.CAT)],
                             sparse_threshold=0)


def _pre_nativo():
    return ColumnTransformer([("num", "passthrough", gold.NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), gold.CAT)],
                             sparse_threshold=0)


def candidatos() -> dict:
    import lightgbm as lgb
    lgbm = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)
    m = {
        "00_base_tasa_plan_segmento": TasaBase(),
        "01_logistica": make_pipeline(_pre(), LogisticRegression(C=0.5, max_iter=3000)),
        "02_logistica_l1": make_pipeline(_pre(), LogisticRegression(C=0.2, penalty="l1", solver="liblinear", random_state=SEED)),
        "03_arbol_decision": make_pipeline(_pre(False), DecisionTreeClassifier(max_depth=5, min_samples_leaf=40,
                                                                               random_state=SEED)),
        "04_random_forest": make_pipeline(_pre(False), RandomForestClassifier(n_estimators=500, min_samples_leaf=10,
                                                                              max_features="sqrt", n_jobs=-1,
                                                                              random_state=SEED)),
        "05_extra_trees": make_pipeline(_pre(False), ExtraTreesClassifier(n_estimators=500, min_samples_leaf=10,
                                                                          max_features="sqrt", n_jobs=-1,
                                                                          random_state=SEED)),
        "06_hist_gradient_boosting": make_pipeline(_pre_nativo(), HistGradientBoostingClassifier(
            learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, max_iter=300,
            random_state=SEED)),
        "07_lightgbm": make_pipeline(_pre_nativo(), lgb.LGBMClassifier(**lgbm)),
        "08_knn": make_pipeline(_pre(), KNeighborsClassifier(n_neighbors=50, weights="distance")),
        "09_naive_bayes": make_pipeline(_pre(), GaussianNB()),
        "10_svm_rbf": make_pipeline(_pre(), SVC(C=1.0, gamma="scale", random_state=SEED)),
        "11_red_neuronal_mlp": make_pipeline(_pre(), MLPClassifier(hidden_layer_sizes=(16,), alpha=3.0,
                                                                   max_iter=2000, random_state=SEED)),
    }
    m["12_ensamble_log_rf_lgbm"] = VotingClassifier(
        [("log", clone(m["01_logistica"])), ("rf", clone(m["04_random_forest"])), ("lgbm", clone(m["07_lightgbm"]))],
        voting="soft")
    m["13_logistica_peso_valor"] = clone(m["01_logistica"])
    m["14_lightgbm_peso_valor"] = clone(m["07_lightgbm"])
    return m


def _pesos(nombre, X, y):
    """Entrenamiento sensible al costo: un churner pesa el valor de retenerlo (normalizado)."""
    if "peso_valor" not in nombre:
        return None
    w = np.where(np.asarray(y) == 1, negocio.GANANCIA * X["precio_plan"].to_numpy(), config.COSTO_CONTACTO)
    return w / w.mean()


def ajustar(nombre, base, X, y, grupos):
    """Modelo calibrado (sigmoide, fuera de pliegue por cliente) entrenado con X, y."""
    if isinstance(base, TasaBase):            # ya es una frecuencia observada: no se calibra
        return clone(base).fit(X, y)
    splits = list(GroupKFold(n_splits=5).split(X, y, grupos))
    w = _pesos(nombre, X, y)
    if w is not None:
        # Los pesos se usan para entrenar; la calibración se ajusta SIN pesos para que la
        # probabilidad final siga siendo una frecuencia real (la regla de decisión la necesita).
        return _ajustar_ponderado(base, X, y, w, splits)
    cal = CalibratedClassifierCV(clone(base), method="sigmoid", cv=splits, ensemble=False)
    return cal.fit(X, y)


def sin_calibrar(base, X, y):
    return clone(base).fit(X, y)


class _Ponderado(BaseEstimator, ClassifierMixin):
    """Estimador entrenado con pesos + calibrador sigmoide sin pesos (ajustado fuera de pliegue)."""

    def __init__(self, estimador, calibrador):
        self.estimador, self.calibrador = estimador, calibrador
        self.classes_ = np.array([0, 1])

    def _score(self, X):
        return self.estimador.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        s = self._score(X).reshape(-1, 1)
        p = self.calibrador.predict_proba(np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6))))[:, 1]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def _ajustar_ponderado(base, X, y, w, splits):
    paso = base.steps[-1][0]
    oof = np.zeros(len(y))
    for tr, te in splits:
        est = clone(base).fit(X.iloc[tr], y.iloc[tr], **{f"{paso}__sample_weight": w[tr]})
        oof[te] = est.predict_proba(X.iloc[te])[:, 1]
    logit = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / (1 - np.clip(oof, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    calib = LogisticRegression(C=1e6).fit(logit, y)
    final = clone(base).fit(X, y, **{f"{paso}__sample_weight": w})
    return _Ponderado(final, calib)


def metricas(y, p, precio) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p),
            "prob_media": p.mean(), "tasa_real": y.mean(), **negocio.resumen(y, p, precio)}


def datos(lake_dir):
    from . import io
    F = io.read(lake_dir, "gold", "features")
    F["fecha_corte"] = pd.to_datetime(F["fecha_corte"])
    tr = F[F["fecha_corte"].isin(pd.to_datetime(CORTES_TRAIN))].reset_index(drop=True)
    va = F[F["fecha_corte"] == pd.Timestamp(CORTE_VALID)].reset_index(drop=True)
    sc = F[F["conjunto"] == "puntuar"].reset_index(drop=True)
    for d in (tr, va):
        d["churn_90d"] = d["churn_90d"].astype(int)
    return tr, va, sc


def ajuste_prior(p, tasa_modelo, tasa_objetivo):
    """Corrige un cambio de prevalencia (prior shift) manteniendo el orden: multiplica las odds."""
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    k = (tasa_objetivo / (1 - tasa_objetivo)) / (tasa_modelo / (1 - tasa_modelo))
    o = p / (1 - p) * k
    return o / (1 + o)


# Validación temporal con origen móvil: (cortes de entrenamiento, corte de validación)
PLIEGUES = [(["2025-03-31"], "2025-06-30"), (["2025-03-31", "2025-06-30"], "2025-09-30")]


def comparar(lake_dir, pliegues=PLIEGUES):
    """Entrena/valida cada candidato en cada pliegue temporal.

    Para cada pliegue se reportan dos versiones de la probabilidad:
      - 'calibrada': calibración fuera de pliegue sobre los cortes de entrenamiento
      - 'ajuste_prior': además lleva la prevalencia a la del ÚLTIMO corte de entrenamiento
        (la información más reciente disponible en ese momento, sin mirar la validación)
    """
    from . import io
    F = io.read(lake_dir, "gold", "features")
    F["fecha_corte"] = pd.to_datetime(F["fecha_corte"])
    filas, preds, modelos = [], {}, {}
    for cortes_tr, corte_va in pliegues:
        tr = F[F["fecha_corte"].isin(pd.to_datetime(cortes_tr))].reset_index(drop=True)
        va = F[F["fecha_corte"] == pd.Timestamp(corte_va)].reset_index(drop=True)
        ytr, yva = tr["churn_90d"].astype(int), va["churn_90d"].astype(int)
        tasa_tr = ytr.mean()
        tasa_ultimo = ytr[tr["fecha_corte"] == pd.Timestamp(cortes_tr[-1])].mean()
        for nombre, base in candidatos().items():
            mod = ajustar(nombre, base, tr[X_COLS], ytr, tr["cliente_id"])
            p = mod.predict_proba(va[X_COLS])[:, 1]
            for version, pv in [("calibrada", p), ("ajuste_prior", ajuste_prior(p, tasa_tr, tasa_ultimo))]:
                preds[(corte_va, nombre, version)] = pv
                filas.append({"valida": corte_va, "modelo": nombre, "version": version,
                              **metricas(yva, pv, va["precio_plan"])})
            modelos[(corte_va, nombre)] = mod
    return pd.DataFrame(filas), preds, modelos


def ranking(res: pd.DataFrame) -> pd.DataFrame:
    """Ranking por % del valor máximo promedio entre pliegues (normaliza escalas distintas)."""
    g = res.groupby(["modelo", "version"])
    r = g.agg(pct_max_prom=("pct_del_maximo", "mean"), pct_max_min=("pct_del_maximo", "min"),
              valor_neto_prom=("valor_neto", "mean"), roc_auc_prom=("roc_auc", "mean"),
              pr_auc_prom=("pr_auc", "mean"), brier_prom=("brier", "mean")).reset_index()
    piv = res.pivot_table(index=["modelo", "version"], columns="valida", values="pct_del_maximo")
    piv.columns = [f"pct_max_{c}" for c in piv.columns]
    return r.merge(piv.reset_index(), on=["modelo", "version"]).sort_values(
        ["pct_max_prom", "pr_auc_prom"], ascending=False).reset_index(drop=True)


def bootstrap_valor(y, precio, preds: dict, ref: str, n=1000, seed=0) -> pd.DataFrame:
    """Diferencia de valor neto (modelo − ref) con IC 95 %, remuestreando clientes de validación."""
    rng = np.random.default_rng(seed)
    y, precio = np.asarray(y), np.asarray(precio, dtype=float)
    dec = {k: negocio.decidir(p, precio) for k, p in preds.items()}
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n)]
    out = []
    for k in preds:
        if k == ref:
            continue
        d = [negocio.valor_realizado(y[i], dec[k][i], precio[i]) - negocio.valor_realizado(y[i], dec[ref][i], precio[i])
             for i in idx]
        out.append({"modelo": k, "delta_valor_vs_" + ref: np.mean(d), "ic95_bajo": np.percentile(d, 2.5),
                    "ic95_alto": np.percentile(d, 97.5), "prob_mejor_que_ref": np.mean(np.array(d) > 0)})
    return pd.DataFrame(out).sort_values("delta_valor_vs_" + ref, ascending=False)


def registrar_mlflow(lake_dir, uri="sqlite:///mlflow.db", experimento="conecta_churn", resultados=None):
    """Registra en MLflow: cada candidato (métricas de ambos pliegues + modelo del pliegue sep),
    el ensamble elegido y el modelo final. Devuelve mlflow.search_runs()."""
    import os
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    import mlflow.sklearn
    from . import io
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experimento)
    res, preds, mods = resultados if resultados is not None else comparar(lake_dir)   # reutiliza si ya se corrió
    F = io.read(lake_dir, "gold", "features")
    F["fecha_corte"] = pd.to_datetime(F["fecha_corte"])
    va = F[F["fecha_corte"] == pd.Timestamp(CORTE_VALID)].reset_index(drop=True)
    tags = {"validacion": "origen movil: mar->jun, mar+jun->sep",
            "calibracion": "sigmoide fuera de pliegue (GroupKFold por cliente)",
            "criterio": "valor neto de la regla de contacto (% del maximo)"}

    def _log(nombre, filas, modelo, fase, extra=None):
        with mlflow.start_run(run_name=nombre):
            mlflow.set_tags({**tags, "fase": fase})
            mlflow.log_params({"modelo": nombre, "n_features": len(X_COLS), **(extra or {})})
            for r in filas:
                suf = "_" + r["valida"][5:7]                   # _06 / _09
                mlflow.log_metrics({k + suf: float(v) for k, v in r.items()
                                    if k not in ("valida", "modelo", "version") and pd.notna(v)})
            mlflow.log_metric("pct_max_prom", float(np.mean([r["pct_del_maximo"] for r in filas])))
            mlflow.log_metric("valor_neto_09", float([r for r in filas if r["valida"] == CORTE_VALID][0]["valor_neto"]))
            # skops pide declarar tipos no estándar; solo son clases propias (conecta.*) y de
            # librerías instaladas (lightgbm), generadas por este mismo código.
            import skops.io as sio
            tipos = sio.get_untrusted_types(data=sio.dumps(modelo))
            assert all(t.split(".")[0] in ("conecta", "lightgbm", "numpy", "sklearn", "builtins", "collections") for t in tipos), tipos
            mlflow.sklearn.log_model(modelo, name="modelo", input_example=va[X_COLS].head(3),
                                     skops_trusted_types=tipos)

    cal = res[res["version"] == "calibrada"]
    for nombre, g in cal.groupby("modelo"):
        _log(nombre, g.to_dict("records"), mods[(CORTE_VALID, nombre)], "comparacion",
             {"peso_valor": "peso_valor" in nombre})
    # ensamble elegido (evaluado en ambos pliegues con los mismos pesos fijos)
    filas = []
    for cortes_tr, corte_va in PLIEGUES:
        v = F[F["fecha_corte"] == pd.Timestamp(corte_va)].reset_index(drop=True)
        p = sum(w * preds[(corte_va, k, "calibrada")] for k, w in ELEGIDOS.items())
        filas.append({"valida": corte_va, **metricas(v["churn_90d"].astype(int), p, v["precio_plan"])})
    tr = F[F["fecha_corte"].isin(pd.to_datetime(CORTES_TRAIN))].reset_index(drop=True)
    _log("20_ensamble_elegido_l1_rf_50_50", filas, entrenar(tr), "eleccion",
         {"pesos": str(ELEGIDOS)})
    return mlflow.search_runs(experiment_names=[experimento], order_by=["metrics.pct_max_prom DESC"])


# ---------------------------------------------------------------------------- modelo elegido
ELEGIDOS = {"02_logistica_l1": 0.5, "04_random_forest": 0.5}   # pesos fijados a priori (50/50)


class EnsambleCalibrado(BaseEstimator, ClassifierMixin):
    """Promedio ponderado de probabilidades de modelos ya calibrados."""

    def __init__(self, modelos: dict, pesos: dict):
        self.modelos, self.pesos = modelos, pesos
        self.classes_ = np.array([0, 1])

    def fit(self, X=None, y=None):
        """Los componentes ya vienen entrenados (usar entrenar()); se define para compatibilidad."""
        return self

    def predict_proba(self, X):
        p = sum(self.pesos[k] * self.modelos[k].predict_proba(X)[:, 1] for k in self.modelos)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def entrenar(df: pd.DataFrame, elegidos=ELEGIDOS) -> EnsambleCalibrado:
    c = candidatos()
    y = df["churn_90d"].astype(int)
    mods = {k: ajustar(k, c[k], df[X_COLS], y, df["cliente_id"]) for k in elegidos}
    return EnsambleCalibrado(mods, elegidos)


def predecir(modelo, df: pd.DataFrame) -> pd.DataFrame:
    p = modelo.predict_proba(df[X_COLS])[:, 1]
    excluir = df["contacto_retencion_previo"].to_numpy() == 1 if config.EXCLUIR_CONTACTADOS_PREVIOS else None
    out = pd.DataFrame({"cliente_id": df["cliente_id"].astype(int), "probabilidad": p.round(6),
                        "contactar": negocio.decidir(p, df["precio_plan"], excluir)})
    out["excluido_por_contacto_previo"] = (excluir if excluir is not None else np.zeros(len(df), bool)).astype(int)
    out["plan"], out["precio_plan"] = df["plan"].values, df["precio_plan"].values
    out["valor_esperado_contacto"] = negocio.valor_esperado(p, df["precio_plan"]).round(0)
    return out


def validar_predicciones(pred: pd.DataFrame, a_puntuar: pd.DataFrame):
    assert list(pred.columns[:3]) == ["cliente_id", "probabilidad", "contactar"]
    assert len(pred) == len(a_puntuar) and set(pred.cliente_id) == set(a_puntuar.cliente_id), "clientes distintos"
    assert pred.cliente_id.is_unique and pred.notna().all().all()
    assert pred.probabilidad.between(0, 1).all() and pred.contactar.isin([0, 1]).all()
    return True
