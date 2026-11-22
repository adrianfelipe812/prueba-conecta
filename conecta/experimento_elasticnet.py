"""Experimento: Ridge (L2), Lasso (L1) y Elastic Net con el mismo protocolo de selección.

Mismos pliegues temporales (mar → jun, mar+jun → sep), misma calibración (sigmoide fuera de
pliegue con GroupKFold por cliente) y mismo criterio (% del valor máximo de la decisión).
Elastic Net se prueba con tres mezclas de L1; elegir la mejor mirando la validación infla
su resultado, así que la comparación se lee como empate salvo diferencias grandes.

Uso:  python -m conecta.experimento_elasticnet
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from . import config, io, modelado as m

warnings.filterwarnings("ignore")


def candidatos():
    c = {
        "ridge_l2": make_pipeline(m._pre(), LogisticRegression(C=0.5, max_iter=3000)),
        "lasso_l1": make_pipeline(m._pre(), LogisticRegression(C=0.2, penalty="l1", solver="liblinear",
                                                               random_state=m.SEED)),
    }
    for r in (0.25, 0.5, 0.75):
        c[f"elasticnet_l1ratio_{r}"] = make_pipeline(m._pre(), LogisticRegression(
            C=0.2, penalty="elasticnet", l1_ratio=r, solver="saga", max_iter=5000, random_state=m.SEED))
    return c


def run(lake_dir=None):
    lake_dir = lake_dir or config.LAKE_DIR
    F = io.read(lake_dir, "gold", "features")
    F["fecha_corte"] = pd.to_datetime(F.fecha_corte)
    filas = []
    for nombre, base in candidatos().items():
        fila = {"modelo": nombre}
        for cortes, cva in m.PLIEGUES:
            tr = F[F.fecha_corte.isin(pd.to_datetime(cortes))].reset_index(drop=True)
            va = F[F.fecha_corte == pd.Timestamp(cva)].reset_index(drop=True)
            mod = m.ajustar(nombre, base, tr[m.X_COLS], tr.churn_90d.astype(int), tr.cliente_id)
            met = m.metricas(va.churn_90d.astype(int), mod.predict_proba(va[m.X_COLS])[:, 1], va.precio_plan)
            fila[f"pct_max_{cva[5:7]}"] = met["pct_del_maximo"]
            fila[f"roc_auc_{cva[5:7]}"] = met["roc_auc"]
        coef = mod.calibrated_classifiers_[0].estimator[-1].coef_[0]     # modelo entrenado con mar+jun
        fila["coef_en_cero"] = int((np.abs(coef) < 1e-8).sum())
        fila["coef_total"] = len(coef)
        fila["pct_max_prom"] = (fila["pct_max_06"] + fila["pct_max_09"]) / 2
        filas.append(fila)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(run().round(4).to_string(index=False))
