"""Experimento: ¿con qué frecuencia conviene actualizar el modelo?

Separa dos cosas que suelen confundirse:
- PUNTUAR más seguido (mismo modelo, lista más fresca).
- REENTRENAR (modelo nuevo con etiquetas recientes).

Simulación de puntuación mensual (sin reentrenar):
  A: puntuar una vez el 30-sep-2025 (como en la validación).
  B: puntuar el 30-sep y volver a puntuar el 31-oct y el 30-nov con el MISMO modelo
     (entrenado con los cortes de marzo y junio), llamando solo a los recién marcados,
     todavía activos y no llamados antes. Valor medido con la realidad hasta el 31-dic.
Limitación: las rondas de oct/nov solo se observan hasta el 31-dic (menos de 90 días),
así que su valor queda algo subestimado.

Uso:  python -m conecta.experimento_frecuencia
"""
import numpy as np
import pandas as pd

from . import config, gold, io, modelado as m, negocio

FIN_DATOS = pd.Timestamp(config.FECHA_HOY)
RONDAS = ["2025-09-30", "2025-10-31", "2025-11-30"]


def run(lake_dir=None):
    lake_dir = lake_dir or config.LAKE_DIR
    S = gold.cargar_silver(lake_dir)
    su = S["suscripciones"]
    F = io.read(lake_dir, "gold", "features")
    F["fecha_corte"] = pd.to_datetime(F.fecha_corte)
    modelo = m.entrenar(F[F.fecha_corte.isin(pd.to_datetime(m.CORTES_TRAIN))].reset_index(drop=True))
    fin_cli = su.groupby("cliente_id").apply(lambda s: pd.NaT if s.activa.any() else s.fecha_fin.max(),
                                             include_groups=False)

    def puntuar(c):
        act = su[(su.fecha_inicio <= c) & (su.fecha_fin.isna() | (su.fecha_fin > c))].cliente_id.unique()
        X = gold.features_corte(S, pd.DataFrame({"cliente_id": act, "fecha_corte": c}))
        p = modelo.predict_proba(X[m.X_COLS])[:, 1]
        ex = X.contacto_retencion_previo.to_numpy() == 1
        X["contactar"] = negocio.decidir(p, X.precio_plan, ex if config.EXCLUIR_CONTACTADOS_PREVIOS else None)
        X["fin"] = X.cliente_id.map(fin_cli)
        X["cancela"] = ((X.fin > c) & (X.fin <= FIN_DATOS)).astype(int)
        X["dias_hasta_baja"] = (X.fin - c).dt.days
        return X

    filas, ya, primera = [], set(), None
    for c in pd.to_datetime(RONDAS):
        X = puntuar(c)
        primera = X if primera is None else primera
        nuevos = X[(X.contactar == 1) & ~X.cliente_id.isin(ya)]
        ya |= set(nuevos.cliente_id)
        filas.append({"ronda": c.date(), "clientes_activos": len(X), "llamadas_nuevas": len(nuevos),
                      "de_ellos_cancelan": int(nuevos.cancela.sum()),
                      "valor_neto": negocio.valor_realizado(nuevos.cancela, np.ones(len(nuevos)), nuevos.precio_plan)})
    rondas = pd.DataFrame(filas)

    # Qué no ve una puntuación trimestral: quienes cancelan oct-dic sin ser marcados el 30-sep
    can = primera[primera.cancela == 1]
    perdidos = can[can.contactar == 0]
    tk = S["tickets"]
    t_cancel = tk[(tk.nlp_intencion == "cancelar") & (tk.creado_en >= pd.Timestamp(RONDAS[0]) + pd.Timedelta(days=1))]
    resumen = {
        "cancelan_oct_dic": len(can),
        "marcados_30sep": int(can.contactar.sum()),
        "no_marcados": len(perdidos),
        "pct_no_marcados_con_ticket_cancelar_posterior": perdidos.cliente_id.isin(t_cancel.cliente_id).mean(),
        "mediana_dias_hasta_baja_no_marcados": float(perdidos.dias_hasta_baja.median()),
        "valor_solo_trimestral": float(rondas.valor_neto.iloc[0]),
        "valor_extra_mensual": float(rondas.valor_neto.iloc[1:].sum()),
        "llamadas_extra_mensual": int(rondas.llamadas_nuevas.iloc[1:].sum()),
    }
    return rondas, resumen


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    r, s = run()
    print(r.assign(valor_neto=lambda d: (d.valor_neto / 1e6).round(2)).to_string(index=False))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
