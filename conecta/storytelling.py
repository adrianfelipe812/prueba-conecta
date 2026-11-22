"""Genera `storytelling.html`: la historia de negocio de los hallazgos (3 actos).

Principios aplicados (curso de Visualización de datos):
- Una Gran Idea y 3 actos: planteamiento → nudo → desenlace / llamada a la acción.
- Bing (título que afirma el hallazgo) · Bang (gráfico, el color dirige la mirada) ·
  Bongo (anotación / conclusión visible).
- Foco vs. contexto: azul institucional = foco, coral = el problema, gris = contexto.
- Gráfico según intención: tendencia → líneas; comparación → barras con eje en cero y
  ordenadas; segmentación → barras apiladas (nunca tortas).
- Mantra de Shneiderman: visión general → filtro → detalle bajo demanda (hovertemplate limpio).
- Ética: ejes honestos, n e intervalos de confianza visibles, supuestos y límites explícitos,
  sin datos personales (solo agregados; el simulador no incluye IDs de clientes).

Uso:  python -m conecta.storytelling   (requiere el lake generado y predicciones.csv)
"""
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs_version

from . import config, io, modelado as m, negocio

FOCO, PROBLEMA, CONTEXTO, CONTEXTO_OSC = "#1f77b4", "#D55E00", "#cbd5e1", "#94a3b8"
TINTA, TINTA_2 = "#0f172a", "#475569"
CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"]}
EJES = dict(showline=True, linewidth=1, linecolor=CONTEXTO_OSC, showgrid=True, gridcolor="#f1f5f9",
            zeroline=False, tickfont=dict(color=TINTA_2, size=12), title_font=dict(color=TINTA_2, size=12))


def _layout(fig, alto=380, **kw):
    kw.setdefault("margin", dict(l=60, r=30, t=30, b=50))
    kw.setdefault("showlegend", False)
    fig.update_layout(template="plotly_white", height=alto,
                      font=dict(family="Inter, Segoe UI, system-ui, sans-serif", color=TINTA, size=13),
                      hoverlabel=dict(bgcolor="white", font_size=13, bordercolor=CONTEXTO_OSC),
                      plot_bgcolor="white", paper_bgcolor="white", separators=",.", **kw)
    fig.update_xaxes(**EJES)
    fig.update_yaxes(**EJES)
    return fig


def pct(x, dec=0):
    """Porcentaje en formato español: 27 %."""
    return f"{x*100:.{dec}f} %".replace(".", ",")


def mill(x, dec=1):
    """Millones COP en formato español: 16,4 M."""
    return f"{x/1e6:.{dec}f} M".replace(".", ",")


def miles(x):
    return f"{x:,.0f}".replace(",", ".")


VENTANA = {"2025-03-31": "abr–jun 2025", "2025-06-30": "jul–sep 2025", "2025-09-30": "oct–dic 2025"}


def _wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


# Paráfrasis propias del tipo de sarcasmo por categoría (no son textos de los clientes)
PARAFRASIS = {
    "app_se_cierra": "Felicita con ironía la «innovación» de una app que se cierra sola.",
    "cobro_doble": "Agradece con sarcasmo que le vuelvan a cobrar dos veces.",
    "lentitud": "Elogia la «velocidad» de una pantalla que tarda minutos en cargar.",
    "cambio_clave": "Llama «excelente» a una recuperación de contraseña que nunca funciona.",
    "cancelar": "Agradece la experiencia con ironía y pide cancelar.",
    "no_reconoce_cobro": "Llama «sorpresa» a un aumento de tarifa que nadie avisó.",
}


def _div(fig, nombre):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config=CONFIG, div_id=nombre)


def _tabla(df, titulo="Ver los datos del gráfico"):
    return (f'<details class="datos"><summary>{titulo}</summary>'
            + df.to_html(index=False, border=0, classes="tabla", float_format=lambda x: f"{x:,.3f}") + "</details>")


# ------------------------------------------------------------------------------------- datos
def calcular(lake_dir, pred_csv):
    F = io.read(lake_dir, "gold", "features")
    F["fecha_corte"] = pd.to_datetime(F["fecha_corte"])
    tr = F[F.conjunto == "entrenamiento"]
    su = io.read(lake_dir, "silver", "suscripciones")
    tk = io.read(lake_dir, "silver", "tickets")
    ev = io.read(lake_dir, "silver", "eventos")
    cl = io.read(lake_dir, "silver", "clientes")
    d = {}

    # Acto 1: tendencia por plataforma
    t = tr.groupby(["fecha_corte", "plataforma_principal"]).churn_90d.agg(["mean", "size"]).reset_index()
    tot = tr.groupby("fecha_corte").churn_90d.agg(["mean", "size"]).reset_index().assign(plataforma_principal="total")
    d["tendencia"] = pd.concat([t, tot])
    d["tendencia"]["ventana"] = d["tendencia"].fecha_corte.dt.strftime("%Y-%m-%d").map(VENTANA)

    # Acto 2a: fallas y tickets por mes
    meses = pd.period_range("2025-01", "2025-12", freq="M")
    d["crash"] = ev[ev.codigo_error == "CRASH_ON_START"].groupby(ev.ts.dt.to_period("M")).size().reindex(meses, fill_value=0)
    d["tickets_app"] = tk[tk.categoria == "app_se_cierra"].groupby(tk.creado_en.dt.to_period("M")).size().reindex(meses, fill_value=0)

    # Acto 2b: churn por grupo (corte sep) con IC de Wilson
    s9 = F[F.fecha_corte == "2025-09-30"].copy()
    s9["grupo"] = np.where(s9.usa_version_540 == 1, "Android con 5.4.0",
                   np.where(s9.plataforma_principal == "android", "Android sin 5.4.0",
                            s9.plataforma_principal.map({"ios": "iOS", "web": "Web"})))
    g = s9.groupby("grupo").churn_90d.agg(clientes="size", cancelaron="sum").reset_index()
    g["tasa"] = g.cancelaron / g.clientes
    g[["ic_bajo", "ic_alto"]] = g.apply(lambda r: pd.Series(_wilson(r.cancelaron, r.clientes)), axis=1)
    d["grupos"] = g.sort_values("tasa")
    d["activos_540"] = int(F[(F.conjunto == "puntuar")].usa_version_540.sum())
    d["exceso"] = (g.set_index("grupo").loc["Android con 5.4.0", "tasa"] - g.set_index("grupo").loc["Android sin 5.4.0", "tasa"]) \
        * g.set_index("grupo").loc["Android con 5.4.0", "clientes"]
    d["precio_540"] = s9.loc[s9.usa_version_540 == 1, "precio_plan"].mean()

    # Acto 2c: motivos de cancelación por trimestre (segmentación)
    c = su[su.fin_es_churn & (su.fecha_fin >= "2025-01-01")].copy()
    c["trimestre"] = c.fecha_fin.dt.to_period("Q").map(lambda q: f"T{q.quarter} {q.year}")
    mot = c.groupby("trimestre").agg(total=("suscripcion_id", "size"),
                                      tecnicos=("motivo_cancelacion", lambda s: (s == "problemas_tecnicos").sum())).reset_index()
    mot["pct_tecnicos"] = mot.tecnicos / mot.total
    d["motivos"] = mot
    # Confidencialidad: no se publican textos de tickets; solo paráfrasis del tipo de queja y conteos
    citas = tk[(tk.nlp_sarcasmo == 1)].groupby("categoria").size().reset_index(name="n")
    citas = citas[citas.categoria.isin(PARAFRASIS)].sort_values("n", ascending=False).head(4)
    citas["parafrasis"] = citas.categoria.map(PARAFRASIS)
    d["citas"] = citas
    d["n_sarcasmo"] = int(tk.nlp_sarcasmo.sum())
    d["n_tickets"] = len(tk)

    # Acto 2d: retención tardía y precio del Básico
    fin = su[su.fin_es_churn].groupby("cliente_id").fecha_fin.max()
    cc = cl.set_index("cliente_id").join(fin.rename("fecha_churn"))
    cc = cc[cc.contactado_por_retencion & ~cc.flag_contacto_antes_alta]
    d["dias_contacto"] = (cc.fecha_churn - cc.fecha_contacto_retencion).dt.days.dropna()
    d["pct_contactados_cancelan"] = cc.fecha_churn.notna().mean()
    d["plan_corte"] = tr.groupby(["fecha_corte", "plan"]).churn_90d.mean().reset_index()
    d["plan_corte"]["ventana"] = d["plan_corte"].fecha_corte.dt.strftime("%Y-%m-%d").map(VENTANA)

    # Acto 3: valor validado (entrenar mar+jun, validar sep) y decisión de diciembre
    tr2 = F[F.fecha_corte.isin(pd.to_datetime(m.CORTES_TRAIN))].reset_index(drop=True)
    va = F[F.fecha_corte == pd.Timestamp(m.CORTE_VALID)].reset_index(drop=True)
    p = m.entrenar(tr2).predict_proba(va[m.X_COLS])[:, 1]
    y, pr = va.churn_90d.astype(int).to_numpy(), va.precio_plan.to_numpy()
    ex = va.contacto_retencion_previo.to_numpy() == 1
    ve = negocio.valor_esperado(p, pr)
    orden = np.argsort(-np.where(ex, -np.inf, ve))                 # los excluidos nunca se contactan
    gan = np.where(ex, 0, y * negocio.GANANCIA * pr - config.COSTO_CONTACTO)[orden]
    curva = np.concatenate([[0], np.cumsum(gan)])
    k = negocio.decidir(p, pr, ex if config.EXCLUIR_CONTACTADOS_PREVIOS else None)
    d["curva"] = pd.DataFrame({"pct": np.arange(len(curva)) / len(y), "valor": curva})
    d["regla"] = {"pct": k.mean(), "valor": negocio.valor_realizado(y, k, pr), "contactos": int(k.sum())}
    d["todos"] = float(curva[-1])        # llamar a todos los que se pueden llamar (misma política)
    d["maximo"] = negocio.valor_realizado(y, y, pr)

    pred = pd.read_csv(pred_csv)
    sc = F[F.conjunto == "puntuar"][["cliente_id", "plan", "precio_plan", "contacto_retencion_previo"]]
    pr_d = pred.merge(sc, on="cliente_id", how="left")
    assert pr_d.plan.notna().all() and len(pr_d) == len(pred)
    pr_d["ve"] = negocio.valor_esperado(pr_d.probabilidad, pr_d.precio_plan)
    d["plan_dic"] = pr_d.groupby("plan").agg(clientes=("cliente_id", "size"), contactar=("contactar", "sum"),
                                             prob_media=("probabilidad", "mean"), precio=("precio_plan", "first")).reset_index()
    d["plan_dic"]["umbral"] = negocio.umbral(d["plan_dic"].precio)
    d["contactos_dic"] = int(pred.contactar.sum())
    d["ve_dic"] = float(pr_d.loc[pr_d.contactar == 1, "ve"].sum())
    # simulador: SIN identificadores (solo probabilidad, precio y bandera de política)
    # orden aleatorio para que las filas no se puedan alinear con los clientes de predicciones.csv
    d["sim"] = (pr_d[["probabilidad", "precio_plan", "contacto_retencion_previo"]].sample(frac=1, random_state=7)
                .round(5).values.tolist())
    return d


# ------------------------------------------------------------------------------------ figuras
def figuras(d):
    f = {}
    # 1. Tendencia: Android en coral, el resto en gris; total en azul (foco del contexto)
    fig = go.Figure()
    t = d["tendencia"]
    nombres = {"android": "Android", "ios": "iOS", "web": "Web", "total": "Total clientes"}
    estilo = {"android": (PROBLEMA, 3.5), "total": (FOCO, 3), "ios": (CONTEXTO_OSC, 2), "web": (CONTEXTO_OSC, 2)}
    for plat in ["ios", "web", "total", "android"]:
        s = t[t.plataforma_principal == plat]
        col, w = estilo[plat]
        fig.add_trace(go.Scatter(x=s.ventana, y=s["mean"] * 100, mode="lines+markers+text", name=nombres[plat],
                                 line=dict(color=col, width=w), marker=dict(size=8, color=col),
                                 text=[""] * (len(s) - 1) + [f"  {nombres[plat]} {pct(s['mean'].iloc[-1])}"],
                                 textposition="middle right", textfont=dict(color=col, size=12),
                                 customdata=s["size"], cliponaxis=False,
                                 hovertemplate=f"<b>{nombres[plat]}</b><br>Ventana %{{x}}<br>"
                                               "Cancelan: %{y:.1f} %<br>Clientes activos al inicio: %{customdata:,}<extra></extra>"))
    fig.add_annotation(x="jul–sep 2025", y=t[(t.plataforma_principal == "android")]["mean"].iloc[1] * 100,
                       text="Esta ventana incluye el<br>lanzamiento de la 5.4.0 (1-ago)", showarrow=True, arrowhead=0,
                       ax=-70, ay=-38, font=dict(color=PROBLEMA, size=12), align="right")
    _layout(fig, 380, xaxis=dict(title="Ventana de 90 días en la que se mide la cancelación", type="category"),
            yaxis=dict(title="% de clientes activos que cancela", range=[0, 23], ticksuffix=" %"),
            margin=dict(l=60, r=130, t=30, b=50))
    f["tendencia"] = fig

    # 2. Fallas y tickets: small multiples con el mismo eje X
    from plotly.subplots import make_subplots
    meses = d["crash"].index.to_timestamp()
    MES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    col = [PROBLEMA if m_ >= pd.Timestamp("2025-08-01") else CONTEXTO for m_ in meses]
    fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.1,
                        subplot_titles=("Cierres de la app al abrirla (CRASH_ON_START)", "Tickets «la app se cierra»"))
    fig.add_trace(go.Bar(x=meses, y=d["crash"].values, marker_color=col,
                         customdata=[MES[m_.month - 1] + " 2025" for m_ in meses], hovertemplate="%{customdata}<br>Cierres: %{y:,}<extra></extra>"), 1, 1)
    fig.add_trace(go.Bar(x=meses, y=d["tickets_app"].values, marker_color=col,
                         customdata=[MES[m_.month - 1] + " 2025" for m_ in meses], hovertemplate="%{customdata}<br>Tickets: %{y:,}<extra></extra>"), 1, 2)
    _layout(fig, 360)
    fig.update_yaxes(rangemode="tozero")
    MES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    fig.update_xaxes(tickvals=list(meses[::2]), ticktext=[MES[m_.month - 1] for m_ in meses[::2]])
    fig.update_annotations(font=dict(size=13, color=TINTA_2))
    antes = d["crash"].loc["2025-01":"2025-07"].mean()
    despues = d["crash"].loc["2025-08":"2025-12"].mean()
    fig.add_annotation(x=meses[3], y=d["crash"].max() * 0.75, xref="x", yref="y", xanchor="center",
                       text=f"<b>~{antes:.0f} → ~{miles(despues)}</b><br>cierres por mes<br>antes / después de agosto",
                       showarrow=False, font=dict(color=PROBLEMA, size=13))
    f["fallas"] = fig

    # 3. Churn por grupo: barras ordenadas, eje en cero, IC 95 %
    g = d["grupos"]
    cols = [PROBLEMA if "con 5.4.0" in x else CONTEXTO_OSC for x in g.grupo]
    fig = go.Figure(go.Bar(
        x=g.tasa * 100, y=g.grupo, orientation="h", marker_color=cols,
        error_x=dict(type="data", symmetric=False, array=(g.ic_alto - g.tasa) * 100, arrayminus=(g.tasa - g.ic_bajo) * 100,
                     color=TINTA_2, thickness=1.2, width=4),
        text=[pct(v) for v in g.tasa], textposition="inside", insidetextanchor="start",
        textfont=dict(color="white", size=13),
        customdata=np.stack([g.clientes, g.cancelaron, g.ic_bajo * 100, g.ic_alto * 100], axis=1),
        hovertemplate="<b>%{y}</b><br>Cancelan: %{x:.1f} % (IC 95 %: %{customdata[2]:.1f}–%{customdata[3]:.1f} %)"
                      "<br>%{customdata[1]:.0f} de %{customdata[0]:,.0f} clientes<extra></extra>"))
    _layout(fig, 300, xaxis=dict(title="% que cancela en 90 días (corte 30-sep-2025)", range=[0, 34], ticksuffix=" %"),
            margin=dict(l=150, r=40, t=20, b=50))
    f["grupos"] = fig

    # 4. Motivos: segmentación con barras apiladas al 100 %
    mo = d["motivos"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mo.trimestre, y=mo.pct_tecnicos * 100, name="Problemas técnicos", marker_color=PROBLEMA,
                         text=[pct(v) for v in mo.pct_tecnicos], textposition="inside", insidetextfont=dict(color="white"),
                         customdata=mo.tecnicos, hovertemplate="%{x}<br>Problemas técnicos: %{y:.0f} % (%{customdata} bajas)<extra></extra>"))
    fig.add_trace(go.Bar(x=mo.trimestre, y=(1 - mo.pct_tecnicos) * 100, name="Otros motivos", marker_color=CONTEXTO,
                         customdata=mo.total - mo.tecnicos, hovertemplate="%{x}<br>Otros motivos: %{y:.0f} % (%{customdata} bajas)<extra></extra>"))
    _layout(fig, 320, barmode="stack", showlegend=True, legend=dict(orientation="h", y=1.12, x=0, traceorder="normal"),
            yaxis=dict(title="% de las cancelaciones", range=[0, 100], ticksuffix=" %"), xaxis=dict(title="Trimestre de la baja"))
    f["motivos"] = fig

    # 5a. Retención tardía: distribución de días
    dc = d["dias_contacto"]
    fig = go.Figure(go.Histogram(x=dc, xbins=dict(start=dc.min() - 0.5, end=dc.max() + 0.5, size=1), marker_color=FOCO,
                                 hovertemplate="%{x} días antes de irse: %{y} clientes<extra></extra>"))
    fig.update_traces(marker_line=dict(color="white", width=1))
    _layout(fig, 300, xaxis=dict(title="Días entre la llamada de retención y la cancelación"),
            yaxis=dict(title="Clientes", rangemode="tozero"), bargap=0.08)
    f["retencion"] = fig

    # 5b. Precio del Básico: tendencia por plan, Básico en foco
    pc = d["plan_corte"]
    fig = go.Figure()
    for plan in ["Estandar", "Premium", "Familiar", "Basico"]:
        s = pc[pc.plan == plan]
        foco = plan == "Basico"
        nombre = {"Basico": "Básico", "Estandar": "Estándar"}.get(plan, plan)
        fig.add_trace(go.Scatter(x=s.ventana, y=s.churn_90d * 100, mode="lines+markers+text",
                                 line=dict(color=FOCO if foco else CONTEXTO_OSC, width=3.5 if foco else 1.8),
                                 marker=dict(size=7, color=FOCO if foco else CONTEXTO_OSC),
                                 text=[""] * (len(s) - 1) + [f"  {nombre}"], textposition="middle right", cliponaxis=False,
                                 textfont=dict(color=FOCO if foco else TINTA_2, size=12),
                                 hovertemplate=f"<b>{nombre}</b><br>%{{x}}: %{{y:.1f}} %<extra></extra>"))
    fig.add_annotation(x="abr–jun 2025", y=21.5, text="1-abr: Básico pasa<br>de 19.900 a 23.900", showarrow=False,
                       font=dict(color=FOCO, size=12), xanchor="left", align="left")
    _layout(fig, 300, xaxis=dict(type="category", title="Ventana de 90 días"),
            yaxis=dict(title="% que cancela", range=[0, 23], ticksuffix=" %"), margin=dict(l=60, r=80, t=30, b=50))
    f["basico"] = fig

    # 6. Curva de valor validada
    cu = d["curva"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cu.pct * 100, y=cu.valor / 1e6, mode="lines", line=dict(color=FOCO, width=3),
                             hovertemplate="Contactando al %{x:.0f} % con más valor esperado<br>Valor neto: %{y:.1f} M COP<extra></extra>"))
    fig.add_hline(y=d["todos"] / 1e6, line=dict(color=CONTEXTO_OSC, dash="dash", width=1.5))
    fig.add_annotation(x=60, y=d["todos"] / 1e6, text=f"Llamar a todos: {mill(d['todos'])}", showarrow=False,
                       yshift=12, xanchor="center", font=dict(color=TINTA_2, size=12))
    r = d["regla"]
    fig.add_trace(go.Scatter(x=[r["pct"] * 100], y=[r["valor"] / 1e6], mode="markers", marker=dict(size=13, color=PROBLEMA,
                             line=dict(color="white", width=2)),
                             hovertemplate=f"<b>Nuestra regla</b><br>{r['contactos']} llamadas · {mill(r['valor'])} COP<extra></extra>"))
    fig.add_annotation(x=r["pct"] * 100, y=r["valor"] / 1e6, text=f"<b>Nuestra regla: {mill(r['valor'])}</b><br>{r['contactos']} llamadas",
                       showarrow=True, arrowhead=0, ax=60, ay=-40, font=dict(color=PROBLEMA, size=12), align="left")
    fig.add_hline(y=0, line=dict(color=CONTEXTO_OSC, width=1))
    _layout(fig, 360, xaxis=dict(title="% de clientes contactados (ordenados por valor esperado)", ticksuffix=" %", range=[0, 100]),
            yaxis=dict(title="Valor neto real (millones COP)"))
    f["curva"] = fig

    # 7. Contactos por plan en diciembre (comparación, eje en cero, ordenado)
    pdi = d["plan_dic"].sort_values("contactar")
    nombres = pdi.plan.map({"Basico": "Básico", "Estandar": "Estándar"}).fillna(pdi.plan)
    fig = go.Figure(go.Bar(x=pdi.contactar, y=nombres, orientation="h", marker_color=FOCO,
                           text=pdi.contactar, textposition="outside",
                           customdata=np.stack([pdi.clientes, pdi.umbral * 100, pdi.precio], axis=1),
                           hovertemplate="<b>%{y}</b> (%{customdata[2]:,.0f} COP/mes)<br>Llamar a %{x} de %{customdata[0]} clientes"
                                         "<br>Se llama si el riesgo supera %{customdata[1]:.1f} %<extra></extra>"))
    _layout(fig, 260, xaxis=dict(title="Clientes a llamar", range=[0, pdi.contactar.max() * 1.18]), margin=dict(l=90, r=30, t=10, b=50))
    f["plan_dic"] = fig
    return f


# --------------------------------------------------------------------------------------- HTML
CSS = """
:root{--foco:#1f77b4;--problema:#D55E00;--ctx:#cbd5e1;--tinta:#0f172a;--tinta2:#475569;--fondo:#f8fafc;--borde:#e2e8f0}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--fondo);color:var(--tinta);font:16px/1.6 Inter,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px}
header.hero{background:#0f172a;color:#fff;padding:56px 0 44px}
.hero .eyebrow{color:#7dd3fc;font-weight:600;letter-spacing:.08em;text-transform:uppercase;font-size:13px}
.hero h1{font-size:clamp(28px,4.2vw,44px);line-height:1.15;margin:10px 0 14px;max-width:880px}
.hero h1 em{color:#fdba74;font-style:normal}
.hero p.lead{color:#cbd5e1;max-width:760px;font-size:18px;margin:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-top:30px}
.kpi{background:#1e293b;border-radius:12px;padding:16px 18px;border-left:4px solid #7dd3fc}
.kpi.alerta{border-left-color:#fdba74}
.kpi b{display:block;font-size:30px;line-height:1.1;color:#fff}.kpi span{color:#cbd5e1;font-size:14px}
nav.toc{position:sticky;top:0;z-index:5;background:rgba(248,250,252,.95);backdrop-filter:blur(6px);border-bottom:1px solid var(--borde)}
nav.toc .wrap{display:flex;gap:18px;overflow-x:auto;padding-top:10px;padding-bottom:10px;font-size:14px}
nav.toc a{color:var(--tinta2);text-decoration:none;white-space:nowrap;font-weight:500}nav.toc a:hover{color:var(--foco)}
section{padding:46px 0 10px}
.acto{color:var(--foco);font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:13px;margin:0}
h2{font-size:clamp(22px,3vw,30px);line-height:1.25;margin:6px 0 10px}
h3{font-size:20px;margin:30px 0 6px;line-height:1.3}
p.contexto{color:var(--tinta2);max-width:820px;margin:0 0 14px}
figure{background:#fff;border:1px solid var(--borde);border-radius:14px;padding:16px 16px 10px;margin:14px 0 8px}
figcaption.bongo{border-left:4px solid var(--problema);background:#fff7ed;padding:10px 14px;border-radius:0 10px 10px 0;margin-top:8px;font-size:15px}
figcaption.bongo.azul{border-left-color:var(--foco);background:#eff6ff}
.dos{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:860px){.dos{grid-template-columns:1fr}}
.citas{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:14px 0}
blockquote{margin:0;background:#fff;border:1px solid var(--borde);border-radius:12px;padding:14px 16px;font-size:15px;position:relative}
blockquote::before{content:"“";position:absolute;top:-6px;left:10px;font-size:44px;color:var(--problema);font-family:Georgia,serif}
blockquote p{margin:12px 0 8px}blockquote small{color:var(--tinta2)}
details.datos{margin:6px 2px 4px;font-size:14px;color:var(--tinta2)}details.datos summary{cursor:pointer}
table.tabla{border-collapse:collapse;margin:8px 0;font-size:13px}table.tabla th,table.tabla td{padding:4px 10px;border-bottom:1px solid var(--borde);text-align:right}
table.tabla th{color:var(--tinta)}
.acciones{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin:16px 0}
.accion{background:#fff;border:1px solid var(--borde);border-top:4px solid var(--foco);border-radius:12px;padding:16px}
.accion.urgente{border-top-color:var(--problema)}.accion h4{margin:4px 0 6px;font-size:17px}.accion p{margin:0;font-size:15px;color:var(--tinta2)}
.accion .tag{font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--problema)}
.sim{background:#fff;border:1px solid var(--borde);border-radius:14px;padding:18px;margin:14px 0}
.sim .controles{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px 26px}
.sim label{font-weight:600;font-size:15px;display:flex;justify-content:space-between}.sim output{color:var(--foco);font-weight:700}
.sim input[type=range]{width:100%;accent-color:var(--foco)}
.sim .res{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0 4px}
.sim .res div{background:var(--fondo);border-radius:10px;padding:10px 14px}.sim .res b{font-size:24px;display:block}
.sim .res span{font-size:13px;color:var(--tinta2)}
.nota{font-size:14px;color:var(--tinta2)}
.limites{background:#fff;border:1px solid var(--borde);border-radius:14px;padding:6px 22px;margin:14px 0}
footer{margin-top:40px;padding:26px 0 40px;border-top:1px solid var(--borde);font-size:14px;color:var(--tinta2)}
footer h4{color:var(--tinta);margin:0 0 6px}
"""


def construir(lake_dir=None, pred_csv=None, salida=None):
    lake_dir = lake_dir or config.LAKE_DIR
    pred_csv = pred_csv or (config.RAIZ / "predicciones.csv")
    salida = Path(salida or (config.RAIZ / "storytelling.html"))
    d = calcular(lake_dir, pred_csv)
    f = figuras(d)
    t = d["tendencia"]
    tot = t[t.plataforma_principal == "total"].sort_values("fecha_corte")["mean"].tolist()
    g = d["grupos"].set_index("grupo")
    r540, rsin = g.loc["Android con 5.4.0", "tasa"], g.loc["Android sin 5.4.0", "tasa"]
    mo = d["motivos"]
    perdida_anual = d["exceso"] * 12 * d["precio_540"]
    ratio = d["regla"]["valor"] / d["todos"]
    # titular calculado (no escrito a mano) para que nunca exagere ni subestime
    veces = (f"{ratio:.0f} veces más" if ratio >= 3 else "más del doble" if ratio >= 2 else f"{pct(ratio - 1)} más")
    esc = html.escape

    citas = "".join(f'<blockquote><p>{esc(r.parafrasis)}</p><small>Paráfrasis · tickets «{esc(r.categoria.replace("_", " "))}» · '
                    f'{r.n} tickets sarcásticos en esta categoría</small></blockquote>' for r in d["citas"].itertuples())
    tabla_grupos = d["grupos"].assign(tasa=lambda x: x.tasa.round(3), ic_bajo=lambda x: x.ic_bajo.round(3),
                                      ic_alto=lambda x: x.ic_alto.round(3))
    tabla_plan = d["plan_dic"].assign(umbral=lambda x: x.umbral.round(3), prob_media=lambda x: x.prob_media.round(3))

    cuerpo = f"""
<header class="hero"><div class="wrap">
  <div class="eyebrow">Conecta+ · Retención de clientes · datos al 31-dic-2025</div>
  <h1>Cada trimestre perdemos más clientes, y la causa más grande <em>está en nuestras manos</em>: la app Android 5.4.0.</h1>
  <p class="lead">Una historia en tres actos sobre por qué se van los clientes y a quién llamar en los próximos 90 días para retenerlos.</p>
  <div class="kpis">
    <div class="kpi alerta"><b>{pct(tot[0])} → {pct(tot[-1])}</b><span>clientes que cancelan en 90 días (abr–jun → oct–dic 2025)</span></div>
    <div class="kpi alerta"><b>{pct(r540)} vs {pct(rsin)}</b><span>cancelación con y sin la versión Android 5.4.0</span></div>
    <div class="kpi"><b>{d['contactos_dic']}</b><span>clientes a llamar en el primer trimestre de 2026</span></div>
    <div class="kpi"><b>~{mill(d['ve_dic'], 0)} COP</b><span>valor neto esperado de la campaña</span></div>
  </div>
</div></header>
<nav class="toc" aria-label="Contenido"><div class="wrap">
  <a href="#acto1">1 · El problema</a><a href="#acto2">2 · Qué lo causa</a><a href="#voz">La voz del cliente</a>
  <a href="#otras">Otras señales</a><a href="#acto3">3 · Qué hacer</a><a href="#simulador">Simulador</a><a href="#limites">Límites</a>
</div></nav>
<main class="wrap">

<section id="acto1">
  <p class="acto">Acto 1 · Planteamiento</p>
  <h2>La cancelación crece cada trimestre, y Android se dispara después de agosto</h2>
  <p class="contexto">Porcentaje de los clientes activos al inicio de cada ventana que cancelaron dentro de ella.
  iOS y Web se mantienen estables; el aumento total lo explica casi por completo Android.</p>
  <figure aria-label="Gráfico de líneas: cancelación a 90 días por plataforma en tres cortes de 2025">
    {_div(f["tendencia"], "g-tendencia")}
    <figcaption class="bongo">Android pasó de {pct(t[(t.plataforma_principal=="android")]["mean"].iloc[0])} a
    {pct(t[(t.plataforma_principal=="android")]["mean"].iloc[-1])} de cancelación, mientras iOS y Web siguen cerca de 10 %.
    Algo cambió para los usuarios de Android.</figcaption>
  </figure>
  {_tabla(t.rename(columns={"mean": "tasa_cancelacion", "size": "clientes"}).assign(tasa_cancelacion=lambda x: x.tasa_cancelacion.round(3)))}
</section>

<section id="acto2">
  <p class="acto">Acto 2 · Nudo</p>
  <h2>El 1 de agosto salió la versión 5.4.0 y la app empezó a cerrarse sola</h2>
  <p class="contexto">Dos sistemas independientes, los registros de la app y la plataforma de soporte, muestran el mismo quiebre el mismo mes.</p>
  <figure aria-label="Dos gráficos de barras por mes de 2025: cierres de la app y tickets de la app se cierra, ambos se disparan desde agosto">
    {_div(f["fallas"], "g-fallas")}
    <figcaption class="bongo">Antes de agosto la app casi no fallaba. Desde agosto se cierra unas 2.000 veces al mes y aparecen
    cientos de reclamos nuevos por «la app se cierra».</figcaption>
  </figure>

  <h3>Quien usa la versión 5.4.0 cancela tres veces más</h3>
  <p class="contexto">Clientes activos al 30-sep-2025 y si cancelaron en los 90 días siguientes. Las líneas negras son el intervalo
  de confianza del 95 %: los grupos no se solapan, así que la diferencia no es casualidad.</p>
  <figure aria-label="Barras horizontales: cancelación de 27 % con la versión 5.4.0 frente a 8 a 12 % en los demás grupos">
    {_div(f["grupos"], "g-grupos")}
    <figcaption class="bongo">Un usuario Android <b>sin</b> la 5.4.0 cancela {pct(rsin)}, incluso menos que iOS. El problema no es Android:
    es la versión. Son ~{d['exceso']:.0f} cancelaciones de más por trimestre, unos <b>{mill(perdida_anual, 0)} COP de ingresos al año</b>,
    y hoy {d['activos_540']} clientes activos la siguen usando.</figcaption>
  </figure>
  {_tabla(tabla_grupos)}

  <h3 id="voz">Los clientes lo dicen, aunque a veces con ironía</h3>
  <p class="contexto">El {pct(d['n_sarcasmo']/d['n_tickets'])} de los tickets usa palabras positivas para quejarse. Un análisis de
  sentimiento automático los leería como clientes felices; nuestro análisis de texto detecta el sarcasmo. Por
  confidencialidad se muestran paráfrasis del tipo de queja, no los textos de los clientes.</p>
  <div class="citas">{citas}</div>
  <figure aria-label="Barras apiladas al 100 % por trimestre: participación de problemas técnicos en las cancelaciones">
    {_div(f["motivos"], "g-motivos")}
    <figcaption class="bongo">Los problemas técnicos pasaron de {pct(mo.pct_tecnicos.iloc[1])} de las bajas en el segundo trimestre
    a {pct(mo.pct_tecnicos.iloc[-1])} en el cuarto: hoy son el motivo de cancelación más frecuente.</figcaption>
  </figure>

  <h3 id="otras">Dos señales más que la gerencia debería ver</h3>
  <div class="dos">
    <figure aria-label="Histograma: la llamada de retención ocurre entre 3 y 19 días antes de que el cliente se vaya">
      <strong>Llamamos cuando el cliente ya decidió irse</strong>
      {_div(f["retencion"], "g-retencion")}
      <figcaption class="bongo azul">La llamada llega entre {int(d['dias_contacto'].min())} y {int(d['dias_contacto'].max())}
      días antes de la baja, y el {pct(d['pct_contactados_cancelan'])} de los contactados se va igual. Hay que llamar <b>antes</b>.</figcaption>
    </figure>
    <figure aria-label="Líneas por plan: el Básico sube su cancelación después del aumento de precio de abril">
      <strong>El Básico subió 20 % sin aviso</strong>
      {_div(f["basico"], "g-basico")}
      <figcaption class="bongo azul">Tras el aumento de abril, la cancelación del Básico pasó de
      {pct(d['plan_corte'].query("plan=='Basico'").churn_90d.iloc[0])} a {pct(d['plan_corte'].query("plan=='Basico'").churn_90d.iloc[-1])},
      y en los tickets aparecen reclamos por el aumento sin aviso previo.</figcaption>
    </figure>
  </div>
</section>

<section id="acto3">
  <p class="acto">Acto 3 · Desenlace y llamada a la acción</p>
  <h2>Llamar a los clientes correctos deja {veces} que llamar a todos</h2>
  <p class="contexto">Un modelo estima para cada cliente la probabilidad de irse. Solo lo llamamos si lo que esperamos recuperar
  (40 % de éxito × 12 meses de su plan) supera los 25.000 COP de la llamada; por eso un plan caro se llama con menos riesgo.
  La curva muestra lo que habría pasado en septiembre de 2025, con datos que el modelo no había visto.</p>
  <figure aria-label="Curva de valor neto según el porcentaje de clientes contactados, con el punto de nuestra regla">
    {_div(f["curva"], "g-curva")}
    <figcaption class="bongo">Con nuestra regla se habrían recuperado <b>{mill(d['regla']['valor'])} COP</b> netos con
    {d['regla']['contactos']} llamadas, frente a {mill(d['todos'])} llamando a todos. En ambos casos no se recontacta a quien ya
    recibió una oferta (casi todos se fueron igual).</figcaption>
  </figure>
  <div class="dos">
    <figure aria-label="Barras: clientes a llamar por plan en el primer trimestre de 2026">
      <strong>A quién llamar ahora ({d['contactos_dic']} clientes)</strong>
      {_div(f["plan_dic"], "g-plan")}
      <figcaption class="bongo azul">La lista completa está en <code>predicciones.csv</code>. Los planes Premium y Familiar
      justifican la llamada con menos riesgo porque retenerlos vale más.</figcaption>
    </figure>
    <div>
      <h3 style="margin-top:14px">Tres decisiones para esta semana</h3>
      <div class="acciones" style="grid-template-columns:1fr">
        <div class="accion urgente"><span class="tag">Urgente</span><h4>Corregir o retirar la versión 5.4.0</h4>
          <p>Es la causa controlable más grande. Avisar a sus {d['activos_540']} usuarios activos cuando esté resuelto, con una compensación.</p></div>
        <div class="accion"><span class="tag" style="color:var(--foco)">Campaña</span><h4>Llamar a los {d['contactos_dic']} clientes de la lista</h4>
          <p>Antes de que pidan cancelar, no después. Dejar un grupo pequeño sin llamar, al azar, para medir el efecto real de la oferta.</p></div>
        <div class="accion"><span class="tag" style="color:var(--foco)">Prevención</span><h4>Alarma de calidad por versión y aviso de precios</h4>
          <p>No liberar versiones con más fallos que la anterior y comunicar cualquier cambio de tarifa antes de cobrarlo.</p></div>
      </div>
    </div>
  </div>

  <h3 id="simulador">¿Y si la oferta funciona menos? Simulador de la campaña</h3>
  <p class="contexto">Mueve los supuestos y mira cómo cambia a quién conviene llamar. Usa las probabilidades del modelo para los
  1.558 clientes activos (sin identificarlos).</p>
  <div class="sim" role="group" aria-label="Simulador de supuestos de la campaña">
    <div class="controles">
      <div><label for="exito">Clientes que la oferta retiene <output id="o-exito">40 %</output></label>
        <input id="exito" type="range" min="10" max="70" step="5" value="40"></div>
      <div><label for="costo">Costo de cada llamada <output id="o-costo">25.000 COP</output></label>
        <input id="costo" type="range" min="10000" max="60000" step="5000" value="25000"></div>
      <div><label for="politica" style="justify-content:flex-start;gap:8px"><input id="politica" type="checkbox" checked>
        No recontactar a quien ya recibió una oferta</label></div>
    </div>
    <div class="res">
      <div><b id="r-n">–</b><span>clientes a llamar</span></div>
      <div><b id="r-costo">–</b><span>costo de la campaña</span></div>
      <div><b id="r-valor">–</b><span>valor neto esperado</span></div>
    </div>
    <p class="nota" id="r-nota"></p>
  </div>
</section>

<section id="limites">
  <p class="acto">Lo que hay que tener en cuenta</p>
  <h2>Límites honestos del análisis</h2>
  <div class="limites"><ul>
    <li><b>El 40 % de éxito y los 25.000 COP son supuestos del área.</b> Sin un grupo de control no sabemos cuánto funciona la oferta de verdad.</li>
    <li><b>La relación entre la 5.4.0 y la cancelación es muy fuerte, pero es observacional.</b> El momento, los fallos y los reclamos apuntan
    en la misma dirección; corregir la versión y comparar es la forma de confirmarlo.</li>
    <li><b>La cancelación viene subiendo.</b> El modelo aprende del pasado y puede quedarse corto si la tendencia sigue; hay que
    actualizarlo cada trimestre.</li>
    <li><b>No recontactar tiene un costo:</b> con los supuestos del área deja de recuperar unos 2 a 3 millones; es una decisión de negocio.</li>
    <li><b>Privacidad:</b> esta página muestra solo cifras agregadas; ningún cliente es identificable.</li>
  </ul></div>
</section>
</main>
<footer><div class="wrap">
  <h4>Ficha técnica</h4>
  Fuentes: CRM, suscripciones y facturación (SQLite), tickets de soporte (dos plataformas unificadas) y registros de uso de la app (2025).
  Población: clientes activos en cada fecha de corte (1.447 · 1.620 · 1.743) y 1.558 a puntuar al 31-dic-2025.
  Modelo: promedio de una regresión logística L1 y un Random Forest, calibrado, validado fuera de tiempo (entrenar mar–jun, evaluar sep).
  Intervalos de confianza: Wilson al 95 %. Detalle técnico y código: <code>solucion.ipynb</code> en el repositorio
  <a href="https://github.com/adrianfelipe812/prueba-conecta">adrianfelipe812/prueba-conecta</a>.
  Página generada con <code>python -m conecta.storytelling</code>.
</div></footer>
"""

    js = """
<script>
const SIM = __SIM__;
const fmt = v => (v/1e6).toLocaleString('es-CO',{minimumFractionDigits:1,maximumFractionDigits:1}) + ' M COP';
function simular(){
  const ex = +document.getElementById('exito').value/100, co = +document.getElementById('costo').value;
  const pol = document.getElementById('politica').checked;
  document.getElementById('o-exito').textContent = Math.round(ex*100) + ' %';
  document.getElementById('o-costo').textContent = co.toLocaleString('es-CO') + ' COP';
  let n=0, valor=0, excl=0;
  for (const [p, precio, previo] of SIM){
    const ve = p*ex*12*precio - co;
    if (ve > 0){ if (pol && previo===1){excl++; continue;} n++; valor += ve; }
  }
  document.getElementById('r-n').textContent = n.toLocaleString('es-CO');
  document.getElementById('r-costo').textContent = fmt(n*co);
  document.getElementById('r-valor').textContent = fmt(valor);
  document.getElementById('r-nota').textContent = pol && excl ? `Se excluyen ${excl} clientes rentables que ya recibieron una oferta.` : '';
}
['exito','costo','politica'].forEach(id => document.getElementById(id).addEventListener('input', simular));
simular();
</script>""".replace("__SIM__", json.dumps(d["sim"]))

    doc = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conecta+ · Por qué se van los clientes</title>
<meta name="description" content="Historia de negocio: hallazgos de cancelación de clientes y plan de retención de Conecta+">
<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" charset="utf-8"></script>
<style>{CSS}</style></head>
<body>{cuerpo}{js}</body></html>"""
    salida.write_text(doc, encoding="utf-8")
    return salida, d


if __name__ == "__main__":
    p, d = construir()
    print("Generado:", p, f"({p.stat().st_size/1024:.0f} KB)")
