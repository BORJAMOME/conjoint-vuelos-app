"""Figuras Plotly. Mismo sistema de color que el resto del portfolio:
azules marino para lo estructural, verde/rojo solo para lo semántico
(aquí, prácticamente no se usan — un part-worth no es "bueno" o "malo",
es solo cuánto vale un nivel frente a su referencia)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

INK = "#1D2638"
NAVY2 = "#273A5F"
NAVY3 = "#4A628E"
NAVY4 = "#B9C5D6"
MUTED = "#6B7280"
LINE = "#E3DFD5"
POSITIVE = "#6E7F5B"
NEGATIVE = "#C2412E"
SUPPORT = "#B8783C"
FONT = "Arial, Helvetica, sans-serif"


def _base_layout(fig, height=380, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=12.5),
        hovermode="closest",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                     font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED)),
    )
    return fig


def rating_distribution(ratings: pd.Series) -> go.Figure:
    fig = go.Figure(go.Histogram(x=ratings, marker_color=NAVY2, opacity=0.85, nbinsx=24))
    fig.update_xaxes(title_text="Rating (1-10)")
    fig.update_yaxes(title_text="Valoraciones")
    return _base_layout(fig, height=380, legend=False)


def rating_by_segment(df: pd.DataFrame, segment_colors: dict, order: list) -> go.Figure:
    fig = go.Figure()
    for seg in order:
        fig.add_trace(go.Box(
            y=df.loc[df["Segment"] == seg, "Rating"], name=seg,
            marker_color=segment_colors.get(seg, MUTED), boxmean=True,
        ))
    fig.update_yaxes(title_text="Rating (1-10)")
    return _base_layout(fig, height=380, legend=False)


def partworth_bars(pw_attr: pd.DataFrame, level_labels: dict) -> go.Figure:
    """Una barra por nivel de un atributo — la referencia (utilidad 0) en gris claro."""
    labels = [level_labels.get(n, n) for n in pw_attr["Nivel"]]
    colors = [NAVY4 if u == 0 else NAVY2 for u in pw_attr["Utilidad"]]
    fig = go.Figure(go.Bar(
        x=labels, y=pw_attr["Utilidad"], marker_color=colors,
        text=[f"{u:+.2f}" if u != 0 else "ref." for u in pw_attr["Utilidad"]],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color=INK, line_width=1)
    fig.update_yaxes(title_text="Utilidad parcial")
    return _base_layout(fig, height=280, legend=False)


def importance_overall(imp_df: pd.DataFrame, attribute_labels: dict) -> go.Figure:
    d = imp_df.sort_values("Importancia", ascending=True)
    labels = [attribute_labels.get(a, a) for a in d["Atributo"]]
    fig = go.Figure(go.Bar(
        x=d["Importancia"], y=labels, orientation="h", marker_color=NAVY2,
        text=[f"{v:.1f}%" for v in d["Importancia"]], textposition="outside",
    ))
    fig.update_xaxes(title_text="Importancia relativa (%)")
    return _base_layout(fig, height=340, legend=False)


def importance_by_segment(imp_seg_df: pd.DataFrame, attribute_labels: dict, segment_colors: dict,
                           order: list) -> go.Figure:
    attrs = (imp_seg_df.groupby("Atributo")["Importancia"].mean().sort_values(ascending=True).index.tolist())
    labels = [attribute_labels.get(a, a) for a in attrs]
    fig = go.Figure()
    for seg in order:
        sub = imp_seg_df[imp_seg_df["Segmento"] == seg].set_index("Atributo").reindex(attrs)
        fig.add_trace(go.Bar(
            y=labels, x=sub["Importancia"], name=seg, orientation="h",
            marker_color=segment_colors.get(seg, MUTED),
        ))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Importancia relativa (%)")
    return _base_layout(fig, height=420)


def price_curve(pw_price: pd.DataFrame) -> go.Figure:
    """La utilidad del precio no es lineal — una línea lo deja ver de un vistazo."""
    x = [f"{n} €" for n in pw_price["Nivel"]]
    fig = go.Figure(go.Scatter(
        x=x, y=pw_price["Utilidad"], mode="lines+markers+text",
        line=dict(color=NEGATIVE, width=2.4), marker=dict(size=9, color=NEGATIVE),
        text=[f"{u:+.2f}" for u in pw_price["Utilidad"]], textposition="top center",
    ))
    fig.add_hline(y=0, line_color=LINE)
    fig.update_yaxes(title_text="Utilidad parcial")
    return _base_layout(fig, height=320, legend=False)


def playground_segment_comparison(predictions: dict, segment_labels: dict, segment_colors: dict,
                                   order: list) -> go.Figure:
    labels = [segment_labels.get(s, s) for s in order]
    values = [predictions[s] for s in order]
    colors = [segment_colors.get(s, MUTED) for s in order]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.1f}" for v in values], textposition="outside",
    ))
    fig.update_yaxes(title_text="Rating predicho (1-10)", range=[0, 10.8])
    return _base_layout(fig, height=340, legend=False)
