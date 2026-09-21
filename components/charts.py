"""Figuras Plotly con la paleta propia de este proyecto (ver assets/style.css):
  - Azul   #3772FF  → datos principales; en utilidades, lo que SUMA a la valoración
  - Rojo   #DF2935  → lo que RESTA (utilidad negativa) y el énfasis
  - Negro  #080708  → tinta: etiquetas, líneas de referencia; también el segmento Business
  - Amarillo #FDCA40 → segmento Low Cost y resaltados; nunca como texto sobre claro (contraste 1,4:1)
  - Gris   #E6E8E6  → cuadrícula y niveles de referencia (y su versión media, GRAY_MID, para «todos los clientes»)
Los segmentos usan SEGMENT_COLORS (utils/data_loader.py): el mismo color en gráficos y en tarjetas. Todas las
barras y cajas llevan un contorno fino en tinta para que el amarillo y el gris se definan sobre fondo claro.
Legibles en móvil: etiquetas largas ajustadas en varias líneas, leyendas arriba, sin anotaciones flotantes que
se solapen con las barras."""
import textwrap

import pandas as pd
import plotly.graph_objects as go

INK = "#080708"
BLUE = "#3772FF"
RED = "#DF2935"
YELLOW = "#FDCA40"
GRAY = "#E6E8E6"
GRAY_MID = "#A3A6A3"
MUTED = "#5E615E"
GRID = "#DCDFDC"
FONT = "Arial, Helvetica, sans-serif"
OUTLINE = dict(color=INK, width=0.7)


def _num(value: float, decimals: int = 1, sign: bool = False) -> str:
    """Número con coma decimal y menos tipográfico, como en el texto de la app."""
    text = f"{value:+.{decimals}f}" if sign else f"{value:.{decimals}f}"
    return text.replace("-", "−").replace(".", ",")


def _wrap(label: str, width: int = 13) -> str:
    return "<br>".join(textwrap.wrap(label, width)) or label


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
                    font=dict(size=11.5), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED)),
    )
    return fig


def rating_distribution(ratings: pd.Series) -> go.Figure:
    fig = go.Figure(go.Histogram(x=ratings, nbinsx=24, marker=dict(color=BLUE, line=OUTLINE), opacity=0.95))
    fig.update_xaxes(title_text="Valoración (1 a 10)")
    fig.update_yaxes(title_text="Valoraciones")
    return _base_layout(fig, height=380, legend=False)


def rating_by_segment(df: pd.DataFrame, segment_colors: dict, order: list) -> go.Figure:
    fig = go.Figure()
    for seg in order:
        color = segment_colors.get(seg, GRAY_MID)
        fig.add_trace(go.Box(
            y=df.loc[df["Segment"] == seg, "Rating"], name=seg, boxmean=True,
            fillcolor=color, line=dict(color=INK, width=1.2), marker=dict(color=INK, size=4),
            opacity=0.95,
        ))
    fig.update_yaxes(title_text="Valoración (1 a 10)")
    return _base_layout(fig, height=380, legend=False)


def partworth_bars(pw_attr: pd.DataFrame, level_labels: dict, title: str = "") -> go.Figure:
    """Una barra por nivel: azul si suma a la valoración, rojo si resta, gris en la referencia (utilidad 0)."""
    labels = [_wrap(level_labels.get(n, n)) for n in pw_attr["Nivel"]]
    colors = [GRAY if u == 0 else (BLUE if u > 0 else RED) for u in pw_attr["Utilidad"]]
    top, bottom = pw_attr["Utilidad"].max(), pw_attr["Utilidad"].min()
    fig = go.Figure(go.Bar(
        x=labels, y=pw_attr["Utilidad"], marker=dict(color=colors, line=OUTLINE),
        text=[_num(u, 2, sign=True) if u != 0 else "ref." for u in pw_attr["Utilidad"]],
        textposition="outside", cliponaxis=False, hoverinfo="skip", textfont=dict(size=11.5),
    ))
    fig.add_hline(y=0, line_color=INK, line_width=1)
    fig.update_yaxes(range=[min(bottom * 1.35, -0.6), max(top * 1.3, 0.6)], title_text="")
    _base_layout(fig, height=290, legend=False)
    fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0, xanchor="left", font=dict(size=13.5), y=0.97),
                      margin=dict(l=10, r=10, t=44 if title else 10, b=10))
    return fig


def importance_overall(imp_df: pd.DataFrame, attribute_labels: dict, highlight: int = 2) -> go.Figure:
    """Importancia relativa; las `highlight` barras más largas en azul y el resto en gris medio."""
    d = imp_df.sort_values("Importancia", ascending=True)
    labels = [_wrap(attribute_labels.get(a, a), 14) for a in d["Atributo"]]
    colors = [BLUE if i >= len(d) - highlight else GRAY_MID for i in range(len(d))]
    fig = go.Figure(go.Bar(
        x=d["Importancia"], y=labels, orientation="h", marker=dict(color=colors, line=OUTLINE),
        text=[f"{_num(v)}%" for v in d["Importancia"]], textposition="outside", cliponaxis=False, hoverinfo="skip",
    ))
    fig.update_xaxes(title_text="Importancia relativa (%)", range=[0, max(d["Importancia"]) * 1.2])
    fig.update_yaxes(showgrid=False, automargin=True)
    return _base_layout(fig, height=360, legend=False)


def importance_by_segment(imp_seg_df: pd.DataFrame, attribute_labels: dict, segment_colors: dict,
                          order: list) -> go.Figure:
    attrs = imp_seg_df.groupby("Atributo")["Importancia"].mean().sort_values(ascending=True).index.tolist()
    labels = [_wrap(attribute_labels.get(a, a), 14) for a in attrs]
    fig = go.Figure()
    for seg in order:
        sub = imp_seg_df[imp_seg_df["Segmento"] == seg].set_index("Atributo").reindex(attrs)
        fig.add_trace(go.Bar(
            y=labels, x=sub["Importancia"], name=seg, orientation="h",
            marker=dict(color=segment_colors.get(seg, GRAY_MID), line=OUTLINE),
            text=[f"{_num(v)}%" for v in sub["Importancia"]], textposition="outside", cliponaxis=False,
            textfont=dict(size=10.5), hoverinfo="skip",
        ))
    fig.update_layout(barmode="group", bargap=0.22, legend=dict(traceorder="reversed"))
    fig.update_xaxes(title_text="Importancia relativa (%)", range=[0, imp_seg_df["Importancia"].max() * 1.18])
    fig.update_yaxes(showgrid=False, automargin=True)
    return _base_layout(fig, height=560)


def price_curve(pw_price: pd.DataFrame) -> go.Figure:
    """La utilidad del precio no es lineal: una línea lo deja ver de un vistazo (rojo: lo que resta)."""
    x = [f"{n} €" for n in pw_price["Nivel"]]
    marker_colors = [GRAY_MID if u == 0 else RED for u in pw_price["Utilidad"]]
    fig = go.Figure(go.Scatter(
        x=x, y=pw_price["Utilidad"], mode="lines+markers+text",
        line=dict(color=INK, width=2.4), marker=dict(size=13, color=marker_colors, line=dict(color=INK, width=1)),
        text=[_num(u, 2, sign=True) if u != 0 else "ref." for u in pw_price["Utilidad"]],
        textposition="top center", cliponaxis=False, hoverinfo="skip", textfont=dict(size=12.5),
    ))
    fig.add_hline(y=0, line_color=GRID)
    fig.update_yaxes(title_text="Utilidad parcial (puntos)", range=[min(pw_price["Utilidad"]) * 1.2, 0.9])
    fig.update_xaxes(title_text="Precio del billete")
    return _base_layout(fig, height=340, legend=False)


def playground_segment_comparison(predictions: dict, segment_labels: dict, segment_colors: dict,
                                  order: list) -> go.Figure:
    labels = [_wrap(segment_labels.get(s, s), 12) for s in order]
    values = [predictions[s] for s in order]
    colors = [segment_colors.get(s, GRAY_MID) for s in order]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker=dict(color=colors, line=OUTLINE), hoverinfo="skip",
        text=[_num(v) for v in values], textposition="outside", cliponaxis=False,
    ))
    fig.update_yaxes(title_text="Valoración estimada (1 a 10)", range=[0, 11])
    return _base_layout(fig, height=380, legend=False)
