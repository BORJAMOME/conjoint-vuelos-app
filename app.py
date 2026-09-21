"""
Análisis Conjoint — Preferencias de Vuelos
Case study interactivo en Streamlit: qué valoran realmente los clientes de una aerolínea en cada
característica de un vuelo, y cómo cambia esa prioridad según quién vuela.

Toda cifra del texto sale de los artefactos de model/artifacts (train.py y export_design_checks.py) o
se calcula de ellos: nada está escrito a mano.

Autor: Borja Mora Méndez
"""
import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

from components import charts, ui
from utils.conjoint import predict_all_segments
from utils.data_loader import (ATTRIBUTE_LABELS, LEVEL_LABELS, SEGMENT_COLORS, SEGMENT_LABELS,
                                artifacts_ready, load_csv, load_json)

# Streamlit recarga app.py al detectar cambios, pero mantiene en memoria los módulos locales ya
# importados. Tras un despliegue que modifica components/*.py y app.py a la vez, eso deja un
# app.py nuevo llamando a un módulo antiguo (AttributeError). Recargarlos en cada ejecución lo
# evita; el coste es despreciable.
importlib.reload(charts)
importlib.reload(ui)

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Análisis Conjoint · Preferencias de Vuelos",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

with open(ROOT / "assets" / "style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if not artifacts_ready():
    st.error(
        "Los artefactos del modelo todavía no se han generado. Ejecuta `py -3.10 model/train.py` y "
        "`py -3.10 model/export_design_checks.py` desde la raíz del proyecto y recarga esta página."
    )
    st.stop()

stats = load_json("dataset_stats.json")
model_summary = load_json("model_summary.json")
playground_model = load_json("playground_model.json")
checks = load_json("design_checks.json")

ratings_df = load_csv("ratings_raw.csv")
pw_overall = load_csv("partworths_overall.csv")
pw_segment = load_csv("partworths_segment.csv")
imp_overall = load_csv("importance_overall.csv")
imp_segment = load_csv("importance_segment.csv")
design_cards = load_csv("design_cards.csv", dtype=str)
card_ratings = load_csv("card_ratings.csv")

ATRIBUTOS = playground_model["atributos"]
ORDEN = playground_model["orden"]
SEGMENT_ORDER = ["Business", "Leisure", "Low Cost"]
PLOT = {"displayModeBar": False}


# ---------------------------------------------------------------- formato en español
def es(value: float, decimals: int = 1) -> str:
    """Número con coma decimal, como se escribe en español."""
    return f"{value:.{decimals}f}".replace("-", "−").replace(".", ",")


def sgn(value: float, decimals: int = 1) -> str:
    """Con signo y coma decimal; el menos tipográfico (−) para no confundirlo con un guion."""
    return f"{value:+.{decimals}f}".replace("-", "−").replace(".", ",")


def pct(value: float, decimals: int = 1) -> str:
    return f"{es(value, decimals)}%"


def miles(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


# ---------------------------------------------------------------- cifras del texto
n_fmt, rows_fmt = miles(stats["n_customers"]), miles(stats["n_rows"])
n_cards, n_combos = stats["n_cards"], stats["n_combinaciones_factorial_completo"]
n_unrated = checks["n_combinaciones_sin_valorar"]
seg_n = stats["segment_counts"]
rating_info, seg_rating = checks["rating"], checks["rating"]["por_segmento"]
scale_min, scale_max = rating_info["min"], rating_info["max"]

imp_o = imp_overall.set_index("Atributo")["Importancia"]
imp_s = imp_segment.set_index(["Segmento", "Atributo"])["Importancia"]
top_attr, second_attr = imp_o.sort_values(ascending=False).index[:2]
top_two = imp_o[top_attr] + imp_o[second_attr]
biz_flight, biz_price = imp_s[("Business", "Flight")], imp_s[("Business", "Price")]
lc_price = imp_s[("Low Cost", "Price")]

pw_o = pw_overall.set_index(["Atributo", "Nivel"])["Utilidad"]
pw_s = pw_segment.set_index(["Segmento", "Atributo", "Nivel"])["Utilidad"]
step_1 = abs(pw_o[("Price", "100")])
step_2 = abs(pw_o[("Price", "150")] - pw_o[("Price", "100")])

baggage_range = [imp_s[(s, "Baggage")] for s in SEGMENT_ORDER]
baggage_third_everywhere = all(
    imp_segment[imp_segment["Segmento"] == s].sort_values("Importancia", ascending=False)["Atributo"].tolist()[2] == "Baggage"
    for s in SEGMENT_ORDER
)
ci_half = max((v[a]["hi"] - v[a]["lo"]) / 2 for s, v in checks["importancia_ci"].items() for a in ATRIBUTOS)
seg_r2 = {s: stats["segments"][s]["r_squared"] for s in SEGMENT_ORDER}

ui.nav()
ui.install_smooth_scroll()

# ============================================================ HERO ==
st.markdown(
    f"""
    <div id="top" class="hero-wrap">
      <p class="hero-kicker">Machine Learning Case Study · Regresión (Análisis Conjoint)</p>
      <h1 class="hero-title">¿Qué hace que un vuelo merezca la pena <em>para cada cliente</em>?</h1>
      <p class="hero-sub">Una aerolínea puede cambiar el precio, añadir equipaje, quitar una escala o dar más
      flexibilidad. Pero no todos los clientes valoran esos extras de la misma manera.</p>
      <p class="hero-sub">Analicé {rows_fmt} valoraciones de {n_fmt} clientes para descubrir cuánto pesa realmente
      cada característica de un vuelo y cómo cambia esa valoración según el tipo de cliente.</p>
      <div class="hero-meta">
        <span class="hero-pill">Borja Mora Méndez</span>
        <span class="hero-pill">Python · statsmodels (OLS)</span>
        <span class="hero-pill">Streamlit</span>
        <span class="hero-pill">{n_fmt} clientes</span>
      </div>
      <div class="hero-scroll-row">
        <a href="#contexto" class="hero-scroll">explorar el caso &#8595;</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================ CONTEXTO ==
ui.section_open("contexto")
ui.eyebrow("Contexto")
ui.h2("El problema")
ui.lead("Una aerolínea puede crear muchas combinaciones distintas cambiando precio, escalas, equipaje, asiento, "
        "flexibilidad u horario.")
ui.lead("El problema no es crear esas combinaciones. Es saber <b>cuáles importan realmente al cliente</b>.")
ui.body("¿Pagaría más por un vuelo directo? ¿Cuánto le importa llevar equipaje incluido? ¿Prefiere ahorrar aunque "
        "tenga que hacer una escala?")
ui.lead("Y, sobre todo: <b>¿las respuestas son las mismas para todos?</b>")
st.write("")
ui.question_block(
    "La pregunta de negocio",
    '¿Cuánto aporta cada característica de un vuelo a su valoración '
    '<span class="accent">y cómo cambia ese valor según el cliente</span>?',
    "En lugar de preguntar directamente qué atributo considera más importante, el experimento pidió a los clientes "
    "que valoraran vuelos completos, y utilicé esas respuestas para descubrir qué había detrás de cada valoración.",
)
ui.section_close()

# ============================================================ DATOS ==
ui.section_open("datos")
ui.eyebrow("Materia prima")
ui.h2("Los datos")
ui.lead(f"Para averiguar qué valora cada cliente, los {n_fmt} participantes evaluaron las mismas {n_cards} "
        f"combinaciones de vuelo del {scale_min:.0f} al {scale_max:.0f}.")
ui.body(f"Eso generó {rows_fmt} valoraciones que permiten comparar cómo cambia la percepción de un vuelo cuando "
        "cambian sus características.")

ui.h3("El experimento")
ui.body(f"Cada vuelo combinaba <b>{stats['n_atributos']} atributos</b>: precio, equipaje, asiento, escalas, "
        "flexibilidad y horario. Cada atributo tenía dos o tres niveles:")
attr_table = pd.DataFrame([
    {"Atributo": "Precio", "Niveles": "50 € / 100 € / 150 €"},
    {"Atributo": "Equipaje", "Niveles": "20 kg incluido / Sin equipaje"},
    {"Atributo": "Selección de asiento", "Niveles": "Con selección / Aleatoria"},
    {"Atributo": "Escalas", "Niveles": "Directo / 1 escala"},
    {"Atributo": "Flexibilidad", "Niveles": "Con flexibilidad / Sin flexibilidad"},
    {"Atributo": "Horario de salida", "Niveles": "Mañana / Tarde / Noche"},
])
st.dataframe(attr_table, use_container_width=True, hide_index=True)

ui.lead(f"El problema es que combinar todos los niveles habría generado <b>{n_combos} vuelos diferentes</b>. "
        "Demasiados para pedirle a una persona que los valore.")
ui.body(f"Por eso el experimento se limita a <b>{n_cards} combinaciones</b>, un diseño fraccionado: suficientes "
        "para variar los atributos y poder separar después el efecto de cada uno.")
ui.body(f"Todos los clientes valoraron exactamente las mismas {n_cards} combinaciones: lo he comprobado uno por uno, "
        f"con los {n_fmt} clientes. Esa condición permite comparar sus respuestas bajo las mismas reglas.")
ui.note(f"El diseño no es un ortogonal estricto: la correlación máxima entre atributos es "
        f"{es(checks['correlacion_max_entre_atributos'], 2)} (un ortogonal exacto tendría 0) y el factor de inflación "
        f"de la varianza (VIF) máximo es {es(checks['vif_max'], 2)} (el ideal es 1). Con esos valores los efectos "
        f"siguen siendo separables. De las {n_combos} combinaciones posibles, {n_unrated} no se valoraron nunca.")

ui.eyebrow("En números", muted=True)
ui.kpi_grid([
    {"num": n_fmt, "label": "clientes"},
    {"num": f"{n_cards}", "label": "vuelos valorados por cliente"},
    {"num": rows_fmt, "label": "valoraciones"},
    {"num": f"{stats['n_atributos']}", "label": "atributos"},
    {"num": f"{scale_min:.0f}–{scale_max:.0f}", "label": "escala de valoración"},
], cols=5)
st.write("")
ui.body(f"Cada cliente pertenece a uno de tres segmentos que ya venían definidos en los datos: "
        f"<b>Business</b> ({miles(seg_n['Business'])} clientes), <b>Leisure</b> ({miles(seg_n['Leisure'])}) y "
        f"<b>Low Cost</b> ({miles(seg_n['Low Cost'])}).")
ui.section_close()

# ============================================================ EXPLORACIÓN ==
ui.section_open("exploracion")
ui.eyebrow("Qué dicen las valoraciones")
ui.h2("¿Qué me dicen las valoraciones?")
ui.lead("Antes de construir el modelo, quería saber si las respuestas tenían suficiente variación como para "
        "encontrar patrones.")
st.plotly_chart(charts.rating_distribution(ratings_df["Rating"]), use_container_width=True, config=PLOT)
ui.finding(
    f"La valoración media fue de <b>{es(ratings_df['Rating'].mean(), 2)} sobre 10</b>, pero detrás de esa media hay "
    "vuelos que gustan mucho y otros que generan una respuesta bastante peor. Esa diferencia es precisamente la que "
    "interesa explicar."
)
ui.note(f"El {es(rating_info['pct_en_el_maximo'])}% de las valoraciones es un {scale_max:.0f}, el máximo de la escala. "
        "Ese techo importa más adelante: un modelo lineal no lo conoce.")

ui.h3("¿Valoran igual los tres segmentos?")
st.plotly_chart(charts.rating_by_segment(ratings_df, SEGMENT_COLORS, SEGMENT_ORDER), use_container_width=True, config=PLOT)
ui.finding(
    f"No. La valoración media es <b>{es(seg_rating['Business']['media'])}</b> en Business, "
    f"<b>{es(seg_rating['Leisure']['media'])}</b> en Leisure y <b>{es(seg_rating['Low Cost']['media'])}</b> en Low "
    f"Cost (medianas: {es(seg_rating['Business']['mediana'])}, {es(seg_rating['Leisure']['mediana'])} y "
    f"{es(seg_rating['Low Cost']['mediana'])})."
)
ui.body("Pero saber que puntúan distinto no dice <b>por qué</b>. Un cliente puede dar un 8 porque el vuelo es directo. "
        "Otro puede darle un 8 porque es barato.")
ui.lead("La valoración final puede ser la misma. <b>Lo que cambia es lo que hay detrás.</b> Y eso es lo que el modelo "
        "tiene que descubrir.")
ui.section_close()

# ============================================================ METODOLOGÍA ==
ui.section_open("metodologia")
ui.eyebrow("Cómo funciona")
ui.h2("De una valoración global al valor de cada característica")
ui.lead("El cliente no me dice directamente cuánto valen el precio, las escalas o el equipaje. Me dice cuánto le "
        "gusta un vuelo completo. El análisis conjoint utiliza esas valoraciones para separar después cuánto aporta "
        "cada característica.")
ui.story_steps([
    ("Primero, el cliente valora el vuelo completo",
     "En lugar de preguntar «¿cuánto te importa el precio?», cada cliente ve una combinación concreta de atributos "
     f"y la puntúa del {scale_min:.0f} al {scale_max:.0f}."),
    ("Todos valoran las mismas combinaciones",
     f"Los {n_cards} vuelos están diseñados para que los atributos cambien entre unas tarjetas y otras. Así puedo "
     "observar qué ocurre con la valoración cuando cambia una característica y las demás se mantienen dentro de un "
     "diseño controlado."),
    ("El modelo separa los efectos",
     "Cada nivel de cada atributo se convierte en una variable que el modelo puede comparar frente a un nivel de "
     "referencia. Así puedo estimar cuánto suma o resta cada característica a la valoración."),
    ("La regresión pone números a esas diferencias",
     f"Ajusté una regresión OLS sobre las {rows_fmt} valoraciones. El resultado son las llamadas <b>utilidades "
     "parciales</b> (<i>part-worths</i>): una estimación de cuánto aporta cada nivel a la valoración del vuelo."),
    ("Y ahora puedo construir vuelos que nadie valoró",
     f"Una vez estimadas esas utilidades, puedo combinarlas para calcular la valoración esperada de una nueva "
     f"configuración. De las {n_combos} combinaciones posibles solo se valoraron {n_cards}: el resto ({n_unrated}) "
     "se estima sumando utilidades. Es lo que permite el Playground."),
])
ui.section_close()

# ============================================================ MODELO ==
ui.section_open("modelo")
ui.eyebrow("Resultado")
ui.h2("¿Cuánto consigue explicar el modelo?")
ui.lead(f"Con los {stats['n_atributos']} atributos incluidos, el modelo explica el "
        f"<b>{pct(model_summary['r_squared'] * 100)}</b> de la variación en las valoraciones.")
ui.body(f"Es decir, una parte importante de las diferencias entre las {rows_fmt} respuestas puede explicarse por las "
        "características de los vuelos incluidas en el análisis. La parte que queda sin explicar son, sobre todo, "
        "diferencias entre clientes: más adelante veremos que el modelo mejora mucho cuando se ajusta por segmento.")
ui.body(f"<b>R² = {es(model_summary['r_squared'], 3)}</b> · {miles(model_summary['n_obs'])} observaciones · "
        f"8 coeficientes significativos (p &lt; 0,001)")
ui.note("El R² se mide sobre los mismos datos con los que se ajustó el modelo: no hay conjunto de prueba. Como cada "
        f"cliente aporta {n_cards} valoraciones, comprobé la significación con errores agrupados por cliente: el "
        f"|t| más bajo de los 8 coeficientes es {es(checks['robustez']['t_min_agrupado_por_cliente'], 0)}.")
ui.lead("Pero saber que el modelo funciona no responde todavía a la pregunta más interesante: <b>¿qué características "
        "hacen que un vuelo guste más o menos?</b>")
ui.body("Y entonces entramos en las utilidades.")

ui.h3("Las utilidades parciales de cada atributo")
ui.body("Cada gráfico muestra cuánto suma o resta cada nivel a la valoración, frente a su nivel de referencia (en "
        "gris, utilidad 0). La unidad son puntos de valoración sobre 10.")
attr_cols = st.columns(3)
for i, atributo in enumerate(ATRIBUTOS):
    with attr_cols[i % 3]:
        st.markdown(f'<p class="co-body" style="font-weight:700; text-align:center;">{ATTRIBUTE_LABELS[atributo]}</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(charts.partworth_bars(pw_overall[pw_overall["Atributo"] == atributo], LEVEL_LABELS),
                        use_container_width=True, config=PLOT)

ui.h3("El precio no pesa igual en todos los tramos")
st.plotly_chart(charts.price_curve(pw_overall[pw_overall["Atributo"] == "Price"]), use_container_width=True, config=PLOT)
lc_step1 = abs(pw_s[("Low Cost", "Price", "100")])
biz_step1 = abs(pw_s[("Business", "Price", "100")])
ui.finding(
    f"Subir el precio de 50 € a 100 € resta <b>{es(step_1, 2)}</b> puntos de utilidad. Pasar de 100 € a 150 € resta "
    f"otros <b>{es(step_2, 2)}</b>: el segundo salto cuesta más del doble. La sensibilidad al precio no es "
    "simplemente «cada euro importa lo mismo»: el modelo detecta un cambio claro a partir de los 100 €."
)
ui.body(f"Pero ese es el patrón del cliente promedio. Por segmentos cambia: Low Cost ya pierde {es(lc_step1, 2)} puntos "
        f"al pasar de 50 € a 100 €, mientras que Business pierde solo {es(biz_step1, 2)}.")
ui.section_close()

# ============================================================ EXPLICABILIDAD ==
ui.section_open("explicabilidad")
ui.eyebrow("Qué importa")
ui.h2("¿Qué características mueven realmente la valoración?")
ui.lead("Las utilidades nos dicen cuánto aporta cada nivel. Pero hay otra pregunta: <b>¿qué atributos tienen más peso "
        "en la decisión?</b>")
ui.body("Para responderla, calculé la <b>importancia relativa</b> de cada atributo comparando la diferencia entre su "
        "nivel mejor y peor valorado.")

ui.h3("El cliente promedio")
st.plotly_chart(charts.importance_overall(imp_overall, ATTRIBUTE_LABELS), use_container_width=True, config=PLOT)
ui.finding(
    f"<b>{ATTRIBUTE_LABELS[top_attr]}</b> ({pct(imp_o[top_attr])}) y <b>{ATTRIBUTE_LABELS[second_attr].lower()}</b> "
    f"({pct(imp_o[second_attr])}) concentran el <b>{pct(top_two, 0)}</b> de la importancia total. La selección de "
    "asiento y la flexibilidad tienen mucho menos peso."
)
ui.note("La importancia es relativa a los niveles que se probaron: el precio solo se probó entre 50 € y 150 €, y con "
        "otro rango su peso sería distinto.")
ui.lead("Pero aquí aparece una limitación importante de mirar solo el promedio: <b>el cliente promedio no existe</b>. "
        "Cuando separo los resultados por segmento, la historia cambia.")
ui.section_close()

# ============================================================ SEGMENTOS ==
ui.section_open("segmentos")
ui.eyebrow("Tres formas de elegir")
ui.h2("El mismo vuelo no significa lo mismo para todos")
ui.lead("Cuando separo las valoraciones por segmento, las prioridades cambian.")
st.plotly_chart(charts.importance_by_segment(imp_segment, ATTRIBUTE_LABELS, SEGMENT_COLORS, SEGMENT_ORDER),
                use_container_width=True, config=PLOT)

direct_biz, price150_biz = pw_s[("Business", "Flight", "Direct")], abs(pw_s[("Business", "Price", "150")])
seg_text = {
    "Business": (f"Para este segmento, volar directo aporta más valoración ({sgn(direct_biz)} puntos) que lo que "
                 f"resta encarecer el billete de 50 € a 150 € ({sgn(-price150_biz)})."
                 if direct_biz > price150_biz else "Las escalas pesan más que el precio."),
    "Leisure": "Busca un equilibrio diferente entre coste y comodidad.",
    "Low Cost": "Aquí el precio domina claramente la valoración.",
}
seg_cols = st.columns(3, gap="medium")
for c, seg in zip(seg_cols, SEGMENT_ORDER):
    with c:
        top2 = imp_segment[imp_segment["Segmento"] == seg].sort_values("Importancia", ascending=False).head(2)
        value = "<br>".join(f"{ATTRIBUTE_LABELS[a]}: {pct(v)}" for a, v in zip(top2["Atributo"], top2["Importancia"]))
        ui.stat_card(seg, value, seg_text[seg], title_color=SEGMENT_COLORS[seg], value_size="1.25rem",
                     min_height="10.5rem")
st.write("")
ui.finding(
    "No estamos ante tres clientes que simplemente puntúan los mismos vuelos de forma diferente. Son <b>tres formas "
    "distintas de valorar</b> las características de un vuelo."
)
ui.body(f"Leisure es el segmento más numeroso ({miles(seg_n['Leisure'])} de {n_fmt} clientes), y por eso se parece tanto "
        "al cliente promedio: Business y Low Cost son los que más se alejan de él.")
ui.body(f"El modelo ajustado solo con los datos de cada segmento explica más variación (R² = {es(seg_r2['Business'], 2)} "
        f"en Business, {es(seg_r2['Leisure'], 2)} en Leisure y {es(seg_r2['Low Cost'], 2)} en Low Cost) que el modelo "
        f"único con todos los clientes mezclados ({es(model_summary['r_squared'], 2)}). Mezclar los tres perfiles diluye "
        "una señal que, por separado, es mucho más clara.")
ui.note(f"Intervalos de confianza del 95% por bootstrap de clientes ({miles(checks['bootstrap']['repeticiones'])} "
        f"repeticiones): la importancia de cada atributo varía menos de ±{es(ci_half, 1)} puntos, así que las "
        "diferencias entre segmentos no son ruido de muestreo. Esos intervalos solo recogen el azar de los clientes "
        "de la muestra, no las limitaciones del diseño.")
ui.section_close()

# ============================================================ PLAYGROUND ==
ui.section_open("playground")
ui.eyebrow("Diseña un vuelo")
ui.h2("¿Y si pudieras diseñar un vuelo y ver quién lo valoraría mejor?")
ui.lead("El modelo ya ha estimado cuánto aporta cada característica. Ahora podemos utilizar esas estimaciones para "
        f"construir un vuelo que quizá ningún cliente haya valorado durante el experimento (de las {n_combos} "
        f"combinaciones posibles, solo {n_cards} se valoraron).")
ui.body("Elige precio, equipaje, asiento, escalas, flexibilidad y horario. El modelo calcula cómo lo valoraría cada "
        "segmento: la misma combinación de atributos, con valoraciones que pueden ser muy distintas.")

pg_left, pg_right = st.columns([1, 1.2], gap="large")
level_options = {a: ORDEN[a] for a in ATRIBUTOS}
default_choice = {"Price": "100", "Baggage": "20 kg", "Seat": "Random", "Flight": "1 Stop",
                   "Flexible": "No", "Departure": "Morning"}
choice = {}
with pg_left:
    for atributo in ATRIBUTOS:
        opts = level_options[atributo]
        nice_opts = [LEVEL_LABELS.get(o, o) for o in opts]
        default_idx = opts.index(default_choice[atributo]) if default_choice[atributo] in opts else 0
        picked = st.selectbox(ATTRIBUTE_LABELS[atributo], nice_opts, index=default_idx, key=f"pg_{atributo}")
        choice[atributo] = opts[nice_opts.index(picked)]

raw_predictions = predict_all_segments(choice, ATRIBUTOS, playground_model["intercept"], playground_model["partworths"])
# El modelo es lineal y la escala va de 1 a 10 (el 22% de las valoraciones reales es un 10): puede calcular valores
# fuera de la escala. Se muestran acotados a 1-10 y se avisa.
predictions = {k: min(max(v, scale_min), scale_max) for k, v in raw_predictions.items()}
out_of_scale = {k: v for k, v in raw_predictions.items() if v > scale_max or v < scale_min}
plot_order = ["Overall"] + SEGMENT_ORDER

with pg_right:
    st.plotly_chart(
        charts.playground_segment_comparison(predictions, SEGMENT_LABELS, SEGMENT_COLORS, plot_order),
        use_container_width=True, config=PLOT,
    )
    badge_cols = st.columns(3)
    for c, seg in zip(badge_cols, SEGMENT_ORDER):
        with c:
            ui.stat_card(seg, f"{es(predictions[seg])}", color=SEGMENT_COLORS[seg], value_size="1.5rem")

match = design_cards[design_cards[ATRIBUTOS].eq(pd.Series(choice)).all(axis=1)]
if len(match):
    card_id = int(match.iloc[0]["CardID"])
    real = card_ratings[card_ratings["CardID"] == card_id].iloc[0]
    ui.note("Este vuelo sí se valoró en el experimento (tarjeta nº " + str(card_id) + "). Valoración media real: "
            + " · ".join(f"{s} {es(real[s])}" for s in SEGMENT_ORDER) + ". El modelo estima: "
            + " · ".join(f"{s} {es(predictions[s])}" for s in SEGMENT_ORDER) + ".", label="Comprobación")
else:
    ui.note(f"Este vuelo no estaba entre las {n_cards} que se valoraron: es una de las {n_unrated} combinaciones que "
            "nadie puntuó, y su valoración es una estimación que suma utilidades.", label="Comprobación")
if out_of_scale:
    names = ", ".join(f"{SEGMENT_LABELS[k]} ({es(v)})" for k, v in out_of_scale.items())
    ui.note(f"El modelo lineal calcula valores fuera de la escala {scale_min:.0f}–{scale_max:.0f} para {names}; se "
            f"muestran acotados. Es el efecto techo: el {es(rating_info['pct_en_el_maximo'], 0)}% de las valoraciones "
            "reales es un 10.", label="Techo de la escala")

best_seg = max(SEGMENT_ORDER, key=lambda s: predictions[s])
worst_seg = min(SEGMENT_ORDER, key=lambda s: predictions[s])
gap = predictions[best_seg] - predictions[worst_seg]
ui.h3("¿Convence este vuelo a todo el mundo por igual?")
if gap < 0.8:
    ui.finding(
        f"Los tres segmentos valoran este vuelo de forma muy parecida (diferencia de solo {es(gap)} puntos entre "
        f"{best_seg} y {worst_seg}): es un diseño de vuelo genérico, sin un ganador ni un perdedor claro."
    )
else:
    ui.finding(
        f"<b>{best_seg}</b> ({es(predictions[best_seg])}) valora este vuelo {es(gap)} puntos por encima de "
        f"<b>{worst_seg}</b> ({es(predictions[worst_seg])}): la misma combinación de atributos no genera el mismo "
        "entusiasmo en todos los perfiles de cliente."
    )
ui.section_close()

# ============================================================ RESULTADOS ==
ui.section_open("resultados")
ui.eyebrow("Qué aprendí")
ui.h2("Tres clientes. Tres formas de valorar un vuelo.")
ui.lead(f"El análisis encuentra una señal clara: <b>{ATTRIBUTE_LABELS[top_attr]} y {ATTRIBUTE_LABELS[second_attr].lower()}</b> "
        f"concentran el {pct(top_two, 0)} de la importancia para el cliente promedio.")
ui.body("Pero cuando separamos los segmentos, el promedio deja de contar toda la historia.")
ui.body(f"<b>Business</b> prioriza las escalas ({pct(biz_flight)}). <b>Leisure</b> da más peso al precio "
        f"({pct(imp_s[('Leisure', 'Price')])}), pero mantiene las escalas como segundo factor "
        f"({pct(imp_s[('Leisure', 'Flight')])}). <b>Low Cost</b> concentra casi el {pct(lc_price, 0)} de su decisión en "
        "el precio.")
ui.lead("La conclusión no es que exista un atributo ganador. Es que <b>el valor de un atributo depende de quién está "
        "tomando la decisión</b>.")
ui.section_close()

# ============================================================ IMPACTO ==
ui.section_open("impacto", tight=True)
ui.impact_banner(
    f'Para <span class="accent">Business</span>, volar directo pesa más que el precio. Para '
    f'<span class="accent">Low Cost</span>, el precio concentra casi el {pct(lc_price, 0)} de la decisión.',
    quote='"No quería saber qué vuelo gusta más. Quería saber por qué gusta y si ese porqué cambia según el cliente."',
)
ui.section_close()

# ============================================================ DECISIONES ==
ui.section_open("decisiones")
ui.eyebrow("Qué podría hacer una empresa")
ui.h2("¿Qué podría hacer una empresa con estos resultados?")
ui.lead("El modelo no decide qué producto lanzar. Pero sí permite plantear <b>hipótesis mucho más concretas</b>.")
ui.hypothesis_flow(
    "Business",
    f"Las escalas representan el {pct(biz_flight)} de la importancia, por encima del precio ({pct(biz_price)}).",
    "Probar una propuesta de vuelo directo orientada a este segmento y medir su respuesta.",
    "Adopción, conversión y disposición a pagar.",
)
ui.hypothesis_flow(
    "Low Cost",
    f"El precio concentra el {pct(lc_price)} de la importancia.",
    "Probar una propuesta centrada en precio antes de añadir extras como argumento principal.",
    "Conversión y coste de adquisición.",
)
ui.hypothesis_flow(
    "Equipaje",
    f"Su importancia se mantiene relativamente estable entre los tres segmentos ({pct(min(baggage_range), 0)} a "
    f"{pct(max(baggage_range), 0)})" + (" y es el tercer atributo en todos." if baggage_third_everywhere else "."),
    "Evaluar si funciona mejor integrado en la propuesta base que como extra diferenciado.",
    "Conversión, aceptación y reclamaciones relacionadas con el equipaje.",
)
ui.finding("Estas no son decisiones que el modelo haya tomado. Son <b>hipótesis que podrían pasar después a un "
           "experimento real</b>.")
ui.section_close()

# ============================================================ LIMITACIONES ==
ui.section_open("limitaciones")
ui.eyebrow("Honestidad ante todo")
ui.h2("Limitaciones")
ui.lead("Un buen análisis también tiene que dejar claro dónde termina lo que sabemos.")
fit = checks["ajuste_tarjeta"]
lc1, lc2 = st.columns(2, gap="large")
with lc1:
    st.markdown('<p class="limit-col-title">Lo que el modelo SÍ puede hacer</p>', unsafe_allow_html=True)
    st.markdown(
        f"""<ul class="limit-list">
        <li>Descomponer una valoración global en el valor de cada atributo individual, con significancia estadística (también con errores agrupados por cliente).</li>
        <li>Simular vuelos que ningún cliente valoró nunca, sumando utilidades ya estimadas ({n_unrated} de las {n_combos} combinaciones no se valoraron).</li>
        <li>Mostrar que la sensibilidad al precio del cliente promedio no es lineal: el segundo tramo pesa más del doble.</li>
        <li>Mostrar que la prioridad de atributos cambia claramente entre segmentos de cliente, con diferencias mayores que el azar de muestreo.</li>
        </ul>""",
        unsafe_allow_html=True,
    )
with lc2:
    st.markdown('<p class="limit-col-title">Lo que el modelo NO puede hacer</p>', unsafe_allow_html=True)
    st.markdown(
        f"""<ul class="limit-list">
        <li>Capturar interacciones entre atributos — p.ej. si el precio importa menos cuando el vuelo es directo, este modelo no lo ve.</li>
        <li>Garantizar que una preferencia declarada en una encuesta se traduzca en una compra real (declared vs. revealed preference).</li>
        <li>Generalizar a atributos o niveles que nunca se incluyeron en el diseño (p.ej. un precio de 200 €).</li>
        <li>Respetar el techo de la escala: es un modelo lineal y puede calcular valoraciones por encima de {scale_max:.0f} (el {es(rating_info['pct_en_el_maximo'], 0)}% de las valoraciones reales es un {scale_max:.0f}).</li>
        <li>Reproducir con exactitud cada vuelo: el error medio por tarjeta es de {es(fit['Overall']['mae_tarjeta'], 1)} puntos con todos los clientes y de {es(fit['Business']['mae_tarjeta'], 1)} en Business.</li>
        <li>Sustituir un test de mercado real antes de un lanzamiento de producto con impacto económico grande.</li>
        </ul>""",
        unsafe_allow_html=True,
    )
st.markdown(
    '<div class="limit-note"><p class="co-body">'
    "Este es un conjoint <i>rating-based</i> clásico: mide cuánto gusta cada vuelo, no si el cliente lo "
    "compraría al precio marcado frente a alternativas reales del mercado. Para decisiones de pricing con "
    "impacto económico alto, conviene contrastar estos resultados con un Choice-Based Conjoint o un test "
    "de mercado antes de fijar precios definitivos. Además, el modelo no se ha evaluado con datos que no hubiera "
    "visto, y los segmentos vienen dados en el dataset: no se ha comprobado que sean la mejor forma de agrupar "
    "a los clientes."
    "</p></div>",
    unsafe_allow_html=True,
)
ui.section_close()

# ============================================================ CONCLUSIÓN ==
ui.section_open("conclusion")
ui.eyebrow("Del dato a la decisión")
ui.h2("No existe un único valor para todos los clientes")
ui.lead(f"A partir de {rows_fmt} valoraciones, el modelo permite descomponer una valoración global y estimar cuánto "
        "aporta cada característica de un vuelo.")
ui.body("El resultado más claro es que <b>precio y escalas dominan la decisión</b>, pero no de la misma manera para "
        "todos.")
ui.body("Para <b>Business</b>, las escalas pesan más que el precio.<br>"
        "Para <b>Leisure</b>, el precio ocupa el primer lugar.<br>"
        f"Para <b>Low Cost</b>, el precio concentra casi el {pct(lc_price, 0)} de la importancia.")
ui.lead("El análisis no dice qué vuelo debe vender una aerolínea. Dice algo más útil: <b>qué está valorando cada tipo "
        "de cliente y dónde cambia esa valoración</b>.")
ui.lead("El siguiente paso ya no sería preguntar qué prefieren. <b>Sería probarlo en el mercado.</b>")
ui.section_close()

ui.footer_minimal(
    name="Borja Mora Méndez",
    repo_url="https://github.com/BORJAMOME/conjoint-vuelos-app",
    linkedin_url="https://www.linkedin.com/in/borja-mora-mendez/",
    email="borja.mora.mendez@gmail.com",
)
