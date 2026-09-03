"""
Análisis Conjoint — Preferencias de Vuelos
Case study interactivo en Streamlit: qué valoran realmente los clientes
de una aerolínea en cada característica de un vuelo, y cómo cambia esa
prioridad según quién vuela.

Autor: Borja Mora Méndez
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from components import charts, ui
from utils.conjoint import predict_all_segments
from utils.data_loader import (ATTRIBUTE_LABELS, LEVEL_LABELS, SEGMENT_COLORS, SEGMENT_LABELS,
                                artifacts_ready, load_csv, load_json)

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
        "Los artefactos del modelo todavía no se han generado. "
        "Ejecuta `py -3.10 model/train.py` desde la raíz del proyecto y recarga esta página."
    )
    st.stop()

stats = load_json("dataset_stats.json")
model_summary = load_json("model_summary.json")
playground_model = load_json("playground_model.json")

ratings_df = load_csv("ratings_raw.csv")
pw_overall = load_csv("partworths_overall.csv")
pw_segment = load_csv("partworths_segment.csv")
imp_overall = load_csv("importance_overall.csv")
imp_segment = load_csv("importance_segment.csv")

ATRIBUTOS = playground_model["atributos"]
ORDEN = playground_model["orden"]
SEGMENT_ORDER = ["Business", "Leisure", "Low Cost"]

n_fmt = f"{stats['n_customers']:,}".replace(",", ".")
rows_fmt = f"{stats['n_rows']:,}".replace(",", ".")


def pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


ui.nav()
ui.install_smooth_scroll()

# ============================================================ HERO ==
st.markdown(
    f"""
    <div id="top" class="hero-wrap">
      <p class="hero-kicker">Machine Learning Case Study · Regresión (Análisis Conjoint)</p>
      <h1 class="hero-title">Una aerolínea diseña vuelos con precio, escalas, equipaje y flexibilidad, pero no sabe cuánto vale cada extra para el cliente, ni si vale lo mismo para todos. Así es como lo averigüé, <em>atributo por atributo</em>.</h1>
      <p class="hero-sub">{n_fmt} clientes valoraron 24 combinaciones distintas de vuelo. Con una regresión sobre esas
      {rows_fmt} valoraciones, descompuse cada rating en el valor exacto que aporta cada característica,
      y descubrí que ese valor cambia por completo según quién compra el billete.</p>
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
ui.lead(
    "Una aerolínea puede combinar precio, equipaje, selección de asiento, escalas, flexibilidad y horario "
    "en cientos de configuraciones de vuelo distintas. Pero combinar variables no es lo mismo que saber "
    "cuáles importan de verdad: sin esa información, cada decisión de producto o de precio es una apuesta, "
    "no una decisión basada en lo que el cliente realmente valora."
)
ui.kpi_grid([
    {"num": n_fmt, "label": "clientes encuestados"},
    {"num": "24", "label": "combinaciones de vuelo valoradas"},
    {"num": f"{stats['n_atributos']}", "label": "atributos del vuelo"},
    {"num": "3", "label": "segmentos de cliente"},
])
st.write("")
ui.question_block(
    "La pregunta de negocio",
    '¿Cuánto vale realmente cada característica de un vuelo para el cliente, '
    '<span class="accent">y es el mismo valor para todo el mundo</span>?',
    "No me bastaba con preguntar directamente \"¿cuánto te importa el precio?\": la gente no siempre "
    "sabe responder eso con precisión. Tenía que observar cómo valora vuelos completos y descomponer "
    "esa valoración, característica a característica.",
)
ui.section_close()

# ============================================================ DATOS ==
ui.section_open("datos")
ui.eyebrow("Materia prima")
ui.h2("Los datos")
ui.lead(
    f"{n_fmt} clientes valoraron, del 1 al 10, las mismas 24 combinaciones de vuelo — un total de "
    f"{rows_fmt} valoraciones. Cada combinación mezcla 6 atributos con 2 o 3 niveles cada uno."
)
ui.kpi_grid([
    {"num": n_fmt, "label": "clientes"},
    {"num": "24", "label": "tarjetas (vuelos) valoradas por cliente"},
    {"num": rows_fmt, "label": "valoraciones totales"},
    {"num": "0-10", "label": "escala del rating"},
])

ui.eyebrow("Los 6 atributos del vuelo", muted=True)
attr_table = pd.DataFrame([
    {"Atributo": "Precio", "Niveles": "50 € / 100 € / 150 €"},
    {"Atributo": "Equipaje", "Niveles": "20 kg incluido / Sin equipaje"},
    {"Atributo": "Selección de asiento", "Niveles": "Con selección / Aleatoria"},
    {"Atributo": "Escalas", "Niveles": "Directo / 1 escala"},
    {"Atributo": "Flexibilidad", "Niveles": "Con flexibilidad / Sin flexibilidad"},
    {"Atributo": "Horario de salida", "Niveles": "Mañana / Tarde / Noche"},
])
st.dataframe(attr_table, use_container_width=True, hide_index=True)

n_combos = stats["n_combinaciones_factorial_completo"]
ui.finding(
    f"Combinar los 6 atributos con todos sus niveles daría <b>{n_combos} vuelos distintos</b>: "
    "imposible pedirle a un cliente que valore tantos. Por eso usé un <b>diseño ortogonal</b> "
    "(fractional factorial): un subconjunto de solo <b>24 tarjetas</b>, elegido estadísticamente para "
    "poder aislar el efecto de cada atributo por separado, sin perder esa capacidad de análisis."
)
same_design = "Sí" if stats["mismo_diseno_para_todos"] else "No"
ui.body(
    f"Verificación técnica: <b>¿todos los clientes valoraron exactamente el mismo diseño de 24 tarjetas?</b> "
    f"{same_design}. Condición necesaria para poder comparar respuestas entre clientes con las mismas "
    "reglas."
)
ui.section_close()

# ============================================================ EXPLORACIÓN ==
ui.section_open("exploracion")
ui.eyebrow("Antes de modelar")
ui.h2("¿Qué me dicen las valoraciones?")
ui.lead("Antes de descomponer nada, me hice dos preguntas: ¿cómo se reparten los ratings? ¿Puntúan "
        "igual los tres segmentos?")

ui.h3("Distribución de los ratings")
st.plotly_chart(charts.rating_distribution(ratings_df["Rating"]), use_container_width=True,
                 config={"displayModeBar": False})
mean_r = ratings_df["Rating"].mean()
ui.finding(
    f"La media de todas las valoraciones es {mean_r:.2f} sobre 10: hay de todo, desde vuelos que "
    "encantan hasta combinaciones que decepcionan. Esa variación es justo lo que el modelo necesita para "
    "poder aprender qué atributos la explican."
)

ui.h3("¿Puntúan igual los tres segmentos?")
st.plotly_chart(charts.rating_by_segment(ratings_df, SEGMENT_COLORS, SEGMENT_ORDER), use_container_width=True,
                 config={"displayModeBar": False})
ui.finding(
    "Las medianas son parecidas entre segmentos: la diferencia real no está en <i>cuánto puntúan</i> de "
    "media, sino en <i>qué</i> hace que puntúen alto o bajo. Eso solo se ve descomponiendo el rating por "
    "atributo, no mirando la distribución global."
)
ui.section_close()

# ============================================================ METODOLOGÍA ==
ui.section_open("metodologia")
ui.eyebrow("Cómo funciona un análisis conjoint")
ui.h2("De la valoración global al valor de cada extra")
ui.lead(
    "Un análisis conjoint es, por dentro, una regresión lineal — pero la lógica del experimento es lo que "
    "lo hace útil."
)
ui.story_steps([
    ("No pregunté directamente",
     "En vez de \"¿cuánto te importa el precio?\", le mostré al cliente un vuelo completo (todos sus "
     "atributos a la vez) y le pedí un rating global del 1 al 10."),
    ("Cada cliente valora las mismas 24 tarjetas",
     "El diseño ortogonal garantiza que, entre las 24 tarjetas, cada nivel de cada atributo aparece "
     "combinado con suficiente variedad de los demás: así se puede aislar su efecto individual."),
    ("Codifiqué cada atributo como variable dummy",
     "Cada nivel se convierte en una variable 0/1 frente a un nivel de referencia (p.ej. \"Precio 100€\" "
     "y \"Precio 150€\" se codifican frente a la referencia \"Precio 50€\")."),
    ("Ajusté una regresión OLS sobre las 24.000 valoraciones",
     "Rating ~ Precio + Equipaje + Asiento + Escalas + Flexibilidad + Horario. El coeficiente de cada "
     "nivel es su <b>utilidad parcial</b> (part-worth): cuánto suma o resta a la valoración total, frente "
     "a su referencia."),
    ("Sumé las utilidades para simular cualquier vuelo",
     "La utilidad total de un vuelo (real o hipotético) es la suma de las utilidades de sus atributos. "
     "Eso permite simular combinaciones que ningún cliente valoró nunca: la base del Playground."),
])
ui.section_close()

# ============================================================ MODELO ==
ui.section_open("modelo")
ui.eyebrow("¿Cómo de bien explica el modelo el rating?")
ui.h2("El modelo global")
ui.lead(
    f"Con los 6 atributos, el modelo explica el {pct(model_summary['r_squared']*100)} de la variación en "
    f"los ratings (R²={model_summary['r_squared']:.3f}), y los 8 coeficientes son estadísticamente "
    f"significativos (p < 0.001 en todos los casos)."
)
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("R² del modelo", f"{model_summary['r_squared']:.3f}")
with m2:
    st.metric("Observaciones", f"{model_summary['n_obs']:,}".replace(",", "."))
with m3:
    st.metric("Intercepto (rating base)", f"{model_summary['intercept']:.2f}")

st.write("")
ui.h3("Las utilidades parciales de cada atributo")
ui.body("Cada gráfico muestra cuánto suma o resta cada nivel frente a su referencia (en gris, utilidad 0).")
attr_cols = st.columns(3)
for i, atributo in enumerate(ATRIBUTOS):
    with attr_cols[i % 3]:
        st.markdown(f'<p class="co-body" style="font-weight:700; text-align:center;">{ATTRIBUTE_LABELS[atributo]}</p>',
                    unsafe_allow_html=True)
        sub = pw_overall[pw_overall["Atributo"] == atributo]
        st.plotly_chart(charts.partworth_bars(sub, LEVEL_LABELS), use_container_width=True,
                         config={"displayModeBar": False})

ui.h3("El precio no pesa lo mismo en cada tramo")
pw_price = pw_overall[pw_overall["Atributo"] == "Price"]
st.plotly_chart(charts.price_curve(pw_price), use_container_width=True, config={"displayModeBar": False})
drop_1 = abs(pw_price[pw_price["Nivel"] == "100"]["Utilidad"].values[0])
drop_2 = abs(pw_price[pw_price["Nivel"] == "150"]["Utilidad"].values[0] - pw_price[pw_price["Nivel"] == "100"]["Utilidad"].values[0])
ui.finding(
    f"Subir de 50€ a 100€ cuesta {drop_1:.2f} puntos de utilidad. Pero subir de 100€ a 150€ cuesta "
    f"{drop_2:.2f}, más del doble. La sensibilidad al precio <b>no es lineal</b>: hay un salto psicológico "
    "a partir de los 100€ que un modelo que asumiera \"cada euro cuesta lo mismo\" no habría detectado."
)
ui.section_close()

# ============================================================ EXPLICABILIDAD ==
ui.section_open("explicabilidad")
ui.eyebrow("¿Qué importa más?")
ui.h2("Explicabilidad")
ui.lead(
    "La utilidad parcial dice cuánto vale un nivel. La <b>importancia relativa</b> —el rango entre el "
    "nivel mejor y peor valorado de cada atributo— dice qué tanto mueve la decisión ese atributo frente "
    "a los demás."
)

ui.h3("Importancia relativa, con todos los clientes juntos")
st.plotly_chart(charts.importance_overall(imp_overall, ATTRIBUTE_LABELS), use_container_width=True,
                 config={"displayModeBar": False})
top_attr = imp_overall.sort_values("Importancia", ascending=False).iloc[0]
second_attr = imp_overall.sort_values("Importancia", ascending=False).iloc[1]
ui.finding(
    f"<b>{ATTRIBUTE_LABELS[top_attr['Atributo']]}</b> ({pct(top_attr['Importancia'])}) y "
    f"<b>{ATTRIBUTE_LABELS[second_attr['Atributo']]}</b> ({pct(second_attr['Importancia'])}) concentran "
    f"casi el {pct(top_attr['Importancia']+second_attr['Importancia'], 0)} de lo que decide la valoración "
    "de un vuelo. Selección de asiento y flexibilidad, en cambio, apenas mueven la aguja: son extras, no "
    "razones de decisión."
)

ui.h3("Pero esa importancia cambia por completo según el segmento")
st.plotly_chart(charts.importance_by_segment(imp_segment, ATTRIBUTE_LABELS, SEGMENT_COLORS, SEGMENT_ORDER),
                 use_container_width=True, config={"displayModeBar": False})

biz_flight = imp_segment[(imp_segment["Segmento"] == "Business") & (imp_segment["Atributo"] == "Flight")]["Importancia"].values[0]
biz_price = imp_segment[(imp_segment["Segmento"] == "Business") & (imp_segment["Atributo"] == "Price")]["Importancia"].values[0]
lc_price = imp_segment[(imp_segment["Segmento"] == "Low Cost") & (imp_segment["Atributo"] == "Price")]["Importancia"].values[0]
ui.finding(
    f"Para <b>Business</b>, las escalas pesan más que el precio ({pct(biz_flight)} frente a "
    f"{pct(biz_price)}): pagan por llegar directo. Para <b>Low Cost</b>, el precio concentra por sí solo "
    f"{pct(lc_price)} de la decisión, casi el triple que las escalas. No es el mismo cliente disfrazado de "
    "tres segmentos: son tres lógicas de decisión distintas."
)
ui.body(
    f"Encontré otra señal en la misma dirección: el modelo ajustado <i>solo</i> con los datos de cada segmento "
    f"explica mucho mejor su comportamiento (R²={stats['segments']['Business']['r_squared']:.2f} en "
    f"Business, {stats['segments']['Leisure']['r_squared']:.2f} en Leisure, "
    f"{stats['segments']['Low Cost']['r_squared']:.2f} en Low Cost) que el modelo único con todos los "
    f"clientes mezclados (R²={model_summary['r_squared']:.2f}). Mezclar los tres perfiles diluye una señal "
    "que, por separado, es mucho más clara."
)
ui.section_close()

# ============================================================ PLAYGROUND ==
ui.section_open("playground")
ui.eyebrow("Pruébalo tú mismo")
ui.h2("Playground — diseña un vuelo y compara cómo lo valora cada segmento")
ui.lead(
    "Elige las características de un vuelo hipotético. El modelo calcula, en vivo, qué rating le daría "
    "cada segmento — para ver si un vuelo pensado para uno también convence a los demás."
)

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

predictions = predict_all_segments(
    choice, ATRIBUTOS, playground_model["intercept"], playground_model["partworths"],
)
plot_order = ["Overall"] + SEGMENT_ORDER

with pg_right:
    st.plotly_chart(
        charts.playground_segment_comparison(predictions, SEGMENT_LABELS, SEGMENT_COLORS, plot_order),
        use_container_width=True, config={"displayModeBar": False},
    )
    badge_cols = st.columns(3)
    for c, seg in zip(badge_cols, SEGMENT_ORDER):
        with c:
            ui.stat_card(seg, f"{predictions[seg]:.1f}", color=SEGMENT_COLORS[seg], value_size="1.5rem")

best_seg = max(SEGMENT_ORDER, key=lambda s: predictions[s])
worst_seg = min(SEGMENT_ORDER, key=lambda s: predictions[s])
gap = predictions[best_seg] - predictions[worst_seg]
ui.h3("¿Convence este vuelo a todo el mundo por igual?")
if gap < 0.8:
    ui.finding(
        f"Los tres segmentos valoran este vuelo de forma muy parecida (diferencia de solo {gap:.1f} puntos "
        f"entre {best_seg} y {worst_seg}): es un diseño de vuelo genérico, sin un ganador ni un perdedor claro."
    )
else:
    ui.finding(
        f"<b>{best_seg}</b> ({predictions[best_seg]:.1f}) valora este vuelo {gap:.1f} puntos por encima de "
        f"<b>{worst_seg}</b> ({predictions[worst_seg]:.1f}): la misma combinación de atributos no genera "
        "el mismo entusiasmo en todos los perfiles de cliente."
    )
ui.section_close()

# ============================================================ RESULTADOS ==
ui.section_open("resultados")
ui.eyebrow("¿Qué aprendí?")
ui.h2("Resultados")
ui.lead(
    f"El precio y las escalas concentran juntos casi el {pct(top_attr['Importancia']+second_attr['Importancia'], 0)} "
    "de lo que decide un cliente promedio — pero ese promedio esconde tres lógicas de decisión distintas."
)
res_cols = st.columns(3)
priority_by_segment = {
    "Business": "Escalas primero, precio segundo",
    "Leisure": "Precio primero, escalas segundo",
    "Low Cost": "El precio lo domina casi todo",
}
for c, seg in zip(res_cols, SEGMENT_ORDER):
    with c:
        top2 = (imp_segment[imp_segment["Segmento"] == seg]
                .sort_values("Importancia", ascending=False).head(2))
        subtitle = " · ".join(f"{ATTRIBUTE_LABELS[a]} {pct(v)}"
                               for a, v in zip(top2["Atributo"], top2["Importancia"]))
        ui.stat_card(seg, priority_by_segment[seg], subtitle,
                     title_color=SEGMENT_COLORS[seg], value_size="1.3rem")
ui.section_close()

# ============================================================ IMPACTO ==
ui.section_open("impacto", tight=True)
ui.impact_banner(
    f'Para <span class="accent-pos">Business</span>, volar directo pesa '
    f'<span class="accent-pos">más que el precio</span>. Para <span class="accent-neg">Low Cost</span>, '
    f'el precio es <span class="accent-neg">casi lo único que importa</span>.',
    quote='"No hay un vuelo perfecto para todo el mundo — hay un vuelo perfecto para cada segmento, y este modelo dice cuál es."',
)
ui.section_close()

# ============================================================ DECISIONES ==
ui.section_open("decisiones")
ui.eyebrow("¿Qué haría con esto?")
ui.h2("Decisiones que habilita")
ui.decision_flow(
    f"Business paga por llegar directo (Escalas {pct(biz_flight)}) casi tanto como por el precio (Precio {pct(biz_price)})",
    "Diseñar y promocionar un producto \"directo garantizado\" a precio premium para este segmento",
    "Capturar disposición a pagar en el atributo que este segmento más valora",
    "Adopción del producto directo en clientes Business",
)
st.write("")
ui.decision_flow(
    f"Low Cost decide casi solo por precio ({pct(lc_price)} de la importancia, el triple que escalas)",
    "No invertir presupuesto de marketing en venderles selección de asiento o flexibilidad a este segmento",
    "Dejar de gastar esfuerzo comercial en atributos que este segmento no valora",
    "Coste de adquisición por segmento",
)
st.write("")
ui.decision_flow(
    "El equipaje pesa de forma consistente en los tres segmentos (13-18% de importancia)",
    "Mantenerlo como atributo universal del producto base, no como extra de pago diferenciado",
    "Evitar fricción en el atributo que ningún segmento perdona que falte",
    "Reclamaciones o cancelaciones ligadas al equipaje",
)
ui.section_close()

# ============================================================ LIMITACIONES ==
ui.section_open("limitaciones")
ui.eyebrow("Honestidad ante todo")
ui.h2("Limitaciones")
lc1, lc2 = st.columns(2, gap="large")
with lc1:
    st.markdown('<p class="limit-col-title">Lo que el modelo SÍ puede hacer</p>', unsafe_allow_html=True)
    st.markdown(
        """<ul class="limit-list">
        <li>Descomponer una valoración global en el valor de cada atributo individual, con significancia estadística.</li>
        <li>Simular vuelos que ningún cliente valoró nunca, sumando utilidades ya estimadas.</li>
        <li>Detectar que la sensibilidad al precio no es lineal — hay un salto a partir de los 100€.</li>
        <li>Mostrar que la prioridad de atributos cambia radicalmente entre segmentos de cliente.</li>
        </ul>""",
        unsafe_allow_html=True,
    )
with lc2:
    st.markdown('<p class="limit-col-title">Lo que el modelo NO puede hacer</p>', unsafe_allow_html=True)
    st.markdown(
        """<ul class="limit-list">
        <li>Capturar interacciones entre atributos — p.ej. si el precio importa menos cuando el vuelo es directo, este modelo no lo ve.</li>
        <li>Garantizar que una preferencia declarada en una encuesta se traduzca en una compra real (declared vs. revealed preference).</li>
        <li>Generalizar a atributos o niveles que nunca se incluyeron en el diseño (p.ej. un precio de 200€).</li>
        <li>Sustituir un test de mercado real antes de un lanzamiento de producto con impacto económico grande.</li>
        </ul>""",
        unsafe_allow_html=True,
    )
st.markdown(
    '<div class="limit-note"><p class="co-body">'
    "Este es un conjoint <i>rating-based</i> clásico: mide cuánto gusta cada vuelo, no si el cliente lo "
    "compraría al precio marcado frente a alternativas reales del mercado. Para decisiones de pricing con "
    "impacto económico alto, conviene contrastar estos resultados con un Choice-Based Conjoint o un test "
    "de mercado antes de fijar precios definitivos."
    "</p></div>",
    unsafe_allow_html=True,
)
ui.section_close()

# ============================================================ CONCLUSIÓN ==
ui.section_open("conclusion")
ui.eyebrow("Del dato a la decisión")
ui.h2("Conclusión")
ui.lead(
    f"Con {rows_fmt} valoraciones y un modelo simple (una regresión lineal), descompuse exactamente "
    "cuánto vale cada característica de un vuelo, y para quién. El precio y las escalas dominan la "
    "decisión media, pero esa media esconde tres clientes distintos: uno que paga por llegar directo, uno "
    "que busca equilibrio, y uno que solo mira el precio. Diseñar un único vuelo \"para todos\" es, en "
    "realidad, diseñar el vuelo que no convence del todo a nadie."
)
ui.section_close()

ui.footer_minimal(
    name="Borja Mora Méndez",
    repo_url="https://github.com/BORJAMOME/conjoint-vuelos-app",
    linkedin_url="https://www.linkedin.com/in/borja-mora-mendez/",
    email="borja.mora.mendez@gmail.com",
)
