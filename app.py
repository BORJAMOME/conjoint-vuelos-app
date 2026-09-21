"""
Análisis Conjoint — Preferencias de Vuelos
Reportaje digital de datos: qué valoran realmente los clientes de una aerolínea en cada característica
de un vuelo, y cómo cambia esa prioridad según quién vuela.

Narrativa (12 beats):
  LEDE → 01 El problema → 02 Los datos → 03 ¿Qué dicen las valoraciones? → 04 De una valoración global al
  valor de cada característica → 05 ¿Cuánto explica el modelo? → 06 ¿Qué mueve la valoración? →
  07 Tres formas de elegir → 08 Diseña un vuelo → 09 Qué aprendí → 10 ¿Qué podría hacer una empresa? →
  11 Lo que sabemos y lo que no → 12 Del dato a la decisión

Regla de copy: primero se explica qué significa, después se pone el nombre técnico (en notas
«Detalle técnico» o entre paréntesis).

La composición vive en assets/editorial.css + components/editorial.py (sistema editorial reutilizable);
aquí solo hay contenido y datos. Toda cifra del texto sale de los artefactos de model/artifacts (train.py y
export_design_checks.py) o se calcula de ellos: nada está escrito a mano.

Autor: Borja Mora Méndez
"""
import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

from components import charts
from components import editorial as ed
from utils.conjoint import predict_all_segments
from utils.data_loader import (ATTRIBUTE_LABELS, LEVEL_LABELS, SEGMENT_COLORS, SEGMENT_LABELS,
                                artifacts_ready, load_csv, load_json)

# Streamlit recarga app.py al detectar cambios, pero mantiene en memoria los módulos locales ya
# importados. Tras un despliegue que modifica components/*.py y app.py a la vez, eso deja un
# app.py nuevo llamando a un módulo antiguo (AttributeError). Recargarlos en cada ejecución lo
# evita; el coste es despreciable.
importlib.reload(charts)
importlib.reload(ed)

ROOT = Path(__file__).resolve().parent
LECTURA = "12 min"
ACTUALIZADO = "Septiembre 2026"
FUENTE = "Elaboración propia"
PLOT = {"displayModeBar": False}

st.set_page_config(
    page_title="Análisis Conjoint · Preferencias de Vuelos",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Identidad del proyecto primero (paleta, tipografías); sistema editorial después
ed.load_css(ROOT, "style.css", "editorial.css")

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
lc_step1 = abs(pw_s[("Low Cost", "Price", "100")])
biz_step1 = abs(pw_s[("Business", "Price", "100")])

baggage_range = [imp_s[(s, "Baggage")] for s in SEGMENT_ORDER]
baggage_third_everywhere = all(
    imp_segment[imp_segment["Segmento"] == s].sort_values("Importancia", ascending=False)["Atributo"].tolist()[2] == "Baggage"
    for s in SEGMENT_ORDER
)
ci_half = max((v[a]["hi"] - v[a]["lo"]) / 2 for s, v in checks["importancia_ci"].items() for a in ATRIBUTOS)
seg_r2 = {s: stats["segments"][s]["r_squared"] for s in SEGMENT_ORDER}
fit = checks["ajuste_tarjeta"]

# ============================================================ LEDE ==
ed.skip_link("contexto")
ed.topbar(
    "Análisis Conjoint · Preferencias de Vuelos",
    [("contexto", "Problema"), ("datos", "Datos"), ("metodo", "Método"), ("modelo", "Modelo"),
     ("segmentos", "Segmentos"), ("playground", "Playground"), ("resultado", "Resultado"),
     ("decisiones", "Decisiones"), ("conclusion", "Conclusión")],
    back_url="https://borjamora.es/",
)
ed.anchor_scroll()
ed.lede(
    kicker="Machine Learning Case Study · Regresión (análisis conjoint)",
    headline="¿Qué hace que un vuelo merezca la pena <em>para cada cliente</em>?",
    deck=(
        "Una aerolínea puede cambiar el precio, añadir equipaje, quitar una escala o dar más flexibilidad. Pero no "
        "todos los clientes valoran esos extras de la misma manera. "
        f"Analicé {rows_fmt} valoraciones de {n_fmt} clientes para descubrir <b>cuánto pesa realmente cada "
        "característica de un vuelo</b> y cómo cambia esa valoración según el tipo de cliente."
    ),
    meta=[
        ("Autor", "Borja Mora Méndez"),
        ("Stack", "Python · statsmodels (OLS) · Streamlit"),
        ("Datos", f"{n_fmt} clientes · {rows_fmt} valoraciones"),
        ("Actualizado", ACTUALIZADO),
        ("Lectura", LECTURA),
    ],
)

# ============================================================ 01 · EL PROBLEMA ==
ed.beat(
    "contexto", "01", "Problema", "El problema",
    deck=[
        "Una aerolínea puede crear muchas combinaciones distintas cambiando precio, escalas, equipaje, asiento, "
        "flexibilidad u horario.",
        "El problema no es crear esas combinaciones. Es saber <b>cuáles importan realmente al cliente</b>.",
        "¿Pagaría más por un vuelo directo? ¿Cuánto le importa llevar equipaje incluido? ¿Prefiere ahorrar aunque "
        "tenga que hacer una escala? Y, sobre todo: <b>¿las respuestas son las mismas para todos?</b>",
    ],
)
ed.band(
    "La pregunta de negocio",
    '¿Cuánto aporta cada característica de un vuelo a su valoración '
    '<span class="accent">y cómo cambia ese valor según el cliente</span>?',
    "En lugar de preguntar directamente qué atributo considera más importante, el experimento pidió a los clientes "
    "que valoraran vuelos completos, y utilicé esas respuestas para descubrir qué había detrás de cada valoración.",
)

# ============================================================ 02 · LOS DATOS ==
ed.beat(
    "datos", "02", "Datos", "Los datos",
    deck=[
        f"Para averiguar qué valora cada cliente, los <b>{n_fmt} participantes</b> evaluaron las mismas {n_cards} "
        f"combinaciones de vuelo del {scale_min:.0f} al {scale_max:.0f}.",
        f"Eso generó {rows_fmt} valoraciones que permiten comparar cómo cambia la percepción de un vuelo cuando "
        "cambian sus características.",
    ],
)
ed.provenance([
    ("Dataset", "Conjoint_Flight.xlsx"),
    ("Clientes", f"{n_fmt} · Business {miles(seg_n['Business'])}, Leisure {miles(seg_n['Leisure'])}, "
                 f"Low Cost {miles(seg_n['Low Cost'])}"),
    ("Diseño", f"{n_cards} tarjetas, las mismas para todos los clientes"),
    ("Valoraciones", f"{rows_fmt} · escala de {scale_min:.0f} a {scale_max:.0f}"),
    ("Atributos", f"{stats['n_atributos']}, con 2 o 3 niveles cada uno"),
    ("Segmentos", "ya venían definidos en los datos"),
    ("Elaboración", FUENTE),
])

ed.subhead("El experimento", level="wide")
with ed.figure("atributos"):
    ed.cols([
        {"tag": "Atributo 1", "title": "Precio", "text": "50 € · 100 € · 150 €"},
        {"tag": "Atributo 2", "title": "Equipaje", "text": "20 kg incluidos · Sin equipaje"},
        {"tag": "Atributo 3", "title": "Selección de asiento", "text": "Con selección · Aleatoria"},
        {"tag": "Atributo 4", "title": "Escalas", "text": "Directo · 1 escala"},
        {"tag": "Atributo 5", "title": "Flexibilidad", "text": "Con flexibilidad · Sin flexibilidad"},
        {"tag": "Atributo 6", "title": "Horario de salida", "text": "Mañana · Tarde · Noche"},
    ], count=3)
ed.passage(
    f"El problema es que combinar todos los niveles habría generado <b>{n_combos} vuelos diferentes</b>. Demasiados "
    "para pedirle a una persona que los valore.",
    f"Por eso el experimento se limita a <b>{n_cards} combinaciones</b>, un diseño fraccionado: suficientes para "
    "variar los atributos y poder separar después el efecto de cada uno.",
    f"Todos los clientes valoraron exactamente las mismas {n_cards} combinaciones: lo he comprobado uno por uno, con "
    f"los {n_fmt} clientes. Esa condición permite comparar sus respuestas bajo las mismas reglas.",
    aside=(
        f"El diseño no es un ortogonal estricto: la correlación máxima entre atributos es "
        f"{es(checks['correlacion_max_entre_atributos'], 2)} (un ortogonal exacto tendría 0) y el factor de "
        f"inflación de la varianza (VIF) máximo es {es(checks['vif_max'], 2)} (el ideal es 1). Con esos valores los "
        f"efectos siguen siendo separables. De las {n_combos} combinaciones posibles, {n_unrated} no se valoraron "
        "nunca."
    ),
)

# ============================================================ 03 · ¿QUÉ DICEN LAS VALORACIONES? ==
ed.beat(
    "exploracion", "03", "Antes de modelar", "¿Qué me dicen las valoraciones?",
    deck="Antes de construir el modelo, quería saber si las respuestas tenían suficiente variación como para "
         "encontrar patrones.",
)
ed.subhead("¿Cómo se reparten las valoraciones?", level="wide")
with ed.split("distribucion", "8-4") as (viz, txt):
    with viz:
        st.plotly_chart(charts.rating_distribution(ratings_df["Rating"]), use_container_width=True, config=PLOT)
        ed.caption("FIG. 01", "Número de valoraciones según la nota (1 a 10).", FUENTE)
    with txt:
        ed.insight(
            f"La valoración media fue de <b>{es(ratings_df['Rating'].mean(), 2)} sobre 10</b>, pero detrás de esa "
            "media hay vuelos que gustan mucho y otros que generan una respuesta bastante peor. Esa diferencia es "
            "precisamente la que interesa explicar.",
        )
        ed.note(f"<b>Detalle técnico.</b> El {es(rating_info['pct_en_el_maximo'])}% de las valoraciones es un "
                f"{scale_max:.0f}, el máximo de la escala. Ese techo importa más adelante: un modelo lineal no lo "
                "conoce.")

ed.subhead("¿Valoran igual los tres segmentos?", level="wide")
with ed.split("segmentos-nota", "5-7") as (txt, viz):
    with txt:
        ed.insight(
            f"No. La valoración media es <b>{es(seg_rating['Business']['media'])}</b> en Business, "
            f"<b>{es(seg_rating['Leisure']['media'])}</b> en Leisure y <b>{es(seg_rating['Low Cost']['media'])}</b> "
            f"en Low Cost (medianas: {es(seg_rating['Business']['mediana'])}, {es(seg_rating['Leisure']['mediana'])} "
            f"y {es(seg_rating['Low Cost']['mediana'])}).",
        )
    with viz:
        st.plotly_chart(charts.rating_by_segment(ratings_df, SEGMENT_COLORS, SEGMENT_ORDER),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 02", "Distribución de las valoraciones de cada segmento. El rombo marca la media.", FUENTE)
ed.passage(
    "Pero saber que puntúan distinto no dice <b>por qué</b>. Un cliente puede dar un 8 porque el vuelo es directo. "
    "Otro puede darle un 8 porque es barato.",
    tight=True,
)
ed.insight("La valoración final puede ser la misma. <b>Lo que cambia es lo que hay detrás.</b> Y eso es lo que el "
           "modelo tiene que descubrir.")

# ============================================================ 04 · CÓMO FUNCIONA ==
ed.beat(
    "metodo", "04", "Método", "De una valoración global al valor de cada característica",
    deck="El cliente no me dice directamente cuánto valen el precio, las escalas o el equipaje. Me dice cuánto le "
         "gusta un vuelo completo. El análisis conjoint utiliza esas valoraciones para <b>separar después cuánto "
         "aporta cada característica</b>.",
)
ed.route(["Valoración del vuelo completo", f"{n_cards} tarjetas iguales para todos", "Una variable por nivel",
          "Regresión OLS", "Utilidades parciales", "Simulación de vuelos nuevos"])
ed.steps([
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

# ============================================================ 05 · EL MODELO ==
ed.beat(
    "modelo", "05", "Resultado", "¿Cuánto consigue explicar el modelo?",
    deck=[
        f"Con los {stats['n_atributos']} atributos incluidos, el modelo explica el "
        f"<b>{pct(model_summary['r_squared'] * 100)}</b> de la variación en las valoraciones.",
        f"Es decir, una parte importante de las diferencias entre las {rows_fmt} respuestas puede explicarse por las "
        "características de los vuelos incluidas en el análisis.",
    ],
)
ed.subhead("En resumen")
ed.metrics([
    ("Variación explicada (R²)", es(model_summary["r_squared"], 3),
     "La parte que queda sin explicar son, sobre todo, diferencias entre clientes: más adelante veremos que el "
     "modelo mejora mucho cuando se ajusta por segmento."),
    ("Observaciones", miles(model_summary["n_obs"]), f"{n_fmt} clientes por {n_cards} tarjetas."),
    ("Coeficientes significativos", "8 de 8",
     "Todos con p &lt; 0,001, también con errores agrupados por cliente (el |t| más bajo es "
     f"{es(checks['robustez']['t_min_agrupado_por_cliente'], 0)})."),
])
ed.insight(
    "Pero saber que el modelo funciona no responde todavía a la pregunta más interesante: <b>¿qué características "
    "hacen que un vuelo guste más o menos?</b> Y entonces entramos en las utilidades.",
    aside="El R² se mide sobre los mismos datos con los que se ajustó el modelo: no hay conjunto de prueba. Como cada "
          f"cliente aporta {n_cards} valoraciones, la significación se comprobó con errores agrupados por cliente.",
)

ed.subhead("Las utilidades parciales de cada atributo", level="wide")
with ed.figure("utilidades"):
    row1, row2 = st.columns(3), st.columns(3)
    for i, atributo in enumerate(ATRIBUTOS):
        with (row1 if i < 3 else row2)[i % 3]:
            st.plotly_chart(
                charts.partworth_bars(pw_overall[pw_overall["Atributo"] == atributo], LEVEL_LABELS,
                                      ATTRIBUTE_LABELS[atributo]),
                use_container_width=True, config=PLOT)
    ed.caption("FIG. 03", "Cuánto suma o resta cada nivel a la valoración (puntos sobre 10), frente a su nivel de "
                          "referencia (utilidad 0).", "Modelo entrenado en este proyecto")

ed.subhead("El precio no pesa igual en todos los tramos", level="wide")
with ed.split("precio", "5-7") as (txt, viz):
    with txt:
        ed.insight(
            f"Subir el precio de 50 € a 100 € resta <b>{es(step_1, 2)}</b> puntos de utilidad. Pasar de 100 € a 150 € "
            f"resta otros <b>{es(step_2, 2)}</b>: el segundo salto cuesta más del doble. La sensibilidad al precio no "
            "es simplemente «cada euro importa lo mismo»: el modelo detecta un cambio claro a partir de los 100 €."
        )
        ed.note(f"Pero ese es el patrón del cliente promedio. Por segmentos cambia: Low Cost ya pierde "
                f"{es(lc_step1, 2)} puntos al pasar de 50 € a 100 €, mientras que Business pierde solo "
                f"{es(biz_step1, 2)}.")
    with viz:
        st.plotly_chart(charts.price_curve(pw_overall[pw_overall["Atributo"] == "Price"]),
                        use_container_width=True, config=PLOT)
        ed.caption("FIG. 04", "Utilidad parcial de cada nivel de precio frente a 50 €.", "Modelo entrenado en este proyecto")

# ============================================================ 06 · ¿QUÉ MUEVE LA VALORACIÓN? ==
ed.beat(
    "explicabilidad", "06", "Explicabilidad", "¿Qué características mueven realmente la valoración?",
    deck=[
        "Las utilidades nos dicen cuánto aporta cada nivel. Pero hay otra pregunta: <b>¿qué atributos tienen más "
        "peso en la decisión?</b>",
        "Para responderla, calculé la <b>importancia relativa</b> de cada atributo comparando la diferencia entre su "
        "nivel mejor y peor valorado.",
    ],
)
ed.subhead("El cliente promedio", level="wide")
with ed.split("importancia", "7-5") as (viz, txt):
    with viz:
        st.plotly_chart(charts.importance_overall(imp_overall, ATTRIBUTE_LABELS), use_container_width=True, config=PLOT)
        ed.caption("FIG. 05", "Importancia relativa de cada atributo con todos los clientes juntos.",
                   "Modelo entrenado en este proyecto")
    with txt:
        ed.insight(
            f"<b>{ATTRIBUTE_LABELS[top_attr]}</b> ({pct(imp_o[top_attr])}) y "
            f"<b>{ATTRIBUTE_LABELS[second_attr].lower()}</b> ({pct(imp_o[second_attr])}) concentran el "
            f"<b>{pct(top_two, 0)}</b> de la importancia total. La selección de asiento y la flexibilidad tienen "
            "mucho menos peso."
        )
        ed.note("<b>Detalle técnico.</b> La importancia es relativa a los niveles que se probaron: el precio solo se "
                "probó entre 50 € y 150 €, y con otro rango su peso sería distinto.")
ed.insight("Pero aquí aparece una limitación importante de mirar solo el promedio: <b>el cliente promedio no "
           "existe</b>. Cuando separo los resultados por segmento, la historia cambia.")

# ============================================================ 07 · SEGMENTOS ==
ed.beat(
    "segmentos", "07", "Tres formas de elegir", "El mismo vuelo no significa lo mismo para todos", weight="major",
    deck="Cuando separo las valoraciones por segmento, las prioridades cambian.",
)
with ed.figure("importancia-segmentos"):
    st.plotly_chart(charts.importance_by_segment(imp_segment, ATTRIBUTE_LABELS, SEGMENT_COLORS, SEGMENT_ORDER),
                    use_container_width=True, config=PLOT)
    ed.caption("FIG. 06", "Importancia relativa de cada atributo en cada segmento.", "Modelos entrenados en este proyecto")

direct_biz, price150_biz = pw_s[("Business", "Flight", "Direct")], abs(pw_s[("Business", "Price", "150")])
seg_text = {
    "Business": (f"Para este segmento, volar directo aporta más valoración ({sgn(direct_biz)} puntos) que lo que "
                 f"resta encarecer el billete de 50 € a 150 € ({sgn(-price150_biz)})."
                 if direct_biz > price150_biz else "Las escalas pesan más que el precio."),
    "Leisure": "Busca un equilibrio diferente entre coste y comodidad.",
    "Low Cost": "Aquí el precio domina claramente la valoración.",
}
seg_cards = []
for seg in SEGMENT_ORDER:
    (a1, v1), (a2, v2) = (imp_segment[imp_segment["Segmento"] == seg]
                          .sort_values("Importancia", ascending=False).head(2)[["Atributo", "Importancia"]].values)
    seg_cards.append({
        "tag": f"{miles(seg_n[seg])} clientes", "title": seg, "color": SEGMENT_COLORS[seg],
        "stat": pct(v1), "stat_label": f"{ATTRIBUTE_LABELS[a1]} · después {ATTRIBUTE_LABELS[a2].lower()} {pct(v2)}",
        "text": seg_text[seg],
    })
with ed.figure("segmentos-cards"):
    ed.cards(seg_cards, count=3)
ed.insight(
    "No estamos ante tres clientes que simplemente puntúan los mismos vuelos de forma diferente. Son <b>tres formas "
    "distintas de valorar</b> las características de un vuelo.",
    aside=(
        f"Intervalos de confianza del 95% por bootstrap de clientes ({miles(checks['bootstrap']['repeticiones'])} "
        f"repeticiones): la importancia de cada atributo varía menos de ±{es(ci_half, 1)} puntos, así que las "
        "diferencias entre segmentos no son ruido de muestreo. Esos intervalos solo recogen el azar de los clientes "
        "de la muestra, no las limitaciones del diseño."
    ),
)
ed.passage(
    f"Leisure es el segmento más numeroso ({miles(seg_n['Leisure'])} de {n_fmt} clientes), y por eso se parece tanto "
    "al cliente promedio: Business y Low Cost son los que más se alejan de él.",
    f"El modelo ajustado solo con los datos de cada segmento explica más variación (R² = {es(seg_r2['Business'], 2)} "
    f"en Business, {es(seg_r2['Leisure'], 2)} en Leisure y {es(seg_r2['Low Cost'], 2)} en Low Cost) que el modelo "
    f"único con todos los clientes mezclados ({es(model_summary['r_squared'], 2)}). Mezclar los tres perfiles diluye "
    "una señal que, por separado, es mucho más clara.",
)

# ============================================================ 08 · PLAYGROUND ==
ed.beat(
    "playground", "08", "Playground", "¿Y si pudieras diseñar un vuelo y ver quién lo valoraría mejor?", weight="major",
    deck=[
        "El modelo ya ha estimado cuánto aporta cada característica. Ahora podemos utilizar esas estimaciones para "
        f"construir un vuelo que quizá ningún cliente haya valorado durante el experimento (de las {n_combos} "
        f"combinaciones posibles, solo {n_cards} se valoraron).",
        "Elige precio, equipaje, asiento, escalas, flexibilidad y horario. El modelo calcula cómo lo valoraría cada "
        "segmento: la misma combinación de atributos, con valoraciones que pueden ser muy distintas.",
    ],
)

level_options = {a: ORDEN[a] for a in ATRIBUTOS}
default_choice = {"Price": "100", "Baggage": "20 kg", "Seat": "Random", "Flight": "1 Stop",
                   "Flexible": "No", "Departure": "Morning"}
choice = {}
with ed.split("playground", "4-8") as (pg_left, pg_right):
    with pg_left:
        for atributo in ATRIBUTOS:
            opts = level_options[atributo]
            nice_opts = [LEVEL_LABELS.get(o, o) for o in opts]
            default_idx = opts.index(default_choice[atributo]) if default_choice[atributo] in opts else 0
            picked = st.selectbox(ATTRIBUTE_LABELS[atributo], nice_opts, index=default_idx, key=f"pg_{atributo}")
            choice[atributo] = opts[nice_opts.index(picked)]

    raw_predictions = predict_all_segments(choice, ATRIBUTOS, playground_model["intercept"],
                                           playground_model["partworths"])
    # El modelo es lineal y la escala va de 1 a 10 (el 22% de las valoraciones reales es un 10): puede calcular
    # valores fuera de la escala. Se muestran acotados a 1-10 y se avisa.
    predictions = {k: min(max(v, scale_min), scale_max) for k, v in raw_predictions.items()}
    out_of_scale = {k: v for k, v in raw_predictions.items() if v > scale_max or v < scale_min}
    plot_order = ["Overall"] + SEGMENT_ORDER

    with pg_right:
        st.plotly_chart(
            charts.playground_segment_comparison(predictions, SEGMENT_LABELS, SEGMENT_COLORS, plot_order),
            use_container_width=True, config=PLOT,
        )
        ed.caption("FIG. 07", "Valoración estimada del vuelo que has diseñado, para todos los clientes y para cada "
                              "segmento.", "Modelo entrenado en este proyecto")

match = design_cards[design_cards[ATRIBUTOS].eq(pd.Series(choice)).all(axis=1)]
real = None
if len(match):
    card_id = int(match.iloc[0]["CardID"])
    real = card_ratings[card_ratings["CardID"] == card_id].iloc[0]

play_cards = []
for seg in SEGMENT_ORDER:
    if real is not None:
        text = f"Valoración real media en el experimento: {es(real[seg])}."
    else:
        text = "Vuelo que nadie valoró: estimación que suma utilidades."
    if seg in out_of_scale:
        text += f" El modelo calcula {es(out_of_scale[seg])}, fuera de la escala; se muestra acotado."
    play_cards.append({"tag": "Valoración estimada", "title": seg, "color": SEGMENT_COLORS[seg],
                       "stat": es(predictions[seg]), "stat_label": "sobre 10", "text": text})
with ed.figure("playground-cards"):
    ed.cards(play_cards, count=3)

if real is not None:
    ed.note(f"<b>Comprobación.</b> Este vuelo sí se valoró en el experimento (tarjeta nº {card_id}), así que puedes "
            "comparar la estimación con la valoración real.")
else:
    ed.note(f"<b>Comprobación.</b> Este vuelo no estaba entre las {n_cards} que se valoraron: es una de las "
            f"{n_unrated} combinaciones que nadie puntuó.")
if out_of_scale:
    ed.note(f"<b>Techo de la escala.</b> El modelo es lineal y no conoce el máximo de la escala: el "
            f"{es(rating_info['pct_en_el_maximo'], 0)}% de las valoraciones reales es un {scale_max:.0f}.")

best_seg = max(SEGMENT_ORDER, key=lambda s: predictions[s])
worst_seg = min(SEGMENT_ORDER, key=lambda s: predictions[s])
gap = predictions[best_seg] - predictions[worst_seg]
ed.subhead("¿Convence este vuelo a todo el mundo por igual?")
if gap < 0.8:
    ed.insight(
        f"Los tres segmentos valoran este vuelo de forma muy parecida (diferencia de solo {es(gap)} puntos entre "
        f"{best_seg} y {worst_seg}): es un diseño de vuelo genérico, sin un ganador ni un perdedor claro."
    )
else:
    ed.insight(
        f"<b>{best_seg}</b> ({es(predictions[best_seg])}) valora este vuelo {es(gap)} puntos por encima de "
        f"<b>{worst_seg}</b> ({es(predictions[worst_seg])}): la misma combinación de atributos no genera el mismo "
        "entusiasmo en todos los perfiles de cliente."
    )

# ============================================================ 09 · QUÉ APRENDÍ ==
ed.beat(
    "resultado", "09", "Resultado", "Tres clientes. Tres formas de valorar un vuelo.", weight="major",
    deck=[
        f"El análisis encuentra una señal clara: <b>{ATTRIBUTE_LABELS[top_attr]} y "
        f"{ATTRIBUTE_LABELS[second_attr].lower()}</b> concentran el {pct(top_two, 0)} de la importancia para el "
        "cliente promedio.",
        "Pero cuando separamos los segmentos, el promedio deja de contar toda la historia.",
    ],
)
ed.band(
    "Lo que quedaría de todo esto",
    'La conclusión no es que exista un atributo ganador. Es que <span class="accent">el valor de un atributo depende '
    "de quién está tomando la decisión</span>.",
    "«No quería saber qué vuelo gusta más. Quería saber por qué gusta y si ese porqué cambia según el cliente.»",
    quote=True,
)
ed.subhead("En resumen")
ed.metrics([
    ("Cliente promedio", pct(top_two, 0),
     f"{ATTRIBUTE_LABELS[top_attr]} y {ATTRIBUTE_LABELS[second_attr].lower()} concentran casi toda la decisión."),
    ("Business", pct(biz_flight),
     f"Prioriza las escalas, por encima del precio ({pct(biz_price)}).", SEGMENT_COLORS["Business"]),
    ("Leisure", pct(imp_s[("Leisure", "Price")]),
     f"Da más peso al precio, pero mantiene las escalas como segundo factor ({pct(imp_s[('Leisure', 'Flight')])}).",
     SEGMENT_COLORS["Leisure"]),
    ("Low Cost", pct(lc_price),
     f"Concentra casi el {pct(lc_price, 0)} de su decisión en el precio.", SEGMENT_COLORS["Low Cost"]),
])

# ============================================================ 10 · DECISIONES ==
ed.beat(
    "decisiones", "10", "Implicaciones", "¿Qué podría hacer una empresa con estos resultados?",
    deck="El modelo no decide qué producto lanzar. Pero sí permite plantear <b>hipótesis mucho más concretas</b>.",
)
with ed.figure("hipotesis"):
    ed.cards([
        {"tag": "Hipótesis · Business", "title": "Vuelo directo para Business", "color": SEGMENT_COLORS["Business"],
         "points": [
             f"<b>Insight.</b> Las escalas representan el {pct(biz_flight)} de la importancia, por encima del precio "
             f"({pct(biz_price)}).",
             "<b>Hipótesis.</b> Probar una propuesta de vuelo directo orientada a este segmento y medir su respuesta.",
             "<b>Qué medir.</b> Adopción, conversión y disposición a pagar.",
         ]},
        {"tag": "Hipótesis · Low Cost", "title": "Precio primero para Low Cost", "color": SEGMENT_COLORS["Low Cost"],
         "points": [
             f"<b>Insight.</b> El precio concentra el {pct(lc_price)} de la importancia.",
             "<b>Hipótesis.</b> Probar una propuesta centrada en precio antes de añadir extras como argumento "
             "principal.",
             "<b>Qué medir.</b> Conversión y coste de adquisición.",
         ]},
        {"tag": "Hipótesis · Equipaje", "title": "Equipaje en la propuesta base",
         "points": [
             f"<b>Insight.</b> Su importancia se mantiene relativamente estable entre los tres segmentos "
             f"({pct(min(baggage_range), 0)} a {pct(max(baggage_range), 0)})"
             + (" y es el tercer atributo en todos." if baggage_third_everywhere else "."),
             "<b>Hipótesis.</b> Evaluar si funciona mejor integrado en la propuesta base que como extra "
             "diferenciado.",
             "<b>Qué medir.</b> Conversión, aceptación y reclamaciones relacionadas con el equipaje.",
         ]},
    ], count=3)
ed.subhead("Una advertencia importante")
ed.insight("Estas no son decisiones que el modelo haya tomado. Son <b>hipótesis que podrían pasar después a un "
           "experimento real</b>.", tight=True)

# ============================================================ 11 · LIMITACIONES ==
ed.beat(
    "limitaciones", "11", "Limitaciones", "Lo que sabemos y lo que no sabemos",
    deck="Un buen análisis también tiene que dejar claro dónde termina lo que sabemos.",
)
ed.cards([
    {"title": "Lo que el modelo sí puede hacer", "color": "var(--positive)", "points": [
        "<b>Descomponer</b> una valoración global en el valor de cada atributo, con significancia estadística "
        "(también con errores agrupados por cliente).",
        f"<b>Simular vuelos que ningún cliente valoró</b>, sumando utilidades ya estimadas ({n_unrated} de las "
        f"{n_combos} combinaciones no se valoraron).",
        "Mostrar que la sensibilidad al precio del cliente promedio <b>no es lineal</b>: el segundo tramo pesa más "
        "del doble.",
        "Mostrar que la prioridad de atributos <b>cambia claramente entre segmentos</b>, con diferencias mayores "
        "que el azar de muestreo.",
    ]},
    {"title": "Lo que el modelo no puede hacer", "color": "var(--negative)", "points": [
        "Capturar <b>interacciones</b> entre atributos: si el precio importa menos cuando el vuelo es directo, este "
        "modelo no lo ve.",
        "Garantizar que una <b>preferencia declarada</b> en una encuesta se traduzca en una compra real (declared vs. "
        "revealed preference).",
        "Generalizar a atributos o niveles que nunca se incluyeron en el diseño (p. ej. un precio de 200 €).",
        f"Respetar el <b>techo de la escala</b>: es un modelo lineal y puede calcular valoraciones por encima de "
        f"{scale_max:.0f} (el {es(rating_info['pct_en_el_maximo'], 0)}% de las reales es un {scale_max:.0f}).",
        f"Reproducir con exactitud cada vuelo: el error medio por tarjeta es de "
        f"{es(fit['Overall']['mae_tarjeta'], 1)} puntos con todos los clientes y de "
        f"{es(fit['Business']['mae_tarjeta'], 1)} en Business.",
        "Sustituir un <b>test de mercado real</b> antes de un lanzamiento con impacto económico grande.",
    ]},
])
ed.passage(
    "Este es un conjoint <i>rating-based</i> clásico: mide cuánto gusta cada vuelo, no si el cliente lo compraría al "
    "precio marcado frente a alternativas reales del mercado. Para decisiones de pricing con impacto económico alto, "
    "conviene contrastar estos resultados con un Choice-Based Conjoint o un test de mercado antes de fijar precios "
    "definitivos.",
    aside="El modelo no se ha evaluado con datos que no hubiera visto, y los segmentos vienen dados en el dataset: no "
          "se ha comprobado que sean la mejor forma de agrupar a los clientes.",
)

# ============================================================ 12 · CONCLUSIÓN ==
ed.beat(
    "conclusion", "12", "Conclusión", "No existe un único valor para todos los clientes", weight="major",
    deck=[
        f"A partir de {rows_fmt} valoraciones, el modelo permite descomponer una valoración global y estimar cuánto "
        "aporta cada característica de un vuelo.",
        "El resultado más claro es que <b>precio y escalas dominan la decisión</b>, pero no de la misma manera para "
        f"todos. Para Business, las escalas pesan más que el precio. Para Leisure, el precio ocupa el primer lugar. "
        f"Para Low Cost, el precio concentra casi el {pct(lc_price, 0)} de la importancia.",
        "El análisis no dice qué vuelo debe vender una aerolínea. Dice algo más útil: <b>qué está valorando cada tipo "
        "de cliente y dónde cambia esa valoración</b>. El siguiente paso ya no sería preguntar qué prefieren. "
        "<b>Sería probarlo en el mercado.</b>",
    ],
)
ed.colophon("Borja Mora Méndez", [
    ("Repositorio del proyecto", "https://github.com/BORJAMOME/conjoint-vuelos-app"),
    ("Portfolio", "https://borjamora.es/"),
    ("LinkedIn", "https://www.linkedin.com/in/borja-mora-mendez/"),
    ("Contacto", "mailto:borja.mora.mendez@gmail.com"),
])
