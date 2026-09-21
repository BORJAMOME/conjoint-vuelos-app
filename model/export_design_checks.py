"""Comprobaciones del diseño del experimento y de la robustez de los resultados, sin tocar el modelo.

Lee el mismo dataset que train.py y guarda dos artefactos nuevos que la narrativa necesita para no
afirmar más de lo que los datos permiten:

  design_cards.csv   las 24 tarjetas (combinaciones) que valoró cada cliente
  design_checks.json escala real del rating, techo, medianas por segmento, calidad del diseño (VIF),
                     robustez de los p-valores (errores agrupados por cliente) e intervalos de confianza
                     de la importancia relativa por segmento (bootstrap de clientes)

No reentrena nada: train.py y sus artefactos no cambian.

Ejecutar una sola vez:
    py -3.10 model/export_design_checks.py
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "model" / "artifacts"

ATRIBUTOS = ["Price", "Baggage", "Seat", "Flight", "Flexible", "Departure"]
ORDEN = {
    "Price": ["50", "100", "150"], "Baggage": ["No baggage", "20 kg"], "Seat": ["Random", "Seat Selection"],
    "Flight": ["1 Stop", "Direct"], "Flexible": ["No", "Yes"], "Departure": ["Morning", "Afternoon", "Evening"],
}
FORMULA = "Rating ~ C(Price) + C(Baggage) + C(Seat) + C(Flight) + C(Flexible) + C(Departure)"
SEGMENTOS = ["Business", "Leisure", "Low Cost"]
N_BOOT = 2000
SEED = 42


def cargar() -> pd.DataFrame:
    df = pd.read_excel(ROOT / "data" / "Conjoint_Flight.xlsx", sheet_name="Transacciones")
    df["Price"] = df["Price"].astype(str)
    for a in ATRIBUTOS:                       # mismas categorías de referencia que train.py
        df[a] = pd.Categorical(df[a], categories=ORDEN[a])
    return df


def importancia(pw_rangos: dict) -> dict:
    total = sum(pw_rangos.values())
    return {a: v / total * 100 for a, v in pw_rangos.items()}


def main():
    df = cargar()
    n_customers = int(df["CustomerID"].nunique())

    # -- ¿Todos los clientes valoraron exactamente las mismas 24 tarjetas? (train.py solo comparaba dos) --
    base = df[df["CustomerID"] == df["CustomerID"].iloc[0]][["CardID"] + ATRIBUTOS].reset_index(drop=True)
    mismo_diseno_todos = all(
        g[["CardID"] + ATRIBUTOS].reset_index(drop=True).equals(base) for _, g in df.groupby("CustomerID")
    )
    base.to_csv(ARTIFACTS / "design_cards.csv", index=False)
    card_means = df.pivot_table(index="CardID", columns="Segment", values="Rating", aggfunc="mean", observed=True)
    card_means.insert(0, "Overall", df.groupby("CardID")["Rating"].mean())
    card_means.reset_index().to_csv(ARTIFACTS / "card_ratings.csv", index=False)

    # -- Calidad del diseño: un diseño ortogonal estricto tendría VIF = 1 y correlación 0 entre atributos --
    X = pd.get_dummies(base[ATRIBUTOS].astype(str), drop_first=True).astype(float)
    atributo_de = {c: c.split("_")[0] for c in X.columns}
    cruzadas = [abs(X[a].corr(X[b])) for a, b in itertools.combinations(X.columns, 2) if atributo_de[a] != atributo_de[b]]
    Xc = X.copy()
    Xc.insert(0, "const", 1.0)
    vif = [variance_inflation_factor(Xc.values, i) for i, c in enumerate(Xc.columns) if c != "const"]
    combos = set(itertools.product(*[ORDEN[a] for a in ATRIBUTOS]))
    probadas = set(tuple(str(r[a]) for a in ATRIBUTOS) for _, r in base.iterrows())

    # -- Robustez de la significación: errores agrupados por cliente (24 valoraciones por cliente) --
    modelo = smf.ols(FORMULA, data=df).fit()
    modelo_cl = smf.ols(FORMULA, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["CustomerID"]})
    t_min = float(modelo_cl.tvalues.drop("Intercept").abs().min())
    p_max_cl = float(modelo_cl.pvalues.drop("Intercept").max())

    # -- Rating: escala real, techo y medianas por segmento --
    rating = df["Rating"]
    por_segmento = {
        seg: {"media": float(g.mean()), "mediana": float(g.median()), "pct_maximo": float((g == rating.max()).mean() * 100)}
        for seg, g in rating.groupby(df["Segment"], observed=True)
    }

    # -- Intervalos de confianza de la importancia relativa por segmento (bootstrap de clientes) --
    # Como todos valoran las mismas tarjetas, los part-worths salen de la media por tarjeta: exacto y rápido.
    rng = np.random.default_rng(SEED)
    niveles_x = pd.get_dummies(base[ATRIBUTOS].astype(str), drop_first=False).astype(float)
    cols_ref = [f"{a}_{ORDEN[a][0]}" for a in ATRIBUTOS]
    Xd = niveles_x.drop(columns=cols_ref)
    Xd.insert(0, "const", 1.0)
    Xd = Xd.values
    proyeccion = np.linalg.pinv(Xd)
    nombres = [c for c in niveles_x.columns if c not in cols_ref]

    def importancias_desde_medias(medias_por_tarjeta: np.ndarray) -> dict:
        beta = proyeccion @ medias_por_tarjeta
        coef = dict(zip(nombres, beta[1:]))
        rangos = {}
        for a in ATRIBUTOS:
            utilidades = [0.0] + [coef[f"{a}_{n}"] for n in ORDEN[a][1:]]
            rangos[a] = max(utilidades) - min(utilidades)
        return importancia(rangos)

    matriz = df.pivot(index="CustomerID", columns="CardID", values="Rating").sort_index(axis=1)
    segmento_cliente = df.drop_duplicates("CustomerID").set_index("CustomerID")["Segment"]
    ci = {}
    for seg in SEGMENTOS:
        m = matriz.loc[segmento_cliente[segmento_cliente == seg].index].values
        puntual = importancias_desde_medias(m.mean(axis=0))
        muestras = {a: [] for a in ATRIBUTOS}
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(m), len(m))
            for a, v in importancias_desde_medias(m[idx].mean(axis=0)).items():
                muestras[a].append(v)
        ci[seg] = {a: {"puntual": puntual[a], "lo": float(np.percentile(muestras[a], 2.5)),
                        "hi": float(np.percentile(muestras[a], 97.5))} for a in ATRIBUTOS}
        # ¿es robusta la diferencia entre los dos atributos que dominan?
        d = np.array(muestras["Flight"]) - np.array(muestras["Price"])
        ci[seg]["flight_minus_price"] = {"puntual": puntual["Flight"] - puntual["Price"],
                                         "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}

    # -- Ajuste a nivel de tarjeta: ¿cuánto se equivoca el modelo aditivo al reproducir cada vuelo valorado? --
    ajuste = {}
    for nombre, datos in [("Overall", df)] + [(s, df[df["Segment"] == s]) for s in SEGMENTOS]:
        ajustado = smf.ols(FORMULA, data=datos).fit().fittedvalues
        por_tarjeta = pd.DataFrame({"real": datos["Rating"], "pred": ajustado}).groupby(datos["CardID"]).mean()
        brecha = (por_tarjeta["pred"] - por_tarjeta["real"]).abs()
        ajuste[nombre] = {"mae_tarjeta": float(brecha.mean()), "max_tarjeta": float(brecha.max()),
                          "tarjetas_por_encima_del_maximo": int((por_tarjeta["pred"] > rating.max()).sum())}

    checks = {
        "n_customers": n_customers,
        "mismo_diseno_para_todos_verificado": bool(mismo_diseno_todos),
        "n_tarjetas": int(len(base)),
        "n_factorial_completo": int(len(combos)),
        "n_combinaciones_sin_valorar": int(len(combos - probadas)),
        "vif_max": float(max(vif)),
        "correlacion_max_entre_atributos": float(max(cruzadas)),
        "rating": {"min": float(rating.min()), "max": float(rating.max()), "media": float(rating.mean()),
                   "pct_en_el_maximo": float((rating == rating.max()).mean() * 100),
                   "por_segmento": por_segmento},
        "robustez": {"t_min_agrupado_por_cliente": t_min, "p_max_agrupado_por_cliente": p_max_cl,
                     "p_max_ols_clasico": float(modelo.pvalues.max())},
        "ajuste_tarjeta": ajuste,
        "importancia_ci": ci, "bootstrap": {"repeticiones": N_BOOT, "semilla": SEED, "unidad": "cliente"},
    }
    with open(ARTIFACTS / "design_checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)

    print("design_cards.csv y design_checks.json guardados")
    print(f"  mismo diseño para los {n_customers} clientes: {mismo_diseno_todos}")
    print(f"  VIF máx. {max(vif):.2f} · correlación máx. entre atributos {max(cruzadas):.2f}")
    print(f"  combinaciones sin valorar: {len(combos - probadas)} de {len(combos)}")
    print(f"  rating {rating.min():.0f}–{rating.max():.0f}, {checks['rating']['pct_en_el_maximo']:.1f}% en el máximo")
    print(f"  |t| mínimo con errores agrupados por cliente: {t_min:.1f} (p máx. {p_max_cl:.1e})")
    for seg in SEGMENTOS:
        d = ci[seg]["flight_minus_price"]
        print(f"  {seg}: Escalas - Precio = {d['puntual']:+.1f} pts (IC95% {d['lo']:+.1f} a {d['hi']:+.1f})")


if __name__ == "__main__":
    main()
