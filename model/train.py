"""
Replica el análisis conjoint del notebook original (regresión OLS sobre
un diseño factorial ortogonal de 24 tarjetas) y guarda los artefactos
que consume la app de Streamlit: part-worths globales y por segmento,
importancia relativa de atributos, y las estadísticas del diseño.

El "modelo" del Playground no necesita ningún pickle: un conjoint es una
suma de utilidades parciales, así que basta con guardar el intercepto y
los part-worths (números pequeños) para poder predecir en vivo.

Ejecutar una sola vez:
    py -3.10 model/train.py
"""
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Conjoint_Flight.xlsx"
ARTIFACTS = ROOT / "model" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

ATRIBUTOS = ["Price", "Baggage", "Seat", "Flight", "Flexible", "Departure"]
ORDEN = {
    "Price": ["50", "100", "150"],
    "Baggage": ["No baggage", "20 kg"],
    "Seat": ["Random", "Seat Selection"],
    "Flight": ["1 Stop", "Direct"],
    "Flexible": ["No", "Yes"],
    "Departure": ["Morning", "Afternoon", "Evening"],
}
FORMULA = "Rating ~ C(Price) + C(Baggage) + C(Seat) + C(Flight) + C(Flexible) + C(Departure)"
SEGMENTOS = ["Business", "Leisure", "Low Cost"]


def extraer_partworths(modelo, atributos, orden):
    filas = []
    params = modelo.params
    for atributo in atributos:
        niveles = orden[atributo]
        filas.append({"Atributo": atributo, "Nivel": str(niveles[0]), "Utilidad": 0.0})
        for nivel in niveles[1:]:
            clave = f"C({atributo})[T.{nivel}]"
            filas.append({"Atributo": atributo, "Nivel": str(nivel), "Utilidad": float(params[clave])})
    return pd.DataFrame(filas)


def importancia_desde_partworths(pw: pd.DataFrame) -> pd.Series:
    rangos = pw.groupby("Atributo")["Utilidad"].agg(lambda x: x.max() - x.min())
    return (rangos / rangos.sum() * 100).reindex(ATRIBUTOS)


def main():
    print("Cargando dataset...")
    df = pd.read_excel(DATA_PATH, sheet_name="Transacciones")

    df["Price"] = pd.Categorical(df["Price"].astype(str), categories=["50", "100", "150"])
    df["Baggage"] = pd.Categorical(df["Baggage"], categories=["No baggage", "20 kg"])
    df["Seat"] = pd.Categorical(df["Seat"], categories=["Random", "Seat Selection"])
    df["Flight"] = pd.Categorical(df["Flight"], categories=["1 Stop", "Direct"])
    df["Flexible"] = pd.Categorical(df["Flexible"], categories=["No", "Yes"])
    df["Departure"] = pd.Categorical(df["Departure"], categories=["Morning", "Afternoon", "Evening"])

    n_customers = int(df["CustomerID"].nunique())
    n_cards = int(df["CardID"].nunique())
    n_rows = int(len(df))
    mismo_diseno = bool(
        df[df["CustomerID"] == df["CustomerID"].iloc[0]][["CardID"] + ATRIBUTOS].reset_index(drop=True)
        .equals(df[df["CustomerID"] == df["CustomerID"].unique()[5]][["CardID"] + ATRIBUTOS].reset_index(drop=True))
    )
    segment_counts = df.groupby("Segment")["CustomerID"].nunique().to_dict()

    # -- Distribución del rating (para el histograma de EDA) ------------------------
    df[["Rating", "Segment"]].to_csv(ARTIFACTS / "ratings_raw.csv", index=False)

    # -- Modelo global ----------------------------------------------------------------
    print("Ajustando modelo OLS global...")
    modelo = smf.ols(FORMULA, data=df).fit()
    partworths = extraer_partworths(modelo, ATRIBUTOS, ORDEN)
    partworths.to_csv(ARTIFACTS / "partworths_overall.csv", index=False)

    importancia = importancia_desde_partworths(partworths)
    importancia.reset_index().rename(columns={"index": "Atributo", "Utilidad": "Importancia"}).to_csv(
        ARTIFACTS / "importance_overall.csv", index=False)

    coef_table = pd.DataFrame({
        "termino": modelo.params.index,
        "coef": modelo.params.values,
        "std_err": modelo.bse.values,
        "p_valor": modelo.pvalues.values,
    })
    coef_table.to_csv(ARTIFACTS / "model_coefficients.csv", index=False)

    with open(ARTIFACTS / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "r_squared": float(modelo.rsquared),
            "r_squared_adj": float(modelo.rsquared_adj),
            "f_statistic": float(modelo.fvalue),
            "n_obs": int(modelo.nobs),
            "intercept": float(modelo.params["Intercept"]),
            "max_p_valor": float(modelo.pvalues.max()),
        }, f, ensure_ascii=False, indent=2)

    # -- Modelos por segmento -----------------------------------------------------------
    print("Ajustando modelos OLS por segmento...")
    pw_segmento_rows = []
    importancia_segmento_rows = []
    segment_summary = {}
    for seg in SEGMENTOS:
        datos_seg = df[df["Segment"] == seg]
        modelo_seg = smf.ols(FORMULA, data=datos_seg).fit()
        pw_seg = extraer_partworths(modelo_seg, ATRIBUTOS, ORDEN)
        pw_seg["Segmento"] = seg
        pw_segmento_rows.append(pw_seg)

        imp_seg = importancia_desde_partworths(pw_seg)
        for atributo, valor in imp_seg.items():
            importancia_segmento_rows.append({"Segmento": seg, "Atributo": atributo, "Importancia": valor})

        segment_summary[seg] = {
            "n_customers": int(segment_counts[seg]),
            "n_obs": int(modelo_seg.nobs),
            "r_squared": float(modelo_seg.rsquared),
            "intercept": float(modelo_seg.params["Intercept"]),
        }
        print(f"  {seg}: R2={modelo_seg.rsquared:.3f}, n_clientes={segment_counts[seg]}")

    pd.concat(pw_segmento_rows, ignore_index=True).to_csv(ARTIFACTS / "partworths_segment.csv", index=False)
    pd.DataFrame(importancia_segmento_rows).to_csv(ARTIFACTS / "importance_segment.csv", index=False)

    # -- Artefactos para el Playground (sin pickles: solo intercepto + part-worths) ----
    def pw_to_dict(pw_df):
        d = {}
        for atributo in ATRIBUTOS:
            sub = pw_df[pw_df["Atributo"] == atributo]
            d[atributo] = dict(zip(sub["Nivel"], sub["Utilidad"]))
        return d

    playground = {
        "intercept": {"Overall": float(modelo.params["Intercept"])},
        "partworths": {"Overall": pw_to_dict(partworths)},
        "orden": ORDEN,
        "atributos": ATRIBUTOS,
    }
    for seg, pw_seg_df in zip(SEGMENTOS, pw_segmento_rows):
        playground["intercept"][seg] = segment_summary[seg]["intercept"]
        playground["partworths"][seg] = pw_to_dict(pw_seg_df)

    with open(ARTIFACTS / "playground_model.json", "w", encoding="utf-8") as f:
        json.dump(playground, f, ensure_ascii=False, indent=2)

    # -- Estadísticas generales del diseño y del dataset -------------------------------
    with open(ARTIFACTS / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_customers": n_customers,
            "n_cards": n_cards,
            "n_rows": n_rows,
            "mismo_diseno_para_todos": mismo_diseno,
            "segment_counts": {k: int(v) for k, v in segment_counts.items()},
            "n_atributos": len(ATRIBUTOS),
            "n_combinaciones_factorial_completo": 3 * 2 * 2 * 2 * 2 * 3,
            "segments": segment_summary,
        }, f, ensure_ascii=False, indent=2)

    print("\nListo. Resumen:")
    print(f"  {n_customers} clientes x {n_cards} tarjetas = {n_rows} valoraciones")
    print(f"  R² global = {modelo.rsquared:.3f}")
    print(f"  Importancia: {importancia.round(1).to_dict()}")


if __name__ == "__main__":
    main()
