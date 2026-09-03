"""Carga de datos y artefactos del modelo, con cache de Streamlit."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "model" / "artifacts"

ATTRIBUTE_LABELS = {
    "Price": "Precio",
    "Baggage": "Equipaje",
    "Seat": "Selección de asiento",
    "Flight": "Escalas",
    "Flexible": "Flexibilidad",
    "Departure": "Horario de salida",
}

LEVEL_LABELS = {
    "50": "50 €", "100": "100 €", "150": "150 €",
    "No baggage": "Sin equipaje", "20 kg": "20 kg incluidos",
    "Random": "Asignación aleatoria", "Seat Selection": "Con selección de asiento",
    "1 Stop": "1 escala", "Direct": "Directo",
    "No": "Sin flexibilidad", "Yes": "Con flexibilidad",
    "Morning": "Mañana", "Afternoon": "Tarde", "Evening": "Noche",
}

SEGMENT_LABELS = {"Overall": "Todos los clientes", "Business": "Business", "Leisure": "Leisure", "Low Cost": "Low Cost"}
# Familia de azules-marino, igual que el resto del sistema visual: los 3
# segmentos son categorías sin jerarquía de "bueno/malo", así que nunca
# usan verde/rojo — esos quedan reservados en exclusiva para lo semántico.
SEGMENT_COLORS = {"Overall": "#B9C5D6", "Business": "#1D2638", "Leisure": "#4A628E", "Low Cost": "#273A5F"}


@st.cache_data(show_spinner=False)
def load_json(name: str):
    with open(ARTIFACTS / name, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / name, **kwargs)


def artifacts_ready() -> bool:
    return (ARTIFACTS / "dataset_stats.json").exists() and (ARTIFACTS / "playground_model.json").exists()
