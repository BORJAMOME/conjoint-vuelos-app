"""Predice el rating de un vuelo hipotético en vivo. Un modelo conjoint
es una suma de utilidades parciales: predecir es sumar el intercepto y
la utilidad de cada nivel elegido — no hace falta ningún pickle."""


def predict_rating(choice: dict, atributos: list, intercept: float, partworths: dict) -> float:
    total = intercept
    for atributo in atributos:
        nivel = choice[atributo]
        total += partworths[atributo][nivel]
    return total


def predict_all_segments(choice: dict, atributos: list, intercepts: dict, partworths_by_segment: dict) -> dict:
    return {
        seg: predict_rating(choice, atributos, intercepts[seg], partworths_by_segment[seg])
        for seg in intercepts
    }
