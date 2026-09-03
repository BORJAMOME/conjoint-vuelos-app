# Análisis Conjoint — Preferencias de Vuelos

**Una aerolínea diseña vuelos con precio, escalas, equipaje y flexibilidad, pero no sabe cuánto vale cada
extra para el cliente, ni si vale lo mismo para todos.**

Una aplicación interactiva en la que descompuse la valoración de un vuelo en el valor exacto que aporta
cada una de sus características (precio, equipaje, asiento, escalas, flexibilidad y horario) y que
muestra cómo esa prioridad cambia por completo según el tipo de cliente. Deja diseñar un vuelo
hipotético y comparar, en vivo, cómo lo valoraría cada segmento.

No hace falta saber nada de estadística para seguirla: empieza por el problema, explica el análisis
conjoint en lenguaje llano, y termina dejándote construir tu propio vuelo para ver a quién convence.

## Ver la app

🔗 **[Abrir la app](https://conjoint-vuelos.streamlit.app)** _(actualizar con la URL real tras el deploy en Streamlit Cloud)_

## De qué trata, en dos frases

1.000 clientes de una aerolínea valoraron, del 1 al 10, las mismas 24 combinaciones de vuelo (diseño
ortogonal sobre 6 atributos). Con una regresión lineal (análisis conjoint) descompuse esas 24.000
valoraciones en la contribución de cada nivel de cada atributo: sus *part-worths*.

**El resultado:** el precio y las escalas concentran el 69% de la decisión media, pero esa media esconde
tres lógicas de cliente muy distintas: **Business** paga por volar directo incluso más que por el precio;
**Low Cost** decide casi exclusivamente por precio (59,7% de su importancia).

## Qué te vas a encontrar al recorrerla

1. **El problema** — por qué diseñar vuelos sin saber qué valora el cliente es una apuesta
2. **Los datos** — 1.000 clientes, 24 tarjetas de un diseño ortogonal, 24.000 valoraciones
3. **Antes de modelar** — cómo se reparten los ratings y si los segmentos puntúan distinto
4. **Cómo funciona un conjoint** — de la valoración global al valor de cada atributo, sin jerga
5. **El modelo** — R²=0,747, utilidades parciales de cada atributo, y por qué el precio no es lineal
6. **Explicabilidad** — importancia relativa de cada atributo, global y por segmento de cliente
7. **Playground** — diseña un vuelo y compara en directo cómo lo valora cada segmento
8. **Resultados y decisiones** — qué haría un equipo de producto o pricing con estos hallazgos

## Cómo está hecho

Python + [Streamlit](https://streamlit.io) para la aplicación, y
[statsmodels](https://www.statsmodels.org) (`OLS`) para el modelo. El análisis completo, en formato
notebook, está en el
[repositorio de portfolio](https://github.com/BORJAMOME/Data-Analytics-Portfolio/tree/main/03-Machine-Learning/01-supervisado/regresion/02-regresion-lineal-multiple/03-preferencias-vuelos-conjoint).

Calculé todos los números que aparecen en la app una vez en `model/train.py` y los guardé como datos:
nada está escrito a mano.

## Ejecutarla en tu ordenador

```bash
pip install -r requirements.txt
streamlit run app.py
```

Los resultados del modelo ya vienen calculados en `model/artifacts/`, así que no hace falta reentrenar
nada para verla funcionar.

Solo si cambias el dataset (`data/Conjoint_Flight.xlsx`) necesitas regenerarlos:

```bash
python model/train.py    # tarda unos segundos: son 4 regresiones OLS sobre datos tabulares
```

<details>
<summary>Estructura del proyecto, para quien quiera curiosear el código</summary>

```
app.py                    la aplicación — toda la narrativa, sección a sección
components/
  ui.py                    bloques visuales reutilizables (tarjetas, títulos, callouts)
  charts.py                gráficos, con la paleta de colores del proyecto
utils/
  data_loader.py            carga de artefactos (con cache de Streamlit)
  conjoint.py                predice el rating de un vuelo hipotético sumando utilidades parciales
model/
  train.py                    ajusta las 4 regresiones OLS (global + 3 segmentos), calcula todo
  artifacts/                   resultados ya calculados (part-worths, importancia, R²...)
data/                      el dataset original
assets/style.css           el sistema visual de la app
```

Hice que el Playground no dependiera de ningún pickle ni de scikit-learn: un modelo conjoint es una
suma de utilidades parciales, así que predecir un rating es sumar el intercepto y la utilidad del nivel
elegido en cada atributo (números pequeños guardados en un JSON, sin fragilidad entre entornos).
</details>

---

**Autor:** Borja Mora Méndez · [LinkedIn](https://www.linkedin.com/in/borja-mora-mendez/) · [GitHub](https://github.com/BORJAMOME)
