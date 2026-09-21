# Análisis Conjoint — Preferencias de Vuelos

**¿Qué hace que un vuelo merezca la pena para cada cliente?**

Una aplicación interactiva en la que descompuse la valoración de un vuelo en el valor exacto que aporta
cada una de sus características (precio, equipaje, asiento, escalas, flexibilidad y horario) y que
muestra cómo esa prioridad cambia por completo según el tipo de cliente. Deja diseñar un vuelo
hipotético y comparar, en vivo, cómo lo valoraría cada segmento.

No hace falta saber nada de estadística para seguirla: empieza por el problema, explica el análisis
conjoint en lenguaje llano, y termina dejándote construir tu propio vuelo para ver a quién convence.

## Ver la app

🔗 **[Abrir la app](https://conjoint-vuelos.streamlit.app)** 

## De qué trata, en dos frases

1.000 clientes de una aerolínea valoraron, del 1 al 10, las mismas 24 combinaciones de vuelo (un diseño
fraccionado, casi ortogonal, sobre 6 atributos: 120 de las 144 combinaciones posibles nunca se valoraron). Con una regresión lineal (análisis conjoint) descompuse esas 24.000
valoraciones en la contribución de cada nivel de cada atributo: sus *part-worths*.

**El resultado:** el precio y las escalas concentran el 69% de la decisión media, pero esa media esconde
tres lógicas de cliente distintas: para **Business**, volar directo pesa más que el precio (37,9% frente a
24,4%); para **Low Cost**, el precio concentra el 59,7% de su importancia. Los intervalos de confianza
(bootstrap de clientes) son estrechos: no es ruido de muestreo.

## Qué te vas a encontrar al recorrerla

1. **El problema** — saber qué características importan de verdad al cliente, y si importan igual a todos
2. **Los datos** — 1.000 clientes, 24 tarjetas de un diseño fraccionado, 24.000 valoraciones
3. **Antes de modelar** — cómo se reparten los ratings y en qué se diferencian los segmentos
4. **Cómo funciona un conjoint** — de la valoración global al valor de cada atributo, sin jerga
5. **El modelo** — R²=0,747, utilidades parciales de cada atributo, y por qué el precio no es lineal
6. **Explicabilidad** — importancia relativa de cada atributo para el cliente promedio
7. **Segmentos** — tres formas distintas de valorar el mismo vuelo
8. **Playground** — diseña un vuelo y compara en directo cómo lo valora cada segmento (avisa si el vuelo se valoró de verdad o es una estimación, y si el modelo se sale de la escala)
9. **Resultados, decisiones y límites** — hipótesis de negocio (no conclusiones del modelo) y dónde termina lo que sabemos

## Cómo está hecho

Python + [Streamlit](https://streamlit.io) para la aplicación, y
[statsmodels](https://www.statsmodels.org) (`OLS`) para el modelo. El análisis completo, en formato
notebook, está en el
[repositorio de portfolio](https://github.com/BORJAMOME/Data-Analytics-Portfolio/tree/main/03-Machine-Learning/01-supervisado/regresion/02-regresion-lineal-multiple/03-preferencias-vuelos-conjoint).

Calculé todos los números que aparecen en la app una vez y los guardé como datos: el modelo en
`model/train.py` y las comprobaciones del experimento en `model/export_design_checks.py` (mismo diseño para
todos los clientes, calidad del diseño, robustez con errores agrupados por cliente, techo de la escala,
intervalos de confianza por segmento). Nada está escrito a mano.

## Ejecutarla en tu ordenador

```bash
pip install -r requirements.txt
streamlit run app.py
```

Los resultados del modelo ya vienen calculados en `model/artifacts/`, así que no hace falta reentrenar
nada para verla funcionar.

Solo si cambias el dataset (`data/Conjoint_Flight.xlsx`) necesitas regenerarlos:

```bash
python model/train.py                  # tarda unos segundos: son 4 regresiones OLS sobre datos tabulares
python model/export_design_checks.py   # comprobaciones del diseño; no cambia el modelo
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
  export_design_checks.py      comprobaciones del diseño y de la robustez (no toca el modelo)
  artifacts/                   resultados ya calculados (part-worths, importancia, R²...)
data/                      el dataset original
assets/style.css           el sistema visual de la app
```

Hice que el Playground no dependiera de ningún pickle ni de scikit-learn: un modelo conjoint es una
suma de utilidades parciales, así que predecir un rating es sumar el intercepto y la utilidad del nivel
elegido en cada atributo (números pequeños guardados en un JSON, sin fragilidad entre entornos).
</details>

---

**Autor:** Borja Mora Méndez · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [GitHub](https://github.com/BORJAMOME)
