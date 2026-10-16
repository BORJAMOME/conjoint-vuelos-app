# ¿Qué hace que un vuelo merezca la pena para cada cliente?**

En este proyecto he analizado cómo influye cada característica de un vuelo (precio, equipaje, asiento, escalas, flexibilidad y horario) en su valoración, y cómo cambian las preferencias según el tipo de cliente

**[➡ Abrir la app](https://preferencias-vuelos.streamlit.app/)** 

## De qué trata

1.000 clientes valoraron del 1 al 10 un total de 24 combinaciones de vuelo, diseñadas para representar 6 atributos. De las 144 combinaciones posibles, solo se evaluaron 24. Mediante una regresión lineal (análisis conjoint), descompuse las 24.000 valoraciones para medir cuánto aporta cada característica y cada uno de sus niveles a la valoración final (part-worths).

**El resultado:** el precio y las escalas representan el 69% de la importancia media en la elección de un vuelo, pero las preferencias varían según el tipo de cliente. Para Business, volar directo importa más que el precio (37,9% frente a 24,4%), mientras que para Low Cost, el precio concentra el 59,7% de la importancia. Los estrechos intervalos de confianza, calculados mediante bootstrap de clientes, respaldan la estabilidad de estas diferencias.

## Qué te encontrarás en este proyecto

1. **El problema**: saber qué características importan de verdad al cliente, y si importan igual a todos
2. **Los datos**: 1.000 clientes, 24 tarjetas de un diseño fraccionado, 24.000 valoraciones
3. **Antes de modelar**: cómo se reparten los ratings y en qué se diferencian los segmentos
4. **Cómo funciona un conjoint**: de la valoración global al valor de cada atributo, sin jerga
5. **El modelo**: R²=0,747, utilidades parciales de cada atributo, y por qué el precio no es lineal
6. **Explicabilidad**: importancia relativa de cada atributo para el cliente promedio
7. **Segmentos**: tres formas distintas de valorar el mismo vuelo
8. **Playground**: diseña un vuelo y compara en directo cómo lo valora cada segmento (avisa si el vuelo se valoró de verdad o es una estimación, y si el modelo se sale de la escala)
9. **Resultado, decisiones y límites**: hipótesis de negocio (no conclusiones del modelo) y dónde termina lo que sabemos

## Cómo está hecho

Python + [Streamlit](https://streamlit.io) para la aplicación, y
[statsmodels](https://www.statsmodels.org) (`OLS`) para el modelo. El análisis completo, en formato
notebook, está en el
[repositorio de portfolio](https://github.com/BORJAMOME/Data-Analytics-Portfolio/tree/main/03-Machine-Learning/01-supervisado/regresion/02-regresion-lineal-multiple/03-preferencias-vuelos-conjoint).


**Autor:** Borja Mora Méndez · [LinkedIn](https://www.linkedin.com/in/borjamoramendez/) · [GitHub](https://github.com/BORJAMOME)
