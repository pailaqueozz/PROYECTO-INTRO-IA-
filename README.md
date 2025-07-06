# 🍄 Clasificación de Hongos: Predicción de Comestibilidad con Naive Bayes
## 🔍 Descripción del problema
El consumo de hongos silvestres puede representar un riesgo para la salud si no se identifican correctamente las especies venenosas. Este proyecto busca desarrollar un modelo de clasificación automática que permita predecir si un hongo es comestible o venenoso, basándose en sus características morfológicas.

El objetivo es implementar un clasificador de Naive Bayes que, a partir de atributos como color, forma, tamaño y otras propiedades observables, determine la comestibilidad de un hongo.

## 🗂️ Descripción del dataset
*Nombre: Mushroom Edibility Classification*

Fuente: Kaggle - Mushroom Classification Dataset

## 🍄 Significado de las columnas del Dataset Mushroom Classification

### Variable objetivo:

**class:** <br>
e = Edible<br>
p = Poisonous

## Atributos descriptivos:<br>
**cap-shape: Forma del sombrero**<br>
b = bell (campana)<br>
c = conical (cónico)<br>
x = convex (convexo)<br>
f = flat (plano)<br>
k = knobbed (abultado)<br>
s = sunken (hundido)

**cap-surface: Textura de la superficie del sombrero**<br>
f = fibrous (fibroso)<br>
g = grooves (acanalado)<br>
y = scaly (escamoso)<br>
s = smooth (liso)

**cap-color: Color del sombrero**<br>
n = brown (marrón)<br>
b = buff (beige)<br>
c = cinnamon (canela)<br>
g = gray (gris)<br>
r = green (verde)<br>
p = pink (rosado)<br>
u = purple (púrpura)<br>
e = red (rojo)<br>
w = white (blanco)<br>
y = yellow (amarillo)

**bruises: Moretones al manipular**<br>
t = bruises (sí)<br>
f = no bruises (no)<br>

**odor: Olor del hongo**<br>
a = almond (almendra)<br>
l = anise (anís)<br>
c = creosote (creosota)<br>
y = fishy (olor a pescado)<br>
f = foul (desagradable)<br>
m = musty (moho)<br>
n = none (sin olor)<br>
p = pungent (picante/penetrante)<br>
s = spicy (especiado)<br>

**gill-attachment: Unión de las láminas al tallo**<br>
a = attached (unidas)<br>
d = descending (descendentes)<br>
f = free (libres)<br>
n = notched (con muescas)

**gill-spacing: Espaciado entre las láminas**<br>
c = close (cercanas)<br>
w = crowded (muy juntas)<br>
d = distant (separadas)<br>

**gill-size: Tamaño de las láminas**<br>
b = broad (anchas)<br>
n = narrow (estrechas)

**gill-color: Color de las láminas**<br>
k = black (negro)<br>
n = brown (marrón)<br>
b = buff (beige)<br>
h = chocolate<br>
g = gray (gris)<br>
r = green (verde)<br>
o = orange (naranja)<br>
p = pink (rosado)<br>
u = purple (púrpura)<br>
e = red (rojo)<br>
w = white (blanco)<br>
y = yellow (amarillo)

**stalk-shape: Forma del tallo**<br>
e = enlarging (engrosado)<br>
t = tapering (estrechándose)

**stalk-root: Tipo de raíz del tallo**<br>
b = bulbous (bulbosa)<br>
c = club (de forma de maza)<br>
u = cup (en forma de copa)<br>
e = equal (uniforme)<br>
z = rhizomorphs (rizomorfos)<br>
r = rooted (enraizado)<br>
? = missing (dato faltante)

**stalk-surface-above-ring: Superficie del tallo por encima del anillo**<br>
f = fibrous (fibroso)<br>
y = scaly (escamoso)<br>
k = silky (sedoso)<br>
s = smooth (liso)

**stalk-surface-below-ring: Superficie del tallo por debajo del anillo**<br>
Mismos valores que stalk-surface-above-ring

**stalk-color-above-ring: Color del tallo por encima del anillo**<br>
n = brown (marrón)<br>
b = buff (beige)<br>
c = cinnamon (canela)<br>
g = gray (gris)<br>
o = orange (naranja)<br>
p = pink (rosado)<br>
e = red (rojo)<br>
w = white (blanco)<br>
y = yellow (amarillo)

**stalk-color-below-ring: Color del tallo por debajo del anillo**<br>
Mismos valores que stalk-color-above-ring

**veil-type: Tipo de velo (protección inicial del hongo)**<br>
p = partial (parcial)<br>
u = universal (completo)

**veil-color: Color del velo**<br>
n = brown (marrón)<br>
o = orange (naranja)<br>
w = white (blanco)<br>
y = yellow (amarillo)

**ring-number: Número de anillos en el tallo**<br>
n = none (ninguno)<br>
o = one (uno)<br>
t = two (dos)

**ring-type: Tipo de anillo en el tallo**<br>
c = cobwebby (tipo telaraña)<br>
e = evanescent (efímero)<br>
f = flaring (acampanado)<br>
l = large (grande)<br>
n = none (ninguno)<br>
p = pendant (colgante)<br>
s = sheathing (envolvente)<br>
z = zone (en zonas)

**spore-print-color: Color de la impresión de esporas**<br>
k = black (negro)<br>
n = brown (marrón)<br>
b = buff (beige)<br>
h = chocolate<br>
r = green (verde)<br>
o = orange (naranja)<br>
u = purple (púrpura)<br>
w = white (blanco)<br>
y = yellow (amarillo)

**population: Densidad de la población de hongos en el área**<br>
a = abundant (abundante)<br>
c = clustered (agrupados)<br>
n = numerous (numerosos)<br>
s = scattered (dispersos)<br>
v = several (varios)<br>
y = solitary (solitarios)

**habitat: Hábitat donde se encuentran los hongos**<br>
g = grasses (pastizales)<br>
l = leaves (hojarasca)<br>
m = meadows (praderas)<br>
p = paths (caminos)<br>
u = urban (urbano)<br>
w = waste (áreas de desecho)<br>
d = woods (bosques)

Este dataset es ampliamente utilizado en problemas de clasificación y es ideal para modelos probabilísticos como Naive Bayes debido a su naturaleza categórica.

🧩 Justificación del modelo seleccionado
Se eligió el algoritmo Naive Bayes, en particular el clasificador de tipo multinomial, por los siguientes motivos:

✅ Los atributos son categóricos y discretos, lo que se ajusta perfectamente al supuesto de Naive Bayes.<br>
✅ Es un modelo simple, rápido y efectivo para tareas de clasificación binaria.<br>
✅ Basado en el Teorema de Bayes, estima la probabilidad de cada clase asumiendo independencia entre los atributos, lo cual, aunque simplifica la realidad, suele ser suficiente para obtener buenos resultados.<br>
✅ Tiene buen desempeño incluso con datasets de tamaño medio como este.

## Limitaciones:

• Todos los atributos son categóricos, lo que limita el uso directo de modelos que requieren datos numéricos. Aunque pueden        codificarse, esta transformación puede introducir ruido, especialmente si se utiliza una codificación ordinal para variables sin orden implícito.<br>

• Algunos atributos (por ejemplo, odor) tienen una fuerte correlación con la clase, mientras que otros pueden ser redundantes o incluso irrelevantes, afectando negativamente a algunos modelos si no se realiza una selección adecuada de variables.<br>

• Los modelos clasificadores solo identifican correlaciones estadísticas. No pueden determinar si un atributo como “odor” causa que un hongo sea venenoso, solo que hay asociación.

## 🛠️ Metodología aplicada<br>
## Carga y Exploración de Datos (EDA)

• Revisión de la distribución de clases (edible vs poisonous).

• Análisis de los valores de los atributos categóricos.

• Visualización de correlaciones entre características.

## Preprocesamiento de Datos

• Codificación de variables categóricas mediante LabelEncoder.

• Revisión de datos faltantes y consistencia.

## División de Datos y Vectorización

• Separación en conjuntos de entrenamiento y prueba (ejemplo: 80% - 20%).

• No se requiere vectorización de texto, ya que todos los atributos son categóricos.

## Entrenamiento del Modelo

• Implementación de Naive Bayes con CategoricalNB de scikit-learn.

## Evaluación y Visualización de Resultados

• Matriz de confusión.

• Métricas de clasificación: Accuracy, Precision, Recall, F1-score.

• Discusión de hallazgos y posibles mejoras.

## Control de Overfitting

• Validación cruzada.

• Comparación de métricas entre entrenamiento y prueba.

## 📊 Resultados y Conclusiones

### Métricas del Modelo Naive Bayes

Después de entrenar y evaluar el modelo CategoricalNB en el dataset elegido, se obtuvieron los siguientes resultados:

#### **Datos del Dataset:**
- **Tamaño total:** 8,124 hongos con 23 características
- **Distribución de clases:** 
  - Comestibles (e): 4,208 (51.8%)
  - Venenosos (p): 3,916 (48.2%)
- **División:** 80% entrenamiento (6,499) / 20% prueba (1,625)

#### **Rendimiento del Modelo:**
**Precisión Global:** 95.08%

**Métricas por Clase:**
- **Hongos Comestibles (Clase 0):**
  - Precisión: 92%
  - Recall: 99%
  - F1-score: 95%

- **Hongos Venenosos (Clase 1):**
  - Precisión: 99%
  - Recall: 91%
  - F1-score: 95%

**Validación Cruzada (5-fold):**
- **Precisión promedio:** 95.27%
- **Desviación estándar:** 0.66%
- **Rango de precisión:** 94.09% - 95.88%

#### **Análisis de la Matriz de Confusión:**

| Real/Predicción | Edible | Poisonous |
|-----------------|--------|-----------|
| **Edible**      | 837    | 6         |
| **Poisonous**   | 74     | 708       |

- **Verdaderos Positivos (Edible):** 837
- **Falsos Negativos (Edible clasificado como Poisonous):** 6 
- **Falsos Positivos (Poisonous clasificado como Edible):** 74 
- **Verdaderos Negativos (Poisonous):** 708

### 🎉 Conclusiones Principales

#### **Fortalezas del Modelo:**

1. **Excelente Rendimiento General:** Con un 95.27% de precisión promedio en validación cruzada, el modelo demuestra una capacidad sobresaliente para clasificar hongos.

2. **Consistencia Robusta:** La baja desviación estándar (0.66%) indica que el modelo es estable y confiable entre diferentes subconjuntos de datos.

3. **Alta Sensibilidad para Hongos Comestibles:** El recall del 99% para hongos comestibles minimiza el riesgo de clasificar incorrectamente un hongo comestible como venenoso.

4. **Precisión Excepcional para Hongos Venenosos:** El 99% de precisión para hongos venenosos significa que cuando el modelo predice "venenoso", es casi siempre correcto.

#### **Consideraciones Críticas:**

**Falsos Negativos (6 casos):** Hongos comestibles clasificados como venenosos. Esto es relativamente seguro, aunque puede llevar a desperdiciar hongos comestibles.

**Falsos Positivos (74 casos):** Hongos venenosos clasificados como comestibles. **Este es el error más peligroso** en este contexto, ya que podría llevar a intoxicaciones.

### Conclusión Final

El modelo CategoricalNB logró una precisión excepcional del **95.27%** en la clasificación de hongos comestibles vs venenosos, validando la efectividad del algoritmo Naive Bayes para problemas de clasificación con variables categóricas. 

Sin embargo, dada la naturaleza crítica de la aplicación (donde un error puede tener consecuencias graves para la salud), se enfatiza que **este modelo debe ser utilizado únicamente como herramienta de apoyo** para especialistas, nunca como sustituto del conocimiento experto humano.

El proyecto demuestra exitosamente cómo los algoritmos de ML pueden contribuir significativamente a problemas de clasificación, entregando una base sólida para futuros proyectos.

## 📂 Estructura del repositorio

    📁 PROYECTO-INTRO-IA-/
    - proyecto_IA.ipynb       # Notebook principal del proyecto 
    - dataset/                        # Carpeta con el dataset
        -mushrooms.csv
    - requirements.yml                # Dependencias del proyecto 
    - README.md                       # Este archivo explicativo

## Librerías principales:

    - numpy
    - pandas
    - scikit-learn
    - seaborn
    - matplotlib

🔗 Fuente del Dataset
Kaggle - Mushroom Classification Dataset
https://www.kaggle.com/datasets/uciml/mushroom-classification