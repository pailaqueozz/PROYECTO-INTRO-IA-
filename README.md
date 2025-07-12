#  Clasificación de Hongos: Predicción de Comestibilidad con Naive Bayes

## **Descripción del problema**
Los hongos en general son poco intuitivos de reconocer y clasificar, por lo que una clasificación manual podría generar errores y terminar en intoxicaciones. En este proyecto abordamos la oportunidad de crear una clasificación con Naive Bayes automática. Proponemos un modelo predictivo que, utilizando atributos descriptivos de los hongos, clasifique cada espécimen como comestible o venenoso.

## **Descripción del dataset**
El análisis se basa en el conjunto de datos "Mushroom Edibility Classification", obtenido de la plataforma Kaggle. En este dataset encontramos datos descriptivos de distintos hongos como lo son, características físicas, textura, sombrero, color, etc.

## **Significado de las columnas del Dataset**
El dataset consta de una variable objetivo y múltiples atributos descriptivos, todos ellos de naturaleza categórica. A continuación, se detalla el significado de cada variable y sus posibles valores:


### Variable objetivo:

**class:** <br>
e = Edible<br>
p = Poisonous

### Atributos descriptivos:<br>
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

## 4. Justificación de los modelos seleccionados

Se decidió implementar y comparar dos algoritmos de clasificación para evaluar cuál ofrece mejor rendimiento en la clasificación de hongos: **Naive Bayes** y **Random Forest**.

### Naive Bayes (CategoricalNB)

Se eligió el algoritmo Naive Bayes, en particular el clasificador de tipo multinomial, por los siguientes motivos:

Los atributos son categóricos y discretos, lo que se ajusta perfectamente al supuesto de Naive Bayes.<br>

Es un modelo simple, rápido y efectivo para tareas de clasificación binaria.<br>

Basado en el Teorema de Bayes, estima la probabilidad de cada clase asumiendo independencia entre los atributos, lo cual, aunque simplifica la realidad, suele ser suficiente para obtener buenos resultados.<br>

Tiene buen desempeño incluso con datasets de tamaño medio como este.

### Random Forest

La inclusión de Random Forest como segundo modelo se justifica por las siguientes razones:

Es un algoritmo robusto que maneja bien variables categóricas sin necesidad de asunciones de independencia entre características.<br>

Al ser un método de ensemble, combina múltiples árboles de decisión, lo que reduce el riesgo de overfitting y mejora la generalización.<br>

Proporciona información valiosa sobre la importancia de las características, permitiendo identificar qué atributos son más relevantes para la clasificación.<br>

Es menos sensible a outliers y ruido en los datos comparado con modelos más simples.<br>

Su capacidad para capturar interacciones complejas entre variables lo hace especialmente adecuado para problemas donde las relaciones entre características pueden ser no lineales.


## Limitaciones:

• Todos los atributos son categóricos, lo que limita el uso directo de modelos que requieren datos numéricos. Aunque pueden codificarse, esta transformación puede introducir ruido, especialmente si se utiliza una codificación ordinal para variables sin orden implícito.<br>

• Algunos atributos (por ejemplo, odor) tienen una fuerte correlación con la clase, mientras que otros pueden ser redundantes o incluso irrelevantes, afectando negativamente a algunos modelos si no se realiza una selección adecuada de variables.<br>

• Los modelos clasificadores solo identifican correlaciones estadísticas. No pueden determinar si un atributo como “odor” causa que un hongo sea venenoso, solo que hay asociación.

# Metodología aplicada<br>
## Carga y Exploración de Datos (EDA)

• Revisión de la distribución de clases (edible vs poisonous).

• Análisis de los valores de los atributos categóricos.

• Visualización de correlaciones entre características.

## Preprocesamiento de Datos

• Codificación de variables categóricas mediante LabelEncoder.

• Revisión de datos faltantes y consistencia.

## División de Datos

• Separación en conjuntos de entrenamiento y prueba (ejemplo: 80% - 20%).

• No se requiere vectorización de texto, ya que todos los atributos son categóricos.

## Entrenamiento del Modelo

• Implementación de Naive Bayes con CategoricalNB y Random Forest con RandomForestClassifier de scikit-learn.

## Evaluación y Visualización de Resultados

• Matriz de confusión.

• Métricas de clasificación: Accuracy, Precision, Recall, F1-score.

• Discusión de hallazgos y posibles mejoras.

## Control de Overfitting

• Validación cruzada.

• Comparación de métricas entre entrenamiento y prueba.

## Resultados y Conclusiones

### Comparación de Modelos: Naive Bayes vs Random Forest

Después de entrenar y evaluar ambos modelos en el dataset de hongos, se obtuvieron los siguientes resultados comparativos:

#### **Datos del Dataset:**
- **Tamaño total:** 8,124 hongos con 23 características
- **Distribución de clases:** 
  - Comestibles (e): 4,208 (51.8%)
  - Venenosos (p): 3,916 (48.2%)
- **División:** 80% entrenamiento (6,499) / 20% prueba (1,625)

#### **Rendimiento Comparativo:**

| Métrica | Naive Bayes | Random Forest |
|---------|-------------|---------------|
| **Accuracy (Test Set)** | 95.1% | 100.0% |
| **Accuracy (Validación Cruzada)** | 95.3% | 100.0% |
| **Desviación Estándar** | 0.7% | 0.0% |

#### **Análisis de Errores Críticos:**

| Tipo de Error | Naive Bayes | Random Forest |
|---------------|-------------|---------------|
| **Falsos Positivos** | 74 casos | 0 casos |
| **Falsos Negativos** | 6 casos | 0 casos |

**Nota:** Los falsos positivos (hongos venenosos clasificados como comestibles) representan el error más crítico en este contexto.

#### **Métricas Detalladas por Modelo:**

**Naive Bayes:**
- **Hongos Comestibles:** Precisión 92%, Recall 99%, F1-score 95%
- **Hongos Venenosos:** Precisión 99%, Recall 91%, F1-score 95%

**Random Forest:**
- **Hongos Comestibles:** Precisión 100%, Recall 100%, F1-score 100%
- **Hongos Venenosos:** Precisión 100%, Recall 100%, F1-score 100%

### Conclusiones Principales

#### **Rendimiento Superior de Random Forest:**

1. **Precisión Perfecta:** Random Forest logró un 100% de precisión tanto en el conjunto de prueba como en validación cruzada, superando significativamente a Naive Bayes.

2. **Eliminación de Errores Críticos:** Random Forest no presentó falsos positivos ni falsos negativos, eliminando completamente el riesgo de clasificar hongos venenosos como comestibles.

3. **Estabilidad Absoluta:** La desviación estándar de 0.0% en validación cruzada indica consistencia perfecta entre diferentes particiones de datos.

#### **Desempeño de Naive Bayes:**

1. **Rendimiento Sólido:** Con 95.3% de precisión promedio, Naive Bayes demostró ser un clasificador competente.

2. **Errores Controlados:** Aunque presentó 74 falsos positivos, mantuvo un número relativamente bajo considerando el tamaño del dataset.

3. **Eficiencia Computacional:** Ofrece entrenamiento más rápido y menor complejidad computacional.

#### **Consideraciones Prácticas:**

**Recomendación:** Random Forest es el modelo recomendado para esta aplicación debido a su precisión perfecta y eliminación total de errores críticos.

**Ventajas de Random Forest:**
- Precisión perfecta en la clasificación
- Eliminación completa de riesgos de seguridad
- Capacidad de identificar características más importantes
- Mayor robustez ante variaciones en los datos

**Ventajas de Naive Bayes:**
- Menor tiempo de entrenamiento
- Mayor simplicidad e interpretabilidad
- Menor requerimiento computacional
- Buen rendimiento general con recursos limitados

### Conclusión Final

El análisis comparativo demuestra la superioridad clara de Random Forest sobre Naive Bayes para la clasificación de hongos comestibles vs venenosos. Con una **precisión perfecta del 100%** y la eliminación total de errores críticos, Random Forest proporciona el nivel de seguridad requerido para una aplicación tan sensible.

Aunque Naive Bayes mostró un rendimiento sólido (95.3%), la presencia de 74 falsos positivos representa un riesgo inaceptable en el contexto de seguridad alimentaria. 

**Random Forest se establece como el modelo de elección** para este problema, ofreciendo la confiabilidad necesaria para una herramienta de apoyo en la identificación de hongos, manteniendo siempre la recomendación de que debe complementarse con conocimiento experto humano.

##  Estructura del repositorio

     PROYECTO-INTRO-IA-/
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