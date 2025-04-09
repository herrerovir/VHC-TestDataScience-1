# Exploratory Data Analysis

En esta sección, se realiza un análisis exploratorio de datos en profundidad.

## **Target Variable**

El primer paso es examinar la variable objetivo para obtener una visión general de su distribución.

El objetivo de este proyecto es predecir fallos en las placas de acero. Este conjunto de datos contiene 7 columnas que representan los 7 posibles fallos. Estas 7 columnas son la variable objetivo de este problema.

```python
# Plot distribution of all target columns on the datasest
fig, ax = plt.subplots(2, 4, figsize = (20, 8))
ax = ax.flatten()
for i, col in enumerate(defect_detection.columns[-7:]):
    sns.histplot(defect_detection[col], ax = ax[i], color = "#74add1")
Verifica si hay observaciones sin fallos.
# List of fault columns
faults = ["Pastry", "Z_Scratch", "K_Scratch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]

# Get the observations with no faults
no_faults = defect_detection[defect_detection[faults].sum(axis = 1) == 0]

# Display the results
if no_faults.empty:
    print("There are no observations with no faults.")
else:
    print("Number of observations with no faults:")
    print(no_faults)
```

De este análisis rápido se puede concluir que no hay observaciones sin fallos. Ahora verifica si hay fallos que ocurren simultáneamente en una misma observación.

```python
# Sum defect values per row
defect_detection[faults].sum(axis = 1).value_counts()
```

Se confirma que todas las observaciones contienen solo un tipo de fallo. Por lo tanto, este problema se mantiene como una clasificación multicategórica en lugar de una clasificación multietiqueta. Lo siguiente será echar un vistazlo a la distribución de todos los fallos para tener una visión clara de cuántas observaciones por fallo hay.

```python
sum_faults = defect_detection[faults].sum().sort_values(ascending=True)
sum_faults
```
La distribución de tipos de fallos en el acero muestra que el fallo más común es el de **other faults**, seguido por **bumps** y **k_scratch**. El fallo menos frecuente en el conjunto de datos es **dirtiness** (suciedad).

El conjunto de datos presenta un claro desequilibrio, ya que el número de instancias entre las diferentes clases de fallos varía ampliamente. Durante la etapa de modelado, será necesario abordar este problema. Algunas técnicas utilizadas para manejar conjuntos de datos desequilibrados incluyen: re-muestreo, el uso de algoritmos robustos contra el desequilibrio de clases, o aplicar métricas que consideren la distribución de clases, como el F1-score y la curva de precisión-recall.

```python
fault_types_distribution_graph = plt.figure(figsize = (10, 4))
ax = sns.barplot(sum_faults, color = "#74add1")
for i in ax.containers:
    ax.bar_label(i,)
plt.xlabel("Tipo de defecto")
plt.ylabel("Frecuencia")
plt.title("Distribución de los tipos de defectos en las placas de acero", size = 12)
fault_types_distribution_pie = plt.figure(figsize = (11, 5))
colors = ["#d73027", "#f46d43", "#fdae61", "#fee090", "#abd9e9", "#74add1", "#4575b4"]
plt.pie(sum_faults, labels = sum_faults.index, startangle = 90, autopct = "%1.0f%%", shadow = True, colors = colors)
plt.axis("equal")
plt.legend()
plt.title("Distribución de los tipos de defectos en las placas de acero", pad = 15)
```

## **Features**

El siguiente paso es examinar cada característica del conjunto de datos para obtener una visión general de sus distribuciones. Dado que todas están en diferentes escalas, será necesario escalar las características en los próximos pasos.

Los histogramas revelan información valiosa sobre la distribución de las características.

- **Distribución normal**: Maximum_of_Luminosity, Empty_Index, Square_Index, Luminosity_Index, Minimum_of_Luminosity y Orientation_Index.
- **Distribución sesgada**: Y_Minimum, Y_Maximum, Pixels_Areas, X_Perimeter, Y_Perimeter, Sum_of_Luminosity, Log_Y_Index y Edges_X_Index.
- **Distribución uniforme**: X_Minimum, X_Maximum, EdgesIndex, Edges_Y_Index y SigmoidOfAreas.

```python
# Plot distribution of all features on the datasest
fig, ax = plt.subplots(9, 3, figsize = (18, 32))
for i, col in enumerate(defect_detection.columns[:27]):
    sns.histplot(defect_detection[col], ax = ax[i//3][i%3], color = "#74add1")
```

## **Matriz de correlación**
```python
correlations = defect_detection.corr()
correlations
correlation_heatmap_graph = plt.figure(figsize = (10, 8))
sns.heatmap(correlations, linewidths = 0.5, cmap = "RdYlBu")
plt.title("Correlation Heatmap", size = 12)
```

La matriz de correlación muestra las relaciones entre variables: el color rojo indica que la relación es negativa, el azul que la relación es positiva y el color amarillo que la relación es baja o nula. Puesto que hay un alto número de características correlacionadas es necesario crear nuevas características a partir de éstas para reducir el número total de características en el dataset y facilitar el modelado.