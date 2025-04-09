# Data cleaning

Limpia y preprocesa el conjunto de datos antes de seguir con el análisis.

```python
df.info()
```

* **Renombrar columnas**

Se renombraron algunas columnas para mejorar la legibilidad y la comprensión del conjunto de datos.

```python
df.columns
df = df.rename(columns = {"TypeOfSteel_A300" : "Steel_Type_A300",
                          "TypeOfSteel_A400" : "Steel_Type_A400",
                          "LogOfAreas" : "Log_of_Areas",
                          "SigmoidOfAreas" : "Sigmoid_of_Areas",
                          "K_Scatch" : "K_Scratch"})
df.head()
```

* **Tipos de datos**

Verifica que todas las columnas tengan los tipos de datos apropiados.

```python
df.dtypes
```

* **Valores nulos**

Identifica y elimina cualquier valor nulo en el conjunto de datos cuando sea necesario.

```python
# Check the total of null values in each column
df.isna().sum()
```
No hay valores nulos en el dataset.

* **Valores duplicados**

Verifica si hay valores duplicados en el conjunto de datos.

```python
df.duplicated().sum()
```

No hay ninguna entrada duplicada en el dataset.

* **Outliers**

Revisa el resumen estadístico del conjunto de datos para detectar posibles valores atípicos. Esta evaluación inicial permitirá identificar cualquier valor inusual que necesite un análisis más detallado.

```python
# Changes float format to display two decimals
pd.set_option("display.float_format", "{:.2f}".format)
df.describe().T
```
A primera vista, columnas como **X_Minimum**, **X_Maximum**, **Y_Minimum**, **Y_Maximum**, **Pixel_Areas**, **X_Perimeter**, **Y_Perimeter**, **Sum_of_Luminosity** y **Steel_plate_thickness** pueden contener valores atípicos. Esta conclusión se basa en la observación de que los valores máximos superan tanto la media, la mediana, y el tercerl cuartil, lo que puede indicar la presencia de outliers. Dado que estos pueden ser claros indicadores de defectos en las placas de acero, he decidido mantener los valores atípicos para no reducir las observaciones del conjunto de datos y contribuir a una mejor predicción del modelo.

* **El conjunto de datos limpio:**

```python
defect_detection = df.copy()
defect_detection.head()

# Save the cleaned dataset
defect_detection.to_csv("../data/processed/Steel-plates-faults-cleaned-dataset.csv")
```