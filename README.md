# Test 1: Classification - Steel Plates Anomaly Detection

Este proyecto consiste en un proceso end-to-end de machine learning con el objetivo de crear un modelo capaz de clasificar el tipo de defecto o anomalía producido en placas de acero industrial. El modelo desarrollado durante estre proyecto es XGBoost al cual se le ha aplicado la metodología de re-muestreo SMOTE para abordar el desequilibrio de los datos. Finalmente, se ha conseguido un modelo robusto capaz de distinguir bien entre todas las clases y con una alta precisión. 

**Motivación**

Elegí este conjunto de datos por su relevancia en la industria 2.0 y en el control de calidad. La detección de anomalías en productos industriales es un excelente ejemplo para demostrar mis habilidades en ingeniería y ciencia de datos. Este proyecto ilustra cómo los enfoques basados en datos pueden optimizar la gestión de calidad y la toma de decisiones, siendo aplicables a diversas industrias.

## Tabla de contenidos

- [Contenido del repositorio](#contenido-del-repositorio)
- [Estructura del notebook](#estructura-del-notebook)
- [Dependencias](#dependencias)
- [Instalar dependencias](#instalar)
- [Como ejecutar el projecto](#como-ejecutar-el-proyecto)
- [Documentacion](#documentación)
- [Dataset](#dataset)
- [Modelado](#modelado)
- [Resultados](#resultados)

## Contenido del repositorio

```plaintext
VHC-TestDataScience-1/
│
├── data/                               # Conjunto de datos
│   ├── raw/                            # Datos originales
│   ├── processed/                      # Datos limpios
│   └── feature-engineering/            # Datos limpios después de feature engineering

├── docs/                               # Documentacion
|   ├── data-cleaning.md                # Limpieza
|   ├── data-exploration                # Exploracion
│   ├── feature-engineering.md          # Feature engineering
|   ├── index.md                        # Indice
|   ├── modeling.md                     # Modelado
|   ├── pre-processing.md               # Preprocesamiento
|   ├── results.md                      # Resultados
|   ├── xgboost.md                      # Modelado xgboost
│
├── models/                             # Modelos entrenados
│   ├── xgboost_model.json              # Modelo guardado en formato json
│   └── xbboost_model.pkl               # Modelo guardado en formato pkl
│
├── notebooks/                          # Jupyter notebooks
│   ├── VHC-TestDataScience-1.ipynb     # Proyecto completo de inicio a fin
│   └── VHC-TestDataScience-1.html      # Proyecto completo en formato HTML
│
├── results/                            # Resultados del proyecto
│   ├── figures/                        # Gráficas generadas
│   └── model_results.txt               # Resultados del modelo
│
├── requirements.txt                    # Dependencias del proyecto
└── README.md                           # Documentación del proyecto
```

## Estructura del notebook

1. **Introducción**: descripción del problema, motivación y objetivo del proyecto.
2. **Dataset**: descripción del conjunto de datos.
2. **Carga de Datos**: importación y exploración inicial del conjunto de datos.
3. **Limpieza de los datos**: limpieza y preprocesado inicial de los datos.
4. **Análisis exploratorio**: exploración de la variable objetivo y las características.
5. **Feature engineering**: creación de nuevas características a partir de los datos obtenidos en la exploración.
6. **Preprocesado**: procesamiento de los datos antes del modelado.
7. **Modelado**: primera aproximación al modelado con la selección del mejor algoritmo para este problema.
8. **XGBoost**: afinación y remuestro de los datos para obtener la mejor versión del modelo xgboost.

9. **Conclusiones**: Resumen del proyecto y los resultados obtenidos.

## Dependencias

Todas las dependencias necesarias para este proyecto están listadas en el fichero **requirements.txt**, entre las que se encuentran: 

- Python
- Jupyter Notebooks
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Instalar dependencias

Para ejecutar este proyecto, asegúrate de tener instaladas las siguientes librerías de Python:

```{shell}
pip install -r requirements.txt
```

## Como ejecutar el proyecto

1. Clona este repositorio:
   ```{shell}
   git clone https://github.com/herrerovir/VHC-TestDataScience-1.git
   ```
2. Navega al directorio del proyecto:
   ```{shell}
   cd VHC-TestDataScience-1
   ```
3. Abre el notebook:
   ```{shell}
   jupyter notebook

## Documentacion

Toda la documentación del proyecto en mkdocs se encuentra alojada en la siguiente dirección [documentación](https://herrerovir.github.io/VHC-TestDataScience-1/)

## Dataset

El conjunto de datos utilizado en este proyecto se obtuvo de la página web Kaggle [here](https://www.kaggle.com/datasets/uciml/faulty-steel-plates). Este dataset consiste en 1941 entradas y 34 columnas. 

## Modelado

El primer paso del proceso de modelado fue encontrar el algoritmo que mejor rendimiento mostraba para resolver este problema de clasificación. Los algoritmos utilizados fueron: **Decision Tree**, **Random Forest**, **XGBoost**, **Support Vector Machine** y **Multilayer Perceptron**. Tras comparar estos modelos se concluyó que el algoritmo **xgboost** es el que mejores métricas presentaba. De esta manera, el siguiente paso consitió en afinar el modelo tuneando sus hiperparámetros. Además, como durante el análisis exploratorio se detectó que el conjunto de datos está desequilibrado, se aplicó al modelo xgboost la metodología de re-muestreo SMOTE. De esta manera, se obtuvo un modelo robusto, capaz de distinguir bien entre todas las clases, incluida las minoritarias, y alcanzar una buenas precisión y rendimiento. 

## Resultados

A continuación se muestran las gráficas que recogen y muestran las metricas y resultados del modelo xgboost:

**Matriz de confusión**

![Xgboost-confusion-matrix](https://github.com/user-attachments/assets/4eae48dd-bb1a-4fc7-bf32-230b7f421d97)

**Importancia de las características**

![Xgboost-feature-importance](https://github.com/user-attachments/assets/674d7f58-fcc8-4be3-896d-17599a970e4a)

**Curvas de aprendizaje**

![Xgboost-learning-curve](https://github.com/user-attachments/assets/cb3682c6-dac3-4054-94ec-d2a8ca4e29df)

**Curva Precision-recall**

![Xgboost-precison-recall-curve](https://github.com/user-attachments/assets/122efac1-fa58-46df-b1c5-575c3b57ce88)

**Curva ROC**

![Xgboost-roc-curve](https://github.com/user-attachments/assets/21939e39-b2ea-4742-a8de-12b3a9a945f5)
