# XGBoost

XGBoost es el modelo seleccionado para abordar esta tarea de clasificación. Ahora es el momento de comprender el modelo y encontrar los mejores hiperparámetros.

## **Hyperparameter Tuning**

Primero, se optimiza el modelo para mejorar su rendimiento. Este modelo ajustado funciona mejor que el original, pero aún presenta dificultades para clasificar las clases minoritarias, como la clase 4, que corresponde al defecto "suciedad" y tiene muy pocas observaciones en el conjunto de datos. Para solucionar este problema, es necesario aplicar una técnica de muestreo. Dado que las observaciones son escasas, decidí utilizar la técnica **SMOTE** para mejorar la clasificación.

```python
# Define the model
model = xgb.XGBClassifier(random_state = 42)

# Define the hyperparameter grid
param_grid = {"max_depth": [3, 5, 7],
              "learning_rate": [0.01, 0.1, 0.2],
              "n_estimators": [100, 200, 300],
              "subsample": [0.6, 0.8, 1.0]}

# Set up Grid Search
grid_search = GridSearchCV(estimator = model, 
                           param_grid = param_grid,
                           scoring = "f1_macro",
                           cv = 3,
                           verbose = 1,
                           n_jobs = -1)

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

## **SMOTE**

SMOTE es una técnica de re-muestreo que crea nuevas muestras ajustando ligeramente los datos existentes hacia sus vecinos. Funciona eligiendo aleatoriamente una muestra de la clase minoritaria, identificando sus k vecinos más cercanos y generando nuevos puntos de datos al escalar la distancia hacia esos vecinos. De esta manera, se mantiene la integridad de la clase minoritaria y se enriquece el conjunto de datos.

Una vez dividido el conjunto de datos en entrenamiento y prueba, aplicamos SMOTE **solamente** al conjunto de entrenamiento. Esto asegura que el conjunto de prueba siga siendo una representación fiel de la distribución original de los datos y previene cualquier filtración de información desde el conjunto de entrenamiento. Así, garantizamos que el proceso de evaluación sea más sólido y confiable.

```python
# Apply SMOTE to the training data
smote = SMOTE(random_state = 42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Fit the XGBoost model on the resampled data
xgb_smote_model = XGBClassifier(random_state = 42)
xgb_smote_model.fit(X_resampled, y_resampled)

# Evaluate the model on the test set
xgb_smote_pred = xgb_smote_model.predict(X_test_scaled)
print(classification_report(y_test, xgb_smote_pred))
```

El modelo XGBoost con SMOTE se destaca claramente como el mejor en comparación con el modelo XGBoost ajustado, logrando una precisión del 80% y un F1-score macro promedio de 0.80. Aunque la precisión de ambos modelos es similar, uno de los puntos fuertes de este modelo es la capacidad para identificar eficazmente instancias de clases minoritarias, como la clase 4, donde logró un valor de recall del 88%, mientras que el modelo original alcanzó 62% y el ajustado solo alcanzó el 50%.

Además, el modelo SMOTE muestra un rendimiento constante en las clases mayoritarias, lo que demuestra su fiabilidad y un rendimiento bastante sólido en general. Esto lo convierte en la mejor opción para aplicaciones del mundo real, especialmente en conjuntos de datos desbalanceados donde cada clase es importante.

```python
# Save the trained model as pickle format
with open("../model/xgboost_model.pkl", "wb") as file:
    pickle.dump(xgb_model, file)
# Save the trained model as a json file
xgb_model.save_model("../model/xgboost_model.json")
```

## **Matriz de Confusión**

Analizar la matriz de confusión nos ayuda a comprender las clasificaciones incorrectas y a identificar áreas potenciales de mejora. 

```python
# Plot XGBoost Confusion Matrix
xgb_smote_model_cm = confusion_matrix(y_test, xgb_predictions)
ConfusionMatrixDisplay(confusion_matrix = xgb_smote_model_cm).plot(cmap = "PuBu")
plt.title("Matriz de confusión del modelo XGBoost")
plt.xlabel("Predicción")
plt.ylabel("Valor real")
```

## **Importancia de las características**

Identificar las características clave que impulsan las predicciones es fundamental para comprender cómo el modelo toma decisiones.

```python
# Plot feature importance
xgboost_feature_importance = plt.figure(figsize = (15, 3))
xgb.plot_importance(xgb_smote_model,
                    importance_type = "weight",
                    color = "#74add1",
                    title = "Importancia de características de XGBoost",
                    xlabel = "Peso",
                    ylabel = "Características")
```

## **ROC AUC**

Evalúa el rendimiento del modelo de manera integral.

```python
# Binarize the output
y_bin = label_binarize(y_test, classes = [0, 1, 2, 3, 4, 5, 6])
y_score = xgb_smote_model.predict_proba(X_test_scaled)

# Compute ROC AUC for each class
roc_auc = roc_auc_score(y_bin, y_score, average = "macro")
print(f"XGBoost Model ROC AUC Score: {roc_auc:.2f}")
# Binarize the output
y_bin = label_binarize(y_test, classes = [0, 1, 2, 3, 4, 5, 6])
n_classes = y_bin.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], xgb_smote_model.predict_proba(X_test_scaled)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the curves
plt.figure(figsize = (11, 4))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color = colors[i], label = "ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva Característica de Operación del Receptor (ROC)")
plt.legend(loc = "lower right")
```

## **Curvas de Aprendizaje**

Evalúa el comportamiento del entrenamiento del modelo y detecta problemas de sesgo.

```python
# Plot the learning curves of the model
plt.figure(figsize = (11, 4))

train_sizes, train_scores, test_scores = learning_curve(xgb_smote_model, X_resampled, y_resampled, cv = 5, n_jobs = -1, train_sizes = np.linspace(0.1, 1.0, 10))
train_scores_mean = train_scores.mean(axis = 1)
test_scores_mean = test_scores.mean(axis = 1)

plt.plot(train_sizes, train_scores_mean, color = "#74add1", label = "Puntuación de entrenamiento")
plt.plot(train_sizes, test_scores_mean, color = "#f46d43", label = "Puntuación de validación cruzada")
plt.title("Curva de aprendizaje del modelo XGBoost")
plt.xlabel("Tamaño de entrenamiento")
plt.ylabel("Puntuación")
plt.legend(loc = "best")
```

## **Curva de Precisión-Recall**

Evalúa el rendimiento del modelo en términos de precisión y recuperación, especialmente útil en casos de clases desbalanceadas.

```python
# Plot the precision-recall curve
plt.figure(figsize=(11, 4))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], xgb_smote_model.predict_proba(X_test_scaled)[:, i])
    plt.plot(recall, precision, color = colors[i], label = f"Curva de Precisión-Recuperación de la clase {i}")
plt.xlabel("Recuperación")
plt.ylabel("Precisión")
plt.title("Curva de Precisión-Recuperación")
plt.legend(loc = "lower left")
```