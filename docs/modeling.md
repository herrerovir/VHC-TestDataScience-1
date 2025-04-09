# Modelado

El objetivo de este ejercicio es construir un modelo que clasifique los defectos de placas de acero industrial. Para ello, el primer paso es encontrar el algoritmo que mejor rendimiento presente y a partir de ahí, se afinarán los hiperparámetros del este modelo para encontrar la mejor versión del modelo ganador. Los algoritmos elegidos para comenzar el modelado son: **Decision Tree**, **Random Forest**, **XGBoost**, **Support Vector Machine** y **Multilayer Perceptron**. 

```python
# 1. Decision Tree
print("Decision Tree Model:")
dt_model = DecisionTreeClassifier(random_state = 42)
dt_model.fit(X_train_scaled, y_train)
dt_predictions = dt_model.predict(X_test_scaled)
print(classification_report(y_test, dt_predictions))
dt_cm = confusion_matrix(y_test, dt_predictions)
ConfusionMatrixDisplay(confusion_matrix = dt_cm).plot(cmap = "PuBu")
plt.title("Matriz de Confusión del modelo Decision Tree")
plt.xlabel("Predicción")
plt.ylabel("Valor real")

# 2. Random Forest
print("Random Forest Model:")
rf_model = RandomForestClassifier(random_state = 42)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
print(classification_report(y_test, rf_predictions))
rf_cm = confusion_matrix(y_test, rf_predictions)
ConfusionMatrixDisplay(confusion_matrix = rf_cm).plot(cmap = "PuBu")
plt.title("Matriz de Confusión del modelo Random Forest")
plt.xlabel("Predicción")
plt.ylabel("Valor real")

# 3. XGBoost
print("XGBoost Model:")
xgb_model = XGBClassifier(random_state = 42)
xgb_model.fit(X_train_scaled, y_train)
xgb_predictions = xgb_model.predict(X_test_scaled)
print(classification_report(y_test, xgb_predictions))
xgb_cm = confusion_matrix(y_test, xgb_predictions)
ConfusionMatrixDisplay(confusion_matrix = xgb_cm).plot(cmap = "PuBu")
plt.title("Matriz de Confusión del modelo XGBoost")
plt.xlabel("Predicción")
plt.ylabel("Valor real")

# 4. Support Vector Machine
print("Support Vector Machine Model:")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)
print(classification_report(y_test, svm_predictions))
svm_cm = confusion_matrix(y_test, svm_predictions)
ConfusionMatrixDisplay(confusion_matrix = svm_cm).plot(cmap = "PuBu")
plt.title("Matriz de Confusión del modelo Support Vector Machine")
plt.xlabel("Predicción")
plt.ylabel("Valor real")

# 5. Multilayer Perceptron
print("Multilayer Perceptron Model:")
mlp_model = MLPClassifier(random_state = 42, max_iter = 500)
mlp_model.fit(X_train_scaled, y_train)
mlp_predictions = mlp_model.predict(X_test_scaled)
print(classification_report(y_test, mlp_predictions))
mlp_cm = confusion_matrix(y_test, mlp_predictions)
ConfusionMatrixDisplay(confusion_matrix = mlp_cm).plot(cmap = "PuBu")
plt.title("Matriz de Confusión del modelo Multilayer Perceptron")
plt.xlabel("Predicción")
plt.ylabel("Valor real")
```

Entre todos los modelos probados, **XGBoost** destaca como el más efectivo, alcanzando una precisión de **0.79** y sobresaliendo tanto en precisión como en recall en varias clases. Su F1-score macro promedio de **0.78** refuerza su rendimiento equilibrado.

**Random Forest** y **Multilayer Perceptron** le siguen de cerca con una precisión de **0.76**. Ambos ofrecen resultados sólidos, pero no logran igualar la efectividad de XGBoost. Por otro lado, **Árbol de Decisión** y **Máquina de Vectores de Soporte** solo alcanzan una precisión de **0.70**, teniendo más dificultades en la clasificación de algunas clases.

**XGBoost** es la opción más acertada si el objetivo es maximizar la precisión del model y tener un rendimiento equilibrado. Además, un ajuste adicional de los hiperparámetros podría llevar su rendimiento a otro nivel.