# Pre-processing

Antes de comenzar con el modelado, será necesario hacer unas transformaciones en el conjunto de datos.

- **Nueva columna de Fallos**

Crea una nueva columna que contenga todos los tipos de fallos. Esta columna será la variable objetivo y facilitará el modelado y la predicción.
```python
# List of faults and their corresponding values
fault_mapping = {"Pastry": 0, "Z_Scratch": 1, "K_Scratch": 2, "Stains": 3, "Dirtiness": 4, "Bumps": 5, "Other_Faults": 6}

# Initialize the Faults column
defect_detection["Faults"] = 0

# Loop through each fault and assign the corresponding value
for fault, value in fault_mapping.items():
    defect_detection.loc[defect_detection[fault] == 1, "Faults"] = value

# Display the first few rows
defect_detection.head()
```
```python
# Drop individual fault columns
defect_detection.drop(faults, axis = 1, inplace=True)
defect_detection.head()
```

- **Dividir los datos**

Al dividir los datos en un conjunto de entrenamiento y uno de prueba, se asegura que el escalado se base únicamente en los datos de entrenamiento, evitando que cualquier información del conjunto de prueba se filtren en el modelo durante el entrenamiento.
```python
# Independent features
X = defect_detection.drop("Faults", axis = 1)

# Dependent or target variable
y = defect_detection["Faults"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("Training set - X_train shape:", X_train.shape)
print("Testing set - X_test shape:", X_test.shape)
print("Training set - y_train shape:", y_train.shape)
print("Testing set - y_test shape:", y_test.shape)
```

- **Escalado de características**

El método utilizado es el de escalado estándar. 
```python
# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)
```