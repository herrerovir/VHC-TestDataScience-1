# Feature Engineering

De acuerdo con los resultados obtenidos de la matrix de correlación, se crean nuevas variables. Las variables que ya no sean necesarias se eliminarán del conjunto de datos.
```python
# Create new variables
defect_detection["X_Range"] = defect_detection["X_Maximum"] - defect_detection["X_Minimum"]
defect_detection["Y_Range"] = defect_detection["Y_Maximum"] - defect_detection["Y_Minimum"]
defect_detection["Defect_Area"] = (defect_detection["X_Perimeter"] * defect_detection["Y_Perimeter"])
defect_detection["Luminosity_Range"] = defect_detection["Maximum_of_Luminosity"] - defect_detection["Minimum_of_Luminosity"]
defect_detection["Edge"] = defect_detection["Edges_Index"] / (defect_detection["Edges_X_Index"] * defect_detection["Edges_Y_Index"])
defect_detection["Outside_X_Range"] = defect_detection["Outside_X_Index"] * defect_detection["X_Range"]
defect_detection["Log_Area"] = defect_detection["Log_of_Areas"] / (0.000001 + defect_detection["Log_X_Index"] * defect_detection ["Log_Y_Index"])
defect_detection["Luminosity_Sum_Range"] = defect_detection["Sum_of_Luminosity"] * defect_detection["Luminosity_Range"]
defect_detection["Log_Area_Sigmoid"] = defect_detection["Log_Area"] * defect_detection["Sigmoid_of_Areas"]
defect_detection.columns
```
```python
# Drop unnecessary columns
columns_to_drop = ["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "X_Perimeter", "Y_Perimeter", "Minimum_of_Luminosity", "Maximum_of_Luminosity",
                   "Outside_X_Index", "Edges_Index", "Edges_X_Index", "Edges_Y_Index","Log_of_Areas", "Log_X_Index", "Log_Y_Index", "Sum_of_Luminosity", 
                   "Luminosity_Range", "Sigmoid_of_Areas"]

defect_detection = defect_detection.drop(columns_to_drop, axis = 1)
defect_detection
```
```python
# Show columns after dropping the unnecessary ones
defect_detection.columns
```
```python
# Plot again the correlation matrix with the new features
correlations_2 = defect_detection.corr()
correlations_2
correlation_heatmap_graph_2 = plt.figure(figsize = (10, 8))
sns.heatmap(correlations_2, linewidths = 0.5, cmap = "RdYlBu")
plt.title("Correlation Heatmap", size = 12)
```
```python
# Save the feature engineered dataset
defect_detection.to_csv("../data/feature-engineering/Steel-plates-faults-feature-engineering-dataset.csv")
```