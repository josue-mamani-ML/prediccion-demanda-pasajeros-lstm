import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Leer el archivo CSV
data = pd.read_csv("Demanda.csv")

# Preprocesamiento inicial
columns_to_drop = ['Unnamed: 10', 'Parada', 'Eventos inesperados']
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1)
data.columns = data.columns.str.strip()
data['Fecha'] = pd.to_datetime(data['Fecha'])
data['Hora'] = pd.to_timedelta(data['Hora'].astype(str).str.replace(":", "."), errors='coerce')
data['Dia de la semana'] = data['Dia de la semana'].str.strip().replace("Miercoles", "Miércoles")

# Crear variable de días feriados
feriados = ['2024-05-01', '2024-06-21', '2024-07-16']
data['Feriado'] = data['Fecha'].dt.strftime('%Y-%m-%d').isin(feriados).astype(int)

# Seleccionar variables predictoras y objetivo
categorical_features = ['Dia de la semana', 'Clima']
numerical_features = ['Flujo de personas', 'Flujo de choferes', 'Eventos especiales']
y = data['Numero de pasajeros']
X = data[numerical_features + categorical_features]

# Preprocesamiento con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Dividir los datos
data_split = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = data_split

# Crear el Pipeline para Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Crear el Pipeline para Regresión Lineal
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Entrenar ambos modelos
rf_pipeline.fit(X_train, y_train)
lr_pipeline.fit(X_train, y_train)

# Hacer predicciones
rf_pred = rf_pipeline.predict(X_test)
lr_pred = lr_pipeline.predict(X_test)

# Evaluar los modelos
metrics = {
    "Random Forest": {
        "MSE": mean_squared_error(y_test, rf_pred),
        "MAE": mean_absolute_error(y_test, rf_pred),
        "R2": r2_score(y_test, rf_pred)
    },
    "Linear Regression": {
        "MSE": mean_squared_error(y_test, lr_pred),
        "MAE": mean_absolute_error(y_test, lr_pred),
        "R2": r2_score(y_test, lr_pred)
    }
}

# Mostrar resultados
for model, result in metrics.items():
    print(f"\n{model}:")
    print(f"  Mean Squared Error (MSE): {result['MSE']:.2f}")
    print(f"  Mean Absolute Error (MAE): {result['MAE']:.2f}")
    print(f"  R² Score: {result['R2']:.2f}")

# Graficar resultados
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Valores reales', color='blue', linestyle='-', linewidth=2)
plt.plot(rf_pred, label='Random Forest', linestyle='--', linewidth=2, color='orange')
plt.plot(lr_pred, label='Linear Regression', linestyle='--', linewidth=2, color='green')
plt.title('Comparación de predicciones: Random Forest vs Regresión Lineal')
plt.xlabel('Índice')
plt.ylabel('Número de pasajeros')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
