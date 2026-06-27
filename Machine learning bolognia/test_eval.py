import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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
target = 'Numero de pasajeros'

X = data[numerical_features + categorical_features]
y = data[target]

# Preprocesamiento con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Regresión Lineal ----------------
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)

# ---------------- Random Forest ----------------
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)

# ---------------- Redes Neuronales Densas ----------------
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

input_dim = X_train_processed.shape[1]

nn_model = Sequential([
    Dense(64, activation='relu', input_dim=input_dim),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])
nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = nn_model.fit(X_train_processed, y_train, epochs=50, batch_size=16, verbose=1, validation_split=0.2)
nn_preds = nn_model.predict(X_test_processed).flatten()

# ---------------- Métricas de evaluación ----------------
def calculate_metrics(y_true, y_pred):
    # Manejar casos donde y_true sea cero para evitar división por cero
    y_true_safe = np.where(y_true == 0, 1, y_true)
    precision = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    exactitud = 1 - (np.sum(np.abs(y_true - y_pred)) / np.sum(y_true))
    exhaustividad = np.sum(y_pred[y_pred >= np.mean(y_true)]) / np.sum(y_true)
    return precision, exactitud, exhaustividad

# Calcular métricas
lr_precision, lr_exactitud, lr_exhaustividad = calculate_metrics(y_test.values, lr_preds)
rf_precision, rf_exactitud, rf_exhaustividad = calculate_metrics(y_test.values, rf_preds)
nn_precision, nn_exactitud, nn_exhaustividad = calculate_metrics(y_test.values, nn_preds)

# Mostrar resultados
print(f"Regresión Lineal - Precisión: {lr_precision:.3f}, Exactitud: {lr_exactitud:.3f}, Exhaustividad: {lr_exhaustividad:.3f}")
print(f"Random Forest - Precisión: {rf_precision:.3f}, Exactitud: {rf_exactitud:.3f}, Exhaustividad: {rf_exhaustividad:.3f}")
print(f"Redes Neuronales - Precisión: {nn_precision:.3f}, Exactitud: {nn_exactitud:.3f}, Exhaustividad: {nn_exhaustividad:.3f}")

# ---------------- Graficar Resultados ----------------
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Real', color='blue')
plt.plot(lr_preds, label='Regresión Lineal', color='orange', linestyle='dotted')
plt.plot(rf_preds, label='Random Forest', color='green', linestyle='dashed')
plt.plot(nn_preds, label='Redes Neuronales', color='red', linestyle='dashdot')
plt.legend()
plt.title('Comparación de modelos predictivos')
plt.xlabel('Índice')
plt.ylabel('Número de Pasajeros')
plt.grid(True)
plt.show()
