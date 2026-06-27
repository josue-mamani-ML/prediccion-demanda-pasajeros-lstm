import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
y = data['Numero de pasajeros']
X = data[numerical_features + categorical_features]

# Preprocesamiento con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

# Dividir los datos
data_split = train_test_split(X_processed, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = data_split

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Definir el modelo de red neuronal
def create_model():
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

model = create_model()
history = model.fit(X_train, y_train_scaled, validation_data=(X_test, y_test_scaled), epochs=300, batch_size=16, verbose=1)

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

# Calcular métricas
y_pred = y_pred.flatten()
mse = mean_squared_error(y_test_actual, y_pred)
mae = mean_absolute_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print(f"Neural Network - Mean Squared Error (MSE): {mse:.2f}")
print(f"Neural Network - Mean Absolute Error (MAE): {mae:.2f}")
print(f"Neural Network - R² Score: {r2:.2f}")

# Graficar resultados
plt.figure(figsize=(14, 7))
plt.plot(y_test_actual, label='Valores reales', color='blue', linestyle='-', linewidth=2)
plt.plot(y_pred, label='Predicción', linestyle='--', linewidth=2, color='orange')
plt.title('Comparación entre demanda de pasajeros reales y predichos (Red Neuronal)')
plt.xlabel('Índice')
plt.ylabel('Número de pasajeros')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Graficar la curva de pérdida
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Curva de pérdida durante el entrenamiento')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
