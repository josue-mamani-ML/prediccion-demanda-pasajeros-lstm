import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle

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

# Crear el Pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Dividir los datos
data_split = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = data_split

# Entrenar el modelo
rf_pipeline.fit(X_train, y_train)

# Hacer predicciones
y_pred = rf_pipeline.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Random Forest - Mean Squared Error (MSE): {mse:.2f}")
print(f"Random Forest - Mean Absolute Error (MAE): {mae:.2f}")
print(f"Random Forest - R² Score: {r2:.2f}")

# Guardar el modelo entrenado
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_pipeline, f)

# Cargar el modelo entrenado
with open('random_forest_model.pkl', 'rb') as f:
    trained_model = pickle.load(f)

# Crear el conjunto de datos para el 29/05/2024 (miércoles)
date_future = pd.date_range(start='2024-05-29 06:00:00', end='2024-05-29 23:59:00', freq='15T')  # Cada 15 minutos
future_data = pd.DataFrame({
    'Fecha': date_future,
    'Hora': date_future.time,
    'Flujo de personas': np.random.randint(5, 30, size=len(date_future)),  # Valores estimados
    'Flujo de choferes': np.random.randint(3, 15, size=len(date_future)),  # Valores estimados
    'Eventos especiales': [0] * len(date_future),  # Suponemos que no hay eventos especiales
    'Dia de la semana': ['Wednesday'] * len(date_future),  # Día fijo (en inglés)
    'Clima': ['sol'] * len(date_future)  # Cambia 'soleado' por una categoría válida, por ejemplo, 'sol'
})

# Convertir días de la semana a español
future_data['Dia de la semana'] = future_data['Dia de la semana'].replace({
    'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
    'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
})

# Preprocesar las variables categóricas
future_data['Dia de la semana'] = future_data['Dia de la semana'].str.strip()
future_data['Clima'] = future_data['Clima'].str.strip()

# Realizar predicciones
predicted_passengers = trained_model.predict(future_data[numerical_features + categorical_features])

# Agregar las predicciones al DataFrame futuro
future_data['Prediccion Numero de pasajeros'] = predicted_passengers

# Mostrar las predicciones para el 29/05/2024
print(future_data[['Fecha', 'Hora', 'Prediccion Numero de pasajeros']])

# Graficar las predicciones a lo largo del día con mejor formato de horas
plt.figure(figsize=(14, 7))
plt.plot(future_data['Fecha'], future_data['Prediccion Numero de pasajeros'], marker='o', linestyle='-', color='blue', label='Predicción de pasajeros')

# Mejorar el formato del eje X
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Mostrar solo horas y minutos
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))   # Etiquetas cada hora
plt.gcf().autofmt_xdate()  # Rotar etiquetas automáticamente para mayor claridad

plt.title('Predicción de pasajeros para el 29/05/2024')
plt.xlabel('Hora del día')
plt.ylabel('Número de pasajeros')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
