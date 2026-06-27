import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Cargar el modelo entrenado
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Función para generar datos futuros
def generate_future_data():
    # Obtener el día actual
    current_date = datetime.now()
    feriados = ['2024-05-01', '2024-06-21', '2024-07-16']
    is_holiday = current_date.strftime('%Y-%m-%d') in feriados

    # Crear el rango de tiempo para el día actual
    date_range = pd.date_range(
        start=current_date.replace(hour=6, minute=0, second=0, microsecond=0),
        end=current_date.replace(hour=23, minute=45, second=0, microsecond=0),
        freq='15T'
    )

    # Generar datos futuros
    future_data = pd.DataFrame({
        'Fecha': date_range,
        'Hora': date_range.time,
        'Flujo de personas': np.random.randint(5, 30, size=len(date_range)),  # Valores estimados
        'Flujo de choferes': np.random.randint(3, 15, size=len(date_range)),  # Valores estimados
        'Eventos especiales': [0] * len(date_range),  # Suponemos que no hay eventos especiales
        'Dia de la semana': [current_date.strftime('%A')] * len(date_range),
        'Clima': ['soleado'] * len(date_range),  # Cambia si tienes datos reales del clima
        'Feriado': [int(is_holiday)] * len(date_range)
    })

    return future_data

def predict_demand(data):
    # Convertir los datos al formato correcto
    features = data[['Flujo de personas', 'Flujo de choferes', 'Eventos especiales', 'Dia de la semana', 'Clima']]
    predictions = model.predict(features)
    data['Prediccion Numero de pasajeros'] = predictions
    return data

if __name__ == "__main__":
    # Generar datos futuros y predecir demanda
    future_data = generate_future_data()
    predicted_data = predict_demand(future_data)

    # Convertir a JSON y enviar como respuesta
    print(predicted_data[['Fecha', 'Hora', 'Prediccion Numero de pasajeros']].to_json(orient='records'))
