# Sistema de Predicción de Demanda de Pasajeros — Trans Bolognia

> Undergraduate Thesis · Universidad Privada La Salle · La Paz, Bolivia · 2024

End-to-end Machine Learning system to predict passenger demand for public transport in La Paz, Bolivia, with 15-minute granularity.

---

## 🎯 Problem

The Trans Bolognia transport association managed passenger demand manually, with no data-driven tools to anticipate peak hours, optimize routes, or plan resources. This project built a complete ML pipeline to solve that.

---

## 🤖 ML Pipeline (CRISP-DM)

### Dataset
- **Source:** Field data collected via interviews (15 drivers) and surveys (40 operators)
- **Size:** Daily records from April 2024, 15-minute intervals (6:00–23:45)
- **Features:** Date, time, bus stop, passenger count, weather, day of week, holiday flag

### Models Compared

| Model | Status | Notes |
|-------|--------|-------|
| Linear Regression | Trained | Baseline |
| Ridge Regression | Trained & saved (.pkl) | Regularized baseline |
| Random Forest | **Deployed in production** | Best overall performance |
| Dense Neural Network | Trained | Architecture: 256→128→64→32→1, 300 epochs |
| LSTM | Trained & saved (.h5) | MAE: 2.57 passengers |

### Results
- Best model deployed: Random Forest (production API)
- Best sequential model: LSTM — MAE: 2.57 passengers
- Serialized: random_forest_model.pkl, model_lstm.h5, ridge_model.pkl, scaler.pkl

---

## 🏛 Full System Stack

| Layer | Technology |
|-------|-----------|
| ML & Data | Python · Pandas · NumPy · Scikit-learn · TensorFlow · Keras |
| Backend API | Node.js · Express · Firebase Realtime Database |
| Mobile app | React Native 0.74 · Expo · Firebase |
| Admin panel | React · Firebase |
| Sales API | NestJS · Prisma · TypeScript |

---

## 📁 Repository Structure

- Demanda de pasajeros/ — ML core: dataset, notebooks, trained models
- Machine learning bolognia/ — Model experiments and production scripts
- Redes neuronales bolognia/ — LSTM and RNN experiments
- Interfaz de usuario/ — Architecture diagrams, mockups, UML
- appbolognia/ — React Native mobile app

---

## 🚀 How to Run

```bash
pip install pandas numpy scikit-learn tensorflow keras matplotlib jupyter
jupyter notebook "Demanda de pasajeros/AnalisiDatos.ipynb"
python "Machine learning bolognia/predict_demand.py"
```

---

## 👤 Author

**Josue Abraham Mamani Huanca**
Systems Engineer · ML Engineer
Universidad Privada La Salle · La Paz, Bolivia

LinkedIn: https://www.linkedin.com/in/josue-a-mamani-h
GitHub: https://github.com/josue-mamani-ML
