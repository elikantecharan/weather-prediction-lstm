# Weather Forecasting Using LSTM

[![Major Project](https://img.shields.io/badge/B.Tech-Major_Project_Report-0ea5e9.svg)](https://github.com/elikantecharan/weather-prediction-lstm)
[![Department](https://img.shields.io/badge/Department-AI_%26_ML-purple.svg)](https://github.com/elikantecharan/weather-prediction-lstm)
[![Live Web App](https://img.shields.io/badge/Live-GitHub_Pages-brightgreen.svg)](https://elikantecharan.github.io/weather-prediction-lstm/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask-black.svg)](https://flask.palletsprojects.com/)

A Deep Learning time-series weather forecasting application using **Long Short-Term Memory (LSTM)** neural networks, developed as a Major Project in the **Department of Artificial Intelligence & Machine Learning** at **J.B Institute of Engineering & Technology** (2025–2026).

---

## 👥 Major Project Team & Mentors
- **Team Members**:
  - **E. CHARAN** (`22671A7312`)
  - M. VISHNUVARDHAN (`22671A7333`)
  - P. VIGNESH GOUD (`22671A7341`)
  - M. TRISHA (`22671A7336`)
- **Guide & Mentor**: Mr. Md. Mahebub Ali (*Assistant Professor*)
- **Head of Department**: Dr. Gurrampally Kumar (*Associate Professor & HOD*)
- **Institution**: J.B Institute of Engineering & Technology (UGC Autonomous), Hyderabad

---

## 🌟 Key System Features
- **Deep Learning LSTM Topology**: Stacked LSTM architecture (`LSTM 64 -> Dropout 0.2 -> LSTM 32 -> Dense 16 -> Dense 5`) with sequence windows (`SEQ_LEN = 3`).
- **Multi-Feature Time-Series Data**: Normalizes and predicts 5 key parameters simultaneously:
  1. Temperature (°C)
  2. Humidity (%)
  3. Wind Speed (km/h)
  4. Atmospheric Pressure (hPa)
  5. Precipitation (mm)
- **Interactive Web Interface**: Sky/slate modern dashboard with Chart.js visualization, 1-Day, 3-Day, and 7-Day forecasting, and fallback simulation mode.
- **RESTful Flask Backend API**: `POST /predict` accepts user atmospheric parameters and returns multi-day time-series forecasts and rain probability.

---

## 🛠️ Project Structure
```
weather-prediction-lstm/
├── app.py                 # Flask RESTful API backend server
├── train_model.py         # Keras/TensorFlow LSTM model training pipeline
├── data/
│   └── weather_data.csv   # Historical time-series weather dataset
├── index.html             # Frontend web dashboard layout
├── style.css              # Sky/slate modern UI styling
├── scripts.js             # Frontend API client, fallback engine & Chart.js renderer
└── README.md              # Project documentation
```

---

## 🚀 Live Demo & Quick Start

- **Live Web App**: [https://elikantecharan.github.io/weather-prediction-lstm/](https://elikantecharan.github.io/weather-prediction-lstm/)

### 1. Run Backend Server (Optional):
```bash
pip install flask flask-cors pandas numpy scikit-learn tensorflow
python app.py
```

### 2. Run Local Frontend:
Open `index.html` in your browser!

---

## 📊 System Evaluation Metrics
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
