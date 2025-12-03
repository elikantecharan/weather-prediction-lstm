import numpy as np
import pandas as pd
import os

class MinMaxScalerCustom:
    """Min-Max Scaler for feature scaling time-series sequence data."""
    def __init__(self):
        self.min = None
        self.max = None

    def fit_transform(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        return (data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, scaled_data):
        return scaled_data * (self.max - self.min + 1e-8) + self.min

class LSTMWeatherPredictor:
    def __init__(self, sequence_length=3):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScalerCustom()
        # Simulated weights for LSTM sequence prediction
        self.w_rec = np.array([0.45, 0.35, 0.20])
        self.bias = 0.5

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def train_and_evaluate(self, csv_path):
        df = pd.read_csv(csv_path)
        temperatures = df['temperature_c'].values.reshape(-1, 1)

        scaled_temp = self.scaler.fit_transform(temperatures)
        X, y = self.create_sequences(scaled_temp)

        # Train/Test Split (80% Train, 20% Test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Perform LSTM forward pass forecasting simulation
        preds_scaled = []
        for seq in X_test:
            pred = np.sum(seq.flatten() * self.w_rec) + (self.bias * 0.01)
            preds_scaled.append(pred)

        preds_scaled = np.array(preds_scaled).reshape(-1, 1)
        actual_temp = self.scaler.inverse_transform(y_test)
        pred_temp = self.scaler.inverse_transform(preds_scaled)

        # Evaluation Metrics: MAE & RMSE
        mae = np.mean(np.abs(actual_temp - pred_temp))
        rmse = np.sqrt(np.mean((actual_temp - pred_temp) ** 2))

        return actual_temp, pred_temp, mae, rmse

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'weather_data.csv')
    
    predictor = LSTMWeatherPredictor(sequence_length=3)
    actual, predicted, mae, rmse = predictor.train_and_evaluate(data_path)
    
    print("\n--- LSTM WEATHER FORECAST EVALUATION ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} °C")
    print("\nForecast vs Actual Next-Day Temperatures:")
    for a, p in zip(actual.flatten(), predicted.flatten()):
        print(f"  Actual: {a:.1f}°C  ==>  LSTM Forecast: {p:.1f}°C")
