import os
from src.lstm_model import LSTMWeatherPredictor

def main():
    print("=" * 65)
    print("     WEATHER PREDICTION SYSTEM USING LSTM NEURAL NETWORKS")
    print("=" * 65)

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'weather_data.csv')
    if not os.path.exists(data_path):
        print(f"Error: Weather dataset not found at {data_path}")
        return

    print("[+] Loading time-series daily temperature dataset...")
    print("[+] Applying MinMaxScaler normalization & sliding sequence windows...")
    
    predictor = LSTMWeatherPredictor(sequence_length=3)
    actual, predicted, mae, rmse = predictor.train_and_evaluate(data_path)

    print("\n📊 MODEL EVALUATION RESULTS:")
    print(f"   • Mean Absolute Error (MAE) : {mae:.2f} °C")
    print(f"   • Root Mean Squared Error (RMSE): {rmse:.2f} °C")
    print(f"   • Prediction Accuracy       : {100 - (mae / 20.0 * 100):.1f}%\n")

    print("🌤️  TEST SEQUENCE FORECAST COMPARISON:")
    print("-" * 50)
    print(f"   {'Day Step':<10} | {'Actual Temp (°C)':<18} | {'LSTM Forecast (°C)':<18}")
    print("-" * 50)
    for i, (a, p) in enumerate(zip(actual.flatten(), predicted.flatten()), 1):
        print(f"   Day {i:<6} | {a:<18.1f} | {p:<18.1f}")
    print("-" * 50)

    print("\n✅ Time-Series Forecast Completed Successfully!")

if __name__ == "__main__":
    main()
