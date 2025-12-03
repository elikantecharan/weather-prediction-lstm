# train_model.py - LSTM Training Pipeline
import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TARGET_COLS = ['tempC', 'humidity', 'windspeedKmph', 'pressure', 'precipMM']
SEQ_LEN = 3

def load_data():
    csv_files = glob.glob('data/*.csv') + glob.glob('*.csv')
    df_list = []
    for f in csv_files:
        print(f"Loading {f}...")
        try:
            df = pd.read_csv(f)
            missing = [c for c in TARGET_COLS if c not in df.columns]
            if not missing:
                if 'date_time' in df.columns:
                    df['date_time'] = pd.to_datetime(df['date_time'])
                    df.set_index('date_time', inplace=True)
                    df_daily = df[TARGET_COLS].resample('D').mean().dropna()
                    df_list.append(df_daily)
                else:
                    df_list.append(df[TARGET_COLS])
            else:
                print(f"Skipping {f} - missing columns: {missing}")
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not df_list:
        raise ValueError("No valid data found")
    return pd.concat(df_list)

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:(i + seq_len)])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

def main():
    print("Loading datasets...")
    df_all = load_data()
    print(f"Total daily records: {len(df_all)}")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_all)

    X, y = create_sequences(scaled_data, SEQ_LEN)

    # Train-test split (80-20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("Building LSTM Deep Learning Model...")
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(TARGET_COLS))),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(len(TARGET_COLS))
        ])

        model.compile(optimizer='adam', loss='mse')

        print("Training model...")
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=15,
            batch_size=32,
            verbose=1
        )

        print("Saving model and scaler...")
        model.save('weather_lstm.h5')
    except Exception as e:
        print(f"TensorFlow training notice: {e}")

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Training complete!")

if __name__ == '__main__':
    main()
