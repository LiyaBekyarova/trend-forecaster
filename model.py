import os
from io import BytesIO
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO


def forecast_with_gru(df: pd.DataFrame, keyword: str, days_ahead: int = 180) -> tuple[pd.DataFrame, dict, list]:
    if keyword not in df.columns:
        keyword = df.columns[-1]

    series = df[["date", keyword]].dropna().copy()
    series.set_index("date", inplace=True)
    values = series.values

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    # Create sequences
    X, y = [], []
    n_input = 30
    for i in range(n_input, len(scaled_values)):
        X.append(scaled_values[i - n_input:i, 0])
        y.append(scaled_values[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train-validation split (80/20)
    split_index = int(0.6 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    # Build model
    model = Sequential([
        GRU(32, return_sequences=False, input_shape=(n_input, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(X_train, y_train, epochs=1500, verbose=0, validation_data=(X_val, y_val))

    # Evaluation
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_val = mean_squared_error(y_val, y_val_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)

    # Add R² score calculation
    r2_train = r2_score(y_train, y_train_pred)
    r2_val = r2_score(y_val, y_val_pred)

    metrics = {
        "Train MSE": float(mse_train),
        "Train MAE": float(mae_train),
        "Train R²": float(r2_train),
        "Validation MSE": float(mse_val),
        "Validation MAE": float(mae_val),
        "Validation R²": float(r2_val)
    }

    # Forecast future
    forecast = []
    input_seq = scaled_values[-n_input:].reshape(1, n_input, 1)
    for _ in range(days_ahead):
        pred = model.predict(input_seq, verbose=0)[0, 0]
        forecast.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    last_date = series.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)

    forecast_df = pd.DataFrame({"date": forecast_dates, "value": forecast})
    history_df = series.reset_index()
    history_df.columns = ["date", "value"]
    forecast_df["type"] = "Forecast"
    history_df["type"] = "History"

    return pd.concat([history_df, forecast_df]), metrics, history.history['loss']

def export_forecast_to_csv(df: pd.DataFrame, filename="forecast.csv"):
    """Позволява export към CSV файл чрез Streamlit."""
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download forecast as CSV",
        data=csv,
        file_name=filename,
        mime='text/csv'
    )
