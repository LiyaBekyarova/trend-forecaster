import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

def forecast_with_gru(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """
    Обучава GRU модел и прогнозира следващите 6 месеца.
    Приема DataFrame с колона 'date' и колона с ключовата дума.
    Връща DataFrame с историческа и прогнозна част.
    """
    if keyword not in df.columns:
        keyword = df.columns[-1]  # използваме последната колона, ако няма съвпадение

    series = df[["date", keyword]].dropna().copy()
    series.set_index("date", inplace=True)
    values = series.values

    # Скалиране
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    # Подготовка на последователности
    X, y = [], []
    n_input = 30
    for i in range(n_input, len(scaled_values)):
        X.append(scaled_values[i - n_input:i, 0])
        y.append(scaled_values[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Модел
    model = Sequential([
        GRU(64, return_sequences=False, input_shape=(n_input, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=30, verbose=0)

    # Прогноза
    forecast = []
    input_seq = scaled_values[-n_input:].reshape(1, n_input, 1)
    for _ in range(180):  # 6 месеца напред
        pred = model.predict(input_seq, verbose=0)[0, 0]
        forecast.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    last_date = series.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=180)

    forecast_df = pd.DataFrame({"date": forecast_dates, "forecast": forecast})

    # Обединяване с историческите данни за визуализация
    history_df = series.reset_index()
    history_df.columns = ["date", "value"]
    forecast_df.columns = ["date", "value"]
    forecast_df["type"] = "Forecast"
    history_df["type"] = "History"

    combined = pd.concat([history_df, forecast_df])
    return combined

def export_forecast_to_csv(df: pd.DataFrame, filename="forecast.csv"):
    """Позволява export към CSV файл чрез Streamlit."""
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download forecast as CSV",
        data=csv,
        file_name=filename,
        mime='text/csv'
    )
