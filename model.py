from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
from tensorflow.keras.callbacks import EarlyStopping

# === original GRU FORECAST ===
def determine_data_frequency(df):
    """Determine if data is daily or weekly and set appropriate parameters"""
    date_diff = df.index[1] - df.index[0]
    
    if date_diff.days >= 7:  # Weekly data
        if len(df) >= 260:  # 5 years of weekly data
            return {
                'frequency': 'weekly',
                'window_size': 52,  # One year of weeks
                'forecast_steps': 24  # 6 months in weeks
            }
        else:  # 12 months of weekly data
            return {
                'frequency': 'weekly',
                'window_size': 24,  # 6 months of weeks
                'forecast_steps': 12   # 3 months in weeks
            }
    else:  # Daily data (90 days)
        return {
            'frequency': 'daily',
            'window_size': 14,  # Two weeks of daily data
            'forecast_steps': 28   # One month in days
        }


def forecast_with_gru(df: pd.DataFrame, keyword: str) -> tuple[pd.DataFrame, dict, list]:
    if keyword not in df.columns:
        keyword = df.columns[-1]

    series = df[["date", keyword]].dropna().copy()
    series["date"] = pd.to_datetime(series["date"])
    series.set_index("date", inplace=True)

    # Determine frequency and parameters dynamically
    freq_params = determine_data_frequency(series)
    window_size = freq_params["window_size"]
    forecast_steps = freq_params["forecast_steps"]

    values = series.values
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    # Create sequences
    X, y = [], []
    for i in range(window_size, len(scaled_values)):
        X.append(scaled_values[i - window_size:i, 0])
        y.append(scaled_values[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split
    split_index = int(0.7 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

     # Model

    model = Sequential([
        GRU(50, return_sequences=True, dropout=0.2, input_shape=(window_size, 1)),
        GRU(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=250,
        batch_size=64,
        validation_data=(X_val, y_val),
        shuffle=False,
        callbacks=[early_stop],
        verbose=1
    )


    # Forecast
    forecast = []
    input_seq = scaled_values[-window_size:].reshape(1, window_size, 1)
    for _ in range(forecast_steps):
        pred = model.predict(input_seq, verbose=0)[0, 0]
        forecast.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    last_date = series.index[-1]
    step_days = 7 if freq_params['frequency'] == 'weekly' else 1
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=step_days), periods=forecast_steps, freq=f'{step_days}D')

    forecast_df = pd.DataFrame({"date": forecast_dates, "value": forecast})
    history_df = series.reset_index().rename(columns={keyword: "value"})

    forecast_df["type"] = "Forecast"
    history_df["type"] = "History"

    # Evaluation
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    metrics = {
        "Train MSE": float(mean_squared_error(y_train, y_train_pred)),
        "Train MAE": float(mean_absolute_error(y_train, y_train_pred)),
        "Train R²": float(r2_score(y_train, y_train_pred, force_finite=False)),
        "Validation MSE": float(mean_squared_error(y_val, y_val_pred)),
        "Validation MAE": float(mean_absolute_error(y_val, y_val_pred)),
        "Validation R²": float(r2_score(y_val, y_val_pred, force_finite=False)),
    }

    return pd.concat([history_df, forecast_df]), metrics, y_val, y_val_pred


def export_forecast_to_csv(df: pd.DataFrame, filename="forecast.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download forecast as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )
