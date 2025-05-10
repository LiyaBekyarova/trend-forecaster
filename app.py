import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
from data_utils import load_prepared_data
from model import forecast_with_gru, export_forecast_to_csv
import urllib.parse

CATEGORIES = {
    "Fashion": ["Y2K", "Old Money", "Couture", "Minimalist", "Streetwear"],
    "Beauty": ["Glowy skin", "Clean girl", "Retinol", "Korean skincare", "Lip oil"],
    "Accessories": ["Chunky rings", "Mini bags", "Pearls", "Hair clips", "Sunglasses"],
    "Footwear": [
        "Platform shoes",
        "Sneakers",
        "Loafers",
        "Ballet flats",
        "Cowboy boots",
    ],
    "Styles": ["Coquette", "Grunge", "Preppy", "Athleisure", "Softcore"],
}

TIME_RANGES = {
    "Last 5 years": "today 5-y",
    "Last 12 months": "today 12-m",
    "Last 90 days": "today 3-m",
}

st.set_page_config(
    page_title="Trend Forecaster",
    page_icon="🔮",
    layout="centered",
)

st.title("🔮 Trend Forecaster")

st.markdown(
    """
Welcome! This app helps you **forecast trends** using Google Trends data.

👉 You can either:
- **Enter a keyword manually** (uses live Google Trends)
- **Choose from prepared categories** (loads local data)

If Google Trends fails, you'll get instructions to upload your own `.csv` file downloaded manually from [trends.google.com](https://trends.google.com).
"""
)

# --- User input ---
mode = st.radio(
    "Input method:", ["Enter keyword manually", "Select from prepared categories"]
)
time_label = st.selectbox("Select a time range:", list(TIME_RANGES.keys()))
time_range = TIME_RANGES[time_label]

keyword = ""
df = None

if mode == "Enter keyword manually":
    keyword = st.text_input("Enter your keyword:")
    if keyword:
        loading_placeholder = st.empty()
        try:
            loading_placeholder.info("⏳ Fetching live data from Google Trends...")
            pytrends = TrendReq(hl="en-US", tz=360)
            pytrends.build_payload(
                [keyword], cat=0, timeframe=time_range, geo="", gprop=""
            )
            df = pytrends.interest_over_time()
            loading_placeholder.empty()

            if df.empty:
                st.warning("No data returned for this keyword.")
                df = None
            else:
                df = df.drop(columns=["isPartial"], errors="ignore")
                df.reset_index(inplace=True)
                df.rename(columns={df.columns[-1]: "value"}, inplace=True)
                st.success("✅ Live data loaded successfully!")

        except TooManyRequestsError or Exception as e:
            loading_placeholder.empty()
            st.error("❌ Too many requests to Google Trends.")
            keyword_url = urllib.parse.quote(keyword)

            st.markdown(
                f"""
            👉 You can manually download data for **`{keyword}`** from [Google Trends](https://trends.google.com/trends/explore?q={keyword_url})  
            Then upload the `.csv` file below.
            """
            )
            uploaded_file = st.file_uploader("Upload the downloaded Google Trends CSV")

            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file, skiprows=1)
                    date_col = next(
                        (
                            col
                            for col in df.columns
                            if col.lower() in ["date", "седмица", "ден"]
                        ),
                        None,
                    )
                    value_col = next(
                        (
                            col
                            for col in df.columns
                            if col != date_col and "ispartial" not in col.lower()
                        ),
                        None,
                    )
                    df.rename(
                        columns={date_col: "date", value_col: "value"}, inplace=True
                    )
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df.dropna(subset=["date"], inplace=True)
                    st.success("✅ CSV file processed successfully!")
                except Exception as e:
                    st.error("❌ Could not process the uploaded file.")
                    st.exception(e)

else:
    category = st.selectbox("Select a category:", list(CATEGORIES.keys()))
    keyword_options = ["-- Select a keyword --"] + CATEGORIES[category]
    selected = st.selectbox("Select a keyword:", keyword_options)
    if selected != "-- Select a keyword --":
        keyword = selected

    try:
        df = load_prepared_data(category, keyword, time_label)
        st.success("✅ Loaded prepared data.")
    except FileNotFoundError:
        df = None

# --- Forecast and display ---
if df is not None and not df.empty:
    st.subheader(f"📈 Trend Forecast for: `{keyword}`")
    data_days = (df["date"].max() - df["date"].min()).days
    if data_days >= 730:
        forecast_days = 180
        horizon_label = "6 months"
    elif data_days >= 180:
        forecast_days = 90
        horizon_label = "3 months"
    else:
        forecast_days = 30
        horizon_label = "1 month"
    st.info(
        f"📊 Detected {data_days} days of historical data. Forecasting for the next **{horizon_label}** ({forecast_days} days)."
    )
    if st.button("Run Forecast Model", key="forecast_button"):
        with st.spinner("Training model and generating forecast..."):
                forecast_df, metrics, y_val, y_val_pred = (
                    forecast_with_gru(df, "value")
                )
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Validation MSE", f"{metrics['Validation MSE']:.4f}")
                with col2:
                    st.metric("Validation MAE", f"{metrics['Validation MAE']:.4f}")
                with col3:
                    st.metric("Validation R²", f"{metrics['Validation R²']:.4f}")
                    st.subheader("🧪 Validation Fit: Actual vs Predicted")

                val_fit_df = pd.DataFrame({
                    "Actual": y_val.flatten(),
                    "Predicted": y_val_pred.flatten()
                })
                st.line_chart(val_fit_df)

                # === Forecast Visualization ===
                st.subheader("Forecast Results")

                historical = forecast_df[forecast_df["type"] == "History"].copy()
                forecast = forecast_df[forecast_df["type"] == "Forecast"].copy()

                chart_df = pd.DataFrame(index=forecast_df["date"].unique())
                chart_df["Historical"] = historical.set_index("date")["value"]
                chart_df["Forecast"] = forecast.set_index("date")["value"]

                st.line_chart(chart_df)

                # Export
                export_forecast_to_csv(forecast_df)

