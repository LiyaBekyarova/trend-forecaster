import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
from data_utils import load_prepared_data
from model import forecast_with_gru, export_forecast_to_csv

# --- Static options ---
CATEGORIES = {
    "Fashion": ["Y2K", "Old Money", "Couture", "Minimalist", "Streetwear"],
    "Beauty": ["Glowy skin", "Clean girl", "Retinol", "Korean skincare", "Lip oil"],
    "Accessories": ["Chunky rings", "Mini bags", "Pearls", "Hair clips", "Sunglasses"],
    "Footwear": ["Platform shoes", "Sneakers", "Loafers", "Ballet flats", "Cowboy boots"],
    "Styles": ["Coquette", "Grunge", "Preppy", "Athleisure", "Softcore"]
}
TIME_RANGES = {
    "Last 5 years": "today 5-y",
    "Last 12 months": "today 12-m",
    "Last 90 days": "today 3-m"
}

# Forecast horizon per time range
FORECAST_DAYS = {
    "Last 5 years": 180,
    "Last 12 months": 90,
    "Last 90 days": 30
}

# Update the page configuration
st.set_page_config(
    page_title="Trend Forecaster",
    page_icon="📊", 
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': """

        This app forecasts trends using Google Trends data.
        """
    }
)

st.title("📊 Trend Forecaster")

# --- User input ---
mode = st.radio("Choose input method:", ["Enter keyword manually", "Select from prepared categories"])
time_label = st.selectbox("Select a time range:", list(TIME_RANGES.keys()))
time_range = TIME_RANGES[time_label]
forecast_days = FORECAST_DAYS[time_label]

keyword = ""
df = None

if mode == "Enter keyword manually":
    keyword = st.text_input("Enter your keyword:")
    if keyword:
        try:
            pytrends = TrendReq(hl='en-US', timeout=(30, 300))
            pytrends.build_payload([keyword], cat=0, timeframe=time_range)
            df = pytrends.interest_over_time()

            if "isPartial" in df.columns:
                df = df[df["isPartial"] == False]
                df.drop(columns=["isPartial"], inplace=True)

            if df.empty:
                st.warning("No data returned for this keyword.")
            else:
                st.success("Live data loaded from Google Trends.")
                st.subheader("📊 Raw Data Preview")
                st.dataframe(df.head())
                st.subheader("📈 Data Statistics")
                st.write({
                    "Total data points": len(df),
                    "Date range": f"{df.index.min()} to {df.index.max()}",
                    "Average interest": f"{df[keyword].mean():.2f}",
                    "Peak interest": df[keyword].max()
                })
                st.subheader("📈 Interest Over Time")
                st.line_chart(df[keyword])

                # Normalize for forecast
                df.reset_index(inplace=True)
                df.rename(columns={keyword: "value"}, inplace=True)
        except TooManyRequestsError:
            st.error("❌ Too many requests to Google Trends.")
        except Exception as e:
            st.error("❌ Failed to load Google Trends data.")
            st.exception(e)
else:
    category = st.selectbox("Select a category:", list(CATEGORIES.keys()))
    keyword = st.selectbox("Select a keyword:", CATEGORIES[category])
    try:
        df = load_prepared_data(category, keyword, time_label)
        st.success("Loaded prepared data.")
    except FileNotFoundError:
        st.error("Prepared data not found for this combination.")

# --- Forecast and display ---
if df is not None and not df.empty:
    st.subheader(f"📈 Trend Forecast for: {keyword}")
    st.write("CSV columns detected:", df.columns.tolist())

    if "date" not in df.columns:
        st.error("❌ Date column not found.")
    else:
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df.dropna(subset=["date"], inplace=True)

        st.write("Processed data preview:", df.head())
        st.line_chart(df.set_index("date")["value"])

        if st.button("Run Forecast Model"):
            forecast = forecast_with_gru(df, "value", days_ahead=forecast_days)
            chart_df = forecast.pivot(index="date", columns="type", values="value")
            st.line_chart(chart_df)
            export_forecast_to_csv(forecast)
