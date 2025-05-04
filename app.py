import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
from data_utils import fetch_trends, load_prepared_data
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

keyword = ""
df = None

if mode == "Enter keyword manually":
    keyword = st.text_input("Enter your keyword:")
    if keyword:
        try:
            df = fetch_trends(keyword=keyword, timeframe=time_range, hl='en-US')
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
                st.line_chart(df)
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

    date_column = next((col for col in ["date", "Седмица", "Ден"] if col in df.columns), None)
    if date_column is None:
        st.error("❌ Date column not found.")
    else:
        df.rename(columns={date_column: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df.dropna(subset=["date"], inplace=True)

        original_value_column = next((col for col in df.columns if keyword.lower() in col.lower()), df.columns[-1])
        df.rename(columns={original_value_column: "value"}, inplace=True)

        st.write("Processed data preview:", df.head())
        st.line_chart(df.set_index("date")[["value"]])

        if st.button("Run Forecast Model"):
            forecast = forecast_with_gru(df, "value")
            chart_df = forecast.pivot(index="date", columns="type", values="value")
            st.line_chart(chart_df)
            export_forecast_to_csv(forecast)
