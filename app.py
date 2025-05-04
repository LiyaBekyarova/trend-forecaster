import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
from data_utils import load_prepared_data
from model import forecast_with_gru, export_forecast_to_csv
import urllib.parse
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

st.set_page_config(
    page_title="Trend Forecaster",
    page_icon="🔮",
    layout="centered",
)

st.title("🔮 Trend Forecaster")

st.markdown("""
Welcome! This app helps you **forecast trends** using Google Trends data.

👉 You can either:
- **Enter a keyword manually** (uses live Google Trends)
- **Choose from prepared categories** (loads local data)

If Google Trends fails, you'll get instructions to upload your own `.csv` file downloaded manually from [trends.google.com](https://trends.google.com).
""")

# --- User input ---
mode = st.radio("Input method:", ["Enter keyword manually", "Select from prepared categories"])
time_label = st.selectbox("Select a time range:", list(TIME_RANGES.keys()))
time_range = TIME_RANGES[time_label]

keyword = ""
df = None

if mode == "Enter keyword manually":
    keyword = st.text_input("Enter your keyword:")
    if keyword:
        # Create a placeholder for the loading message
        loading_placeholder = st.empty()
        try:
            loading_placeholder.info("⏳ Fetching live data from Google Trends...")
            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload([keyword], cat=0, timeframe=time_range, geo='', gprop='')
            df = pytrends.interest_over_time()
            # Clear the loading message
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

            st.markdown(f"""
            👉 You can manually download data for **`{keyword}`** from [Google Trends](https://trends.google.com/trends/explore?q={keyword_url})  
            Then upload the `.csv` file below.
            """)
            uploaded_file = st.file_uploader("Upload the downloaded Google Trends CSV")

            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file, skiprows=1)
                    date_col = next((col for col in df.columns if col.lower() in ["date", "седмица", "ден"]), None)
                    value_col = next((col for col in df.columns if col != date_col and "ispartial" not in col.lower()), None)
                    df.rename(columns={date_col: "date", value_col: "value"}, inplace=True)
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df.dropna(subset=["date"], inplace=True)
                    st.success("✅ CSV file processed successfully!")
                except Exception as e:
                    st.error("❌ Could not process the uploaded file.")
                    st.exception(e)

else:
    category = st.selectbox("Select a category:", list(CATEGORIES.keys()))

    # Add a placeholder option
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

    date_column = next((col for col in ["date", "Седмица", "Ден"] if col in df.columns), None)
    if date_column is None:
        st.error("❌ Date column not found in the data.")
    else:
        df.rename(columns={date_column: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df.dropna(subset=["date"], inplace=True)

        # Show preview
        st.write("Here’s the preview of your cleaned data:")
        st.dataframe(df.head())

        # Auto determine forecast horizon
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

        st.info(f"📊 Detected {data_days} days of historical data. Forecasting for the next **{horizon_label}** ({forecast_days} days).")

        st.line_chart(df.set_index("date")[["value"]])

        if st.button("Run Forecast Model"):
            with st.spinner("Training model and generating forecast..."):
                forecast = forecast_with_gru(df, "value", days_ahead=forecast_days)
                chart_df = forecast.pivot(index="date", columns="type", values="value")
                st.line_chart(chart_df)
                export_forecast_to_csv(forecast)
