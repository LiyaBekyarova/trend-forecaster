import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
from data_utils import load_prepared_data
from model import forecast_with_gru, export_forecast_to_csv
import urllib.parse
import matplotlib.pyplot as plt
import random
import time


FUN_FACTS = [
    "👠 Y2K fashion trends are making a major comeback!",
    "🧠 GRU stands for Gated Recurrent Unit – a smart version of RNNs!",
    "👗 'Old Money' aesthetic is trending on TikTok and Instagram.",
    "📈 The GRU model learns from past data to predict the future.",
    "💄 'Clean Girl' makeup look is one of the most searched beauty trends.",
    "🌍 Google Trends reflects real-time user interests across the globe.",
    "📉 MSE punishes large errors more than MAE does.",
    "👡 Chunky sandals and ballet flats returned on Paris runways.",
    "🎯 R² helps you know how well the model explains the trend!"
]

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
                # Create a single placeholder for fun facts
                fact_placeholder = st.empty()
                for fact in FUN_FACTS:
                    fact_placeholder.info(f"💡 Fun Fact: {fact}")
                    time.sleep(2.5)
                forecast, metrics, loss_values = forecast_with_gru(df, "value", days_ahead=forecast_days)

            # Display the forecast chart
            st.subheader("📈 Historical Data + Forecast")
            chart_df = forecast.pivot(index="date", columns="type", values="value")
            st.line_chart(chart_df)

            # 📈 Display metrics
            st.subheader("📏 Model Performance Metrics")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🧪 Train MSE", f"{metrics['Train MSE']:.4f}")
                st.metric("📏 Train MAE", f"{metrics['Train MAE']:.4f}")
                st.metric("🎯 Train R²", f"{metrics['Train R²']:.4f}")
            with col2:
                st.metric("🧪 Val MSE", f"{metrics['Validation MSE']:.4f}")
                st.metric("📏 Val MAE", f"{metrics['Validation MAE']:.4f}")
                st.metric("🎯 Val R²", f"{metrics['Validation R²']:.4f}")
            with st.expander("ℹ️ What do these metrics mean?"):
                st.markdown("""
                **🔹 Mean Squared Error (MSE)**  
                Measures how far off predictions are, with larger errors weighted more.

                **🔹 Mean Absolute Error (MAE)**  
                The average error, easier to interpret since it's in the same scale as the data.

                **🔹 R² Score (Coefficient of Determination)**  
                Tells how much of the variation in the trend is explained by the model.  
                - **1.00** means perfect prediction  
                - **0.00** means model is no better than guessing the average  
                - Negative means it performs worse than a flat average prediction

                **Training vs Validation**  
                - **Training**: performance on data the model learned from.  
                - **Validation**: performance on unseen data (simulates the future).
                """)


            # 📉 Plot the training loss
            st.subheader("📉 Training Loss Curve")
            fig, ax = plt.subplots()
            ax.plot(loss_values)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("GRU Training Loss Over Epochs")
            st.pyplot(fig)

            # 📥 Export CSV
            export_forecast_to_csv(forecast)

