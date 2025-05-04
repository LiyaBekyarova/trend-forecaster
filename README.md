# 🔮 Trend Forecaster

The **Trend Forecaster** is a Streamlit-based web app that uses historical interest data from **Google Trends** to **predict  trends** using a GRU-based neural network.

It is designed for:
- Fashion tech researchers and analysts
- Brands tracking search interest
- Trend enthusiasts and data lovers

---

## 🧠 Key Features

- 📡 **Live Google Trends integration** (with fallback support)
- 📁 **Use prepared datasets** or **upload your own CSVs**
- 📊 **Visualize search interest** with clean line charts
- 🤖 **Forecast next 1, 3, or 6 months** using a trained GRU model
- 💾 **Export results as CSV** for further analysis

---

## 🏗️ Tech Stack

| Component     | Tech Used             |
|---------------|------------------------|
| Frontend      | Streamlit              |
| Backend       | Python, Pandas         |
| Forecasting   | TensorFlow (GRU)       |
| Data source   | PyTrends (Google API)  |
| Visualization | Altair, Matplotlib     |

---

## 📁 Project Structure

```
trend-forecaster/
│
├── app.py              # Streamlit app UI and logic
├── data_utils.py       # Data loading and preprocessing functions
├── model.py            # GRU model definition and forecasting logic
├── requirements.txt    # Full list of Python dependencies
└── data/
    └── prepared/       # CSVs for preloaded keywords
```

---

## 🚀 Getting Started

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/trend-forecaster.git
cd trend-forecaster
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

---

## 💡 How It Works

1. **Choose an input mode**:
   - Type your own keyword (e.g. `"Al dente"`, `"Y2K"`).
   - Select a keyword from categories (like `"Beauty"` → `"Glowy skin"`).

2. **Select a time range**:
   - Last 90 days (short-term)
   - Last 12 months
   - Last 5 years (long-term)

3. **Forecasting Logic**:
   - If you provide **< 6 months** of data → forecasts **1 month** ahead
   - If you provide **~6–24 months** → forecasts **3 months**
   - If you provide **> 2 years** → forecasts **6 months**

4. **Visualizations**:
   - Original time series data
   - Forecasted vs. historical interest
   - Exportable CSV download

---

## 📄 Accepted Data Formats

The app can parse CSVs from:
- Google Trends (exported manually)
- Your custom files
- Prepared datasets in `/data/prepared/`

### ✅ Format Example (Google Trends):

```
Week,Fashion (Worldwide)
2023-01-01,56
2023-01-08,60
```

### ✅ Format Example (Custom):

```
Ден,Cowboy boots: (В цял свят)
2025-02-01,78
```

The app will auto-detect columns like `"Week"`, `"Date"`, `"Ден"` or `"Седмица"`.

---

## 📉 Model: GRU (Gated Recurrent Unit)

- Input: Time series (30-day windows)
- Output: Forecasted values
- Framework: Keras (TensorFlow backend)
- Trained directly on normalized historical data
- Forecast horizon adjusts to input range:
  - 90+ days: 1 month
  - 180+ days: 3 months
  - 2+ years: 6 months

---

## 🧪 Error Handling & Fallbacks

- If **Google Trends fails** (rate limit or offline), users are prompted to:
  1. Download the data manually from [trends.google.com](https://trends.google.com)
  2. Upload the `.csv` file into the app

- Invalid CSVs are validated and rejected with clear errors.

---

## 📦 Dependencies

See [`requirements.txt`](./requirements.txt) for full list, including:

- `streamlit`
- `pandas`
- `tensorflow`
- `pytrends`
- `matplotlib`
- `altair`

Install with:

```bash
pip install -r requirements.txt
```

---

## 🧠 Tips for Better Forecasts

- Use **at least 6 months of data** for stable long-term predictions.
- Avoid overly niche keywords — no data → no forecast!
- Daily data (from Google Trends) works best when exporting with region: Worldwide.

---

## ✨ Example Use Cases

- Predict how long **“Clean girl”** will stay trendy
- Compare trend strength between **“Y2K”** and **“Couture”**
- Help brands plan collections or marketing campaigns
- Explore macro trend cycles in fashion

---

## 📥 Contributions

Have ideas to improve this forecaster?
- Support multiple keywords?
- Compare categories?
- Add LSTM or Transformer options?

Open an issue or submit a pull request!

---

## 🛠 Troubleshooting

| Problem                         | Solution                                                   |
|----------------------------------|-------------------------------------------------------------|
| `KeyError: 'date'`              | Ensure your CSV has a proper date column (Week, Date, etc.)|
| No data from Google Trends      | Wait a few minutes or download CSV manually                |
| `TooManyRequestsError`          | You’ve hit the rate limit — try again later                |
| Forecast is flat or unrealistic | Use longer historical periods if available                 |

---

## 🔗 Links

- [Google Trends](https://trends.google.com/trends/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Keras GRU Layer](https://keras.io/api/layers/recurrent_layers/gru/)
- [PyTrends Library](https://github.com/GeneralMills/pytrends)
