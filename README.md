# 🔮 Trend Forecaster

**Trend Forecaster** is an interactive Streamlit web app that predicts future search trends using data from **Google Trends** and a deep learning model based on **GRU (Gated Recurrent Unit)** networks.

It’s built for:

* Fashion tech analysts and researchers
* Marketing teams and e-commerce brands
* Curious users exploring trend dynamics

---

## 🧠 Key Features

* 📡 **Live Google Trends integration** via PyTrends API
* 📁 **Use prepared datasets** or upload your own Google Trends CSVs
* 🧮 **Forecast future interest** for the next 1, 3, or 6 months
* 🧠 Powered by a **2-layer GRU neural network**
* 📊 **Line charts** of historical + forecasted trends
* 💾 Export results as **CSV**

---

## 🏗️ Tech Stack

| Component     | Technology                  |
| ------------- | --------------------------- |
| Frontend      | Streamlit                   |
| Backend       | Python, Pandas              |
| ML Model      | TensorFlow (2× GRU + Dense) |
| Data Source   | PyTrends / Google Trends    |
| Visualization | Streamlit, Matplotlib       |

---

## 📁 Project Structure

```
trend-forecaster/
│
├── app.py              # Main Streamlit app logic
├── model.py            # GRU forecasting model logic
├── data_utils.py       # CSV parsing, normalization, loaders
├── requirements.txt    # All project dependencies
└── data/
    └── prepared/       # Preloaded keyword CSVs
```

---

## 🚀 Getting Started

### 1. Install & Run

```bash
git clone https://github.com/yourusername/trend-forecaster.git
cd trend-forecaster
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 💡 How It Works

1. **Choose how to input data**:

   * Manually type a keyword (real-time fetch from Google Trends)
   * Choose a keyword from the predefined categories
   * Upload a CSV exported from [trends.google.com](https://trends.google.com)

2. **Select a time range**:

   * Last 90 days → forecasts 1 month (daily granularity)
   * Last 12 months → forecasts 3 months (weekly)
   * Last 5 years → forecasts 6 months (weekly)

3. **The model processes the data**:

   * Normalizes and windows the series
   * Trains a GRU model with `window_size` based on time range
   * Forecasts future values step-by-step (autoregressively)

4. **View & export results**:

   * Actual vs predicted validation chart
   * Forecast line chart (future values)
   * CSV download

---

## 📄 Accepted CSV Formats

Supported formats include manually exported Google Trends files and localized labels.

### Example A (English):

```
Week,Fashion (Worldwide)
2023-01-01,56
2023-01-08,60
```

### Example B (Bulgarian):

```
Седмица,Sunglasses: (В цял свят)
2025-01-01,91
```

App auto-detects:

* Date columns like `Date`, `Week`, `Седмица`, `Ден`
* Value columns (trend values)

---

## 🤖 GRU Forecasting Model

* Architecture: `GRU(50, return_sequences=True)` → `GRU(50)` → `Dense(1)`
* Input: Sliding windows of past values (e.g. 14/24/52 time steps)
* Output: One-step prediction (autoregressively repeated N times)
* Optimizer: Adam
* Loss: Mean Squared Error (MSE)
* Validation: 30% of dataset
* Scaling: MinMaxScaler

Forecast horizon depends on detected time range:

* **< 180 days** → 30-day forecast
* **180–730 days** → 3-month forecast
* **> 2 years** → 6-month forecast

---

## 🛡 Fallbacks & Errors

* If PyTrends fails or rate limits:

  * App prompts you to download `.csv` from [trends.google.com](https://trends.google.com)
  * File upload UI is shown

* Invalid CSV formats:

  * App will show a clear error (e.g. missing date column)

---

## 🧪 Validation Metrics

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* R² (explained variance)

All metrics are shown for validation data (30% holdout).

---

## 📦 Dependencies

Main packages:

* `streamlit`
* `tensorflow==2.19.0`
* `pandas`
* `scikit-learn`
* `pytrends`
* `matplotlib`
* `altair`

Install all via:

```bash
pip install -r requirements.txt
```

---

## 🧠 Forecasting Tips

* Use longer time ranges (6+ months) for more stable results
* Forecasts may flatten if keyword is inactive or noisy
* Avoid rare terms that don’t appear in Google Trends
* Daily granularity works best with recent short-term data

---

## ✨ Use Case Examples

* Forecast future popularity of "Sunglasses" in summer
* See when "Clean girl" trend might fade
* Help fashion brands align product drops with demand
* Explore trend evolution (Y2K, Coquette, Old Money, etc.)

---

## 📥 Contribute

Want to improve this tool?

* Add LSTM or Transformer models
* Visualize multiple keyword comparisons
* Build a deployment pipeline (Docker, HuggingFace Spaces)

Pull requests welcome!

---

## 🛠 Troubleshooting

| Issue                  | Solution                             |
| ---------------------- | ------------------------------------ |
| No data or empty chart | Try a broader keyword or timeframe   |
| `KeyError: 'date'`     | Make sure your CSV has a date column |
| `TooManyRequestsError` | Wait or switch to CSV upload         |
| Forecast looks flat    | Try longer time range or weekly data |

---

## 🔗 Resources

* [Google Trends](https://trends.google.com/trends/)
* [Keras GRU Docs](https://keras.io/api/layers/recurrent_layers/gru/)
* [Streamlit](https://streamlit.io/)
* [PyTrends](https://github.com/GeneralMills/pytrends)
