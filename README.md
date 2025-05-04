# Google Trends Forecaster 📈

Това е уеб приложение, разработено със Streamlit, което:
- Взима търсения за ключова дума от Google Trends
- Обучава GRU невронна мрежа на тези данни
- Прогнозира интереса за следващите 6 месеца
- Показва резултатите в лесен за разбиране графичен вид

## Технологии:
- Python
- Streamlit
- TensorFlow / Keras
- Pytrends
- Pandas
- Matplotlib

## Стартиране локално:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
