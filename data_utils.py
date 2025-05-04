from pytrends.request import TrendReq
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from time import sleep
from random import uniform
import os
import pandas as pd

def load_prepared_data(category: str, keyword: str, time_label: str) -> pd.DataFrame:
    import os
    import pandas as pd

    category = category.replace(" ", "_").lower()
    keyword = keyword.replace(" ", "_").lower()
    time_label = time_label.replace(" ", "_").lower()

    file_path = f"data/prepared/{category}_{keyword}_{time_label}.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    # FIX: skip the first row (metadata), then parse the actual header
    df = pd.read_csv(file_path, skiprows=1)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

def fetch_trends(keyword: str, timeframe="today 5-y", geo="", hl="en-US") -> pd.DataFrame:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            pytrends = TrendReq(hl=hl, timeout=(30,300))
            
            
            pytrends.build_payload(
                [keyword], 
                cat=0, 
                timeframe=timeframe, 
                geo=geo, 
                gprop=''
            )
          
            
            df = pytrends.interest_over_time()
            
            if df.empty:
                raise ValueError("Няма налични данни за тази ключова дума.")
            df = pytrends.interest_over_time()



            # Използваме само редове с пълни данни
            if "isPartial" in df.columns:
                df = df[df["isPartial"] == False]
                df = df.drop(columns=["isPartial"])


            df = df.astype(float)
            return df
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception("Моля, изчакайте няколко минути и опитайте отново.")
            sleep(uniform(20.0, 30.0))


