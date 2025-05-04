import pandas as pd
import os

def load_prepared_data(category: str, keyword: str, time_label: str) -> pd.DataFrame:
    category = category.replace(" ", "_").lower()
    keyword = keyword.replace(" ", "_").lower()
    time_label = time_label.replace(" ", "_").lower()

    file_path = f"data/prepared/{category}_{keyword}_{time_label}.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файлът не съществува: {file_path}")

    # Пропускаме метаданните
    df = pd.read_csv(file_path, skiprows=1)

    # Премахваме частични данни, ако има isPartial
    if "isPartial" in df.columns:
        df = df[df["isPartial"] == False]
        df.drop(columns=["isPartial"], inplace=True)

    # Откриване на колоната за дата
    date_col = next((col for col in df.columns if col.lower() in ["date", "седмица", "ден"]), None)
    if not date_col:
        raise ValueError("❌ Не е открита колона с дата.")

    # Откриване на стойностната колона
    value_col = next((col for col in df.columns if col != date_col), None)
    if not value_col:
        raise ValueError("❌ Не е открита колона със стойности.")

    # Преименуване и обработка
    df.rename(columns={date_col: "date", value_col: "value"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    return df
