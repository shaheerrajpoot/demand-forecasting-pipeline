import pandas as pd

def load_sales_csv(path: str, date_col: str = "date") -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(by=[date_col]).reset_index(drop=True)
