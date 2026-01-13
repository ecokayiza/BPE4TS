import os
import requests
import pandas as pd
from typing import Optional

DATA_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ETTm1.csv")

def download_ettm1(url: str = DATA_URL, save_path: str = DATA_PATH):
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}")
        return

    print(f"Downloading ETTm1 from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(response.content)
    print("Download complete.")

def load_data(path: str = DATA_PATH, target_col: str = 'OT') -> pd.DataFrame:
    if not os.path.exists(path):
        download_ettm1(save_path=path)
    
    df = pd.read_csv(path)
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    return df

def get_series(df: pd.DataFrame, col: str = 'OT') -> pd.Series:
    return df[col]
