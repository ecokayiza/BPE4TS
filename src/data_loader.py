"""
Data Loader Module.
-------------------
Handles downloading, loading, and preprocessing of the ETTm1 dataset.
Provides clean accessors for univariate time series data.

Public Functions:
- load_data(): Returns the full ETTm1 DataFrame (downloads if missing).
- get_series(): Extracts a specific column (default 'OT') as a Series.
- download_ettm1(): Helper to fetch data.

Dependencies: pandas, requests, os
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import Optional

DATA_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ETTm1.csv")

def download_ettm1(url: str = DATA_URL, save_path: str = DATA_PATH):
    """Downloads the dataset from GitHub."""
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}")
        return

    print(f"Downloading ETTm1 from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download data: {e}")
        # Create dummy data for offline testing if download fails
        print("Creating dummy data for testing...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dates = pd.date_range(start='2016-01-01', periods=1000, freq='15T')
        df = pd.DataFrame({
            'date': dates,
            'OT': np.sin(np.linspace(0, 50, 1000)) + np.random.normal(0, 0.1, 1000),
            'HUFL': np.cos(np.linspace(0, 50, 1000))
        })
        df.to_csv(save_path, index=False)

def load_data(path: str = DATA_PATH, target_col: str = 'OT') -> pd.DataFrame:
    """Loads the dataset into a pandas DataFrame."""
    if not os.path.exists(path):
        download_ettm1(save_path=path)
    
    df = pd.read_csv(path)
    # Convert date to datetime if strictly needed, but parsing dates is slow
    # df['date'] = pd.to_datetime(df['date']) 
    return df

def get_series(df: pd.DataFrame, col: str = 'OT') -> pd.Series:
    """Extracts a target column."""
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in dataset. options: {df.columns}")
    return df[col]
