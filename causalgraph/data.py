
from __future__ import annotations
import os
import pandas as pd
from typing import Tuple

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)

def split_df(df: pd.DataFrame, train_ratio: float = 0.8, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    train_df = df.sample(frac=train_ratio, random_state=random_state)
    valid_df = df.drop(train_df.index)
    return train_df, valid_df

def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
