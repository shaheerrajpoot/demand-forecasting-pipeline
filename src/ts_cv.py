from typing import Iterator, Tuple
import numpy as np
import pandas as pd

def rolling_origin_splits(df: pd.DataFrame, date_col: str, n_splits: int, horizon: int, min_train_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    dates = df[date_col].sort_values().unique()
    # Ensure enough history
    start_idx = max(0, len(dates) - (n_splits * horizon) - min_train_size)
    for i in range(n_splits):
        split_point = len(dates) - (n_splits - i) * horizon
        train_end = split_point
        test_end = split_point + horizon
        if train_end <= 0 or test_end > len(dates):
            continue
        train_mask = (df[date_col] <= dates[train_end-1])
        test_mask = (df[date_col] > dates[train_end-1]) & (df[date_col] <= dates[test_end-1])
        yield df.index[train_mask].values, df.index[test_mask].values
