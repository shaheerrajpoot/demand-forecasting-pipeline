import numpy as np
import pandas as pd

def naive_forecast(y: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(y.iloc[-1], horizon)

def moving_average_forecast(y: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    avg = y.iloc[-window:].mean()
    return np.repeat(avg, horizon)

def seasonal_naive_forecast(y: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    # Repeat the last full season to cover horizon
    last_season = y.iloc[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season.values, reps)[:horizon]
