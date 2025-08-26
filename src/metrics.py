import numpy as np
import pandas as pd

def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred, eps=1e-9):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

def smape(y_true, y_pred, eps=1e-9):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0

def metrics_frame(y_true, preds: dict) -> pd.DataFrame:
    rows = []
    for name, y_hat in preds.items():
        rows.append({
            "model": name,
            "MAE": mae(y_true, y_hat),
            "RMSE": rmse(y_true, y_hat),
            "MAPE": mape(y_true, y_hat),
            "SMAPE": smape(y_true, y_hat),
        })
    return pd.DataFrame(rows).sort_values("RMSE")
