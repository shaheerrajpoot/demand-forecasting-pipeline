import argparse, yaml, os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from src.data_loading import load_sales_csv

def make_features(df, date_col, target_col, lags=(1,7,14,28), rollings=(7,14,28)):
    df = df.copy()
    df = df.sort_values(date_col)
    for L in lags:
        df[f"lag_{L}"] = df[target_col].shift(L)
    for W in rollings:
        df[f"roll_mean_{W}"] = df[target_col].shift(1).rolling(W).mean()
        df[f"roll_std_{W}"] = df[target_col].shift(1).rolling(W).std()
    # calendar
    d = df[date_col]
    df["dow"] = d.dt.weekday
    df["dom"] = d.dt.day
    df["month"] = d.dt.month
    df = df.dropna().reset_index(drop=True)
    return df

def evaluate(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    return rmse, mae

def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    raw_path = cfg["data"]["raw_path"]
    date_col = cfg["data"]["date_col"]
    target_col = cfg["data"]["target_col"]

    df = load_sales_csv(raw_path, date_col=date_col)
    df_agg = df.groupby(date_col, as_index=False)[target_col].sum()
    df_feat = make_features(df_agg, date_col, target_col)
    y = df_feat[target_col].values
    X = df_feat.drop(columns=[target_col, date_col]).values

    tscv = TimeSeriesSplit(n_splits=4)
    rows = []
    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        models = {
            "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42),
            "LightGBM": LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=42),
        }
        for name, model in models.items():
            rmse, mae = evaluate(model, X_tr, y_tr, X_va, y_va)
            rows.append({"fold": fold, "model": name, "RMSE": rmse, "MAE": mae})

    res = pd.DataFrame(rows)
    os.makedirs("reports/metrics", exist_ok=True)
    out = "reports/metrics/ml_comparison.csv"
    res.to_csv(out, index=False)
    print(f"Saved ML metrics to {out}")
    print(res.groupby("model")[["RMSE","MAE"]].mean().sort_values("RMSE"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
