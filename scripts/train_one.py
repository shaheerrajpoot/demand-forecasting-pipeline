import os
from pathlib import Path
import pandas as pd
import numpy as np
import awswrangler as wr
import mlflow
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# ---- Config ----
S3_PATH = "s3://demandcast-sajan-eu-west-2/data/curated/m5/sales_long/validation_test/"
SERIES_ID = os.getenv("SERIES_ID", "FOODS_3_090_CA_1_validation")
TEST_DAYS = int(os.getenv("TEST_DAYS", 28))

# ---- Helpers ----
def add_time_features(df):
    df["dow"] = df["date"].dt.weekday
    df["dom"] = df["date"].dt.day
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df

def add_sales_lags_rolls(df):
    df = df.sort_values(["id","date"])
    for L in (7,14,28):
        df[f"lag_{L}"] = df.groupby("id")["sales"].shift(L).astype(float)
    for W in (7,28,56):
        df[f"roll_{W}"] = df.groupby("id")["sales"].transform(lambda s: s.shift(1).rolling(W).mean())
    return df

def add_snap_flags(df, state_col="state_id"):
    if {"snap_CA","snap_TX","snap_WI"}.issubset(df.columns):
        df["snap"] = 0
        df.loc[df[state_col]=="CA","snap"] = df.loc[df[state_col]=="CA","snap_CA"]
        df.loc[df[state_col]=="TX","snap"] = df.loc[df[state_col]=="TX","snap_TX"]
        df.loc[df[state_col]=="WI","snap"] = df.loc[df[state_col]=="WI","snap_WI"]
    else:
        df["snap"] = 0
    return df

def build_feature_matrix(df):
    df = df.copy()
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = add_time_features(df)
    df = add_sales_lags_rolls(df)
    df = add_snap_flags(df)
    df["is_event"] = df["event_name_1"].notna().astype(int) if "event_name_1" in df.columns else 0
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["roll_std_28"] = df.groupby("id")["sales"].transform(lambda s: s.shift(1).rolling(28).std())
    df["cum_sales"]   = df.groupby("id")["sales"].cumsum()
    df = df.dropna(subset=["lag_7","roll_7"]).reset_index(drop=True)
    feature_cols = [
        "lag_7","lag_14","lag_28",
        "roll_7","roll_28","roll_56","roll_std_28",
        "snap","is_event","is_weekend",
        "dow","dom","week","quarter","is_month_start","is_month_end",
        "cum_sales",
    ]
    return df, feature_cols

# ---- Data load ----
cols = ["id","item_id","dept_id","cat_id","store_id","state_id","date","wm_yr_wk","year","month","sales"]
df = wr.s3.read_parquet(S3_PATH, dataset=True, columns=cols)
df["date"] = pd.to_datetime(df["date"])
df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

g = df[df["id"] == SERIES_ID].copy().sort_values("date")
g_feat, feats = build_feature_matrix(g)

# ---- MLflow setup ----
Path("mlruns").mkdir(exist_ok=True)
mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
mlflow.set_experiment("DemandCast - single series")
Path("artifacts").mkdir(exist_ok=True)

# ---- Train & log ----
dfm = g_feat.dropna(subset=feats + ["sales"]).sort_values("date")
train, test = dfm.iloc[:-TEST_DAYS], dfm.iloc[-TEST_DAYS:]
Xtr, ytr = train[feats], train["sales"]
Xte, yte = test[feats], test["sales"]

params = dict(n_estimators=900, learning_rate=0.05, max_depth=-1,
              subsample=0.9, colsample_bytree=0.9, random_state=42)

with mlflow.start_run(run_name=f"LGBM_{SERIES_ID}_trend_volatility"):
    mlflow.log_params(params)
    mlflow.log_param("features", ",".join(feats))
    mlflow.log_param("horizon_days", TEST_DAYS)
    mlflow.set_tag("series_id", SERIES_ID)

    model = LGBMRegressor(**params).fit(Xtr, ytr)
    preds = model.predict(Xte)
    rmse = sqrt(mean_squared_error(yte, preds))
    mlflow.log_metric("rmse", rmse)

    out = test[["date","id","sales"]].copy()
    out["pred"] = preds
    out_path = f"artifacts/test_preds_{SERIES_ID}.csv"
    out.to_csv(out_path, index=False)
    mlflow.log_artifact(out_path)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.plot(out["date"], out["sales"], label="Actual")
    plt.plot(out["date"], out["pred"], label="Predicted")
    plt.title(f"{SERIES_ID} — Last {TEST_DAYS} days — RMSE {rmse:.2f}")
    plt.xlabel("Date"); plt.ylabel("Sales"); plt.legend(); plt.tight_layout()
    png_path = f"artifacts/test_preds_{SERIES_ID}.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    mlflow.log_artifact(png_path)

print("DONE. RMSE:", rmse)
