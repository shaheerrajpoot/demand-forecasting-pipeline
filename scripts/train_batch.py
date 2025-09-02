# scripts/train_batch.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import awswrangler as wr
import mlflow
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

# ---------- Config ----------
S3_PATH   = os.getenv("S3_PATH", "s3://demandcast-sajan-eu-west-2/data/curated/m5/sales_long/validation_test/")
TOP_N     = int(os.getenv("TOP_N", 10))
TEST_DAYS = int(os.getenv("TEST_DAYS", 28))

# ---------- Feature utils (same as train_one.py) ----------
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

# ---------- Load data ----------
cols = ["id","item_id","dept_id","cat_id","store_id","state_id","date","wm_yr_wk","year","month","sales"]
df = wr.s3.read_parquet(S3_PATH, dataset=True, columns=cols)
df["date"] = pd.to_datetime(df["date"])
df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

# pick top-N high volume series
top_ids = (
    df.groupby("id")["sales"].mean()
      .sort_values(ascending=False)
      .head(TOP_N)
      .index.tolist()
)

# ---------- MLflow ----------
Path("mlruns").mkdir(exist_ok=True)
mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
mlflow.set_experiment("DemandCast - batch rich")
Path("artifacts").mkdir(exist_ok=True)

results = []
for sid in top_ids:
    try:
        g = df[df["id"] == sid].copy().sort_values("date")
        g_feat, feats = build_feature_matrix(g)

        dfx = g_feat.dropna(subset=feats + ["sales"]).sort_values("date")
        train, test = dfx.iloc[:-TEST_DAYS], dfx.iloc[-TEST_DAYS:]
        Xtr, ytr = train[feats], train["sales"]
        Xte, yte = test[feats], test["sales"]

        params = dict(n_estimators=900, learning_rate=0.05, max_depth=-1,
                      subsample=0.9, colsample_bytree=0.9, random_state=42)

        with mlflow.start_run(run_name=f"LGBM_{sid}_rich"):
            mlflow.log_params(params)
            mlflow.log_param("features", ",".join(feats))
            mlflow.log_param("horizon_days", TEST_DAYS)
            mlflow.set_tag("series_id", sid)

            model = LGBMRegressor(**params).fit(Xtr, ytr)
            preds = model.predict(Xte)
            rmse = sqrt(mean_squared_error(yte, preds))
            mlflow.log_metric("rmse", rmse)

            out = test[["date","id","sales"]].copy()
            out["pred"] = preds
            csv_path = f"artifacts/test_preds_{sid}_rich.csv"
            out.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)

            plt.figure(figsize=(10,4))
            plt.plot(out["date"], out["sales"], label="Actual")
            plt.plot(out["date"], out["pred"], label="Predicted")
            plt.title(f"{sid} — Last {TEST_DAYS} days — RMSE {rmse:.2f}")
            plt.xlabel("Date"); plt.ylabel("Sales"); plt.legend(); plt.tight_layout()
            png_path = f"artifacts/test_preds_{sid}_rich.png"
            plt.savefig(png_path, dpi=150)
            plt.close()
            mlflow.log_artifact(png_path)

        results.append({"id": sid, "rmse": rmse})
        print(f"Done: {sid}  RMSE={rmse:.3f}")

    except Exception as e:
        results.append({"id": sid, "rmse": None, "error": str(e)})
        print(f"Error: {sid} -> {e}")

# summary
res_df = pd.DataFrame(results).sort_values("rmse", ascending=True)
out_sum = "artifacts/multi_series_rmse_rich.csv"
res_df.to_csv(out_sum, index=False)
print(f"\nSaved summary -> {out_sum}")
print(res_df.head(10))
