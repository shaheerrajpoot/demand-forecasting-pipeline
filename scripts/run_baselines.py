import argparse, yaml, os
import pandas as pd
from src.data_loading import load_sales_csv
from src.baselines import naive_forecast, moving_average_forecast, seasonal_naive_forecast
from src.metrics import metrics_frame
from src.ts_cv import rolling_origin_splits

def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    raw_path = cfg["data"]["raw_path"]
    date_col = cfg["data"]["date_col"]
    target_col = cfg["data"]["target_col"]
    horizon = cfg["cv"]["horizon"]
    n_splits = cfg["cv"]["n_splits"]
    min_train = cfg["cv"]["min_train_size"]

    df = load_sales_csv(raw_path, date_col=date_col)
    # For a univariate baseline: aggregate over ids (or pick one SKU)
    if "id_cols" in cfg["data"] and cfg["data"]["id_cols"]:
        # Example: aggregate total sales across all ids
        df_agg = df.groupby(date_col, as_index=False)[target_col].sum()
    else:
        df_agg = df[[date_col, target_col]].copy()

    results = []
    for tr_idx, te_idx in rolling_origin_splits(df_agg, date_col, n_splits, horizon, min_train):
        y_tr = df_agg.iloc[tr_idx][target_col]
        y_te = df_agg.iloc[te_idx][target_col]

        preds = {
            "naive": naive_forecast(y_tr, horizon),
            "moving_avg_7": moving_average_forecast(y_tr, horizon, window=7),
            "seasonal_naive_7": seasonal_naive_forecast(y_tr, horizon, season=7),
        }
        mf = metrics_frame(y_te.values, preds)
        mf["fold_start"] = df_agg.iloc[tr_idx][date_col].min()
        mf["fold_end"] = df_agg.iloc[tr_idx][date_col].max()
        results.append(mf)

    all_results = pd.concat(results, ignore_index=True)
    os.makedirs("reports/metrics", exist_ok=True)
    out_path = os.path.join("reports/metrics", "baseline_comparison.csv")
    all_results.to_csv(out_path, index=False)
    print(f"Saved baseline metrics to {out_path}")
    print(all_results.groupby("model")[["MAE","RMSE","MAPE","SMAPE"]].mean().sort_values("RMSE"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
