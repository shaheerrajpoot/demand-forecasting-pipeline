# m5_melt_validation_small.py  — dry run on 500 rows
import pandas as pd
from pathlib import Path

root = Path(r"D:\Projects\DemandCast\demand_forecasting_pipeline")
raw = root / "data" / "raw" / "m5"
out = root / "data" / "curated" / "m5" / "sales_long" / "validation_test"
out.mkdir(parents=True, exist_ok=True)

# 1) Load calendar once (maps d_### -> date/year/month/week)
cal = pd.read_csv(raw / "calendar.csv")
cal.columns = cal.columns.str.lower()
cal = cal[["d", "date", "wm_yr_wk", "year", "month"]]

# 2) Read only first 500 rows of the wide sales file
sales_path = raw / "sales_train_validation.csv"
df = pd.read_csv(sales_path, nrows=5000,low_memory=False)

# 3) Identify id (meta) vs day columns
meta = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
day_cols = [c for c in df.columns if c.startswith("d_")]

# 4) Melt wide -> long (creates columns: d, sales)
long_df = df.melt(id_vars=meta, value_vars=day_cols, var_name="d", value_name="sales")

# 5) Join to calendar to get real dates + partitions
long_df = long_df.merge(cal, on="d", how="left")

# 6) Sort (optional)
long_df = long_df.sort_values(["store_id", "item_id", "date"])

# 7) Write a small Parquet sample partitioned by year/month
for (yr, mo), part in long_df.groupby(["year", "month"]):
    part_path = out / f"year={yr}" / f"month={mo}"
    part_path.mkdir(parents=True, exist_ok=True)
    part.to_parquet(part_path / "part.parquet", index=False)

print("✅ Dry run done. Output at:", out)
