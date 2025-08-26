from pathlib import Path

root = Path(r"D:\Projects\DemandCast\demand_forecasting_pipeline")
raw = root / "data" / "raw" / "m5"

print("raw folder:", raw, "exists:", raw.exists())
print("calendar.csv:", raw / "calendar" / "calendar.csv", 
      "exists:", (raw / "calendar" / "calendar.csv").exists())
print("validation.csv:", raw / "sales_train_validation" / "sales_train_validation.csv",
      "exists:", (raw / "sales_train_validation" / "sales_train_validation.csv").exists())
