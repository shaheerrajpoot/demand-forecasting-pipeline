# Demand Forecasting & Business Intelligence Pipeline

**Industrial-grade, end-to-end project** that forecasts product demand and exposes results for BI dashboards (Tableau/Power BI) via a reproducible ML pipeline.

## What you’ll build
- Data ingestion → cleaning → feature engineering
- Multiple forecasting models (baselines, ARIMA/SARIMA, gradient boosting, optional LSTM)
- Time-series cross-validation + **MAE / RMSE / (S)MAPE** comparison
- Persist best model → batch forecast job
- Optional API (FastAPI) and cloud storage (S3/GCS) integration
- BI-ready outputs (tables for Tableau/Power BI)

## Folder structure
```
demand_forecasting_pipeline/
├─ data/
│  ├─ raw/            # Put your original CSV here (e.g., sales.csv)
│  ├─ processed/
│  └─ external/
├─ models/            # Saved model artifacts (joblib)
├─ notebooks/         # EDA & experiments (Jupyter)
├─ reports/
│  ├─ figures/        # Charts & plots
│  └─ metrics/        # CSVs of model comparison
├─ scripts/           # CLI scripts to run pipeline steps
├─ src/               # Reusable Python modules
│  ├─ models/         # Model classes & wrappers
│  └─ __init__.py
├─ .github/workflows/ # CI/CD (GitHub Actions)
├─ config.yaml        # Central config (paths, column names, parameters)
├─ requirements.txt   # Python dependencies
└─ README.md
```

## Dataset options
- **Option A (quick start):** Use a simple time-series CSV with columns like: `date, store_id, item_id, sales`.
- **Option B:** Kaggle datasets (e.g., *Store Item Demand Forecasting*). Place the CSV into `data/raw/` as `sales.csv`.

> Configure the column names in `config.yaml`.

## Quick start
```bash
# 1) Create & activate venv (example with python -m venv)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix/Mac:
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Place your dataset
# Put a CSV into data/raw/sales.csv

# 4) Run baseline models (naive / moving average / seasonal naive)
python scripts/run_baselines.py --config config.yaml

# 5) (After EDA/feature engineering) run classic ML benchmarks
python scripts/run_ml.py --config config.yaml
```

## BI Output
- The scripts will write **tables** to `reports/metrics/` (for model comparison) and **forecast outputs** to `reports/` that you can **import to Tableau/Power BI**.
- Connect Tableau/Power BI to the CSVs (or to a cloud warehouse you add later).

## Roadmap (steps)
1. Setup & scaffold ✅
2. EDA + data cleaning
3. Baseline models (naive, moving average, seasonal naive)
4. Statistical models (ARIMA/SARIMA, ETS)
5. ML models (XGBoost/LightGBM) with exogenous features
6. Time-series CV + model comparison
7. Persist best model + batch forecasting
8. (Optional) API + Cloud storage + BI wiring
9. (Optional) DL (LSTM/TFT) for a premium variant

---

*Generated on 2025-08-17*
