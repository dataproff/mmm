# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Commerce MMM (Marketing Mix Modeling) Accelerator — a platform for e-commerce budget optimization. Three-layer architecture: BigQuery/DBT data pipeline → Meta Robyn model training → Streamlit interactive dashboard.

## Commands

```bash
make setup          # Install all deps (dbt deps + pip install for both components)
make dbt-run        # Seed + run DBT models: cd dbt_project && dbt seed && dbt run
make dbt-test       # Run DBT data quality tests: cd dbt_project && dbt test
make train-model    # Train Robyn model: cd robyn_training && python train_model.py
make run-app        # Start Streamlit: cd streamlit_app && streamlit run app.py
make docker-build   # Build training Docker image
make deploy-app     # Deploy Streamlit to Cloud Run
make clean          # Remove target/, dbt_packages/, models/*, __pycache__
```

Note: The Streamlit entry point is `streamlit_app/Home.py` but Makefile references `app.py` — use `cd streamlit_app && streamlit run Home.py` to run directly.

## Architecture & Data Flow

```
Raw ad platform data (TikTok, Meta, Google, Bing, Pinterest, Snapchat)
  → BigQuery raw tables
  → DBT staging views (stg_revenue, stg_channel_spend, stg_refinancing_rate, stg_email_push)
  → DBT mart table (mmm_datamart) — long format, one row per channel per day
  → Robyn training (train_model.py pivots long→wide, trains model, outputs JSON artifacts)
  → Streamlit app (loads robyn_results.json, runs SciPy optimization for budget allocation)
```

### DBT Layer (`dbt_project/`)
- **Staging models** are views; **mart models** are tables (materialized in `dbt_project.yml`)
- The datamart uses **long format** (scalable to unlimited channels) — Robyn training pivots to wide
- Seed CSVs in `dbt_project/seeds/` provide reference data (calendar events, raw ad data, transactions)
- BigQuery dataset configured via `profiles.yml` (dev: `mmm_dev`, prod: `mmm_prod`)

### Robyn Training (`robyn_training/`)
- `train_model.py` — `RobynMMM` class orchestrates: fetch → pivot → prepare → train → allocate → save
- Config in `config/robyn_config.yaml` (data source, channels, hyperparameters, adstock settings)
- Outputs: `models/robyn_results.json` (channel ROI, contributions, response curves) and `robyn_model.pkl`
- Uses `robynpy` (Meta's Python API) primarily; optional R integration via `rpy2`
- Has fallback dev mode when R/Robyn not available

### Streamlit App (`streamlit_app/`)
- Multi-page app: `Home.py` (about), `pages/1_🧮_Calculator.py` (budget optimizer), `pages/2_📅_Context_Calendar.py`, `pages/3_📈_Saturation_Curves.py`
- `utils/optimizer.py` — `BudgetOptimizer` class using SciPy SLSQP: geometric adstock + Hill saturation curves
- `utils/robyn_optimizer.py` — `RobynOptimizer` wraps native Robyn model with context variable support
- `utils/model_loader.py` — loads `robyn_results.json`, provides fallback defaults if missing
- `utils/i18n.py` — English/Russian translations from `locales/{en,ru}.json`
- `utils/context_calendar.py` — manages calendar events (holidays, promotions, macro factors)

## Key Conventions

- **Long format datamart**: The DBT mart stores data as one row per (date, channel). Robyn training pivots this to wide format. When adding channels, add them in `stg_channel_spend.sql` and the pipeline propagates automatically.
- **Model artifacts as JSON**: The Streamlit app reads `robyn_results.json` for channel parameters (alpha, gamma, carryover, ROI). This is the contract between training and serving.
- **Environment**: Requires `GCP_PROJECT_ID` and `GOOGLE_APPLICATION_CREDENTIALS` for BigQuery access. See `.env.example`.
- **Python 3.11** (configured in `.devcontainer/devcontainer.json`).
