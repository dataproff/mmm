# Commerce MMM Accelerator

Incrementality-based Marketing Mix Modeling (MMM) system for e-commerce budget optimization powered by BigQuery, DBT, Streamlit, and Robyn (Meta).

## Overview

This project implements a complete MMM pipeline that:
1. **Transforms raw data** into MMM-ready datamart using DBT (long format - scalable!)
2. **Trains MMM models** using Meta's Robyn framework
3. **Provides interactive tools** for budget optimization via Streamlit

## 🚀 Key Features

✅ **Scalable Architecture** - Long format data model: add new channels without code changes
✅ **Auto-Channel Detection** - Channels automatically discovered from data
✅ **Production Ready** - Docker, Cloud Run, monitoring included
✅ **Interactive** - Streamlit app for budget planning
✅ **Three Use Cases** - Budget for leads, maximize leads, scenario analysis
✅ **Industry Standard** - Based on Meta's Robyn MMM framework

## Architecture

```
┌─────────────────┐
│   BigQuery      │  Raw data from ad platforms
│   (Raw Data)    │  (TikTok, Meta, Google, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   DBT Models    │  Data transformation
│  (Staging/Mart) │  → mmm_datamart table
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Robyn Training │  MMM model training
│  (Python + R)   │  → Model artifacts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit App  │  Interactive optimization
│ (Budget Planner)│  → Business decisions
└─────────────────┘
```

## Project Structure

```
.
├── dbt_project/                # DBT transformation layer
│   ├── models/
│   │   ├── staging/           # Staging models
│   │   │   ├── stg_revenue.sql
│   │   │   └── stg_channel_spend.sql
│   │   └── mart/              # Final datamart
│   │       └── mmm_datamart.sql
│   ├── seeds/                 # Static data (holidays, etc.)
│   │   └── calendar_events.csv
│   ├── dbt_project.yml
│   └── profiles.yml
│
├── robyn_training/            # MMM model training
│   ├── config/
│   │   └── robyn_config.yaml
│   ├── utils/
│   │   └── bigquery_client.py
│   ├── train_model.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── streamlit_app/            # Interactive web app
│   ├── utils/
│   │   ├── model_loader.py
│   │   └── optimizer.py
│   ├── app.py
│   └── requirements.txt
│
├── .env.example
└── README.md
```

## Prerequisites

- **Google Cloud Project** with BigQuery enabled
- **Service Account** with BigQuery access
- **Python 3.9+**
- **R 4.0+** (for Robyn)
- **DBT** (Data Build Tool)
- **Docker** (optional, for containerized training)

## Setup Instructions

### 1. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your GCP credentials
nano .env
```

Required environment variables:
- `GCP_PROJECT_ID`: Your GCP project ID
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account JSON key

### 2. DBT Setup

```bash
cd dbt_project

# Install DBT with BigQuery adapter
pip install dbt-bigquery

# Configure connection
# Edit profiles.yml with your GCP settings

# Run DBT to create datamart
dbt deps
dbt seed  # Load calendar data
dbt run   # Build models
dbt test  # Run data quality tests
```

**Important:** Update the source table references in [stg_revenue.sql](dbt_project/models/staging/stg_revenue.sql) and [stg_channel_spend.sql](dbt_project/models/staging/stg_channel_spend.sql) to point to your actual BigQuery tables.

### 3. Robyn Model Training

#### Option A: Local Training (requires R)

```bash
cd robyn_training

# Install Python dependencies
pip install -r requirements.txt

# Install R and Robyn
# In R console:
# install.packages("Robyn")

# Configure model
nano config/robyn_config.yaml

# Train model
python train_model.py
```

#### Option B: Docker Training

```bash
cd robyn_training

# Build Docker image
docker build -t mmm-robyn-trainer .

# Run training
docker run \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/config:/app/config \
  -e GCP_PROJECT_ID=$GCP_PROJECT_ID \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  -v /path/to/service-account.json:/app/credentials.json \
  mmm-robyn-trainer
```

### 4. Streamlit App

```bash
cd streamlit_app

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

Access the app at `http://localhost:8501`

## 🎯 Scalable Architecture

### Long Format Data Model

The system uses a **long format** (also called "tall" or "narrow") data structure for maximum scalability:

**Data Structure:**
```sql
date       | channel    | spend | impressions | revenue | is_holiday | ...
2024-01-01 | tiktok     | 1000  | 50000       | 50000   | 0          | ...
2024-01-01 | meta       | 2000  | 80000       | 50000   | 0          | ...
2024-01-01 | google_ads | 1500  | 60000       | 50000   | 0          | ...
```

### Adding New Channels

**No code changes needed!** Just add data to your source:

```sql
-- In dbt_project/models/staging/stg_channel_spend.sql
-- Add a new CTE for your channel:

youtube_spend AS (
  SELECT
    DATE(date) AS date,
    'youtube' AS channel,  -- Channel name
    SUM(spend) AS spend,
    SUM(impressions) AS impressions,
    SUM(clicks) AS clicks
  FROM `{{ env_var('GCP_PROJECT_ID') }}.raw_data.youtube_ads`
  WHERE DATE(date) >= '{{ var("start_date") }}'
  GROUP BY 1, 2
),

-- Add to the union:
SELECT * FROM youtube_spend
UNION ALL
...
```

**That's it!** The rest of the pipeline automatically:
- ✅ Detects the new channel
- ✅ Includes it in model training
- ✅ Shows it in the Streamlit app
- ✅ Optimizes budget allocation for it

### Benefits

| Aspect | Long Format | Wide Format (Old) |
|--------|-------------|-------------------|
| **Scalability** | ✅ Unlimited channels | ❌ Code change per channel |
| **Maintenance** | ✅ One pattern for all | ❌ Multiple places to update |
| **Flexibility** | ✅ Easy filtering/grouping | ❌ Complex SQL |
| **Database** | ✅ Efficient storage | ❌ Wide tables |
| **Auto-detection** | ✅ Channels from data | ❌ Hardcoded lists |

## Use Cases

### 1️⃣ Budget for Target Leads

**Goal:** Find the minimum budget required to achieve a specific number of leads/revenue.

**Inputs:**
- Target leads/revenue (e.g., 50,000 leads)
- Channel budget constraints (min/max per channel)

**Outputs:**
- Optimal budget allocation by channel
- Total budget required
- Predicted lead achievement

### 2️⃣ Maximize Leads for Budget

**Goal:** Determine the optimal channel mix to maximize leads within a fixed budget.

**Inputs:**
- Total available budget (e.g., $100,000)
- Channel budget constraints

**Outputs:**
- Optimal budget allocation
- Maximum predicted leads
- ROI metrics

### 3️⃣ Scenario Analysis

**Goal:** Explore how performance scales across different budget levels.

**Inputs:**
- Budget range (min to max)
- Number of scenarios

**Outputs:**
- Response curves (budget vs. leads)
- ROI curves (diminishing returns)
- Comparative scenario table

## Configuration

### DBT Variables

Edit [dbt_project.yml](dbt_project/dbt_project.yml:27-38):
```yaml
vars:
  start_date: '2022-01-01'
  channels:
    - tiktok
    - meta
    - google_ads
```

### Robyn Configuration

Edit [robyn_config.yaml](robyn_training/config/robyn_config.yaml):
```yaml
model:
  dep_var: "revenue"  # or "orders"
  paid_media_spends:
    - "tiktok_spend"
    - "meta_spend"
  adstock: "geometric"
```

## Data Requirements

The system expects daily-level data with:

**Revenue/Orders:**
- Date
- Revenue or order count
- Transaction metadata

**Channel Spend:**
- Date
- Channel name
- Spend amount
- Impressions, clicks (optional)

**Calendar Events:**
- Holidays (New Year, Thanksgiving, etc.)
- Promotions (Black Friday, sales events)

Minimum history: **24-36 months** for reliable MMM

## Deployment

### Google Cloud Run (Streamlit)

```bash
# Build container
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/mmm-app streamlit_app/

# Deploy
gcloud run deploy mmm-app \
  --image gcr.io/$GCP_PROJECT_ID/mmm-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Cloud Scheduler (Training)

```bash
# Schedule weekly model retraining
gcloud scheduler jobs create http mmm-training \
  --schedule="0 2 * * 0" \
  --uri="https://YOUR-CLOUD-RUN-URL/train" \
  --http-method=POST
```

## Model Outputs

After training, Robyn generates:

- **Response curves** - Channel saturation curves
- **Channel contribution** - Incremental revenue by channel
- **Adstock effects** - Carryover impact of spend
- **ROI metrics** - Return on ad spend by channel
- **Decomposition** - Base vs. incremental revenue

These are stored in `robyn_training/models/` and loaded by the Streamlit app.

## Best Practices

1. **Data Quality:** Ensure clean, consistent daily data with no gaps
2. **Calibration:** Use incrementality tests to calibrate the model when available
3. **Regular Updates:** Retrain monthly or quarterly as new data arrives
4. **Validation:** Compare model predictions against holdout periods
5. **Business Context:** Incorporate seasonality, promotions, and external factors

## Troubleshooting

### DBT Errors

- **"Relation not found"**: Update source table paths in staging models
- **"Permission denied"**: Verify service account has BigQuery Data Editor role

### Robyn Training Issues

- **R package errors**: Ensure R version 4.0+ and Robyn is installed
- **Memory issues**: Reduce hyperparameter iterations or use larger machine

### Streamlit Issues

- **Model not found**: Train model first using `train_model.py`
- **BigQuery timeout**: Increase timeout in `bigquery_client.py`

## Pricing Estimate

Based on project scope:

- **Pilot project:** $15,000 – $25,000
- **Monthly support:** $2,000 – $5,000 (optional)

Includes:
- Custom datamart setup
- Model calibration
- Interactive planning tool
- Documentation and training

## Out of Scope

- User-level attribution
- Campaign or creative optimization
- Media buying or execution
- Real-time bidding integration

## References

- [Meta Robyn](https://facebookexperimental.github.io/Robyn/)
- [DBT Documentation](https://docs.getdbt.com/)
- [Streamlit Docs](https://docs.streamlit.io/)

## Support

For issues or questions:
- Create an issue in the repository
- Review the documentation
- Contact the development team

## License

Proprietary - All rights reserved
