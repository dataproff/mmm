# DBT MMM Datamart

This DBT project transforms raw advertising and revenue data into an MMM-ready datamart.

## Models

### Staging Models

**[stg_revenue.sql](models/staging/stg_revenue.sql)**
- Aggregates daily revenue and orders from transaction data
- Source: `raw_data.transactions` (update with your actual table)
- Output: Daily revenue, orders, average order value

**[stg_channel_spend.sql](models/staging/stg_channel_spend.sql)**
- Unions spend data from multiple advertising platforms
- Sources: TikTok, Meta, Google Ads, Bing, Pinterest, Snapchat
- Output: Daily spend, impressions, clicks by channel

### Mart Models

**[mmm_datamart.sql](models/mart/mmm_datamart.sql)**
- Final MMM-ready dataset combining all features
- **Long format** with `channel` column for scalability
- Includes: Revenue, channel spend, calendar events, time features
- This is the table used by Robyn for modeling

**Why Long Format?**
- ✅ Add new channels without code changes
- ✅ Scalable to 10s or 100s of channels
- ✅ More efficient for BigQuery storage
- ✅ Easier to query and maintain

### Seeds

**[calendar_events.csv](seeds/calendar_events.csv)**
- Holiday and promotion calendar
- Update with your market-specific events

## Configuration

Edit [dbt_project.yml](dbt_project.yml) to configure:
- Date range (`start_date` variable)
- Channel list
- Schema names

## Setup

1. Install DBT with BigQuery adapter:
```bash
pip install dbt-bigquery
```

2. Configure [profiles.yml](profiles.yml) with your GCP credentials

3. Update source table references in staging models

4. Run the project:
```bash
dbt deps
dbt seed
dbt run
dbt test
```

## Updating Source Tables

Before running, update these references in the staging models:

**stg_revenue.sql:**
```sql
FROM `{{ env_var('GCP_PROJECT_ID') }}.raw_data.transactions`
```
Change `raw_data.transactions` to your actual revenue table.

**stg_channel_spend.sql:**
```sql
FROM `{{ env_var('GCP_PROJECT_ID') }}.raw_data.tiktok_ads`
FROM `{{ env_var('GCP_PROJECT_ID') }}.raw_data.meta_ads`
-- etc.
```
Change `raw_data.*` to your actual ad platform tables.

## Output Schema (Long Format)

The final `mmm_datamart` table contains:

| Column | Type | Description |
|--------|------|-------------|
| date | DATE | Daily date dimension |
| channel | STRING | Marketing channel name (e.g., 'tiktok', 'meta') |
| spend | FLOAT | Daily spend for this channel |
| impressions | FLOAT | Daily impressions for this channel |
| clicks | FLOAT | Daily clicks for this channel |
| ctr | FLOAT | Click-through rate |
| cpc | FLOAT | Cost per click |
| revenue | FLOAT | Daily revenue (same for all channels on same date) |
| orders | INT | Daily order count |
| is_holiday | BOOL | Holiday flag |
| is_promotion | BOOL | Promotion flag |
| year, month, quarter | INT | Time features |

**Example Data:**
```
date       | channel    | spend | impressions | revenue | ...
2024-01-01 | tiktok     | 1000  | 50000       | 50000   | ...
2024-01-01 | meta       | 2000  | 80000       | 50000   | ...
2024-01-01 | google_ads | 1500  | 60000       | 50000   | ...
```

Note: Revenue and calendar features are repeated for each channel on the same date.

## Adding New Channels

To add a new channel (e.g., YouTube):

1. Add a new CTE in [stg_channel_spend.sql](models/staging/stg_channel_spend.sql):
```sql
youtube_spend AS (
  SELECT
    DATE(date) AS date,
    'youtube' AS channel,
    SUM(spend) AS spend,
    SUM(impressions) AS impressions,
    SUM(clicks) AS clicks
  FROM `{{ env_var('GCP_PROJECT_ID') }}.raw_data.youtube_ads`
  WHERE DATE(date) >= '{{ var("start_date") }}'
  GROUP BY 1, 2
)
```

2. Add to the `all_channels` union:
```sql
SELECT * FROM youtube_spend
UNION ALL
```

3. Run DBT:
```bash
dbt run
```

**That's it!** The channel will automatically appear in:
- ✅ mmm_datamart table
- ✅ Robyn model training
- ✅ Streamlit optimization app

## Best Practices

1. **Data Quality:** Ensure no gaps in daily data
2. **Consistency:** Use consistent currency and timezone
3. **History:** Include 24-36 months for reliable MMM
4. **Updates:** Schedule daily or weekly refreshes
