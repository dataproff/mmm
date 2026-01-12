{{
  config(
    materialized='view',
    description='Central Bank refinancing rate (key rate) - macroeconomic control variable'
  )
}}

-- Reading from seed file for testing/development
-- For production, replace {{ ref('raw_central_bank_rates') }} with actual BigQuery table
-- The rate typically changes infrequently (a few times per year)
-- We forward-fill to get daily values

WITH rate_changes AS (
  SELECT
    DATE(effective_date) AS date,
    rate_value AS refinancing_rate  -- Rate as decimal (e.g., 0.16 for 16%)
  FROM {{ ref('raw_central_bank_rates') }}
  WHERE rate_type = 'key_rate'
),

-- Generate date spine for forward-filling
date_spine AS (
  SELECT date
  FROM UNNEST(GENERATE_DATE_ARRAY('{{ var("start_date") }}', CURRENT_DATE())) AS date
),

-- Forward-fill rates to get daily values
daily_rates AS (
  SELECT
    ds.date,
    LAST_VALUE(rc.refinancing_rate IGNORE NULLS) OVER (
      ORDER BY ds.date
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS refinancing_rate
  FROM date_spine ds
  LEFT JOIN rate_changes rc ON ds.date = rc.date
)

SELECT
  date,
  refinancing_rate,
  -- Rate change indicator (useful for modeling regime changes)
  CASE
    WHEN refinancing_rate != LAG(refinancing_rate) OVER (ORDER BY date)
    THEN 1
    ELSE 0
  END AS rate_changed
FROM daily_rates
WHERE refinancing_rate IS NOT NULL
ORDER BY date
