{{
  config(
    materialized='view',
    description='Daily revenue/orders aggregated from source data'
  )
}}

-- Reading from seed file for testing/development
-- For production, replace with actual BigQuery table reference

WITH source_data AS (
  SELECT
    DATE(date) AS date,
    SUM(revenue) AS revenue,
    SUM(orders) AS orders
  FROM {{ ref('raw_transactions') }}
  WHERE DATE(date) >= '{{ var("start_date") }}'
  GROUP BY 1
)

SELECT
  date,
  revenue,
  orders,
  revenue / NULLIF(orders, 0) AS avg_order_value
FROM source_data
ORDER BY date
