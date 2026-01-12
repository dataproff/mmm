{{
  config(
    materialized='view',
    description='Daily marketing spend by channel from advertising platforms'
  )
}}

-- Reading from seed files for testing/development
-- For production, replace ref('raw_...') with actual BigQuery table references

WITH tiktok_spend AS (
  SELECT
    DATE(date) AS date,
    'tiktok' AS channel,
    SUM(spend) AS spend,
    SUM(impressions) AS impressions,
    SUM(clicks) AS clicks
  FROM {{ ref('raw_tiktok_ads') }}
  WHERE DATE(date) >= '{{ var("start_date") }}'
  GROUP BY 1, 2
),

meta_spend AS (
  SELECT
    DATE(date) AS date,
    'meta' AS channel,
    SUM(spend) AS spend,
    SUM(impressions) AS impressions,
    SUM(clicks) AS clicks
  FROM {{ ref('raw_meta_ads') }}
  WHERE DATE(date) >= '{{ var("start_date") }}'
  GROUP BY 1, 2
),

google_ads_spend AS (
  SELECT
    DATE(date) AS date,
    'google_ads' AS channel,
    SUM(cost) AS spend,
    SUM(impressions) AS impressions,
    SUM(clicks) AS clicks
  FROM {{ ref('raw_google_ads') }}
  WHERE DATE(date) >= '{{ var("start_date") }}'
  GROUP BY 1, 2
),

bing_spend AS (
  SELECT
    DATE(date) AS date,
    'bing' AS channel,
    SUM(spend) AS spend,
    SUM(impressions) AS impressions,
    SUM(clicks) AS clicks
  FROM {{ ref('raw_bing_ads') }}
  WHERE DATE(date) >= '{{ var("start_date") }}'
  GROUP BY 1, 2
),

pinterest_spend AS (
  SELECT
    DATE(date) AS date,
    'pinterest' AS channel,
    SUM(spend) AS spend,
    SUM(impressions) AS impressions,
    SUM(clicks) AS clicks
  FROM {{ ref('raw_pinterest_ads') }}
  WHERE DATE(date) >= '{{ var("start_date") }}'
  GROUP BY 1, 2
),

snapchat_spend AS (
  SELECT
    DATE(date) AS date,
    'snapchat' AS channel,
    SUM(spend) AS spend,
    SUM(impressions) AS impressions,
    SUM(swipes) AS clicks
  FROM {{ ref('raw_snapchat_ads') }}
  WHERE DATE(date) >= '{{ var("start_date") }}'
  GROUP BY 1, 2
),

all_channels AS (
  SELECT * FROM tiktok_spend
  UNION ALL
  SELECT * FROM meta_spend
  UNION ALL
  SELECT * FROM google_ads_spend
  UNION ALL
  SELECT * FROM bing_spend
  UNION ALL
  SELECT * FROM pinterest_spend
  UNION ALL
  SELECT * FROM snapchat_spend
)

SELECT
  date,
  channel,
  spend,
  impressions,
  clicks,
  SAFE_DIVIDE(clicks, impressions) AS ctr,
  SAFE_DIVIDE(spend, clicks) AS cpc
FROM all_channels
ORDER BY date, channel
