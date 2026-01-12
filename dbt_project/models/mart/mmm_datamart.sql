{{
  config(
    materialized='table',
    description='Final MMM-ready datamart with all features for Robyn modeling'
  )
}}

WITH revenue_daily AS (
  SELECT * FROM {{ ref('stg_revenue') }}
),

channel_spend AS (
  SELECT * FROM {{ ref('stg_channel_spend') }}
),

-- Generate date spine based on actual data range (not hardcoded to CURRENT_DATE)
date_spine AS (
  SELECT date
  FROM UNNEST(GENERATE_DATE_ARRAY(
    (SELECT MIN(date) FROM channel_spend),
    (SELECT MAX(date) FROM channel_spend)
  )) AS date
),

calendar AS (
  SELECT
    DATE(date) AS date,
    MAX(CASE WHEN is_holiday = 1 THEN 1 ELSE 0 END) AS is_holiday,
    MAX(CASE WHEN event_type = 'promotion' THEN 1 ELSE 0 END) AS is_promotion,
    STRING_AGG(DISTINCT event_name, ', ') AS events
  FROM {{ ref('calendar_events') }}
  GROUP BY 1
),

-- Refinancing rate (macroeconomic control variable)
refinancing AS (
  SELECT
    date,
    refinancing_rate,
    rate_changed
  FROM {{ ref('stg_refinancing_rate') }}
),

-- Email and push campaigns (owned media)
email_push AS (
  SELECT
    date,
    -- Aggregate email metrics
    SUM(CASE WHEN channel_type = 'email' THEN sends ELSE 0 END) AS email_sends,
    SUM(CASE WHEN channel_type = 'email' THEN opens ELSE 0 END) AS email_opens,
    SUM(CASE WHEN channel_type = 'email' THEN clicks ELSE 0 END) AS email_clicks,
    -- Aggregate push metrics
    SUM(CASE WHEN channel_type = 'push' THEN sends ELSE 0 END) AS push_sends,
    SUM(CASE WHEN channel_type = 'push' THEN opens ELSE 0 END) AS push_opens,
    SUM(CASE WHEN channel_type = 'push' THEN clicks ELSE 0 END) AS push_clicks,
    -- Combined metrics
    SUM(sends) AS total_crm_sends,
    SUM(clicks) AS total_crm_clicks
  FROM {{ ref('stg_email_push') }}
  GROUP BY 1
),

-- Combine date spine with revenue, calendar, macro, and CRM data
base_data AS (
  SELECT
    ds.date,

    -- KPIs
    COALESCE(r.revenue, 0) AS revenue,
    COALESCE(r.orders, 0) AS orders,
    COALESCE(r.avg_order_value, 0) AS avg_order_value,

    -- Calendar features
    COALESCE(c.is_holiday, 0) AS is_holiday,
    COALESCE(c.is_promotion, 0) AS is_promotion,
    c.events,

    -- Macroeconomic control variables
    COALESCE(ref.refinancing_rate, 0) AS refinancing_rate,
    COALESCE(ref.rate_changed, 0) AS rate_changed,

    -- Email/Push CRM metrics (owned media)
    COALESCE(ep.email_sends, 0) AS email_sends,
    COALESCE(ep.email_opens, 0) AS email_opens,
    COALESCE(ep.email_clicks, 0) AS email_clicks,
    COALESCE(ep.push_sends, 0) AS push_sends,
    COALESCE(ep.push_opens, 0) AS push_opens,
    COALESCE(ep.push_clicks, 0) AS push_clicks,
    COALESCE(ep.total_crm_sends, 0) AS total_crm_sends,
    COALESCE(ep.total_crm_clicks, 0) AS total_crm_clicks,

    -- Time features for trend/seasonality
    EXTRACT(YEAR FROM ds.date) AS year,
    EXTRACT(MONTH FROM ds.date) AS month,
    EXTRACT(QUARTER FROM ds.date) AS quarter,
    EXTRACT(DAYOFWEEK FROM ds.date) AS day_of_week,
    EXTRACT(WEEK FROM ds.date) AS week_of_year

  FROM date_spine ds
  LEFT JOIN revenue_daily r ON ds.date = r.date
  LEFT JOIN calendar c ON ds.date = c.date
  LEFT JOIN refinancing ref ON ds.date = ref.date
  LEFT JOIN email_push ep ON ds.date = ep.date
),

-- Join channel spend data with base data
final AS (
  SELECT
    b.date,

    -- Channel information (scalable to new channels)
    channels.channel,
    COALESCE(cs.spend, 0) AS spend,
    COALESCE(cs.impressions, 0) AS impressions,
    COALESCE(cs.clicks, 0) AS clicks,
    COALESCE(cs.ctr, 0) AS ctr,
    COALESCE(cs.cpc, 0) AS cpc,

    -- KPIs (same for all channels on same date)
    b.revenue,
    b.orders,
    b.avg_order_value,

    -- Calendar features (same for all channels on same date)
    b.is_holiday,
    b.is_promotion,
    b.events,

    -- Macroeconomic control variables (same for all channels on same date)
    b.refinancing_rate,
    b.rate_changed,

    -- Email/Push CRM metrics (same for all channels on same date)
    b.email_sends,
    b.email_opens,
    b.email_clicks,
    b.push_sends,
    b.push_opens,
    b.push_clicks,
    b.total_crm_sends,
    b.total_crm_clicks,

    -- Time features (same for all channels on same date)
    b.year,
    b.month,
    b.quarter,
    b.day_of_week,
    b.week_of_year

  FROM base_data b
  CROSS JOIN (SELECT DISTINCT channel FROM channel_spend) channels
  LEFT JOIN channel_spend cs
    ON b.date = cs.date
    AND channels.channel = cs.channel
)

SELECT * FROM final
ORDER BY date
