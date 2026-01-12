{{
  config(
    materialized='view',
    description='Email and push notification campaigns - owned media channels'
  )
}}

-- Reading from seed files for testing/development
-- For production, replace ref('raw_...') with actual BigQuery table references

WITH email_campaigns AS (
  SELECT
    DATE(sent_date) AS date,
    'email' AS channel_type,
    SUM(emails_sent) AS sends,
    SUM(emails_delivered) AS delivered,
    SUM(emails_opened) AS opens,
    SUM(emails_clicked) AS clicks,
    SUM(unsubscribes) AS unsubscribes,
    SUM(revenue_attributed) AS attributed_revenue
  FROM {{ ref('raw_email_campaigns') }}
  WHERE DATE(sent_date) >= '{{ var("start_date") }}'
  GROUP BY 1, 2
),

push_campaigns AS (
  SELECT
    DATE(sent_date) AS date,
    'push' AS channel_type,
    SUM(pushes_sent) AS sends,
    SUM(pushes_delivered) AS delivered,
    SUM(pushes_opened) AS opens,
    SUM(pushes_clicked) AS clicks,
    0 AS unsubscribes,  -- Push typically doesn't track unsubscribes the same way
    SUM(revenue_attributed) AS attributed_revenue
  FROM {{ ref('raw_push_campaigns') }}
  WHERE DATE(sent_date) >= '{{ var("start_date") }}'
  GROUP BY 1, 2
),

combined AS (
  SELECT * FROM email_campaigns
  UNION ALL
  SELECT * FROM push_campaigns
)

SELECT
  date,
  channel_type,
  sends,
  delivered,
  opens,
  clicks,
  unsubscribes,
  attributed_revenue,

  -- Calculated metrics
  SAFE_DIVIDE(delivered, sends) AS delivery_rate,
  SAFE_DIVIDE(opens, delivered) AS open_rate,
  SAFE_DIVIDE(clicks, opens) AS click_to_open_rate,
  SAFE_DIVIDE(clicks, delivered) AS ctr

FROM combined
ORDER BY date, channel_type
