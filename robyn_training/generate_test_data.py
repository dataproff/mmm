"""
Generate realistic test data for Robyn MMM training

Creates synthetic data with:
- S-shaped saturation curves (Hill function) for each channel
- Adstock/carryover effects
- Different ROI and saturation parameters per channel
- Seasonality and trend
- Control variables (holidays, refinancing rate, CRM)
- Realistic noise
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


def hill_saturation(spend: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """
    Apply Hill saturation curve (S-shaped when alpha > 1)

    Args:
        spend: Spend values
        alpha: Shape parameter (>1 for S-curve, <1 for concave)
        gamma: Inflection point (spend level at 50% saturation)

    Returns:
        Saturated response (0 to 1 scale)
    """
    x = spend / gamma
    return (x ** alpha) / (1 + x ** alpha)


def apply_adstock(spend_series: np.ndarray, theta: float) -> np.ndarray:
    """
    Apply geometric adstock transformation (carryover effect)

    Args:
        spend_series: Daily spend values
        theta: Decay rate (0-1, higher = longer memory)

    Returns:
        Adstocked spend series
    """
    adstocked = np.zeros_like(spend_series)
    adstocked[0] = spend_series[0]

    for t in range(1, len(spend_series)):
        adstocked[t] = spend_series[t] + theta * adstocked[t-1]

    return adstocked


def generate_test_data(output_path: str = "./robyn_training/data/mmm_datamart_test.csv"):
    """
    Generate synthetic MMM data with realistic response curves

    The data simulates:
    - 6 marketing channels with different characteristics
    - Non-linear response to spend (S-curves)
    - Carryover effects (adstock)
    - Seasonality and trend
    - Control variables
    """
    np.random.seed(42)

    # Date range - 1 year of daily data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)

    # Channel configuration with realistic parameters
    # SCENARIO: Company is in GROWTH PHASE - spending BELOW optimal inflection points
    # This allows us to recommend INCREASING budgets for most channels
    # gamma_factor is set to 3-5x of avg spend so inflection is AHEAD of current spend
    channel_config = {
        'google_ads': {
            'spend_range': (2000, 4000),       # Conservative daily spend
            'max_contribution': 120000,        # HIGH potential - not yet tapped
            'adstock_theta': 0.40,             # Medium carryover
            'saturation_alpha': 2.2,           # Strong S-curve (steep growth phase)
            'gamma_factor': 4.0,               # Inflection at 4x avg spend - lots of room!
            'noise_std': 0.10,
        },
        'meta': {
            'spend_range': (2500, 5000),       # Conservative spend
            'max_contribution': 150000,        # Highest potential
            'adstock_theta': 0.55,             # Higher carryover (brand effect)
            'saturation_alpha': 2.0,           # Strong S-curve
            'gamma_factor': 3.5,               # Inflection at 3.5x avg spend
            'noise_std': 0.08,
        },
        'tiktok': {
            'spend_range': (800, 1800),        # New channel, testing phase
            'max_contribution': 50000,         # Big potential for growth
            'adstock_theta': 0.15,             # Low carryover (impulse)
            'saturation_alpha': 1.8,           # S-curve
            'gamma_factor': 5.0,               # Lots of room to scale - early stage
            'noise_std': 0.15,
        },
        'bing': {
            'spend_range': (400, 800),         # Small but efficient channel
            'max_contribution': 20000,         # Room to grow
            'adstock_theta': 0.38,             # Medium carryover
            'saturation_alpha': 1.6,           # Moderate S-curve
            'gamma_factor': 4.0,               # Room to scale
            'noise_std': 0.10,
        },
        'pinterest': {
            'spend_range': (300, 700),         # Small channel
            'max_contribution': 15000,         # Potential to grow
            'adstock_theta': 0.30,             # Medium carryover
            'saturation_alpha': 1.8,           # S-curve
            'gamma_factor': 4.5,               # Room to scale
            'noise_std': 0.12,
        },
        'snapchat': {
            'spend_range': (200, 500),         # Smallest channel - testing
            'max_contribution': 10000,         # Potential to grow
            'adstock_theta': 0.10,             # Very low carryover
            'saturation_alpha': 2.5,           # Strong S-curve (viral potential)
            'gamma_factor': 5.0,               # Room to scale
            'noise_std': 0.18,
        },
    }

    channels = list(channel_config.keys())

    # Generate base spend for each channel (with weekly/seasonal patterns AND monthly growth)
    channel_spends = {}
    for channel, cfg in channel_config.items():
        base = np.random.uniform(cfg['spend_range'][0], cfg['spend_range'][1], n_days)

        # Add weekly pattern (less on weekends for B2B-ish channels)
        weekday_factor = np.array([1.1, 1.15, 1.1, 1.05, 0.95, 0.75, 0.70])
        weekly = np.array([weekday_factor[d.weekday()] for d in dates])

        # Add monthly variation (budget cycles)
        monthly_cycle = 1 + 0.10 * np.sin(2 * np.pi * np.arange(n_days) / 30)

        # MINIMAL growth trend - company is conservative, not yet scaling
        # Only ~15% growth over year to keep spend BELOW inflection points
        monthly_growth = 1 + 0.15 * (np.arange(n_days) / n_days)  # 1.0 at start -> 1.15 at end

        # Add some random spikes (campaign bursts)
        spikes = np.ones(n_days)
        spike_days = np.random.choice(n_days, size=20, replace=False)
        spikes[spike_days] = np.random.uniform(1.3, 1.8, 20)

        channel_spends[channel] = base * weekly * monthly_cycle * monthly_growth * spikes

        # Ensure non-negative
        channel_spends[channel] = np.maximum(channel_spends[channel], 100)

    # Calculate channel contributions using response curves
    channel_contributions = {}
    for channel, cfg in channel_config.items():
        spend = channel_spends[channel]

        # Apply adstock (carryover effect)
        adstocked = apply_adstock(spend, cfg['adstock_theta'])

        # Calculate gamma (inflection point) based on average spend
        avg_spend = spend.mean()
        gamma = avg_spend * cfg['gamma_factor']

        # Apply saturation curve
        saturated = hill_saturation(adstocked, cfg['saturation_alpha'], gamma)

        # Scale to contribution (0-1 -> 0-max_contribution)
        base_contribution = saturated * cfg['max_contribution']

        # Add multiplicative noise
        noise = 1 + np.random.normal(0, cfg['noise_std'], n_days)
        contribution = base_contribution * noise

        # Ensure non-negative
        channel_contributions[channel] = np.maximum(contribution, 0)

    # Generate base revenue (organic + trend + seasonality)
    # Organic baseline that doesn't depend on marketing
    organic_base = 20000  # Daily organic revenue (lower base, more growth)

    # Trend (strong growth over the year - matching spend growth)
    trend = 1 + 0.6 * np.arange(n_days) / n_days  # 60% growth over year

    # Annual seasonality
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    annual_season = 1 + 0.25 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak in spring

    # Weekly seasonality
    weekday = np.array([d.weekday() for d in dates])
    weekly_season = np.where(weekday < 5, 1.0, 0.75)  # Lower on weekends

    organic_revenue = organic_base * trend * annual_season * weekly_season

    # Control variables
    # Holidays
    is_holiday = np.zeros(n_days)
    holiday_dates = [
        (1, 1), (1, 16), (2, 20), (5, 29), (7, 4), (9, 4),
        (10, 9), (11, 11), (11, 23), (12, 25)
    ]
    for m, d in holiday_dates:
        try:
            idx = dates.get_loc(datetime(2023, m, d))
            is_holiday[idx] = 1
            # Holiday effect extends a few days
            for offset in [-1, 1, 2]:
                if 0 <= idx + offset < n_days:
                    is_holiday[idx + offset] = 0.5
        except KeyError:
            pass

    # Holiday shopping season (Black Friday to Christmas)
    holiday_season_mask = (dates >= datetime(2023, 11, 24)) & (dates <= datetime(2023, 12, 31))
    is_holiday[holiday_season_mask] = 1

    # Holiday multiplier effect on revenue
    holiday_multiplier = 1 + 0.5 * is_holiday  # Up to 50% boost

    # Refinancing rate (macro factor - slowly varying)
    refinancing_rate = 7.5 + 1.5 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    refinancing_rate += 0.3 * np.cumsum(np.random.randn(n_days) * 0.1)  # Random walk
    refinancing_rate = np.clip(refinancing_rate, 5.0, 10.0)

    # Refinancing effect (higher rate = lower demand for some products)
    refinancing_effect = 1 - 0.02 * (refinancing_rate - 7.5)

    # CRM variables (email and push)
    email_sends = np.random.uniform(8000, 15000, n_days).astype(int)
    email_open_rate = np.random.uniform(0.15, 0.25, n_days)
    email_click_rate = np.random.uniform(0.02, 0.05, n_days)
    email_clicks = (email_sends * email_open_rate * email_click_rate).astype(int)

    push_sends = np.random.uniform(5000, 12000, n_days).astype(int)
    push_click_rate = np.random.uniform(0.03, 0.08, n_days)
    push_clicks = (push_sends * push_click_rate).astype(int)

    # CRM contribution to revenue
    crm_contribution = email_clicks * 2.5 + push_clicks * 1.5  # Simplified attribution

    # Promotion days (10% of days have promotions)
    is_promotion = (np.random.random(n_days) < 0.10).astype(int)
    promotion_effect = 1 + 0.3 * is_promotion  # 30% boost on promotion days

    # Calculate total revenue
    # Revenue = Organic + Sum(Channel Contributions) + CRM + Effects
    total_channel_contribution = sum(channel_contributions.values())

    total_revenue = (
        organic_revenue * refinancing_effect * holiday_multiplier * promotion_effect
        + total_channel_contribution * holiday_multiplier
        + crm_contribution
    )

    # Add final noise (measurement error, unexplained variance)
    total_revenue *= (1 + np.random.normal(0, 0.05, n_days))
    total_revenue = np.maximum(total_revenue, 0)

    # Build the output dataframe (long format - one row per channel-date)
    records = []
    for i, date in enumerate(dates):
        for channel in channels:
            records.append({
                'date': date.strftime('%Y-%m-%d'),
                'channel': channel,
                'spend': round(channel_spends[channel][i], 2),
                'impressions': int(channel_spends[channel][i] * np.random.uniform(80, 150)),
                'clicks': int(channel_spends[channel][i] * np.random.uniform(0.5, 1.5)),
                'revenue': round(total_revenue[i], 2),  # Daily revenue (same for all channels)
                'is_holiday': int(is_holiday[i] >= 0.5),
                'is_promotion': int(is_promotion[i]),
                'refinancing_rate': round(refinancing_rate[i], 2),
                'email_sends': int(email_sends[i]),
                'email_clicks': int(email_clicks[i]),
                'push_sends': int(push_sends[i]),
                'push_clicks': int(push_clicks[i]),
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    # Print summary
    print(f"Generated {len(df)} rows to {output_path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Channels: {df['channel'].unique().tolist()}")
    print(f"\nChannel spend summary:")
    for channel in channels:
        ch_df = df[df['channel'] == channel]
        cfg = channel_config[channel]
        total_spend = ch_df['spend'].sum()
        total_contrib = sum(channel_contributions[channel])
        implied_roi = total_contrib / total_spend if total_spend > 0 else 0
        print(f"  {channel:12s}: spend=${total_spend:>12,.0f}, contrib=${total_contrib:>12,.0f}, "
              f"ROI={implied_roi:.2f}, theta={cfg['adstock_theta']:.2f}, alpha={cfg['saturation_alpha']:.2f}")

    print(f"\nTotal spend: ${df['spend'].sum():,.0f}")
    print(f"Total revenue: ${df['revenue'].sum():,.0f}")
    print(f"Organic revenue: ${organic_revenue.sum():,.0f}")
    print(f"Marketing contribution: ${sum(c.sum() for c in channel_contributions.values()):,.0f}")

    return df


if __name__ == "__main__":
    generate_test_data()
