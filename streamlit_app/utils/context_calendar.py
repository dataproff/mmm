"""
Context Calendar Loader

Loads and manages context variables calendar for MMM predictions.
Context variables include: holidays, promotions, Federal Funds Rate,
competitor pressure index.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ContextCalendar:
    """Manages context calendar data for MMM predictions"""

    def __init__(self, calendar_path: Optional[str] = None):
        """
        Initialize context calendar

        Args:
            calendar_path: Path to the CSV calendar file (default: data/context_calendar_2026.csv relative to streamlit_app)
        """
        if calendar_path is None:
            utils_dir = Path(__file__).parent
            calendar_path = utils_dir / ".." / "data" / "context_calendar_2026.csv"
            calendar_path = calendar_path.resolve()

        self.calendar_path = Path(calendar_path)
        self.df: Optional[pd.DataFrame] = None
        self._load_calendar()

    def _load_calendar(self):
        """Load calendar data from CSV"""
        if not self.calendar_path.exists():
            logger.warning(f"Calendar file not found: {self.calendar_path}")
            self.df = pd.DataFrame()
            return

        self.df = pd.read_csv(self.calendar_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        logger.info(f"Loaded calendar with {len(self.df)} days")

    def get_available_months(self) -> list:
        """Get list of available months in calendar"""
        if self.df is None or self.df.empty:
            return []

        months = self.df['date'].dt.to_period('M').unique()
        return sorted([str(m) for m in months])

    def get_month_data(self, year: int, month: int) -> pd.DataFrame:
        """Get calendar data for a specific month"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        mask = (self.df['date'].dt.year == year) & (self.df['date'].dt.month == month)
        return self.df[mask].copy()

    def get_weekly_data(self, year: int, month: int) -> pd.DataFrame:
        """
        Get calendar data aggregated by week for a specific month.
        Weeks are ISO weeks (Mon-Sun).
        """
        month_df = self.get_month_data(year, month)
        if month_df.empty:
            return pd.DataFrame()

        df = month_df.copy()
        df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.weekday, unit='D')
        df['week_label'] = df['week_start'].dt.strftime('%b %d') + ' – ' + (df['week_start'] + pd.Timedelta(days=6)).dt.strftime('%b %d')

        weekly = df.groupby(['week_start', 'week_label'], sort=True).agg(
            days_in_week=('date', 'count'),
            holidays=('is_holiday', 'sum'),
            promotion_days=('is_promotion', 'sum'),
            fed_funds_rate=('fed_funds_rate', 'mean'),
            competitor_pressure_index=('competitor_pressure_index', 'mean'),
            holiday_names=('holiday_name', lambda x: ', '.join(x.dropna().unique()) if x.dropna().any() else ''),
            promotion_names=('promotion_name', lambda x: ', '.join(x.dropna().unique()) if x.dropna().any() else ''),
        ).reset_index()

        weekly['competitor_pressure_index'] = weekly['competitor_pressure_index'].round(1)
        weekly = weekly.sort_values('week_start').reset_index(drop=True)
        return weekly

    def get_month_summary(self, year: int, month: int) -> Dict[str, Any]:
        """Get aggregated context summary for a month"""
        month_df = self.get_month_data(year, month)

        if month_df.empty:
            return self._get_default_month_summary()

        n_days = len(month_df)

        return {
            'year': year,
            'month': month,
            'n_days': n_days,
            'n_holidays': int(month_df['is_holiday'].sum()),
            'holiday_names': month_df[month_df['is_holiday'] == 1]['holiday_name'].dropna().unique().tolist(),
            'n_promotion_days': int(month_df['is_promotion'].sum()),
            'promotion_names': month_df[month_df['is_promotion'] == 1]['promotion_name'].dropna().unique().tolist(),
            'avg_fed_funds_rate': float(month_df['fed_funds_rate'].mean()),
            'min_fed_funds_rate': float(month_df['fed_funds_rate'].min()),
            'max_fed_funds_rate': float(month_df['fed_funds_rate'].max()),
            'avg_competitor_pressure': float(month_df['competitor_pressure_index'].mean()),
            'min_competitor_pressure': float(month_df['competitor_pressure_index'].min()),
            'max_competitor_pressure': float(month_df['competitor_pressure_index'].max()),
        }

    def _get_default_month_summary(self) -> Dict[str, Any]:
        """Return default summary when no data available"""
        return {
            'year': 2026,
            'month': 1,
            'n_days': 30,
            'n_holidays': 2,
            'holiday_names': [],
            'n_promotion_days': 3,
            'promotion_names': [],
            'avg_fed_funds_rate': 3.84,
            'min_fed_funds_rate': 3.50,
            'max_fed_funds_rate': 3.75,
            'avg_competitor_pressure': 50.0,
            'min_competitor_pressure': 33.0,
            'max_competitor_pressure': 96.0,
        }

    def calculate_context_multipliers(self, year: int, month: int) -> Dict[str, float]:
        """
        Calculate context-based multipliers for response prediction.
        All effects are multiplicative.
        """
        summary = self.get_month_summary(year, month)
        n_days = summary['n_days']

        # Holiday effect: ~50% boost on holiday days
        holiday_fraction = summary['n_holidays'] / n_days
        holiday_multiplier = 1 + (0.5 * holiday_fraction)

        # Promotion effect: ~30% boost on promotion days
        promotion_fraction = summary['n_promotion_days'] / n_days
        promotion_multiplier = 1 + (0.3 * promotion_fraction)

        # Fed Funds Rate effect: higher rate = lower demand
        baseline_rate = 3.84
        rate_diff = summary['avg_fed_funds_rate'] - baseline_rate
        fed_funds_multiplier = 1 - (0.02 * rate_diff / 0.25)
        fed_funds_multiplier = max(0.8, min(1.2, fed_funds_multiplier))

        # Competitor pressure effect: higher pressure = lower conversion
        baseline_pressure = 50.0
        pressure_diff = summary['avg_competitor_pressure'] - baseline_pressure
        competitor_multiplier = 1 - (0.01 * pressure_diff / 10)
        competitor_multiplier = max(0.85, min(1.15, competitor_multiplier))

        combined_multiplier = holiday_multiplier * promotion_multiplier * fed_funds_multiplier * competitor_multiplier

        return {
            'holiday_multiplier': holiday_multiplier,
            'promotion_multiplier': promotion_multiplier,
            'fed_funds_multiplier': fed_funds_multiplier,
            'competitor_multiplier': competitor_multiplier,
            'combined_multiplier': combined_multiplier,
            'n_holidays': summary['n_holidays'],
            'n_promotion_days': summary['n_promotion_days'],
            'avg_fed_funds_rate': summary['avg_fed_funds_rate'],
            'avg_competitor_pressure': summary['avg_competitor_pressure'],
        }

    def get_daily_context_for_robyn(self, year: int, month: int) -> pd.DataFrame:
        """Get daily context data formatted for Robyn prediction."""
        month_df = self.get_month_data(year, month)

        if month_df.empty:
            import calendar
            n_days = calendar.monthrange(year, month)[1]
            dates = pd.date_range(f'{year}-{month:02d}-01', periods=n_days, freq='D')
            month_df = pd.DataFrame({
                'date': dates,
                'is_holiday': 0,
                'is_promotion': 0,
                'fed_funds_rate': 3.84,
                'competitor_pressure_index': 50,
            })

        robyn_df = month_df[['date', 'is_holiday', 'is_promotion',
                            'fed_funds_rate', 'competitor_pressure_index']].copy()

        return robyn_df
