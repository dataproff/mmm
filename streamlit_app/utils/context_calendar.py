"""
Context Calendar Loader

Loads and manages context variables calendar for MMM predictions.
Context variables include: holidays, promotions, refinancing rate, CRM metrics.
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
            # Use absolute path relative to this file
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
        """
        Get calendar data for a specific month

        Args:
            year: Year (e.g., 2026)
            month: Month number (1-12)

        Returns:
            DataFrame with daily context data for the month
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        mask = (self.df['date'].dt.year == year) & (self.df['date'].dt.month == month)
        return self.df[mask].copy()

    def get_month_summary(self, year: int, month: int) -> Dict[str, Any]:
        """
        Get aggregated context summary for a month

        Args:
            year: Year
            month: Month number

        Returns:
            Dictionary with aggregated context variables
        """
        month_df = self.get_month_data(year, month)

        if month_df.empty:
            return self._get_default_month_summary()

        n_days = len(month_df)

        return {
            'year': year,
            'month': month,
            'n_days': n_days,
            # Holiday metrics
            'n_holidays': int(month_df['is_holiday'].sum()),
            'holiday_names': month_df[month_df['is_holiday'] == 1]['holiday_name'].dropna().unique().tolist(),
            # Promotion metrics
            'n_promotion_days': int(month_df['is_promotion'].sum()),
            'promotion_names': month_df[month_df['is_promotion'] == 1]['promotion_name'].dropna().unique().tolist(),
            # Refinancing rate
            'avg_refinancing_rate': float(month_df['refinancing_rate'].mean()),
            'min_refinancing_rate': float(month_df['refinancing_rate'].min()),
            'max_refinancing_rate': float(month_df['refinancing_rate'].max()),
            # CRM metrics (monthly totals)
            'total_email_sends': int(month_df['email_sends'].sum()),
            'total_email_clicks': int(month_df['email_clicks'].sum()),
            'total_push_sends': int(month_df['push_sends'].sum()),
            'total_push_clicks': int(month_df['push_clicks'].sum()),
            # Daily averages for CRM
            'avg_daily_email_sends': float(month_df['email_sends'].mean()),
            'avg_daily_email_clicks': float(month_df['email_clicks'].mean()),
            'avg_daily_push_sends': float(month_df['push_sends'].mean()),
            'avg_daily_push_clicks': float(month_df['push_clicks'].mean()),
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
            'avg_refinancing_rate': 7.5,
            'min_refinancing_rate': 7.0,
            'max_refinancing_rate': 8.0,
            'total_email_sends': 350000,
            'total_email_clicks': 14000,
            'total_push_sends': 220000,
            'total_push_clicks': 8800,
            'avg_daily_email_sends': 11667,
            'avg_daily_email_clicks': 467,
            'avg_daily_push_sends': 7333,
            'avg_daily_push_clicks': 293,
        }

    def calculate_context_multipliers(self, year: int, month: int) -> Dict[str, float]:
        """
        Calculate context-based multipliers for response prediction

        These multipliers adjust the base response based on context variables.
        Based on coefficients learned during model training.

        Args:
            year: Year
            month: Month number

        Returns:
            Dictionary with multipliers for different effects
        """
        summary = self.get_month_summary(year, month)
        n_days = summary['n_days']

        # Base multiplier = 1.0 (no effect)
        # These coefficients are simplified estimates based on training data patterns

        # Holiday effect: holidays typically boost revenue
        # ~50% boost on holiday days (from generate_test_data.py)
        holiday_fraction = summary['n_holidays'] / n_days
        holiday_multiplier = 1 + (0.5 * holiday_fraction)

        # Promotion effect: promotions boost revenue by ~30%
        promotion_fraction = summary['n_promotion_days'] / n_days
        promotion_multiplier = 1 + (0.3 * promotion_fraction)

        # Refinancing effect: higher rate = lower demand
        # ~2% reduction per 0.5% rate increase above baseline (7.5%)
        baseline_rate = 7.5
        rate_diff = summary['avg_refinancing_rate'] - baseline_rate
        refinancing_multiplier = 1 - (0.02 * rate_diff / 0.5)
        refinancing_multiplier = max(0.8, min(1.2, refinancing_multiplier))  # Clamp

        # CRM contribution (additive, not multiplicative)
        # From training data: email_clicks * 2.5 + push_clicks * 1.5
        crm_contribution = (
            summary['total_email_clicks'] * 2.5 +
            summary['total_push_clicks'] * 1.5
        )

        # Combined multiplier for marketing response
        combined_multiplier = holiday_multiplier * promotion_multiplier * refinancing_multiplier

        return {
            'holiday_multiplier': holiday_multiplier,
            'promotion_multiplier': promotion_multiplier,
            'refinancing_multiplier': refinancing_multiplier,
            'combined_multiplier': combined_multiplier,
            'crm_contribution': crm_contribution,
            # Summary info
            'n_holidays': summary['n_holidays'],
            'n_promotion_days': summary['n_promotion_days'],
            'avg_refinancing_rate': summary['avg_refinancing_rate'],
        }

    def get_daily_context_for_robyn(self, year: int, month: int) -> pd.DataFrame:
        """
        Get daily context data formatted for Robyn prediction

        Returns DataFrame with columns matching model training features.

        Args:
            year: Year
            month: Month number

        Returns:
            DataFrame ready for Robyn predict
        """
        month_df = self.get_month_data(year, month)

        if month_df.empty:
            # Generate default data
            import calendar
            n_days = calendar.monthrange(year, month)[1]
            dates = pd.date_range(f'{year}-{month:02d}-01', periods=n_days, freq='D')
            month_df = pd.DataFrame({
                'date': dates,
                'is_holiday': 0,
                'is_promotion': 0,
                'refinancing_rate': 7.5,
                'email_sends': 12000,
                'email_clicks': 480,
                'push_sends': 8000,
                'push_clicks': 320,
            })

        # Ensure proper column names for Robyn
        robyn_df = month_df[['date', 'is_holiday', 'is_promotion',
                            'refinancing_rate', 'email_sends', 'email_clicks',
                            'push_sends', 'push_clicks']].copy()

        return robyn_df
