"""
Robyn-based budget optimizer that uses the trained MMM model

This wrapper uses the pickled Robyn model for optimization when available,
falling back to the manual Hill curve implementation otherwise.

Supports context variables (holidays, promotions, refinancing rate, CRM metrics)
for more accurate predictions.
"""
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default context coefficients (used if not found in model results)
DEFAULT_CONTEXT_COEFFICIENTS = {
    'is_holiday': {
        'coefficient': 0.5,  # 50% boost on holiday days
        'effect_type': 'multiplicative',
        'baseline': 0
    },
    'is_promotion': {
        'coefficient': 0.3,  # 30% boost on promotion days
        'effect_type': 'multiplicative',
        'baseline': 0
    },
    'refinancing_rate': {
        'coefficient': -0.04,  # -4% per 1% rate increase above baseline
        'effect_type': 'linear',
        'baseline': 7.5
    },
    'email_clicks': {
        'coefficient': 2.5,  # $2.5 revenue per email click
        'effect_type': 'additive'
    },
    'push_clicks': {
        'coefficient': 1.5,  # $1.5 revenue per push click
        'effect_type': 'additive'
    },
    'email_sends': {
        'coefficient': 0.0,  # Sends captured through clicks
        'effect_type': 'additive'
    },
    'push_sends': {
        'coefficient': 0.0,  # Sends captured through clicks
        'effect_type': 'additive'
    }
}


class RobynOptimizer:
    """
    Budget optimizer using Robyn's native allocator

    Uses the pickled Robyn model for accurate predictions and optimization
    based on the trained MMM parameters.

    Integrates context variables for more accurate predictions.
    """

    def __init__(
        self,
        model_path: str = "../robyn_training/models",
        channel_params: Optional[Dict[str, Dict[str, float]]] = None,
        context_coefficients: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize optimizer

        Args:
            model_path: Path to directory containing robyn_model.pkl
            channel_params: Fallback channel parameters if model not available
            context_coefficients: Coefficients for context variables
        """
        self.model_path = Path(model_path)
        self.robyn_model = None
        self.channel_params = channel_params or {}
        self.channels = list(self.channel_params.keys()) if channel_params else []
        self.context_coefficients = context_coefficients or {}

        # Try to load pickled Robyn model
        self._load_robyn_model()

        # Load context coefficients from model results if not provided
        if not self.context_coefficients:
            self._load_context_coefficients()

    def _load_robyn_model(self):
        """Load the pickled Robyn model if available"""
        pickle_path = self.model_path / 'robyn_model.pkl'

        if pickle_path.exists():
            try:
                with open(pickle_path, 'rb') as f:
                    self.robyn_model = pickle.load(f)
                logger.info(f"Loaded Robyn model from {pickle_path}")

                # Extract channels from model
                if hasattr(self.robyn_model, 'mmm_data') and self.robyn_model.mmm_data:
                    spend_cols = self.robyn_model.mmm_data.mmmdata_spec.paid_media_spends
                    self.channels = [c.replace('_spend', '') for c in spend_cols]
                    logger.info(f"Model channels: {self.channels}")

            except Exception as e:
                logger.warning(f"Could not load Robyn model: {e}")
                self.robyn_model = None
        else:
            logger.info(f"No pickled model found at {pickle_path}, using fallback optimizer")

    def _load_context_coefficients(self):
        """Load context coefficients from model results JSON"""
        results_path = self.model_path / 'robyn_results.json'

        if results_path.exists():
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)

                # Check if context_coefficients exist in results
                if 'context_coefficients' in results:
                    self.context_coefficients = results['context_coefficients']
                    logger.info(f"Loaded context coefficients from model: {list(self.context_coefficients.keys())}")
                else:
                    # Use defaults
                    self.context_coefficients = DEFAULT_CONTEXT_COEFFICIENTS.copy()
                    logger.info("Using default context coefficients (not found in model results)")

            except Exception as e:
                logger.warning(f"Could not load context coefficients: {e}")
                self.context_coefficients = DEFAULT_CONTEXT_COEFFICIENTS.copy()
        else:
            self.context_coefficients = DEFAULT_CONTEXT_COEFFICIENTS.copy()
            logger.info("Using default context coefficients (no results file)")

    def is_robyn_available(self) -> bool:
        """Check if native Robyn model is available"""
        return self.robyn_model is not None

    def _apply_constraints(
        self,
        allocation: Dict[str, float],
        constraints: Dict[str, Tuple[float, float]],
        total_budget: float
    ) -> Dict[str, float]:
        """
        Apply min/max constraints to allocation and redistribute excess.

        Args:
            allocation: Current allocation dict
            constraints: Dict mapping channel to (min, max) budget
            total_budget: Total budget to allocate

        Returns:
            Constrained allocation that respects min/max per channel
        """
        constrained = {}

        # First pass: apply min/max constraints
        for ch, spend in allocation.items():
            ch_min, ch_max = constraints.get(ch, (0, total_budget))
            # Clamp to [min, max]
            constrained[ch] = max(ch_min, min(ch_max, spend))

        # Calculate how much we've allocated vs total_budget
        current_total = sum(constrained.values())

        # If we need to redistribute (scale up or down)
        if abs(current_total - total_budget) > 0.01:
            if current_total < total_budget:
                # Need to add more - find channels with room to grow
                deficit = total_budget - current_total
                for _ in range(10):  # Iterate to fill deficit
                    if deficit <= 0.01:
                        break
                    absorbable = []
                    for ch in constrained:
                        ch_min, ch_max = constraints.get(ch, (0, total_budget))
                        room = ch_max - constrained[ch]
                        if room > 0:
                            absorbable.append((ch, room))

                    if not absorbable:
                        break  # No room left

                    total_room = sum(r for _, r in absorbable)
                    for ch, room in absorbable:
                        share = room / total_room if total_room > 0 else 0
                        add = min(deficit * share, room)
                        constrained[ch] += add
                        deficit -= add

            elif current_total > total_budget:
                # Need to reduce - find channels that can give
                excess = current_total - total_budget
                for _ in range(10):
                    if excess <= 0.01:
                        break
                    reducible = []
                    for ch in constrained:
                        ch_min, ch_max = constraints.get(ch, (0, total_budget))
                        room = constrained[ch] - ch_min
                        if room > 0:
                            reducible.append((ch, room))

                    if not reducible:
                        break

                    total_room = sum(r for _, r in reducible)
                    for ch, room in reducible:
                        share = room / total_room if total_room > 0 else 0
                        reduce = min(excess * share, room)
                        constrained[ch] -= reduce
                        excess -= reduce

        # Ensure no negatives
        constrained = {ch: max(0, spend) for ch, spend in constrained.items()}

        return constrained

    def optimize_for_budget(
        self,
        total_budget: float,
        budget_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        period_days: int = 30,
        context_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Optimize budget allocation to maximize response

        Uses Hill curve optimization with parameters from trained Robyn model.
        This ensures consistent results using the exact same Hill formula as Robyn.

        Args:
            total_budget: Total budget to allocate
            budget_constraints: Optional dict mapping channels to (min, max) budget
            period_days: Number of days in the period (for response calculation)
            context_data: Optional DataFrame with daily context variables

        Returns:
            Dictionary with allocation results
        """
        # Always use Hill curve optimizer with Robyn parameters
        # This ensures consistent interpretation of model parameters
        # and better budget distribution across channels
        return self._optimize_with_fallback(total_budget, budget_constraints, period_days, context_data)

    def _optimize_with_robyn(
        self,
        total_budget: float,
        budget_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        period_days: int = 30,
        context_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Use Robyn's native budget allocator"""
        try:
            from robyn.allocator.entities.allocation_params import AllocatorParams

            # Get spend columns from model
            spend_cols = self.robyn_model.mmm_data.mmmdata_spec.paid_media_spends

            # Prepare constraints as fractions of total budget
            n_channels = len(spend_cols)
            if budget_constraints:
                channel_constr_low = [
                    budget_constraints.get(ch.replace('_spend', ''), (0, total_budget))[0] / total_budget
                    for ch in spend_cols
                ]
                channel_constr_up = [
                    budget_constraints.get(ch.replace('_spend', ''), (0, total_budget))[1] / total_budget
                    for ch in spend_cols
                ]
            else:
                # No constraints - allow full range (0-100%)
                channel_constr_low = [0.0] * n_channels
                channel_constr_up = [1.0] * n_channels

            # Create allocator params
            allocator_params = AllocatorParams(
                scenario='max_response',
                total_budget=total_budget,
                channel_constr_low=channel_constr_low,
                channel_constr_up=channel_constr_up,
                constr_mode='eq',  # Use entire budget
                plots=False
            )

            # Run optimization
            result = self.robyn_model.optimize_budget(
                allocator_params=allocator_params,
                display_plots=False,
                export_plots=False
            )

            # Extract results from DataFrame (dt_optim_out, not dt_optimOut)
            optimized_allocation = {}
            predicted_response = 0

            if hasattr(result, 'dt_optim_out') and result.dt_optim_out is not None:
                df = result.dt_optim_out

                # Use optimized shares to calculate allocation for requested budget
                # Robyn returns optimal shares regardless of the budget level
                if 'optmSpendShareUnit' in df.columns:
                    for idx in df.index:
                        ch_name = idx.replace('_spend', '')
                        share = float(df.loc[idx, 'optmSpendShareUnit'])
                        optimized_allocation[ch_name] = total_budget * share

                # Apply budget constraints to allocation (Robyn may not respect them)
                if budget_constraints:
                    optimized_allocation = self._apply_constraints(
                        optimized_allocation, budget_constraints, total_budget
                    )

                # Calculate response using Hill curves with context
                predicted_response = self._predict_response_fallback(
                    optimized_allocation, period_days, context_data
                )

            # Fallback if result parsing failed or response is too low
            if not optimized_allocation or predicted_response < 1:
                logger.warning("Could not parse Robyn result or response too low, using fallback")
                return self._optimize_with_fallback(total_budget, budget_constraints, period_days, context_data)

            return {
                'allocation': optimized_allocation,
                'total_budget': sum(optimized_allocation.values()),
                'predicted_leads': predicted_response,
                'roi': predicted_response / total_budget if total_budget > 0 else 0,
                'success': True,
                'method': 'robyn_native'
            }

        except Exception as e:
            logger.warning(f"Robyn optimization failed: {e}, using fallback")
            return self._optimize_with_fallback(total_budget, budget_constraints, period_days, context_data)

    def _optimize_with_fallback(
        self,
        total_budget: float,
        budget_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        period_days: int = 30,
        context_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Fallback optimization using manual Hill curves with context"""
        from scipy.optimize import minimize

        n_channels = len(self.channels)

        # If no constraints provided, use realistic default bounds:
        # - Minimum: 5% of budget per channel (ensures diversification)
        # - Maximum: 50% of budget per channel (prevents over-concentration)
        if not budget_constraints:
            min_per_channel = total_budget * 0.05  # 5% minimum
            max_per_channel = total_budget * 0.50  # 50% maximum
            budget_constraints = {ch: (min_per_channel, max_per_channel) for ch in self.channels}

        def objective(x):
            allocation = {ch: x[i] for i, ch in enumerate(self.channels)}
            # Optimize on base response (without context) for efficiency
            # Context is applied to final result
            return -self._calculate_hill_response(allocation, period_days)

        def constraint_budget_eq(x):
            return np.sum(x) - total_budget

        bounds = [budget_constraints.get(ch, (0, total_budget)) for ch in self.channels]
        x0 = np.array([total_budget / n_channels] * n_channels)

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': constraint_budget_eq},
            options={'maxiter': 1000}
        )

        optimal_allocation = {
            ch: max(0, result.x[i]) for i, ch in enumerate(self.channels)
        }

        # Calculate final response with context
        predicted = self._predict_response_fallback(optimal_allocation, period_days, context_data)

        return {
            'allocation': optimal_allocation,
            'total_budget': total_budget,
            'predicted_leads': predicted,
            'roi': predicted / total_budget if total_budget > 0 else 0,
            'success': result.success,
            'method': 'robyn_hill_curve'
        }

    def _predict_response_fallback(
        self,
        spend_allocation: Dict[str, float],
        period_days: int = 30,
        context_data: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Predict response using Hill curves with context variable integration.

        The Hill function models saturation: response = max_response * (x^alpha) / (1 + x^alpha)
        where x = adstocked_spend / gamma (normalized spend)

        Context variables are integrated as:
        - Multiplicative effects (is_holiday, is_promotion): applied per-day
        - Linear effects (refinancing_rate): continuous adjustment
        - Additive effects (email_clicks, push_clicks): direct contribution

        Args:
            spend_allocation: Dict mapping channel to period budget
            period_days: Number of days in the period (default 30 for monthly)
            context_data: Optional DataFrame with daily context variables

        Note: spend_allocation contains period budget (e.g., monthly),
        and we convert to daily for consistent saturation modeling.
        """
        # Calculate base marketing response from Hill curves
        base_response = self._calculate_hill_response(spend_allocation, period_days)

        # Apply context effects if data provided
        if context_data is not None and not context_data.empty:
            context_adjusted = self._apply_context_effects(base_response, context_data)
            return context_adjusted
        else:
            return base_response

    def _calculate_hill_response(
        self,
        spend_allocation: Dict[str, float],
        period_days: int = 30
    ) -> float:
        """
        Calculate base response using Hill saturation curves (no context).

        Uses EXACTLY the same Hill formula as Robyn:
        inflexion = x_min * (1 - gamma) + x_max * gamma
        response = (x^alpha) / (x^alpha + inflexion^alpha)

        Where gamma is normalized (0-1) and inflexion is computed from spend range.
        """
        total_response = 0

        for channel, period_spend in spend_allocation.items():
            if channel not in self.channel_params:
                continue

            params = self.channel_params[channel]

            # Convert period spend to daily for saturation calculation
            daily_spend = period_spend / period_days

            # Apply adstock (carryover effect) - same as Robyn geometric adstock
            theta = params.get('adstock_theta', 0.5)
            # For daily spend, adstock effect is approximately (1 + theta + theta^2 + ...) = 1/(1-theta)
            # But simplified to 1 + theta for single-day calculation
            adstocked = daily_spend * (1 + theta)

            # Get saturation parameters
            alpha = params.get('saturation_alpha', 2.0)
            gamma_normalized = params.get('saturation_gamma', 0.5)  # Now normalized 0-1

            # Historical spend data for calculating inflexion point range
            historical_spend = params.get('total_spend', 100000)
            historical_daily_spend = historical_spend / 365

            # Calculate inflexion point using Robyn's formula:
            # inflexion = x_min * (1 - gamma) + x_max * gamma
            # where x_min/x_max are from potential spend range
            # x_max = 10x historical allows gamma=0.6-0.7 to place inflexion at 6-7x current spend
            x_min = 0
            x_max = historical_daily_spend * 10  # 10x growth potential
            inflexion = x_min * (1 - gamma_normalized) + x_max * gamma_normalized

            # Ensure inflexion is positive
            if inflexion <= 0:
                inflexion = historical_daily_spend * 0.5

            # Calculate historical contribution for scaling
            historical_contribution = params.get('contribution', historical_spend * params.get('roi', 2.0))
            historical_daily_contribution = historical_contribution / 365

            # At historical daily spend level, what fraction of max are we at?
            # Using Robyn's Hill formula: (x^alpha) / (x^alpha + inflexion^alpha)
            hist_adstocked = historical_daily_spend * (1 + theta)
            if hist_adstocked > 0 and inflexion > 0:
                saturation_at_hist = (hist_adstocked ** alpha) / (hist_adstocked ** alpha + inflexion ** alpha)
            else:
                saturation_at_hist = 0.5

            # Calculate max daily response (when saturation = 1)
            if saturation_at_hist > 0.01:
                max_daily_response = historical_daily_contribution / saturation_at_hist
            else:
                max_daily_response = historical_daily_contribution * 2

            # Apply Robyn's Hill saturation to current daily spend
            if adstocked > 0 and inflexion > 0:
                # Robyn formula: (x^alpha) / (x^alpha + inflexion^alpha)
                daily_response = max_daily_response * (adstocked ** alpha) / (adstocked ** alpha + inflexion ** alpha)
                period_response = daily_response * period_days
                total_response += period_response

        return total_response

    def _apply_context_effects(
        self,
        base_response: float,
        context_data: pd.DataFrame
    ) -> float:
        """
        Apply context variable effects to base marketing response.

        Processes daily context data and applies:
        - Multiplicative effects: holiday/promotion boosts
        - Linear effects: refinancing rate adjustment
        - Additive effects: CRM contribution

        Args:
            base_response: Base marketing response from Hill curves
            context_data: DataFrame with daily context (date, is_holiday, is_promotion, etc.)

        Returns:
            Context-adjusted total response
        """
        n_days = len(context_data)
        if n_days == 0:
            return base_response

        # Calculate daily base response
        daily_base = base_response / n_days

        total_response = 0
        total_additive = 0

        for _, day_row in context_data.iterrows():
            # Start with daily base
            day_response = daily_base

            # Apply multiplicative effects
            for var in ['is_holiday', 'is_promotion']:
                if var in self.context_coefficients and var in day_row:
                    coef_info = self.context_coefficients[var]
                    if coef_info.get('effect_type') == 'multiplicative':
                        value = float(day_row[var])
                        coefficient = coef_info.get('coefficient', 0)
                        # Multiplier = 1 + coefficient * value
                        # e.g., is_holiday=1, coefficient=0.5 → 1.5x boost
                        multiplier = 1 + (coefficient * value)
                        day_response *= multiplier

            # Apply linear effects (refinancing rate)
            if 'refinancing_rate' in self.context_coefficients and 'refinancing_rate' in day_row:
                coef_info = self.context_coefficients['refinancing_rate']
                if coef_info.get('effect_type') == 'linear':
                    value = float(day_row['refinancing_rate'])
                    baseline = coef_info.get('baseline', 7.5)
                    coefficient = coef_info.get('coefficient', -0.04)
                    # Effect = 1 + coefficient * (value - baseline)
                    # e.g., rate=8.0, baseline=7.5, coef=-0.04 → 1 + (-0.04)*(0.5) = 0.98
                    effect = 1 + (coefficient * (value - baseline))
                    effect = max(0.7, min(1.3, effect))  # Clamp to reasonable range
                    day_response *= effect

            total_response += day_response

            # Calculate additive effects (CRM)
            for var in ['email_clicks', 'push_clicks', 'email_sends', 'push_sends']:
                if var in self.context_coefficients and var in day_row:
                    coef_info = self.context_coefficients[var]
                    if coef_info.get('effect_type') == 'additive':
                        value = float(day_row[var])
                        coefficient = coef_info.get('coefficient', 0)
                        total_additive += value * coefficient

        return total_response + total_additive

    def predict_with_context(
        self,
        spend_allocation: Dict[str, float],
        context_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Predict response with full context integration.

        Args:
            spend_allocation: Dict mapping channel to period budget
            context_data: DataFrame with daily context variables

        Returns:
            Dict with response breakdown
        """
        period_days = len(context_data) if context_data is not None and not context_data.empty else 30

        # Calculate base response (no context)
        base_response = self._calculate_hill_response(spend_allocation, period_days)

        # Calculate with context
        if context_data is not None and not context_data.empty:
            total_response = self._apply_context_effects(base_response, context_data)

            # Calculate context breakdown
            holiday_days = int(context_data['is_holiday'].sum()) if 'is_holiday' in context_data else 0
            promo_days = int(context_data['is_promotion'].sum()) if 'is_promotion' in context_data else 0
            avg_refi_rate = float(context_data['refinancing_rate'].mean()) if 'refinancing_rate' in context_data else 7.5

            # Calculate additive CRM contribution
            crm_contribution = 0
            for var in ['email_clicks', 'push_clicks']:
                if var in self.context_coefficients and var in context_data:
                    coef = self.context_coefficients[var].get('coefficient', 0)
                    crm_contribution += float(context_data[var].sum()) * coef

            return {
                'total_response': total_response,
                'base_response': base_response,
                'context_multiplier': (total_response - crm_contribution) / base_response if base_response > 0 else 1.0,
                'crm_contribution': crm_contribution,
                'holiday_days': holiday_days,
                'promotion_days': promo_days,
                'avg_refinancing_rate': avg_refi_rate,
                'period_days': period_days
            }
        else:
            return {
                'total_response': base_response,
                'base_response': base_response,
                'context_multiplier': 1.0,
                'crm_contribution': 0,
                'holiday_days': 0,
                'promotion_days': 0,
                'avg_refinancing_rate': 7.5,
                'period_days': period_days
            }

    def optimize_for_target(
        self,
        target_response: float,
        budget_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        period_days: int = 30,
        context_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Find minimum budget to achieve target response

        Uses Hill curve optimization with parameters from trained Robyn model.

        Args:
            target_response: Target leads/revenue to achieve
            budget_constraints: Optional channel constraints
            period_days: Number of days in the period
            context_data: Optional DataFrame with daily context variables

        Returns:
            Dictionary with allocation results
        """
        # Always use Hill curve optimizer with Robyn parameters
        return self._optimize_target_with_fallback(target_response, budget_constraints, period_days, context_data)

    def _optimize_target_with_robyn(
        self,
        target_response: float,
        budget_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        period_days: int = 30,
        context_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Use Robyn for target optimization via binary search.

        Since Robyn's target_efficiency doesn't work reliably,
        we use binary search with max_response to find minimum budget.
        """
        # Calculate minimum possible budget from constraints
        if budget_constraints:
            min_budget = sum(c[0] for c in budget_constraints.values())
        else:
            min_budget = 1000 * len(self.channels)  # Default $1000 per channel

        max_budget = 50000000  # 50M

        # First check if target is achievable at max budget
        result_max = self._optimize_with_robyn(max_budget, budget_constraints, period_days, context_data)
        if result_max['predicted_leads'] < target_response:
            return {
                'allocation': result_max['allocation'],
                'total_budget': max_budget,
                'predicted_leads': result_max['predicted_leads'],
                'target_leads': target_response,
                'success': False,
                'method': 'robyn_native',
                'message': 'Target not achievable with maximum budget'
            }

        # Binary search
        tolerance = 0.02  # 2% tolerance
        max_iterations = 30

        best_result = result_max
        best_budget = max_budget

        for _ in range(max_iterations):
            mid_budget = (min_budget + max_budget) / 2
            result = self._optimize_with_robyn(mid_budget, budget_constraints, period_days, context_data)
            predicted = result['predicted_leads']

            if predicted >= target_response:
                # Can achieve target with this budget
                best_result = result
                best_budget = mid_budget
                max_budget = mid_budget
            else:
                # Need more budget
                min_budget = mid_budget

            # Check convergence
            if (max_budget - min_budget) / max_budget < tolerance:
                break

        return {
            'allocation': best_result['allocation'],
            'total_budget': best_budget,
            'predicted_leads': best_result['predicted_leads'],
            'target_leads': target_response,
            'success': True,
            'method': 'robyn_native'
        }

    def _optimize_target_with_fallback(
        self,
        target_response: float,
        budget_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        period_days: int = 30,
        context_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Fallback target optimization using binary search on budget.

        Since we can optimize for a given budget (maximize response),
        we use binary search to find the minimum budget that achieves the target.
        """
        # Calculate minimum possible budget from constraints
        if budget_constraints:
            min_budget = sum(c[0] for c in budget_constraints.values())
        else:
            min_budget = 1000 * len(self.channels)  # Default $1000 per channel

        max_budget = 50000000  # Maximum total budget (50M)

        # First check if target is achievable at max budget
        result_max = self._optimize_with_fallback(max_budget, budget_constraints, period_days, context_data)
        if result_max['predicted_leads'] < target_response:
            # Target not achievable - find optimal budget that maximizes response
            # instead of returning max budget with poor allocation
            # Use binary search to find the "knee" of the response curve
            # where increasing budget yields diminishing returns

            # Find the budget that achieves ~95% of max response (near saturation point)
            best_budget = min_budget
            best_result = self._optimize_with_fallback(min_budget, budget_constraints, period_days, context_data)

            # Binary search for optimal budget (where marginal ROI drops significantly)
            test_budgets = [min_budget]
            current = min_budget
            while current < max_budget:
                current = min(current * 1.5, max_budget)  # Grow by 50%
                test_budgets.append(current)

            prev_response = 0
            for budget in test_budgets:
                result = self._optimize_with_fallback(budget, budget_constraints, period_days, context_data)
                response = result['predicted_leads']

                # Track best result (highest response with reasonable budget)
                if response > best_result['predicted_leads']:
                    best_result = result
                    best_budget = budget

                # Check for diminishing returns (marginal response < 50% of expected linear)
                if prev_response > 0 and budget > min_budget:
                    marginal_response = response - prev_response
                    expected_linear = prev_response * (budget / test_budgets[test_budgets.index(budget) - 1] - 1)
                    if marginal_response < expected_linear * 0.3:
                        # Heavy diminishing returns - stop here
                        break

                prev_response = response

            return {
                'allocation': best_result['allocation'],
                'total_budget': best_budget,
                'predicted_leads': best_result['predicted_leads'],
                'target_leads': target_response,
                'success': False,
                'method': 'robyn_hill_curve',
                'message': f'Target not achievable. Max achievable: ${best_result["predicted_leads"]:,.0f}'
            }

        # Binary search
        tolerance = 0.01  # 1% tolerance
        max_iterations = 50

        for _ in range(max_iterations):
            mid_budget = (min_budget + max_budget) / 2
            result = self._optimize_with_fallback(mid_budget, budget_constraints, period_days, context_data)
            predicted = result['predicted_leads']

            if abs(predicted - target_response) / target_response < tolerance:
                # Close enough to target
                break

            if predicted < target_response:
                # Need more budget
                min_budget = mid_budget
            else:
                # Can achieve with less budget
                max_budget = mid_budget

        # Final optimization at found budget
        final_result = self._optimize_with_fallback(mid_budget, budget_constraints, period_days, context_data)

        return {
            'allocation': final_result['allocation'],
            'total_budget': mid_budget,
            'predicted_leads': final_result['predicted_leads'],
            'target_leads': target_response,
            'success': True,
            'method': 'robyn_hill_curve'
        }

    def scenario_analysis(
        self,
        budget_range: Tuple[float, float],
        steps: int = 10,
        period_days: int = 30,
        context_data: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """Run scenario analysis across budget range with context"""
        budgets = np.linspace(budget_range[0], budget_range[1], steps)
        scenarios = []

        for budget in budgets:
            result = self.optimize_for_budget(
                budget,
                period_days=period_days,
                context_data=context_data
            )
            scenarios.append(result)

        return scenarios

    def get_response_curves(self) -> Dict[str, Dict[str, Any]]:
        """
        Get response curve data for visualization

        Returns spend-response curves for each channel (daily basis)
        """
        curves = {}

        for channel in self.channels:
            params = self.channel_params.get(channel, {})
            historical_spend = params.get('total_spend', 100000)
            daily_spend = historical_spend / 365

            # Generate spend range (daily spend levels)
            spend_range = np.linspace(0, daily_spend * 3, 50)
            responses = []

            for spend in spend_range:
                allocation = {ch: 0 for ch in self.channels}
                allocation[channel] = spend
                # Use period_days=1 since we're working with daily spend
                response = self._predict_response_fallback(allocation, period_days=1)
                responses.append(response)

            curves[channel] = {
                'spend': spend_range.tolist(),
                'response': responses,
                'params': {
                    'theta': params.get('adstock_theta', 0.5),
                    'alpha': params.get('saturation_alpha', 2.0),
                    'gamma': params.get('saturation_gamma', daily_spend),
                    'roi': params.get('roi', 2.0)
                }
            }

        return curves
