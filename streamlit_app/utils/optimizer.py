"""
Budget optimization utilities using Robyn MMM response curves
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BudgetOptimizer:
    """Optimize marketing budget allocation using MMM response curves"""

    def __init__(self, channel_params: Dict[str, Dict[str, float]]):
        """
        Initialize optimizer

        Args:
            channel_params: Dictionary with channel parameters
                            (saturation curves, adstock, etc.)
        """
        self.channel_params = channel_params
        self.channels = list(channel_params.keys())

    def apply_adstock(self, spend: float, theta: float) -> float:
        """
        Apply adstock transformation (simplified geometric adstock)

        Args:
            spend: Spend amount
            theta: Adstock parameter (0-1)

        Returns:
            Adstocked spend
        """
        # Simplified - in production, apply full adstock transformation
        return spend * (1 + theta)

    def apply_saturation(
        self,
        adstocked_spend: float,
        alpha: float,
        gamma: float,
        max_response: float = 1.0
    ) -> float:
        """
        Apply modified Hill saturation curve (S-shaped)

        This creates realistic S-curve behavior:
        - Low spend: slow growth (learning/cold start phase)
        - Medium spend: accelerating growth (optimization phase)
        - High spend: diminishing returns (saturation)

        Args:
            adstocked_spend: Adstocked spend
            alpha: Shape parameter (>1 for S-curve, <1 for concave)
            gamma: Inflection point (spend level at half-saturation)
            max_response: Maximum achievable response

        Returns:
            Saturated response scaled by max_response
        """
        if adstocked_spend <= 0:
            return 0

        # Hill equation with proper scaling
        # S = max_response * (spend^alpha) / (gamma^alpha + spend^alpha)
        # When alpha > 1: S-shaped curve (slow start, fast middle, slow end)
        # When alpha = 1: Michaelis-Menten (always diminishing)
        # When alpha < 1: Concave (fast start, slow end)

        x_norm = adstocked_spend / gamma  # Normalize by inflection point
        numerator = x_norm ** alpha
        denominator = 1 + numerator

        return max_response * numerator / denominator

    def predict_response(self, spend_allocation: Dict[str, float]) -> float:
        """
        Predict total response (revenue/leads) for given spend allocation

        Args:
            spend_allocation: Dictionary mapping channels to spend amounts

        Returns:
            Total predicted response
        """
        total_response = 0

        for channel, spend in spend_allocation.items():
            if channel not in self.channel_params:
                continue

            params = self.channel_params[channel]

            # Apply adstock (carryover effect)
            adstocked = self.apply_adstock(
                spend,
                params.get('adstock_theta', 0.5)
            )

            # Calculate max response for this channel based on historical data
            # max_response = historical_contribution = total_spend * ROI
            historical_spend = params.get('total_spend', 100000)
            roi = params.get('roi', 2.0)
            # Max response is approximately 2x historical (room to grow)
            max_response = historical_spend * roi * 2

            # Apply saturation curve
            # alpha > 1 gives S-shaped curve (realistic for digital marketing)
            saturated = self.apply_saturation(
                adstocked,
                alpha=params.get('saturation_alpha', 2.0),  # S-curve shape
                gamma=params.get('saturation_gamma', historical_spend / 365),  # Daily inflection
                max_response=max_response
            )

            total_response += saturated

        return total_response

    def optimize_for_target_leads(
        self,
        target_leads: float,
        budget_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, any]:
        """
        Find optimal budget allocation to achieve target leads

        Args:
            target_leads: Target number of leads/revenue
            budget_constraints: Optional dict mapping channels to (min, max) budget

        Returns:
            Dictionary with optimal allocation and metrics
        """
        n_channels = len(self.channels)

        if budget_constraints is None:
            # Default: min 1000 per channel, max 100k
            budget_constraints = {ch: (1000, 100000) for ch in self.channels}

        # Objective: minimize total spend while achieving target
        def objective(x):
            return np.sum(x)  # Total spend

        # Constraint: achieve target leads
        def constraint_target(x):
            allocation = {ch: x[i] for i, ch in enumerate(self.channels)}
            predicted = self.predict_response(allocation)
            return predicted - target_leads  # >= 0

        # Bounds for each channel
        bounds = [budget_constraints.get(ch, (0, 100000)) for ch in self.channels]

        # Initial guess (equal distribution)
        x0 = np.array([10000] * n_channels)

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_target},
            options={'maxiter': 1000}
        )

        optimal_allocation = {
            ch: result.x[i] for i, ch in enumerate(self.channels)
        }

        return {
            'allocation': optimal_allocation,
            'total_budget': np.sum(result.x),
            'predicted_leads': self.predict_response(optimal_allocation),
            'target_leads': target_leads,
            'success': result.success,
        }

    def optimize_for_budget(
        self,
        total_budget: float,
        budget_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, any]:
        """
        Find optimal budget allocation to maximize leads for given budget

        Args:
            total_budget: Total available budget
            budget_constraints: Optional dict mapping channels to (min, max) budget

        Returns:
            Dictionary with optimal allocation and metrics
        """
        n_channels = len(self.channels)

        if budget_constraints is None:
            # Default: each channel gets between 5% and 50% of budget
            min_per_channel = total_budget * 0.05
            max_per_channel = total_budget * 0.50
            budget_constraints = {ch: (min_per_channel, max_per_channel) for ch in self.channels}

        # Objective: maximize response (minimize negative response)
        def objective(x):
            allocation = {ch: x[i] for i, ch in enumerate(self.channels)}
            return -self.predict_response(allocation)  # Negative for maximization

        # Constraint: use entire budget (equality constraint)
        def constraint_budget_eq(x):
            return np.sum(x) - total_budget  # == 0

        # Bounds for each channel
        bounds = [budget_constraints.get(ch, (0, total_budget)) for ch in self.channels]

        # Initial guess (equal distribution)
        x0 = np.array([total_budget / n_channels] * n_channels)

        # Optimize with equality constraint to use full budget
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': constraint_budget_eq},
            options={'maxiter': 1000}
        )

        optimal_allocation = {
            ch: max(0, result.x[i]) for i, ch in enumerate(self.channels)  # Ensure non-negative
        }

        return {
            'allocation': optimal_allocation,
            'total_budget': total_budget,
            'predicted_leads': self.predict_response(optimal_allocation),
            'roi': self.predict_response(optimal_allocation) / total_budget if total_budget > 0 else 0,
            'success': result.success,
        }

    def scenario_analysis(
        self,
        budget_range: Tuple[float, float],
        steps: int = 10
    ) -> List[Dict[str, any]]:
        """
        Run scenario analysis across budget range

        Args:
            budget_range: (min_budget, max_budget)
            steps: Number of budget points to evaluate

        Returns:
            List of optimization results for each budget level
        """
        budgets = np.linspace(budget_range[0], budget_range[1], steps)
        scenarios = []

        for budget in budgets:
            result = self.optimize_for_budget(budget)
            scenarios.append(result)

        return scenarios
