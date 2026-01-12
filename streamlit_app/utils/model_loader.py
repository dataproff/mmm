"""
Model loader utilities for loading Robyn MMM model results
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage Robyn MMM model artifacts"""

    def __init__(self, model_path: str = "./models"):
        """
        Initialize model loader

        Args:
            model_path: Path to model directory
        """
        self.model_path = Path(model_path)

    def load_model_results(self, results_file: str = "robyn_results.json") -> Dict[str, Any]:
        """
        Load model results from JSON file

        Args:
            results_file: Name of results file

        Returns:
            Dictionary with model results
        """
        file_path = self.model_path / results_file

        if not file_path.exists():
            raise FileNotFoundError(f"Model results not found at {file_path}")

        logger.info(f"Loading model results from {file_path}")
        with open(file_path, 'r') as f:
            results = json.load(f)

        return results

    def get_channel_parameters(self, results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Extract channel-specific parameters from model results

        Args:
            results: Model results dictionary

        Returns:
            Dictionary mapping channel names to their parameters
            (adstock, saturation, contribution, etc.)
        """
        channels = results.get('channels', [])
        channel_results = results.get('channel_results', {})
        hyper_cfg = results.get('config', {}).get('hyperparameters', {})

        # Get hyperparameter ranges from config
        theta_mid = (hyper_cfg.get('thetas', {}).get('min', 0) + hyper_cfg.get('thetas', {}).get('max', 0.8)) / 2
        alpha_mid = (hyper_cfg.get('alphas', {}).get('min', 0.5) + hyper_cfg.get('alphas', {}).get('max', 3)) / 2

        # Calculate total spend for contribution percentages
        total_spend = sum(
            channel_results.get(ch, {}).get('total_spend', 0)
            for ch in channels
        )

        channel_params = {}
        for channel in channels:
            ch_data = channel_results.get(channel, {})
            ch_spend = ch_data.get('total_spend', 0)

            # Get ROI - use actual or estimate from spend share
            roi = ch_data.get('roi', 0)
            if roi == 0 and total_spend > 0:
                # Estimate based on typical ROI range (1.5-4.0)
                spend_share = ch_spend / total_spend if total_spend > 0 else 1 / len(channels)
                roi = 2.0 + spend_share  # Higher spend channels often have lower marginal ROI

            # Get contribution percentage
            contrib_pct = ch_data.get('contribution_pct', 0)
            if contrib_pct == 0 and total_spend > 0:
                contrib_pct = ch_spend / total_spend

            # Calculate contribution coefficient for optimizer
            # This scales the Hill curve output to actual revenue
            contribution_coef = roi * 1000  # Scale factor for optimization

            # Get channel-specific adstock and saturation params from model results
            # Fall back to config midpoints if not available
            adstock_theta = ch_data.get('adstock_theta', theta_mid)
            saturation_alpha = ch_data.get('saturation_alpha', alpha_mid)

            # Gamma from model is normalized (0-1) - we keep it as-is
            # The conversion to inflexion point happens in the optimizer using Robyn's formula:
            # inflexion = x_min * (1 - gamma) + x_max * gamma
            # This is applied when we have the actual spend range

            # Get gamma from model results (normalized 0-1)
            gamma_from_model = ch_data.get('saturation_gamma', None)

            if gamma_from_model is not None:
                # Store normalized gamma as-is (0-1 range)
                saturation_gamma = gamma_from_model
                logger.info(f"Channel {channel}: gamma_normalized={gamma_from_model:.4f}")
            else:
                # Default: gamma=0.5 means inflexion at midpoint of spend range
                saturation_gamma = 0.5

            channel_params[channel] = {
                'adstock_theta': adstock_theta,
                'saturation_alpha': saturation_alpha,
                'saturation_gamma': saturation_gamma,
                'contribution': ch_data.get('contribution', ch_spend * roi),
                'contribution_pct': contrib_pct,
                'contribution_coef': contribution_coef,
                'roi': roi,
                'mroi': roi * 0.8,  # Marginal ROI typically lower than avg ROI
                'total_spend': ch_spend,
            }

        logger.info(f"Loaded parameters for {len(channel_params)} channels")
        return channel_params

    def get_response_curves(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get response curves for each channel

        Args:
            results: Model results dictionary

        Returns:
            Dictionary with response curve data for each channel
        """
        # Placeholder - in production, load actual response curves from Robyn
        channels = results.get('channels', [])

        response_curves = {}
        for channel in channels:
            response_curves[channel] = {
                'spend_range': [0, 10000, 20000, 50000, 100000],
                'response': [0, 5000, 9000, 18000, 30000],
            }

        return response_curves
