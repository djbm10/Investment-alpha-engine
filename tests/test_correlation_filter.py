import numpy as np

from src.config_loader import Phase2Config
from src.correlation_filter import node_average_correlations, node_tradeable_mask


def test_node_average_correlations_excludes_diagonal() -> None:
    correlation = np.array(
        [
            [1.0, 0.6, 0.2],
            [0.6, 1.0, 0.4],
            [0.2, 0.4, 1.0],
        ]
    )

    averages = node_average_correlations(correlation)

    assert np.allclose(averages, np.array([0.4, 0.5, 0.3]))


def test_node_tradeable_mask_blocks_decorrelated_assets() -> None:
    correlation = np.array(
        [
            [1.0, 0.5, 0.1],
            [0.5, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ]
    )
    config = Phase2Config(
        lookback_window=60,
        diffusion_alpha=0.05,
        diffusion_steps=3,
        sigma_scale=1.0,
        min_weight=0.1,
        zscore_lookback=60,
        signal_threshold=2.0,
        tier2_fraction=0.5,
        tier2_size_fraction=0.5,
        full_size_zscore=3.0,
        max_position_size=0.2,
        risk_budget_utilization=0.5,
        max_drawdown_limit=0.20,
        enforce_dollar_neutral=False,
        max_holding_days=10,
        stop_loss=0.05,
        min_training_months=12,
        annualization_days=252,
        commission_bps=0.0,
        bid_ask_bps=2.0,
        market_impact_bps=2.0,
        slippage_bps=1.0,
        node_corr_floor=0.20,
    )

    tradeable = node_tradeable_mask(correlation, config)

    assert tradeable.tolist() == [True, True, False]
