import numpy as np

from src.config_loader import Phase2Config
from src.correlation_filter import REDUCED_REGIME, TRADEABLE_REGIME, node_average_correlations, node_tradeable_mask


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


def _make_config(node_corr_floor: float = 0.20, reduced_node_corr_multiplier: float = 0.75) -> Phase2Config:
    return Phase2Config(
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
        node_corr_floor=node_corr_floor,
        reduced_node_corr_multiplier=reduced_node_corr_multiplier,
    )


def test_node_tradeable_mask_tradeable_regime_unchanged() -> None:
    """TRADEABLE regime must not change behaviour — floor stays at node_corr_floor."""
    # node_avg_corr for node 2 = (0.1 + 0.1) / 2 = 0.10 < 0.20 → blocked
    correlation = np.array(
        [
            [1.0, 0.5, 0.1],
            [0.5, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ]
    )
    config = _make_config(node_corr_floor=0.20, reduced_node_corr_multiplier=0.75)

    tradeable = node_tradeable_mask(correlation, config, regime_state=TRADEABLE_REGIME)

    assert tradeable.tolist() == [True, True, False]


def test_node_tradeable_mask_reduced_regime_relaxes_floor() -> None:
    """REDUCED regime applies multiplier: effective floor = 0.20 * 0.75 = 0.15.
    Node 2 avg_corr = 0.10 is still below 0.15 → still blocked.
    Node 0 avg_corr = 0.30, node 1 avg_corr = 0.30 → both tradeable.
    """
    correlation = np.array(
        [
            [1.0, 0.5, 0.1],
            [0.5, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ]
    )
    config = _make_config(node_corr_floor=0.20, reduced_node_corr_multiplier=0.75)

    tradeable = node_tradeable_mask(correlation, config, regime_state=REDUCED_REGIME)

    # effective floor = 0.15; node 0 corr=0.30 ✓, node 1 corr=0.30 ✓, node 2 corr=0.10 ✗
    assert tradeable.tolist() == [True, True, False]


def test_node_tradeable_mask_reduced_regime_unlocks_marginal_node() -> None:
    """Node that was just below floor in TRADEABLE regime becomes tradeable in REDUCED regime.
    node_corr_floor=0.20, multiplier=0.75 → effective floor=0.15.
    Node with avg_corr=0.18 should be blocked in TRADEABLE but tradeable in REDUCED.
    """
    # 3 nodes: node 2 has avg_corr ≈ 0.18 (blocked at 0.20, passes at 0.15)
    correlation = np.array(
        [
            [1.0, 0.50, 0.22],
            [0.50, 1.0, 0.14],
            [0.22, 0.14, 1.0],
        ]
    )
    # node 2 avg_corr = (0.22 + 0.14) / 2 = 0.18
    config = _make_config(node_corr_floor=0.20, reduced_node_corr_multiplier=0.75)

    tradeable_normal = node_tradeable_mask(correlation, config, regime_state=TRADEABLE_REGIME)
    tradeable_reduced = node_tradeable_mask(correlation, config, regime_state=REDUCED_REGIME)

    assert tradeable_normal.tolist() == [True, True, False], "node 2 blocked in TRADEABLE regime"
    assert tradeable_reduced.tolist() == [True, True, True], "node 2 unlocked in REDUCED regime"


def test_node_tradeable_mask_default_regime_is_tradeable() -> None:
    """Omitting regime_state arg should behave identically to TRADEABLE regime."""
    correlation = np.array(
        [
            [1.0, 0.50, 0.22],
            [0.50, 1.0, 0.14],
            [0.22, 0.14, 1.0],
        ]
    )
    config = _make_config(node_corr_floor=0.20, reduced_node_corr_multiplier=0.75)

    tradeable_default = node_tradeable_mask(correlation, config)
    tradeable_explicit = node_tradeable_mask(correlation, config, regime_state=TRADEABLE_REGIME)

    assert (tradeable_default == tradeable_explicit).all()
