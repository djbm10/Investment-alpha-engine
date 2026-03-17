import pandas as pd

from src.config_loader import Phase2Config
from src.phase2_sweep import build_base_phase2_configs


def test_build_base_phase2_configs_uses_top_unique_previous_rows() -> None:
    previous_results = pd.DataFrame(
        [
            {
                "lookback_window": 60,
                "diffusion_alpha": 0.05,
                "diffusion_steps": 3,
                "sigma_scale": 1.0,
                "min_weight": 0.1,
                "zscore_lookback": 75,
                "signal_threshold": 2.8,
                "sharpe_ratio": 0.64,
            },
            {
                "lookback_window": 60,
                "diffusion_alpha": 0.05,
                "diffusion_steps": 3,
                "sigma_scale": 1.0,
                "min_weight": 0.1,
                "zscore_lookback": 75,
                "signal_threshold": 2.8,
                "sharpe_ratio": 0.63,
            },
            {
                "lookback_window": 60,
                "diffusion_alpha": 0.05,
                "diffusion_steps": 3,
                "sigma_scale": 1.0,
                "min_weight": 0.2,
                "zscore_lookback": 75,
                "signal_threshold": 2.8,
                "sharpe_ratio": 0.61,
            },
            {
                "lookback_window": 60,
                "diffusion_alpha": 0.05,
                "diffusion_steps": 3,
                "sigma_scale": 1.0,
                "min_weight": 0.1,
                "zscore_lookback": 60,
                "signal_threshold": 2.5,
                "sharpe_ratio": 0.55,
            },
        ]
    )
    default_config = Phase2Config(
        lookback_window=60,
        diffusion_alpha=0.05,
        diffusion_steps=3,
        sigma_scale=1.0,
        min_weight=0.1,
        zscore_lookback=75,
        signal_threshold=2.8,
        tier2_fraction=0.65,
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
    )

    bases = build_base_phase2_configs(previous_results, default_config, top_n=3)

    assert len(bases) == 3
    assert bases[0][1].min_weight == 0.1
    assert bases[1][1].min_weight == 0.2
    assert bases[2][1].zscore_lookback == 60
