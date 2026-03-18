import pandas as pd

from src.config_loader import Phase5Config
from src.portfolio_allocator import DynamicAllocator


def _phase5_config() -> Phase5Config:
    return Phase5Config(
        trend_tickers=["SPY", "TLT", "GLD", "UUP"],
        trend_short_ma=50,
        trend_long_ma=200,
        trend_vol_lookback=20,
        trend_cost_bps=2.0,
        cash_return_annual=0.0,
        utility_lambda_uncertainty=1.0,
        utility_lambda_cost=2.0,
        softmax_temperature=1.0,
        min_allocation=0.15,
        max_allocation=0.85,
        rebalance_frequency_days=5,
        performance_lookback=20,
        daily_loss_limit=0.02,
    )


def test_compute_allocations_clips_and_renormalizes() -> None:
    allocator = DynamicAllocator(_phase5_config())
    weights = allocator.compute_allocations(
        pd.Timestamp("2024-01-31"),
        {"strategy_a": 0.20, "strategy_b": -0.10},
    )

    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert all(0.15 <= value <= 0.85 for value in weights.values())


def test_should_rebalance_uses_business_day_frequency() -> None:
    allocator = DynamicAllocator(_phase5_config())
    assert allocator.should_rebalance(pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-01")) is True
    assert allocator.should_rebalance(pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-01")) is False
