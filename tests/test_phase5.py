import pandas as pd

from src.phase5 import _allocator_responsiveness


def test_allocator_responsiveness_tracks_relative_performance() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    allocation_history = pd.DataFrame(
        {
            "date": dates,
            "allocation_strategy_a": [0.7] * 10 + [0.5] * 10 + [0.3] * 10,
            "allocation_strategy_b": [0.3] * 10 + [0.5] * 10 + [0.7] * 10,
        }
    )
    strategy_a_returns = pd.Series([0.01] * 10 + [0.0] * 10 + [-0.01] * 10, index=dates)
    strategy_b_returns = pd.Series([-0.01] * 10 + [0.0] * 10 + [0.01] * 10, index=dates)

    responsiveness, corr_a, corr_b = _allocator_responsiveness(
        allocation_history=allocation_history,
        strategy_a_returns=strategy_a_returns,
        strategy_b_returns=strategy_b_returns,
        lookback=5,
    )

    assert responsiveness > 0.2
    assert corr_a > 0.2
    assert corr_b > 0.2
