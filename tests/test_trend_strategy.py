import numpy as np
import pandas as pd

from src.trend_strategy import compute_trend_target_weights


def test_compute_trend_target_weights_normalizes_long_signals() -> None:
    dates = pd.bdate_range("2020-01-01", periods=260)
    price_matrix = pd.DataFrame(
        {
            "SPY": np.linspace(100, 140, len(dates)),
            "TLT": np.linspace(100, 120, len(dates)),
            "GLD": np.linspace(120, 150, len(dates)),
            "UUP": np.linspace(100, 90, len(dates)),
        },
        index=dates,
    )

    weights, signals, forward_returns = compute_trend_target_weights(
        price_matrix=price_matrix,
        short_window=50,
        long_window=200,
        vol_lookback=20,
    )

    assert not weights.empty
    assert weights.index.equals(forward_returns.index)
    assert signals.index.equals(weights.index)
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (weights.loc[:, ["SPY", "TLT", "GLD"]] >= 0.0).all().all()
    assert (weights["UUP"] == 0.0).all()


def test_compute_trend_target_weights_goes_flat_without_uptrends() -> None:
    dates = pd.bdate_range("2020-01-01", periods=260)
    price_matrix = pd.DataFrame(
        {
            "SPY": np.linspace(140, 100, len(dates)),
            "TLT": np.linspace(120, 90, len(dates)),
            "GLD": np.linspace(150, 130, len(dates)),
            "UUP": np.linspace(100, 95, len(dates)),
        },
        index=dates,
    )

    weights, _, _ = compute_trend_target_weights(
        price_matrix=price_matrix,
        short_window=50,
        long_window=200,
        vol_lookback=20,
    )

    assert (weights.sum(axis=1) == 0.0).all()
