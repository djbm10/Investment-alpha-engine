from dataclasses import replace

import numpy as np
import pandas as pd

from src.backtest import apply_phase2_geo_overlay, scale_signals_to_risk_budget
from src.config_loader import load_config
from src.graph_engine import compute_graph_signals
from src.trend_strategy import compute_trend_target_weights
from src.trend_strategy import build_strategy_a_signal_history


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


def test_build_strategy_a_signal_history_uses_shared_geo_overlay_path() -> None:
    dates = pd.bdate_range("2023-01-02", periods=150)
    tickers = ["XLK", "XLE"]
    rows = []
    for idx, ticker in enumerate(tickers):
        base_series = 100 + np.linspace(0, 8, len(dates))
        noise = np.sin(np.arange(len(dates)) / (4 + idx)) * (1 + idx * 0.1)
        prices = base_series + noise
        for date, price in zip(dates, prices):
            rows.append({"date": date, "ticker": ticker, "adj_close": price})

    price_history = pd.DataFrame(rows)
    config = load_config("config/phase8.yaml")
    config = replace(config, tickers=tickers, geo=replace(config.geo, enabled=True))

    geo_snapshot = pd.DataFrame(
        {
            "trade_date": [dates[-1].date()],
            "asset": ["XLK"],
            "geo_net_score": [-0.9],
            "geo_structural_score": [-0.9],
            "avg_mapping_confidence": [0.95],
            "coverage_score": [0.95],
            "data_freshness_minutes": [30],
        }
    )

    built = build_strategy_a_signal_history(config=config, price_history=price_history, geo_snapshot=geo_snapshot)
    manual = compute_graph_signals(price_history, tickers, config.phase2)
    manual = apply_phase2_geo_overlay(manual, config.phase2, config.geo, geo_snapshot=geo_snapshot)
    manual = scale_signals_to_risk_budget(manual, config.phase2).scaled_signals

    pd.testing.assert_frame_equal(
        built.sort_index(axis=1).reset_index(drop=True),
        manual.sort_index(axis=1).reset_index(drop=True),
        check_dtype=False,
    )
