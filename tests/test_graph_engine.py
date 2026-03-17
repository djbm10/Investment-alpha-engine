import numpy as np
import pandas as pd

from src.config_loader import Phase2Config
from src.graph_engine import compute_graph_signals


def test_compute_graph_signals_produces_residuals_and_positions() -> None:
    dates = pd.bdate_range("2023-01-02", periods=150)
    tickers = ["XLK", "XLF", "XLE"]

    rows = []
    for idx, ticker in enumerate(tickers):
        base_series = 100 + np.linspace(0, 10, len(dates))
        noise = np.sin(np.arange(len(dates)) / (5 + idx)) * (1 + idx * 0.1)
        prices = base_series + noise
        for date, price in zip(dates, prices):
            rows.append({"date": date, "ticker": ticker, "adj_close": price})

    price_history = pd.DataFrame(rows)
    config = Phase2Config(
        lookback_window=60,
        diffusion_alpha=0.05,
        diffusion_steps=3,
        sigma_scale=1.0,
        min_weight=0.1,
        zscore_lookback=60,
        signal_threshold=1.5,
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

    signals = compute_graph_signals(price_history, tickers, config)

    assert not signals.empty
    assert {"residual", "zscore", "signal_direction", "target_position"}.issubset(signals.columns)
    assert signals["sigma"].gt(0).all()
    assert signals["edge_density"].between(0, 1).all()
