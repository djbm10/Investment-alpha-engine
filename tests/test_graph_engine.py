import numpy as np
import pandas as pd

from src.config_loader import Phase2Config
from src.graph_engine import apply_signal_rules, compute_graph_signals


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

    signals = compute_graph_signals(price_history, tickers, config)

    assert not signals.empty
    assert {"residual", "zscore", "signal_direction", "target_position"}.issubset(signals.columns)
    assert signals["sigma"].gt(0).all()
    assert signals["edge_density"].between(0, 1).all()
    assert signals["graph_density"].between(0, 1).all()
    assert signals["avg_pairwise_corr"].between(-1, 1).all()
    assert signals["graph_regime"].isin({"TRADEABLE", "REDUCED", "NO_TRADE"}).all()


def test_apply_signal_rules_assigns_moderate_and_strong_tiers() -> None:
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
        tier2_enabled=True,
    )
    signals = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "ticker": "XLK", "zscore": -2.2},
            {"date": pd.Timestamp("2024-01-02"), "ticker": "XLF", "zscore": 1.2},
            {"date": pd.Timestamp("2024-01-02"), "ticker": "XLE", "zscore": 0.4},
        ]
    )

    applied = apply_signal_rules(signals, config)

    strong = applied.loc[applied["ticker"] == "XLK"].iloc[0]
    moderate = applied.loc[applied["ticker"] == "XLF"].iloc[0]
    neutral = applied.loc[applied["ticker"] == "XLE"].iloc[0]
    assert strong["signal_tier"] == 2
    assert strong["target_position"] == 0.2
    assert moderate["signal_tier"] == 1
    assert moderate["target_position"] == -0.1
    assert neutral["signal_tier"] == 0
    assert neutral["target_position"] == 0.0


def test_apply_signal_rules_disables_moderate_tier_when_tier2_is_off() -> None:
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
        tier2_enabled=False,
    )
    signals = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "ticker": "XLK", "zscore": -2.2},
            {"date": pd.Timestamp("2024-01-02"), "ticker": "XLF", "zscore": 1.2},
        ]
    )

    applied = apply_signal_rules(signals, config)

    strong = applied.loc[applied["ticker"] == "XLK"].iloc[0]
    below_threshold = applied.loc[applied["ticker"] == "XLF"].iloc[0]
    assert strong["signal_tier"] == 2
    assert strong["target_position"] == 0.2
    assert below_threshold["signal_tier"] == 0
    assert below_threshold["target_position"] == 0.0


def test_apply_signal_rules_respects_reduced_and_no_trade_regimes() -> None:
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
        tier2_enabled=False,
        corr_floor=0.30,
        density_floor=0.40,
    )
    signals = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "ticker": "XLK",
                "zscore": -2.3,
                "regime_threshold_multiplier": 1.25,
                "regime_position_scale": 0.5,
                "allow_new_entries": True,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "ticker": "XLF",
                "zscore": -2.3,
                "regime_threshold_multiplier": 1.0,
                "regime_position_scale": 0.0,
                "allow_new_entries": False,
            },
        ]
    )

    applied = apply_signal_rules(signals, config)

    reduced = applied.loc[applied["ticker"] == "XLK"].iloc[0]
    no_trade = applied.loc[applied["ticker"] == "XLF"].iloc[0]
    assert reduced["signal_tier"] == 0
    assert reduced["target_position"] == 0.0
    assert no_trade["signal_tier"] == 2
    assert no_trade["signal_direction"] == 0
    assert no_trade["target_position"] == 0.0
