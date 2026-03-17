import pandas as pd

from src.backtest import run_walk_forward_backtest
from src.backtest import scale_signals_to_risk_budget
from src.config_loader import Phase2Config


def test_walk_forward_backtest_returns_summary_and_logs() -> None:
    dates = pd.bdate_range("2022-01-03", periods=320)
    rows = []
    for idx, date in enumerate(dates):
        paired_signal = idx % 40 == 0
        for ticker in ["XLK", "XLF"]:
            if paired_signal and ticker == "XLK":
                zscore = -2.0
            elif paired_signal and ticker == "XLF":
                zscore = 2.0
            else:
                zscore = 0.0
            signal_direction = 1 if zscore <= -1.5 else -1 if zscore >= 1.5 else 0
            target_position = signal_direction * 0.1
            forward_return = 0.002 if signal_direction != 0 else 0.0
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "current_return": 0.001,
                    "expected_return": 0.0,
                    "residual": 0.001,
                    "sigma": 1.0,
                    "edge_density": 0.5,
                    "forward_return": forward_return,
                    "zscore": zscore,
                    "signal_direction": signal_direction,
                    "target_position": target_position,
                }
            )

    daily_signals = pd.DataFrame(rows)
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
        min_training_months=6,
        annualization_days=252,
        commission_bps=0.0,
        bid_ask_bps=2.0,
        market_impact_bps=2.0,
        slippage_bps=1.0,
    )

    result = run_walk_forward_backtest(daily_signals, config, run_id="phase2-test")

    assert not result.daily_results.empty
    assert result.summary_metrics["total_signals"] > 0
    assert result.summary_metrics["total_trades"] >= 0
    assert "sharpe_ratio" in result.summary_metrics


def test_walk_forward_backtest_keeps_single_sided_positions_when_not_neutral() -> None:
    dates = pd.bdate_range("2022-01-03", periods=320)
    rows = []
    for idx, date in enumerate(dates):
        zscore = -2.0 if idx % 20 == 0 else 0.0
        signal_direction = 1 if zscore <= -1.5 else 0
        target_position = signal_direction * 0.1
        rows.append(
            {
                "date": date,
                "ticker": "XLK",
                "current_return": 0.001,
                "expected_return": 0.0,
                "residual": 0.001,
                "sigma": 1.0,
                "edge_density": 0.5,
                "forward_return": 0.002 if signal_direction != 0 else 0.0,
                "zscore": zscore,
                "signal_direction": signal_direction,
                "target_position": target_position,
            }
        )

    daily_signals = pd.DataFrame(rows)
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
        min_training_months=6,
        annualization_days=252,
        commission_bps=0.0,
        bid_ask_bps=2.0,
        market_impact_bps=2.0,
        slippage_bps=1.0,
    )

    result = run_walk_forward_backtest(daily_signals, config, run_id="phase2-single-sided")

    assert result.summary_metrics["total_signals"] > 0
    assert result.summary_metrics["total_trades"] > 0


def test_scale_signals_to_risk_budget_increases_target_positions_when_drawdown_is_small() -> None:
    dates = pd.bdate_range("2022-01-03", periods=320)
    rows = []
    for idx, date in enumerate(dates):
        paired_signal = idx % 25 == 0
        for ticker in ["XLK", "XLF"]:
            if paired_signal and ticker == "XLK":
                zscore = -2.0
            elif paired_signal and ticker == "XLF":
                zscore = 2.0
            else:
                zscore = 0.0
            signal_direction = 1 if zscore <= -1.5 else -1 if zscore >= 1.5 else 0
            target_position = signal_direction * 0.05
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "current_return": 0.001,
                    "expected_return": 0.0,
                    "residual": 0.001,
                    "sigma": 1.0,
                    "edge_density": 0.5,
                    "forward_return": 0.0015 if signal_direction != 0 else 0.0,
                    "zscore": zscore,
                    "signal_direction": signal_direction,
                    "target_position": target_position,
                }
            )

    daily_signals = pd.DataFrame(rows)
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
        min_training_months=6,
        annualization_days=252,
        commission_bps=0.0,
        bid_ask_bps=2.0,
        market_impact_bps=2.0,
        slippage_bps=1.0,
    )

    scaling_result = scale_signals_to_risk_budget(daily_signals, config)

    assert scaling_result.scale_factor > 1.0
    assert scaling_result.target_max_drawdown == 0.10
    assert scaling_result.scaled_signals["target_position"].abs().max() > daily_signals["target_position"].abs().max()


def test_walk_forward_backtest_keeps_existing_position_during_no_trade_regime() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    daily_signals = pd.DataFrame(
        [
            {
                "date": dates[0],
                "ticker": "XLK",
                "current_return": 0.001,
                "expected_return": 0.0,
                "residual": 0.001,
                "sigma": 1.0,
                "edge_density": 0.5,
                "forward_return": 0.01,
                "zscore": -2.5,
                "signal_direction": 1,
                "target_position": 0.1,
            },
            {
                "date": dates[1],
                "ticker": "XLK",
                "current_return": 0.001,
                "expected_return": 0.0,
                "residual": 0.001,
                "sigma": 1.0,
                "edge_density": 0.1,
                "forward_return": 0.01,
                "zscore": -0.5,
                "signal_direction": 0,
                "target_position": 0.0,
            },
            {
                "date": dates[2],
                "ticker": "XLK",
                "current_return": 0.001,
                "expected_return": 0.0,
                "residual": 0.001,
                "sigma": 1.0,
                "edge_density": 0.1,
                "forward_return": 0.0,
                "zscore": 0.1,
                "signal_direction": 0,
                "target_position": 0.0,
            },
        ]
    )
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
        min_training_months=0,
        annualization_days=252,
        commission_bps=0.0,
        bid_ask_bps=2.0,
        market_impact_bps=2.0,
        slippage_bps=1.0,
        tier2_enabled=False,
        corr_floor=0.30,
        density_floor=0.40,
    )

    result = run_walk_forward_backtest(daily_signals, config, run_id="phase2-no-trade")

    assert len(result.trade_log) == 1
    assert result.trade_log.iloc[0]["holding_days"] == 2
