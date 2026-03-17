import pandas as pd

from src.backtest import BacktestResult
from src.diagnostics.monthly_analysis import build_monthly_breakdown, build_monthly_summary_text


def test_build_monthly_breakdown_includes_cost_flip_and_trade_stats() -> None:
    daily_results = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "gross_portfolio_return": 0.010,
                "net_portfolio_return": -0.001,
                "portfolio_return": -0.001,
                "gross_exposure": 0.5,
                "turnover": 0.2,
                "transaction_cost": 0.011,
            },
            {
                "date": pd.Timestamp("2024-02-01"),
                "gross_portfolio_return": 0.005,
                "net_portfolio_return": 0.004,
                "portfolio_return": 0.004,
                "gross_exposure": 0.4,
                "turnover": 0.1,
                "transaction_cost": 0.001,
            },
        ]
    )
    trade_log = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "run_id": "run-1",
                "ticker": "XLK",
                "entry_date": pd.Timestamp("2024-01-02"),
                "exit_date": pd.Timestamp("2024-01-05"),
                "position_direction": 1,
                "entry_zscore": -2.4,
                "exit_zscore": 0.2,
                "holding_days": 3,
                "entry_weight": 0.1,
                "gross_return": 0.01,
                "net_return": -0.001,
            }
        ]
    )
    monthly_results = pd.DataFrame(
        [
            {
                "test_month": pd.Timestamp("2024-01-01"),
                "training_end_date": pd.Timestamp("2023-12-31"),
                "monthly_return": -0.001,
                "profitable": False,
                "active_month": True,
            },
            {
                "test_month": pd.Timestamp("2024-02-01"),
                "training_end_date": pd.Timestamp("2024-01-31"),
                "monthly_return": 0.004,
                "profitable": True,
                "active_month": True,
            },
        ]
    )

    result = BacktestResult(
        daily_results=daily_results,
        trade_log=trade_log,
        monthly_results=monthly_results,
        summary_metrics={},
    )

    breakdown = build_monthly_breakdown(result)

    assert len(breakdown) == 2
    january = breakdown.loc[breakdown["test_month"] == pd.Timestamp("2024-01-01")].iloc[0]
    assert january["trade_count"] == 1
    assert bool(january["cost_flipped_month"]) is True
    assert january["avg_entry_zscore"] == -2.4


def test_build_monthly_summary_text_mentions_losing_month_patterns() -> None:
    breakdown = pd.DataFrame(
        [
            {
                "test_month": pd.Timestamp("2024-01-01"),
                "gross_monthly_return": 0.01,
                "net_monthly_return": -0.001,
                "transaction_cost": 0.011,
                "trade_count": 1,
                "avg_entry_zscore": -2.0,
                "avg_holding_days": 2.0,
                "profitable": False,
                "active_month": True,
                "cost_flipped_month": True,
                "inactive_month": False,
                "unprofitable_month": True,
            },
            {
                "test_month": pd.Timestamp("2024-02-01"),
                "gross_monthly_return": 0.02,
                "net_monthly_return": 0.018,
                "transaction_cost": 0.002,
                "trade_count": 6,
                "avg_entry_zscore": -2.5,
                "avg_holding_days": 3.0,
                "profitable": True,
                "active_month": True,
                "cost_flipped_month": False,
                "inactive_month": False,
                "unprofitable_month": False,
            },
            {
                "test_month": pd.Timestamp("2024-03-01"),
                "gross_monthly_return": 0.0,
                "net_monthly_return": 0.0,
                "transaction_cost": 0.0,
                "trade_count": 0,
                "avg_entry_zscore": float("nan"),
                "avg_holding_days": float("nan"),
                "profitable": False,
                "active_month": False,
                "cost_flipped_month": False,
                "inactive_month": True,
                "unprofitable_month": False,
            },
        ]
    )

    summary = build_monthly_summary_text(breakdown)

    assert "1 of 2 active out-of-sample months were unprofitable." in summary
    assert "1 inactive months were flat and excluded" in summary
    assert "1 losing months had fewer than 5 entered trades." in summary
    assert "1 losing months were cost-flipped" in summary
