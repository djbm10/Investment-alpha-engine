from pathlib import Path

from src.performance_tracker import PerformanceTracker


def test_performance_tracker_records_and_computes_statistics(tmp_path: Path) -> None:
    tracker = PerformanceTracker(tmp_path / "performance.db")
    try:
        tracker.record_daily(
            date="2026-03-16",
            portfolio_value=100_000.0,
            daily_pnl=0.0,
            positions={"XLK": {"quantity": 10}},
            allocation_weights={"strategy_a": 0.6, "strategy_b": 0.4},
            regime_state="TRADEABLE",
            risk_headroom={"daily_loss_pct_remaining": 0.02, "weekly_loss_pct_remaining": 0.05, "monthly_loss_pct_remaining": 0.1},
            spy_return=0.001,
        )
        tracker.record_daily(
            date="2026-03-17",
            portfolio_value=101_000.0,
            daily_pnl=1_000.0,
            positions={"XLK": {"quantity": 10}},
            allocation_weights={"strategy_a": 0.7, "strategy_b": 0.3},
            regime_state="TRADEABLE",
            risk_headroom={"daily_loss_pct_remaining": 0.02, "weekly_loss_pct_remaining": 0.05, "monthly_loss_pct_remaining": 0.1},
            spy_return=0.002,
        )
        tracker.record_daily(
            date="2026-03-18",
            portfolio_value=100_500.0,
            daily_pnl=-500.0,
            positions={"SPY": {"quantity": 5}},
            allocation_weights={"strategy_a": 0.5, "strategy_b": 0.5},
            regime_state="REDUCED",
            risk_headroom={"daily_loss_pct_remaining": 0.015, "weekly_loss_pct_remaining": 0.04, "monthly_loss_pct_remaining": 0.09},
            spy_return=-0.001,
        )

        stats = tracker.compute_statistics()
    finally:
        tracker.close()

    assert stats["row_count"] == 3
    assert "2026-03" in stats["monthly_returns"]
    assert "strategy_a" in stats["strategy_attribution"]
    assert "beta" in stats["spy_comparison"]


def test_performance_tracker_generates_report(tmp_path: Path) -> None:
    tracker = PerformanceTracker(tmp_path / "performance.db")
    try:
        tracker.record_daily(
            date="2026-03-16",
            portfolio_value=100_000.0,
            daily_pnl=0.0,
            positions={},
            allocation_weights={"strategy_a": 0.5, "strategy_b": 0.5},
            regime_state="TRADEABLE",
            risk_headroom={"daily_loss_pct_remaining": 0.02, "weekly_loss_pct_remaining": 0.05, "monthly_loss_pct_remaining": 0.1},
        )
        output_path = tracker.generate_report(tmp_path / "performance_report.md")
    finally:
        tracker.close()

    assert output_path.exists()
    report_text = output_path.read_text(encoding="utf-8")
    assert "Performance Report" in report_text
    assert "Sharpe ratio" in report_text
