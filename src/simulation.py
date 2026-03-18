from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .broker_mock import MockBrokerClient
from .config_loader import load_config
from .pipeline import DailyPipeline


@dataclass(frozen=True)
class SimulationValidationResult:
    summary_metrics: dict[str, object]
    output_path: Path


def run_simulation(config_path: str | Path, days: int = 30) -> SimulationValidationResult:
    config = load_config(config_path)
    state_path = config.paths.processed_dir / "phase7_simulation_state.json"
    journal_path = config.paths.project_root / "data/trade_journal_simulation.db"
    if state_path.exists():
        state_path.unlink()
    if journal_path.exists():
        journal_path.unlink()

    broker = MockBrokerClient(
        starting_equity=100_000.0,
        min_slippage_bps=config.phase7.mock_slippage_bps_min,
        max_slippage_bps=config.phase7.mock_slippage_bps_max,
        seed=42,
    )
    pipeline = DailyPipeline(
        config_path,
        broker_client=broker,
        state_path=state_path,
        journal_path=journal_path,
    )

    try:
        simulation_dates = [pd.Timestamp(value) for value in pipeline.available_dates[-days:]]
        if not simulation_dates:
            raise ValueError("No dates were available for the requested simulation window.")

        reference_cash = 100_000.0
        reference_positions: dict[str, float] = {}
        daily_rows: list[dict[str, object]] = []
        aborted_days = 0
        manual_review_days = 0
        risk_checked_days = 0
        summary_key_failures = 0

        for date in simulation_dates:
            result = pipeline.run_daily(date)
            summary = result.daily_summary
            if result.aborted:
                aborted_days += 1
            if result.manual_review:
                manual_review_days += 1
            if "risk_limit_headroom" in summary:
                risk_checked_days += 1
            if not {"positions", "day_pnl", "allocation_weights", "risk_limit_headroom"}.issubset(summary):
                summary_key_failures += 1

            current_prices = pipeline._price_map_for_date(date)
            for order in result.approved_orders:
                signed_quantity = float(order["quantity"]) if order["side"] == "buy" else -float(order["quantity"])
                reference_positions[order["asset"]] = reference_positions.get(order["asset"], 0.0) + signed_quantity
                reference_cash -= signed_quantity * float(order["price"])
            reference_value = reference_cash + sum(
                float(quantity) * float(current_prices.get(asset, 0.0))
                for asset, quantity in reference_positions.items()
            )
            broker_value = float(summary["portfolio_value"])
            tracking_error_pct = 0.0 if reference_value == 0.0 else abs(broker_value - reference_value) / abs(reference_value)
            daily_rows.append(
                {
                    "date": date.date().isoformat(),
                    "broker_portfolio_value": broker_value,
                    "reference_portfolio_value": float(reference_value),
                    "tracking_error_pct": float(tracking_error_pct),
                    "approved_orders": len(result.approved_orders),
                    "rejected_orders": len(result.rejected_orders),
                    "manual_review": bool(result.manual_review),
                }
            )

        daily_frame = pd.DataFrame(daily_rows)
        trades = pipeline.trade_journal.get_trades()
        open_trade_count = int(len(pipeline.state.get("open_trades", {})))
    finally:
        pipeline.close()

    trade_journal_ok = True
    trade_journal_issues: list[str] = []
    if not trades.empty:
        required_columns = ["trade_id", "strategy", "asset", "entry_date", "exit_date", "net_pnl"]
        for column in required_columns:
            if trades[column].isna().any():
                trade_journal_ok = False
                trade_journal_issues.append(f"missing_values_in_{column}")

    max_tracking_error_pct = float(daily_frame["tracking_error_pct"].max()) if not daily_frame.empty else 0.0
    mean_tracking_error_pct = float(daily_frame["tracking_error_pct"].mean()) if not daily_frame.empty else 0.0
    summary_metrics = {
        "simulation_days": int(len(simulation_dates)),
        "aborted_days": int(aborted_days),
        "manual_review_days": int(manual_review_days),
        "risk_checked_days": int(risk_checked_days),
        "summary_key_failures": int(summary_key_failures),
        "trade_count": int(len(trades)),
        "open_trade_count": open_trade_count,
        "trade_journal_ok": bool(trade_journal_ok),
        "trade_journal_issues": trade_journal_issues,
        "max_tracking_error_pct": max_tracking_error_pct,
        "mean_tracking_error_pct": mean_tracking_error_pct,
        "tracking_within_tolerance": bool(max_tracking_error_pct <= 0.01),
        "simulation_passed": bool(
            aborted_days == 0
            and risk_checked_days == len(simulation_dates)
            and summary_key_failures == 0
            and trade_journal_ok
            and max_tracking_error_pct <= 0.01
        ),
    }

    output_path = config.paths.project_root / "diagnostics" / "simulation_validation.md"
    output_path.write_text(
        _render_simulation_report(summary_metrics, daily_frame),
        encoding="utf-8",
    )
    summary_path = config.paths.processed_dir / "simulation_validation_summary.json"
    summary_path.write_text(json.dumps(summary_metrics, indent=2, default=str), encoding="utf-8")
    return SimulationValidationResult(summary_metrics=summary_metrics, output_path=output_path)


def _render_simulation_report(summary_metrics: dict[str, object], daily_frame: pd.DataFrame) -> str:
    lines = [
        "# Simulation Validation",
        "",
        "## Summary",
        f"- Simulation days: `{summary_metrics['simulation_days']}`",
        f"- Aborted days: `{summary_metrics['aborted_days']}`",
        f"- Manual-review days: `{summary_metrics['manual_review_days']}`",
        f"- Risk checked days: `{summary_metrics['risk_checked_days']}`",
        f"- Trade count: `{summary_metrics['trade_count']}`",
        f"- Open trade count: `{summary_metrics['open_trade_count']}`",
        f"- Trade journal OK: `{summary_metrics['trade_journal_ok']}`",
        f"- Max tracking error: `{summary_metrics['max_tracking_error_pct']:.2%}`",
        f"- Mean tracking error: `{summary_metrics['mean_tracking_error_pct']:.2%}`",
        f"- Tracking within 1% tolerance: `{summary_metrics['tracking_within_tolerance']}`",
        f"- Simulation passed: `{summary_metrics['simulation_passed']}`",
        "",
        "## Daily Tracking Preview",
    ]
    if daily_frame.empty:
        lines.append("- No simulation days were available.")
    else:
        preview = daily_frame.tail(10).copy()
        lines.extend(_frame_to_markdown_table(preview))
    return "\n".join(lines) + "\n"


def _frame_to_markdown_table(frame: pd.DataFrame) -> list[str]:
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = [header, separator]
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in frame.columns]
        rows.append("| " + " | ".join(values) + " |")
    return rows
