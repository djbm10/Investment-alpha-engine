from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config_loader import PipelineConfig, load_config
from .deployment import DeploymentManager
from .performance_tracker import PerformanceTracker


@dataclass(frozen=True)
class Phase7GateVerification:
    calendar_days_observed: int
    expected_trading_days: int
    actual_trading_days: int
    missed_trading_days: list[str]
    annualized_tracking_error: float
    critical_day_count: int
    learning_loops_on_schedule: bool
    gate_passed: bool
    output_path: Path


def verify_phase7_gate(config_path: str | Path) -> Phase7GateVerification:
    config = load_config(config_path)
    state_path = config.deployment.paper_state_path
    records = _load_state_records(state_path)
    if records.empty:
        return _write_gate_report(
            config,
            calendar_days_observed=0,
            expected_trading_days=0,
            actual_trading_days=0,
            missed_trading_days=[],
            annualized_tracking_error=float("inf"),
            critical_day_count=0,
            learning_loops_on_schedule=False,
            gate_passed=False,
            issues=["No paper trading state history was found."],
        )

    end_date = records["date"].max()
    start_date = end_date - pd.Timedelta(days=89)
    window_records = records.loc[records["date"].between(start_date, end_date)].copy()
    expected_trading_days = _expected_trading_dates(config, start_date, end_date)
    actual_trading_days = sorted({value.date().isoformat() for value in window_records["date"]})
    missed_trading_days = [value for value in expected_trading_days if value not in actual_trading_days]

    tracker = PerformanceTracker(config.paths.project_root / "data" / "performance.db", config=config)
    try:
        perf_frame = tracker._load_frame(start_date=start_date.date().isoformat(), end_date=end_date.date().isoformat())
    finally:
        tracker.close()
    if perf_frame.empty:
        annualized_tracking_error = float("inf")
        critical_day_count = 0
    else:
        tracking_errors = perf_frame["tracking_error_pct"].fillna(0.0)
        annualized_tracking_error = float(((tracking_errors.pow(2).mean()) ** 0.5) * (252 ** 0.5))
        critical_day_count = int((perf_frame["health_status"] == "CRITICAL").sum())

    learning_loops_on_schedule = _learning_loops_on_schedule(window_records, expected_trading_days)
    calendar_days_observed = int((end_date - start_date).days + 1)
    issues: list[str] = []
    if calendar_days_observed < 90:
        issues.append("Less than 90 calendar days of paper trading history are available.")
    if missed_trading_days:
        issues.append(f"Missed trading days: {', '.join(missed_trading_days[:10])}")
    if annualized_tracking_error >= 0.05:
        issues.append(f"Annualized tracking error {annualized_tracking_error:.2%} exceeds 5.00%.")
    if critical_day_count > 0:
        issues.append(f"Critical system days detected: {critical_day_count}.")
    if not learning_loops_on_schedule:
        issues.append("Weekly/monthly learning loops did not run on every scheduled boundary.")

    gate_passed = (
        calendar_days_observed >= 90
        and not missed_trading_days
        and annualized_tracking_error < 0.05
        and critical_day_count == 0
        and learning_loops_on_schedule
    )
    result = _write_gate_report(
        config,
        calendar_days_observed=calendar_days_observed,
        expected_trading_days=len(expected_trading_days),
        actual_trading_days=len(actual_trading_days),
        missed_trading_days=missed_trading_days,
        annualized_tracking_error=annualized_tracking_error,
        critical_day_count=critical_day_count,
        learning_loops_on_schedule=learning_loops_on_schedule,
        gate_passed=gate_passed,
        issues=issues,
    )
    if gate_passed:
        _write_phase7_clearance_artifact(config, result)
    return result


def emergency_halt(config_path: str | Path, *, reason: str = "manual_halt") -> dict[str, Any]:
    config = load_config(config_path)
    halt_flag_path = config.paths.log_dir / "emergency_halt.flag"
    halt_flag_path.parent.mkdir(parents=True, exist_ok=True)
    halt_flag_path.write_text(
        f"timestamp: {datetime.now(timezone.utc).isoformat()}\nreason: {reason}\n",
        encoding="utf-8",
    )
    manager = DeploymentManager(config)
    broker_client = manager.get_broker_client()
    closed_positions = broker_client.close_all_positions() if hasattr(broker_client, "close_all_positions") else []
    return {
        "halt_flag_path": halt_flag_path,
        "closed_positions": closed_positions,
        "mode": config.deployment.mode,
    }


def clear_emergency_halt(config_path: str | Path) -> bool:
    config = load_config(config_path)
    halt_flag_path = config.paths.log_dir / "emergency_halt.flag"
    if halt_flag_path.exists():
        halt_flag_path.unlink()
        return True
    return False


def _load_state_records(state_path: Path) -> pd.DataFrame:
    if not state_path.exists():
        return pd.DataFrame(columns=["date"])
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    records = pd.DataFrame(payload.get("daily_records", []))
    if records.empty:
        return records
    records["date"] = pd.to_datetime(records["date"])
    return records.sort_values("date").reset_index(drop=True)


def _expected_trading_dates(config: PipelineConfig, start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[str]:
    sector_path = config.paths.processed_dir / "sector_etf_prices_validated.csv"
    trend_path = config.paths.processed_dir / "trend_universe_prices_validated.csv"
    if not sector_path.exists() or not trend_path.exists():
        return []
    sector_dates = pd.read_csv(sector_path, usecols=["date"], parse_dates=["date"])["date"].drop_duplicates()
    trend_dates = pd.read_csv(trend_path, usecols=["date"], parse_dates=["date"])["date"].drop_duplicates()
    intersection = sorted(set(sector_dates) & set(trend_dates))
    return [
        pd.Timestamp(value).date().isoformat()
        for value in intersection
        if start_date <= pd.Timestamp(value) <= end_date
    ]


def _learning_loops_on_schedule(records: pd.DataFrame, expected_dates: list[str]) -> bool:
    if not expected_dates:
        return False
    expected_frame = pd.DataFrame({"date": pd.to_datetime(expected_dates)})
    expected_frame["week"] = expected_frame["date"].dt.strftime("%G-%V")
    expected_frame["month"] = expected_frame["date"].dt.strftime("%Y-%m")
    weekly_due = expected_frame.groupby("week")["date"].max().dt.date.astype(str).tolist()
    monthly_due = expected_frame.groupby("month")["date"].max().dt.date.astype(str).tolist()

    action_map = {
        pd.Timestamp(row["date"]).date().isoformat(): list(row.get("learning_actions", []) or [])
        for _, row in records.iterrows()
    }
    weekly_ok = all(
        any(str(action).startswith("weekly_mistake_analysis:") for action in action_map.get(date, []))
        for date in weekly_due
    )
    monthly_ok = all(
        any(str(action).startswith("monthly_bayesian_update:") for action in action_map.get(date, []))
        for date in monthly_due
    )
    return weekly_ok and monthly_ok


def _write_gate_report(
    config: PipelineConfig,
    *,
    calendar_days_observed: int,
    expected_trading_days: int,
    actual_trading_days: int,
    missed_trading_days: list[str],
    annualized_tracking_error: float,
    critical_day_count: int,
    learning_loops_on_schedule: bool,
    gate_passed: bool,
    issues: list[str],
) -> Phase7GateVerification:
    output_path = config.paths.project_root / "diagnostics" / "phase7_gate_verification.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 7 Gate Verification",
        "",
        f"- Calendar days observed: `{calendar_days_observed}`",
        f"- Expected trading days: `{expected_trading_days}`",
        f"- Actual trading days: `{actual_trading_days}`",
        f"- Missed trading days: `{len(missed_trading_days)}`",
        f"- Annualized tracking error: `{annualized_tracking_error:.2%}`" if annualized_tracking_error != float("inf") else "- Annualized tracking error: `n/a`",
        f"- Critical system days: `{critical_day_count}`",
        f"- Learning loops on schedule: `{learning_loops_on_schedule}`",
        f"- Gate passed: `{gate_passed}`",
        "",
        "## Issues",
    ]
    if not issues:
        lines.append("- None")
    else:
        for issue in issues:
            lines.append(f"- {issue}")
    if missed_trading_days:
        lines.extend(["", "## Missed Trading Days"] + [f"- {value}" for value in missed_trading_days[:20]])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Phase7GateVerification(
        calendar_days_observed=calendar_days_observed,
        expected_trading_days=expected_trading_days,
        actual_trading_days=actual_trading_days,
        missed_trading_days=missed_trading_days,
        annualized_tracking_error=annualized_tracking_error,
        critical_day_count=critical_day_count,
        learning_loops_on_schedule=learning_loops_on_schedule,
        gate_passed=gate_passed,
        output_path=output_path,
    )


def _write_phase7_clearance_artifact(config: PipelineConfig, result: Phase7GateVerification) -> None:
    artifact_path = config.paths.project_root / "config" / "phase7_cleared.yaml"
    payload = {
        "cleared_at": datetime.now(timezone.utc).date().isoformat(),
        "summary": {
            "calendar_days_observed": result.calendar_days_observed,
            "expected_trading_days": result.expected_trading_days,
            "actual_trading_days": result.actual_trading_days,
            "annualized_tracking_error": result.annualized_tracking_error,
            "critical_day_count": result.critical_day_count,
            "learning_loops_on_schedule": result.learning_loops_on_schedule,
        },
    }
    artifact_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
