from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path

import pandas as pd

from ..backtest import BacktestResult, run_walk_forward_backtest
from ..config_loader import Phase2Config, load_config
from ..database import Phase2RunArtifacts, PostgresStore
from ..logging_utils import setup_logger


@dataclass(frozen=True)
class MonthlyDiagnosticResult:
    run_id: str
    output_path: Path
    summary_text: str
    breakdown: pd.DataFrame


def diagnose_monthly_performance(
    config_path: str | Path,
    run_id: str | None = None,
) -> MonthlyDiagnosticResult:
    config = load_config(config_path)
    latest_summary_type = _resolve_latest_summary_type(config.paths.processed_dir)
    if latest_summary_type == "phase3" and run_id is None:
        backtest_result, resolved_run_id = _load_phase3_backtest_result(config.paths.processed_dir)
        breakdown = build_monthly_breakdown(backtest_result)

        output_dir = config.paths.project_root / "diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "monthly_breakdown.csv"
        breakdown.to_csv(output_path, index=False)

        summary_text = build_monthly_summary_text(breakdown)
        print(summary_text)

        return MonthlyDiagnosticResult(
            run_id=resolved_run_id,
            output_path=output_path,
            summary_text=summary_text,
            breakdown=breakdown,
        )

    logger = setup_logger(config.paths.pipeline_log_file, task="diagnose-monthly", phase="phase2")
    store = PostgresStore(config.database, config.paths, logger)

    try:
        store.initialize()
        store.ensure_phase2_schema()
        resolved_run_id = run_id or _resolve_run_id(config.paths.processed_dir)
        if resolved_run_id is None:
            latest_run = store.fetch_latest_phase2_run_summary()
            if latest_run is None:
                raise ValueError("No Phase 2 run found for monthly diagnostics.")
            resolved_run_id = latest_run.run_id
        artifacts = store.fetch_phase2_run_artifacts(resolved_run_id)
    finally:
        store.stop()

    phase2_config = _phase2_config_from_snapshot(config.phase2, artifacts.config_snapshot)
    backtest_result = run_walk_forward_backtest(artifacts.daily_signals, phase2_config, artifacts.run_id)
    breakdown = build_monthly_breakdown(backtest_result)

    output_dir = config.paths.project_root / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "monthly_breakdown.csv"
    breakdown.to_csv(output_path, index=False)

    summary_text = build_monthly_summary_text(breakdown)
    print(summary_text)

    return MonthlyDiagnosticResult(
        run_id=artifacts.run_id,
        output_path=output_path,
        summary_text=summary_text,
        breakdown=breakdown,
    )


def build_monthly_breakdown(backtest_result: BacktestResult) -> pd.DataFrame:
    if backtest_result.daily_results.empty or backtest_result.monthly_results.empty:
        return pd.DataFrame(
            columns=[
                "test_month",
                "gross_monthly_return",
                "net_monthly_return",
                "transaction_cost",
                "trade_count",
                "avg_entry_zscore",
                "avg_holding_days",
                "profitable",
                "active_month",
                "cost_flipped_month",
            ]
        )

    daily_results = backtest_result.daily_results.copy()
    daily_results["test_month"] = daily_results["date"].dt.to_period("M").dt.to_timestamp()
    oos_months = backtest_result.monthly_results["test_month"].tolist()
    daily_results = daily_results.loc[daily_results["test_month"].isin(oos_months)].copy()

    monthly_returns = (
        daily_results.groupby("test_month")
        .apply(
            lambda frame: pd.Series(
                {
                    "gross_monthly_return": float((1.0 + frame["gross_portfolio_return"]).prod() - 1.0),
                    "net_monthly_return": float((1.0 + frame["net_portfolio_return"]).prod() - 1.0),
                    "transaction_cost": float(frame["transaction_cost"].sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    trade_log = backtest_result.trade_log.copy()
    if trade_log.empty:
        trade_stats = pd.DataFrame(
            {"test_month": oos_months, "trade_count": 0, "avg_entry_zscore": None, "avg_holding_days": None}
        )
    else:
        trade_log["test_month"] = pd.to_datetime(trade_log["entry_date"]).dt.to_period("M").dt.to_timestamp()
        trade_stats = (
            trade_log.groupby("test_month")
            .agg(
                trade_count=("trade_id", "count"),
                avg_entry_zscore=("entry_zscore", "mean"),
                avg_holding_days=("holding_days", "mean"),
            )
            .reset_index()
        )

    breakdown = (
        backtest_result.monthly_results.loc[:, ["test_month", "profitable", "active_month"]]
        .merge(monthly_returns, on="test_month", how="left")
        .merge(trade_stats, on="test_month", how="left")
        .sort_values("test_month")
        .reset_index(drop=True)
    )

    breakdown["trade_count"] = breakdown["trade_count"].fillna(0).astype(int)
    breakdown["avg_entry_zscore"] = breakdown["avg_entry_zscore"].astype(float)
    breakdown["avg_holding_days"] = breakdown["avg_holding_days"].astype(float)
    if "active_month" not in breakdown.columns:
        breakdown["active_month"] = True
    breakdown["active_month"] = breakdown["active_month"].fillna(False).astype(bool)
    breakdown["cost_flipped_month"] = (
        breakdown["active_month"]
        & breakdown["gross_monthly_return"].gt(0)
        & breakdown["net_monthly_return"].le(0)
    )
    breakdown["inactive_month"] = ~breakdown["active_month"]
    breakdown["unprofitable_month"] = breakdown["active_month"] & ~breakdown["profitable"].astype(bool)
    return breakdown


def build_monthly_summary_text(breakdown: pd.DataFrame) -> str:
    if breakdown.empty:
        return "No out-of-sample monthly results were available for analysis."

    losing_months = breakdown.loc[breakdown["unprofitable_month"]].copy()
    losing_count = int(len(losing_months))
    active_months = int(breakdown["active_month"].sum())
    inactive_months = int(breakdown["inactive_month"].sum())
    thin_months = int((losing_months["trade_count"] < 5).sum())
    cost_flips = int(losing_months["cost_flipped_month"].sum())
    avg_cost = float(losing_months["transaction_cost"].mean()) if losing_count else 0.0
    avg_entry_z = float(losing_months["avg_entry_zscore"].dropna().mean()) if losing_count else 0.0
    avg_holding = float(losing_months["avg_holding_days"].dropna().mean()) if losing_count else 0.0

    summary_lines = [
        f"{losing_count} of {active_months} active out-of-sample months were unprofitable.",
        f"{inactive_months} inactive months were flat and excluded from the profitable-month calculation.",
        f"{thin_months} losing months had fewer than 5 entered trades.",
        f"{cost_flips} losing months were cost-flipped from positive gross to non-positive net returns.",
        f"Losing months averaged transaction cost {avg_cost:.6f}, entry z-score {avg_entry_z:.3f}, and holding period {avg_holding:.2f} days.",
    ]
    return " ".join(summary_lines)


def _resolve_run_id(processed_dir: Path) -> str | None:
    summary_path = processed_dir / "phase2_summary.json"
    if not summary_path.exists():
        return None

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    run_id = payload.get("run_id")
    if run_id is None:
        return None
    return str(run_id)


def _resolve_latest_summary_type(processed_dir: Path) -> str:
    phase3_summary = processed_dir / "phase3_summary.json"
    phase2_summary = processed_dir / "phase2_summary.json"
    if phase3_summary.exists() and (
        not phase2_summary.exists() or phase3_summary.stat().st_mtime >= phase2_summary.stat().st_mtime
    ):
        return "phase3"
    return "phase2"


def _load_phase3_backtest_result(processed_dir: Path) -> tuple[BacktestResult, str]:
    summary_path = processed_dir / "phase3_summary.json"
    if not summary_path.exists():
        raise ValueError("Phase 3 summary was not found for monthly diagnostics.")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id", "unknown"))
    daily_results = pd.read_csv(processed_dir / "phase3_daily_results.csv", parse_dates=["date"])
    trade_log = pd.read_csv(processed_dir / "phase3_trade_log.csv", parse_dates=["entry_date", "exit_date"])
    monthly_results = pd.read_csv(processed_dir / "phase3_monthly_results.csv", parse_dates=["test_month", "training_end_date"])
    summary_metrics = {key: value for key, value in payload.items() if key != "config_snapshot"}
    return BacktestResult(daily_results=daily_results, trade_log=trade_log, monthly_results=monthly_results, summary_metrics=summary_metrics), run_id


def _phase2_config_from_snapshot(default_config: Phase2Config, config_snapshot: dict[str, object]) -> Phase2Config:
    snapshot_phase2 = config_snapshot.get("phase2", {})
    if not isinstance(snapshot_phase2, dict):
        return default_config

    valid_fields = {field.name for field in fields(Phase2Config)}
    values = {key: value for key, value in snapshot_phase2.items() if key in valid_fields}
    return replace(default_config, **values)
