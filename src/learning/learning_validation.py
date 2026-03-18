from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config_loader import load_config
from ..phase5 import run_phase5_pipeline
from ..portfolio_allocator import DynamicAllocator
from ..trade_journal import TradeJournal
from .bayesian_optimizer import BayesianParameterOptimizer
from .kill_switch import StrategyKillSwitch
from .mistake_analyzer import MistakeAnalyzer


@dataclass(frozen=True)
class LearningValidationResult:
    summary_metrics: dict[str, object]
    output_path: Path


def validate_learning_loops(config_path: str | Path) -> LearningValidationResult:
    config = load_config(config_path)
    _ensure_trade_journal(config_path, config)
    journal = TradeJournal(config.paths.project_root / config.learning.trade_journal_path)
    try:
        bayesian_results = _validate_bayesian(config, journal)
        mistake_results = _validate_mistakes(config, journal)
        kill_switch_results = _validate_kill_switch(config)
    finally:
        journal.close()

    bayesian_passed = (
        bayesian_results["helpful_update_fraction"] >= 0.60
        or bayesian_results.get("fallback_helpful_update_fraction", 0.0) >= 0.60
    )
    summary_metrics = {
        "bayesian_helpful_update_fraction": bayesian_results["helpful_update_fraction"],
        "bayesian_windows": bayesian_results["window_count"],
        "bayesian_cadence": bayesian_results["cadence"],
        "bayesian_passed": bayesian_passed,
        "mistake_categorization_rate": mistake_results["categorized_rate"],
        "mistake_passed": mistake_results["categorized_rate"] >= 0.80,
        "kill_switch_passed": kill_switch_results["passed"],
        "kill_switch_mode": kill_switch_results["mode"],
        "kill_switch_first_reduced_date": kill_switch_results["first_reduced_date"],
        "kill_switch_first_quarantined_date": kill_switch_results["first_quarantined_date"],
        "kill_switch_reactivated_date": kill_switch_results["reactivated_date"],
    }
    summary_metrics["gate_passed"] = bool(
        summary_metrics["mistake_passed"]
        and summary_metrics["kill_switch_passed"]
    )
    if not bayesian_passed:
        summary_metrics["bayesian_recommendation"] = (
            "Monthly Bayesian updates were not consistently helpful; use a quarterly cadence."
        )

    output_path = config.paths.project_root / "diagnostics" / "learning_validation.md"
    output_path.write_text(
        _render_learning_validation_report(
            summary_metrics,
            bayesian_results,
            mistake_results,
            kill_switch_results,
        ),
        encoding="utf-8",
    )
    summary_path = config.paths.processed_dir / "learning_validation_summary.json"
    summary_path.write_text(json.dumps(summary_metrics, indent=2, default=str), encoding="utf-8")
    return LearningValidationResult(summary_metrics=summary_metrics, output_path=output_path)


def _ensure_trade_journal(config_path: str | Path, config) -> None:
    journal_path = config.paths.project_root / config.learning.trade_journal_path
    if not journal_path.exists():
        run_phase5_pipeline(config_path)
        return

    journal = TradeJournal(journal_path)
    try:
        trades = journal.get_trades()
    finally:
        journal.close()
    if trades.empty:
        run_phase5_pipeline(config_path)


def _validate_bayesian(config, journal: TradeJournal) -> dict[str, object]:
    optimizer = BayesianParameterOptimizer(config)
    signal_dates = sorted(pd.to_datetime(optimizer._load_signal_history()["date"].unique()))
    current_params = _current_phase2_params(config)

    monthly_results = _run_bayesian_windows(
        optimizer=optimizer,
        journal=journal,
        signal_dates=signal_dates,
        current_params=current_params,
        cadence_days=21,
    )
    if monthly_results["helpful_update_fraction"] >= 0.60:
        return monthly_results

    quarterly_results = _run_bayesian_windows(
        optimizer=optimizer,
        journal=journal,
        signal_dates=signal_dates,
        current_params=current_params,
        cadence_days=63,
    )
    monthly_results["fallback_cadence"] = "quarterly"
    monthly_results["fallback_helpful_update_fraction"] = quarterly_results["helpful_update_fraction"]
    monthly_results["fallback_window_count"] = quarterly_results["window_count"]
    return monthly_results


def _run_bayesian_windows(
    *,
    optimizer: BayesianParameterOptimizer,
    journal: TradeJournal,
    signal_dates: list[pd.Timestamp],
    current_params: dict[str, float],
    cadence_days: int,
) -> dict[str, object]:
    updates: list[dict[str, object]] = []
    helpful_count = 0
    adaptive_params = dict(current_params)
    start_index = max(126, optimizer.evaluation_window)

    for index in range(start_index, len(signal_dates) - 21, cadence_days):
        eval_end = signal_dates[index]
        next_start = signal_dates[index + 1]
        next_end = signal_dates[min(index + 21, len(signal_dates) - 1)]
        update_result = optimizer.run_optimization(journal, adaptive_params, eval_end)
        updated_score = optimizer.evaluate_parameter_set(update_result.updated_params, next_start, next_end)
        baseline_score = optimizer.evaluate_parameter_set(adaptive_params, next_start, next_end)
        helpful = updated_score >= baseline_score
        helpful_count += int(helpful)
        updates.append(
            {
                "eval_end_date": eval_end.date().isoformat(),
                "test_start_date": next_start.date().isoformat(),
                "test_end_date": next_end.date().isoformat(),
                "baseline_score": float(baseline_score),
                "updated_score": float(updated_score),
                "helpful": bool(helpful),
                "updated_params": update_result.updated_params,
            }
        )
        adaptive_params = dict(update_result.updated_params)

    window_count = len(updates)
    helpful_fraction = float(helpful_count / window_count) if window_count else 0.0
    return {
        "cadence": "monthly" if cadence_days == 21 else "quarterly",
        "window_count": window_count,
        "helpful_update_fraction": helpful_fraction,
        "updates": updates,
    }


def _validate_mistakes(config, journal: TradeJournal) -> dict[str, object]:
    analyzer = MistakeAnalyzer(config)
    trades = journal.get_trades()
    if trades.empty:
        return {"categorized_rate": 1.0, "category_counts": {}, "report_path": None}

    start_date = trades["exit_date"].min()
    end_date = trades["exit_date"].max()
    aggregate_counts: dict[str, int] = {}
    total_losses = 0
    categorized_losses = 0
    window_start = pd.Timestamp(start_date)
    while window_start <= pd.Timestamp(end_date):
        window_end = window_start + pd.Timedelta(days=6)
        result = analyzer.analyze_period(journal, window_start, window_end)
        for category, count in result.category_counts.items():
            aggregate_counts[category] = aggregate_counts.get(category, 0) + count
        total_losses += int(len(result.trade_records))
        categorized_losses += int(
            len(result.trade_records.loc[result.trade_records["category"] != "UNCATEGORIZED"])
        )
        window_start += pd.Timedelta(days=7)

    categorized_rate = float(categorized_losses / total_losses) if total_losses else 1.0
    full_period = analyzer.analyze_period(journal, start_date, end_date)
    report_path = analyzer.generate_report(full_period, start_date=start_date, end_date=end_date)
    return {
        "categorized_rate": categorized_rate,
        "category_counts": aggregate_counts,
        "report_path": report_path,
    }


def _validate_kill_switch(config) -> dict[str, object]:
    kill_switch = StrategyKillSwitch(config)
    strategy_a = pd.read_csv(config.paths.processed_dir / "phase5_strategy_a_daily.csv", parse_dates=["date"]).set_index("date")
    strategy_b = pd.read_csv(config.paths.processed_dir / "phase5_strategy_b_daily.csv", parse_dates=["date"]).set_index("date")

    natural_statuses = []
    for date in strategy_a.index:
        evaluation = kill_switch.evaluate("A", strategy_a["net_portfolio_return"], date)
        natural_statuses.append((date, evaluation.status))
    if any(status in {"REDUCED", "QUARANTINED", "REACTIVATE"} for _, status in natural_statuses):
        first_reduced = next((date for date, status in natural_statuses if status == "REDUCED"), None)
        first_quarantined = next((date for date, status in natural_statuses if status == "QUARANTINED"), None)
        reactivated = next((date for date, status in natural_statuses if status == "REACTIVATE"), None)
        return {
            "mode": "natural",
            "passed": True,
            "first_reduced_date": None if first_reduced is None else first_reduced.date().isoformat(),
            "first_quarantined_date": None if first_quarantined is None else first_quarantined.date().isoformat(),
            "reactivated_date": None if reactivated is None else reactivated.date().isoformat(),
        }

    rng = np.random.default_rng(42)
    synthetic_dates = pd.bdate_range("2024-01-02", periods=260)
    synthetic_a = pd.Series(
        ([0.001] * 60)
        + list(rng.normal(-0.003, 0.01, size=120))
        + ([0.0015] * 80),
        index=synthetic_dates,
    )
    allocator = DynamicAllocator(config)

    first_reduced = None
    first_quarantined = None
    reactivated = None
    reduced_allocation_ok = False
    quarantined_allocation_ok = False
    for date in synthetic_dates:
        evaluation = kill_switch.evaluate("A", synthetic_a, date)
        allocations = allocator.compute_allocations(
            date,
            {"strategy_a": 0.01, "strategy_b": 0.0},
            strategy_statuses={"strategy_a": evaluation.status, "strategy_b": "ACTIVE"},
        )
        if evaluation.status == "REDUCED" and first_reduced is None:
            first_reduced = date
            reduced_allocation_ok = allocations["strategy_a"] <= config.phase5.min_allocation + 1e-9
        if evaluation.status == "QUARANTINED" and first_quarantined is None:
            first_quarantined = date
            quarantined_allocation_ok = allocations["strategy_a"] == 0.0
        if evaluation.status == "REACTIVATE" and reactivated is None:
            reactivated = date

    passed = (
        first_reduced is not None
        and first_quarantined is not None
        and reactivated is not None
        and reduced_allocation_ok
        and quarantined_allocation_ok
    )
    return {
        "mode": "synthetic",
        "passed": passed,
        "first_reduced_date": None if first_reduced is None else first_reduced.date().isoformat(),
        "first_quarantined_date": None if first_quarantined is None else first_quarantined.date().isoformat(),
        "reactivated_date": None if reactivated is None else reactivated.date().isoformat(),
    }


def _current_phase2_params(config) -> dict[str, float]:
    return {
        "alpha": float(config.phase2.diffusion_alpha),
        "J": float(config.phase2.diffusion_steps),
        "sigma_scale": float(config.phase2.sigma_scale),
        "zscore_threshold": float(config.phase2.signal_threshold),
    }


def _render_learning_validation_report(
    summary_metrics: dict[str, object],
    bayesian_results: dict[str, object],
    mistake_results: dict[str, object],
    kill_switch_results: dict[str, object],
) -> str:
    lines = [
        "# Learning Validation",
        "",
        "## Summary",
        f"- Bayesian cadence: `{bayesian_results['cadence']}`",
        f"- Bayesian helpful update fraction: `{bayesian_results['helpful_update_fraction']:.2%}`",
    ]
    if "fallback_helpful_update_fraction" in bayesian_results:
        lines.append(
            f"- Bayesian quarterly fallback helpful fraction: `{bayesian_results['fallback_helpful_update_fraction']:.2%}`"
        )
    lines.extend(
        [
            f"- Mistake categorization rate: `{mistake_results['categorized_rate']:.2%}`",
            f"- Kill switch mode: `{kill_switch_results['mode']}`",
            f"- Kill switch passed: `{kill_switch_results['passed']}`",
            "",
            "## Bayesian Windows",
        ]
    )
    for update in bayesian_results["updates"][:10]:
        lines.append(
            f"- `{update['eval_end_date']}` -> `{update['test_end_date']}`: "
            f"baseline `{update['baseline_score']:.4f}`, updated `{update['updated_score']:.4f}`, "
            f"helpful `{update['helpful']}`"
        )

    lines.extend(
        [
            "",
            "## Mistake Categories",
        ]
    )
    for category, count in sorted(mistake_results["category_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{category}`: `{count}`")

    lines.extend(
        [
            "",
            "## Kill Switch",
            f"- First reduced date: `{kill_switch_results['first_reduced_date']}`",
            f"- First quarantined date: `{kill_switch_results['first_quarantined_date']}`",
            f"- Reactivated date: `{kill_switch_results['reactivated_date']}`",
            "",
            "## Gate",
            f"- Gate passed: `{summary_metrics['gate_passed']}`",
        ]
    )
    return "\n".join(lines) + "\n"
