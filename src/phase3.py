from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .backtest import (
    BacktestResult,
    apply_phase3_regime_overlay,
    run_walk_forward_backtest,
    scale_signals_to_risk_budget,
)
from .config_loader import config_to_dict, load_config
from .diagnostics.regime_validation import (
    KNOWN_CRISIS_WINDOWS,
    compute_false_positive_rate,
    evaluate_crisis_detection,
)
from .graph_engine import compute_daily_graph_matrices, compute_graph_signals
from .storage import load_validated_price_data
from .tda_regime import RegimeState, TDARegimeDetector, stable_regime_observations_from_price_history


@dataclass(frozen=True)
class Phase3Result:
    run_id: str
    summary_metrics: dict[str, object]
    output_paths: dict[str, Path]


@dataclass(frozen=True)
class Phase3VerificationResult:
    gate_passed: bool
    summary_metrics: dict[str, object]
    output_path: Path


def run_phase3_pipeline(config_path: str | Path) -> Phase3Result:
    config = load_config(config_path)
    run_id = str(uuid4())
    price_history = _load_validated_price_history(config)
    regime_observations = _compute_regime_observations(price_history, config)
    regime_states = {date: observation.state.value for date, observation in regime_observations.items()}
    freeze_dates = _build_freeze_dates(regime_observations, config.phase3)
    signal_variant_by_date = _build_signal_variant_schedule(regime_observations, config.phase3)
    phase3_signals = _build_phase3_signal_frame(price_history, config, signal_variant_by_date, regime_states, freeze_dates)

    scaling_result = scale_signals_to_risk_budget(phase3_signals, config.phase2)
    backtest_result = run_walk_forward_backtest(scaling_result.scaled_signals, config.phase2, run_id)
    crisis_records, hit_rate = evaluate_crisis_detection(regime_observations, KNOWN_CRISIS_WINDOWS)
    false_positive_rate = compute_false_positive_rate(regime_observations, KNOWN_CRISIS_WINDOWS)

    backtest_result.summary_metrics.update(
        {
            "baseline_max_drawdown": scaling_result.baseline_max_drawdown,
            "target_max_drawdown": scaling_result.target_max_drawdown,
            "position_scale_factor": scaling_result.scale_factor,
            "regime_hit_rate": hit_rate,
            "regime_false_positive_rate": false_positive_rate,
            "stable_days": int(sum(observation.state == RegimeState.STABLE for observation in regime_observations.values())),
            "transitioning_days": int(sum(observation.state == RegimeState.TRANSITIONING for observation in regime_observations.values())),
            "new_regime_days": int(sum(observation.state == RegimeState.NEW_REGIME for observation in regime_observations.values())),
        }
    )
    output_paths = _save_phase3_outputs(
        processed_dir=config.paths.processed_dir,
        run_id=run_id,
        config_snapshot=config_to_dict(config),
        daily_signals=scaling_result.scaled_signals,
        daily_results=backtest_result.daily_results,
        trade_log=backtest_result.trade_log,
        monthly_results=backtest_result.monthly_results,
        regime_observations=regime_observations,
        summary_metrics=backtest_result.summary_metrics,
        crisis_records=crisis_records,
    )
    return Phase3Result(run_id=run_id, summary_metrics=backtest_result.summary_metrics, output_paths=output_paths)


def verify_phase3_gate(config_path: str | Path) -> Phase3VerificationResult:
    config = load_config(config_path)
    summary_path = config.paths.processed_dir / "phase3_summary.json"
    if not summary_path.exists():
        raise ValueError("Phase 3 summary was not found. Run `python3 -m src.main run-phase3` first.")

    phase3_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline_metrics = _compute_phase2_baseline_metrics(config.paths.project_root / "config/phase2_cleared.yaml")
    hit_rate = float(phase3_summary["regime_hit_rate"])
    phase3_sharpe = float(phase3_summary["sharpe_ratio"])
    phase3_drawdown = float(phase3_summary["max_drawdown"])
    phase3_annualized_return = float(phase3_summary["annualized_return"])
    baseline_drawdown = float(baseline_metrics["max_drawdown"])
    baseline_annualized_return = float(baseline_metrics["annualized_return"])

    drawdown_reduction = (baseline_drawdown - phase3_drawdown) / baseline_drawdown if baseline_drawdown > 0 else 0.0
    annualized_return_ratio = phase3_annualized_return / baseline_annualized_return if baseline_annualized_return != 0 else 0.0

    gate_passed = (
        hit_rate >= 0.70
        and drawdown_reduction >= 0.20
        and annualized_return_ratio >= 0.90
        and phase3_sharpe > 0.80
    )

    summary_metrics = {
        **phase3_summary,
        "phase2_baseline_sharpe": baseline_metrics["sharpe_ratio"],
        "phase2_baseline_annualized_return": baseline_annualized_return,
        "phase2_baseline_max_drawdown": baseline_drawdown,
        "drawdown_reduction_relative": drawdown_reduction,
        "annualized_return_ratio": annualized_return_ratio,
        "gate_passed": gate_passed,
    }
    summary_path.write_text(json.dumps(summary_metrics, indent=2, default=str), encoding="utf-8")
    return Phase3VerificationResult(gate_passed=gate_passed, summary_metrics=summary_metrics, output_path=summary_path)


def _compute_regime_observations(price_history: pd.DataFrame, config) -> dict[pd.Timestamp, object]:
    detector = TDARegimeDetector(config)
    if not detector.enabled:
        return stable_regime_observations_from_price_history(price_history)

    graph_matrices = compute_daily_graph_matrices(price_history, config.tickers, config.phase3.rolling_window)
    return detector.compute_daily_regime({date: snapshot.distance_matrix for date, snapshot in graph_matrices.items()})


def _build_phase3_signal_frame(
    price_history: pd.DataFrame,
    config,
    signal_variant_by_date: dict[pd.Timestamp, str],
    regime_states: dict[pd.Timestamp, str],
    freeze_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    transition_lookback = min(config.phase2.lookback_window, config.phase3.transition_lookback_cap)
    stable_signals = compute_graph_signals(price_history, config.tickers, config.phase2)
    transition_signals = compute_graph_signals(
        price_history,
        config.tickers,
        replace(config.phase2, lookback_window=transition_lookback),
    )
    emergency_signals = compute_graph_signals(
        price_history,
        config.tickers,
        replace(config.phase2, lookback_window=config.phase3.emergency_lookback),
    )

    signal_variants = {
        "stable": stable_signals,
        "transition": transition_signals,
        "emergency": emergency_signals,
    }
    stable_dates = sorted(pd.to_datetime(stable_signals["date"]).unique().tolist())
    selected_frames: list[pd.DataFrame] = []
    for date in stable_dates:
        timestamp = pd.Timestamp(date)
        variant = signal_variant_by_date.get(timestamp, "stable")
        chosen = signal_variants[variant]
        day_frame = chosen.loc[chosen["date"] == timestamp].copy()
        if day_frame.empty:
            day_frame = stable_signals.loc[stable_signals["date"] == timestamp].copy()
            variant = "stable"
        day_frame["phase3_signal_source"] = variant
        selected_frames.append(day_frame)

    combined = pd.concat(selected_frames, ignore_index=True)
    return apply_phase3_regime_overlay(
        combined,
        config.phase2,
        config.phase3,
        regime_states,
        freeze_dates=freeze_dates,
    )


def _build_signal_variant_schedule(
    regime_observations: dict[pd.Timestamp, object],
    phase3_config,
) -> dict[pd.Timestamp, str]:
    schedule: dict[pd.Timestamp, str] = {}
    consecutive_new_regime_days = 0
    for date in sorted(regime_observations):
        observation = regime_observations[date]
        if observation.state == RegimeState.NEW_REGIME:
            consecutive_new_regime_days += 1
            if consecutive_new_regime_days >= phase3_config.emergency_recalib_days:
                schedule[pd.Timestamp(date)] = "emergency"
            else:
                schedule[pd.Timestamp(date)] = "stable"
        else:
            consecutive_new_regime_days = 0
            if observation.state == RegimeState.TRANSITIONING:
                schedule[pd.Timestamp(date)] = "transition"
            else:
                schedule[pd.Timestamp(date)] = "stable"
    return schedule


def _build_freeze_dates(
    regime_observations: dict[pd.Timestamp, object],
    phase3_config,
) -> set[pd.Timestamp]:
    freeze_dates: set[pd.Timestamp] = set()
    consecutive_new_regime_days = 0
    for date in sorted(regime_observations):
        observation = regime_observations[date]
        if observation.state == RegimeState.NEW_REGIME:
            consecutive_new_regime_days += 1
            if consecutive_new_regime_days > phase3_config.new_regime_freeze_days:
                freeze_dates.add(pd.Timestamp(date))
        else:
            consecutive_new_regime_days = 0
    return freeze_dates


def _save_phase3_outputs(
    *,
    processed_dir: Path,
    run_id: str,
    config_snapshot: dict[str, object],
    daily_signals: pd.DataFrame,
    daily_results: pd.DataFrame,
    trade_log: pd.DataFrame,
    monthly_results: pd.DataFrame,
    regime_observations: dict[pd.Timestamp, object],
    summary_metrics: dict[str, object],
    crisis_records,
) -> dict[str, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    signals_path = processed_dir / "phase3_daily_signals.csv"
    daily_results_path = processed_dir / "phase3_daily_results.csv"
    trades_path = processed_dir / "phase3_trade_log.csv"
    monthly_path = processed_dir / "phase3_monthly_results.csv"
    regimes_path = processed_dir / "phase3_regime_states.csv"
    summary_path = processed_dir / "phase3_summary.json"

    daily_signals.to_csv(signals_path, index=False)
    daily_results.to_csv(daily_results_path, index=False)
    trade_log.to_csv(trades_path, index=False)
    monthly_results.to_csv(monthly_path, index=False)
    pd.DataFrame(
        [
            {
                "date": date,
                "state": observation.state.value,
                "total_distance": observation.total_distance,
                "h0_distance": observation.h0_distance,
                "h1_distance": observation.h1_distance,
                "rolling_mean": observation.rolling_mean,
                "rolling_std": observation.rolling_std,
            }
            for date, observation in sorted(regime_observations.items())
        ]
    ).to_csv(regimes_path, index=False)
    summary_payload = {
        "run_id": run_id,
        **summary_metrics,
        "config_snapshot": config_snapshot,
        "crisis_records": [
            {
                "name": record.name,
                "start": record.start.isoformat(),
                "end": record.end.isoformat(),
                "first_transition_date": record.first_transition_date.isoformat() if record.first_transition_date is not None else None,
                "first_new_regime_date": record.first_new_regime_date.isoformat() if record.first_new_regime_date is not None else None,
                "first_flag_date": record.first_flag_date.isoformat() if record.first_flag_date is not None else None,
                "detected_within_window": record.detected_within_window,
                "days_before_window_end": record.days_before_window_end,
            }
            for record in crisis_records
        ],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")

    return {
        "signals": signals_path,
        "daily_results": daily_results_path,
        "trades": trades_path,
        "monthly_results": monthly_path,
        "regimes": regimes_path,
        "summary": summary_path,
    }


def _load_validated_price_history(config) -> pd.DataFrame:
    price_history = load_validated_price_data(config, dataset="sector")
    return price_history.loc[
        price_history["is_valid"] & price_history["ticker"].isin(config.tickers),
        ["date", "ticker", "adj_close"],
    ].copy()


def _compute_phase2_baseline_metrics(config_path: str | Path) -> dict[str, object]:
    baseline_config = load_config(config_path)
    price_history = _load_validated_price_history(baseline_config)
    signals = compute_graph_signals(price_history, baseline_config.tickers, baseline_config.phase2)
    scaled_signals = scale_signals_to_risk_budget(signals, baseline_config.phase2).scaled_signals
    result = run_walk_forward_backtest(scaled_signals, baseline_config.phase2, run_id="phase2-cleared-baseline")
    return result.summary_metrics
