from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .backtest import run_walk_forward_backtest, scale_signals_to_risk_budget
from .config_loader import PipelineConfig, load_config
from .database import PostgresStore
from .graph_engine import apply_signal_rules, compute_graph_signals
from .logging_utils import setup_logger
from .storage import ensure_output_directories


@dataclass(frozen=True)
class Phase2SweepResult:
    best_metrics: dict[str, object]
    output_paths: dict[str, Path]
    total_candidates: int


def run_phase2_sweep(config_path: str | Path) -> Phase2SweepResult:
    config = load_config(config_path)
    run_id = str(uuid4())
    ensure_output_directories(config.paths)
    logger = setup_logger(config.paths.pipeline_log_file, run_id=run_id, phase="phase2")
    store = PostgresStore(config.database, config.paths, logger)
    started_at = datetime.now(timezone.utc)

    try:
        store.initialize()
        store.ensure_phase2_schema()
        price_history = store.fetch_validated_price_history(config.tickers)
    finally:
        store.stop()

    if price_history.empty:
        raise ValueError("No validated Phase 1 price history found in PostgreSQL.")

    logger.info(
        "Starting Phase 2 parameter sweep",
        extra={
            "context": {
                "lookback_windows": config.phase2_sweep.lookback_windows,
                "diffusion_alphas": config.phase2_sweep.diffusion_alphas,
                "diffusion_steps": config.phase2_sweep.diffusion_steps,
                "sigma_scales": config.phase2_sweep.sigma_scales,
                "min_weights": config.phase2_sweep.min_weights,
                "zscore_lookbacks": config.phase2_sweep.zscore_lookbacks,
                "risk_budget_utilizations": config.phase2_sweep.risk_budget_utilizations,
                "signal_thresholds": config.phase2_sweep.signal_thresholds,
            }
        },
    )

    results = _evaluate_candidates(price_history, config)
    output_paths = _save_sweep_outputs(config.paths.processed_dir, run_id, results)

    best_metrics = results.iloc[0].to_dict() if not results.empty else {}
    completed_at = datetime.now(timezone.utc)
    logger.info(
        "Completed Phase 2 parameter sweep",
        extra={
            "context": {
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "candidate_count": len(results),
                "gate_pass_count": int(results["gate_passed"].sum()) if not results.empty else 0,
                "best_metrics": best_metrics,
                "output_paths": {name: str(path) for name, path in output_paths.items()},
            }
        },
    )

    return Phase2SweepResult(
        best_metrics=best_metrics,
        output_paths=output_paths,
        total_candidates=len(results),
    )


def _evaluate_candidates(price_history: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    thresholds = sorted(set(config.phase2_sweep.signal_thresholds))
    if not thresholds:
        raise ValueError("Phase 2 sweep requires at least one signal threshold.")

    base_threshold = thresholds[0]
    rows: list[dict[str, object]] = []
    structural_candidates = list(
        product(
            config.phase2_sweep.lookback_windows,
            config.phase2_sweep.diffusion_alphas,
            config.phase2_sweep.diffusion_steps,
            config.phase2_sweep.sigma_scales,
            config.phase2_sweep.min_weights,
            config.phase2_sweep.zscore_lookbacks,
            config.phase2_sweep.risk_budget_utilizations,
        )
    )

    for (
        lookback_window,
        diffusion_alpha,
        diffusion_steps,
        sigma_scale,
        min_weight,
        zscore_lookback,
        risk_budget_utilization,
    ) in structural_candidates:
        structural_config = replace(
            config.phase2,
            lookback_window=lookback_window,
            diffusion_alpha=diffusion_alpha,
            diffusion_steps=diffusion_steps,
            sigma_scale=sigma_scale,
            min_weight=min_weight,
            zscore_lookback=zscore_lookback,
            risk_budget_utilization=risk_budget_utilization,
            signal_threshold=base_threshold,
        )
        structural_signals = compute_graph_signals(price_history, config.tickers, structural_config)

        for threshold in thresholds:
            candidate_config = replace(structural_config, signal_threshold=threshold)
            candidate_signals = apply_signal_rules(structural_signals, candidate_config)
            scaling_result = scale_signals_to_risk_budget(candidate_signals, candidate_config)
            backtest_result = run_walk_forward_backtest(
                scaling_result.scaled_signals,
                candidate_config,
                run_id=f"sweep-{uuid4()}",
            )
            summary_metrics = backtest_result.summary_metrics.copy()
            summary_metrics["baseline_max_drawdown"] = scaling_result.baseline_max_drawdown
            summary_metrics["target_max_drawdown"] = scaling_result.target_max_drawdown
            summary_metrics["position_scale_factor"] = scaling_result.scale_factor
            summary_metrics["mean_edge_density"] = float(structural_signals["edge_density"].mean())
            summary_metrics["median_edge_density"] = float(structural_signals["edge_density"].median())
            rows.append(summary_metrics)

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    return results.sort_values(
        by=[
            "gate_passed",
            "sharpe_ratio",
            "profitable_month_fraction",
            "profit_factor",
            "max_drawdown",
            "annual_turnover",
        ],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)


def _save_sweep_outputs(
    processed_dir: Path,
    run_id: str,
    results: pd.DataFrame,
) -> dict[str, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_path = processed_dir / "phase2_sweep_results.csv"
    best_path = processed_dir / "phase2_sweep_best.json"

    results.to_csv(results_path, index=False)
    best_payload = {
        "run_id": run_id,
        "candidate_count": int(len(results)),
        "gate_pass_count": int(results["gate_passed"].sum()) if not results.empty else 0,
        "best_candidate": results.iloc[0].to_dict() if not results.empty else {},
    }
    best_path.write_text(json.dumps(best_payload, indent=2, default=str), encoding="utf-8")

    return {
        "results": results_path,
        "best": best_path,
    }
