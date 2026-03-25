from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .backtest import run_walk_forward_backtest, scale_signals_to_risk_budget
from .config_loader import Phase2Config, PipelineConfig, load_config
from .database import PostgresStore
from .graph_engine import compute_graph_signals
from .logging_utils import setup_logger
from .storage import ensure_output_directories, load_validated_price_data


@dataclass(frozen=True)
class Phase2SweepResult:
    best_metrics: dict[str, object]
    output_paths: dict[str, Path]
    total_candidates: int


TOP_BASE_CONFIGURATION_COUNT = 1


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
        validated_prices = load_validated_price_data(config, dataset="sector", logger=logger)
        price_history = validated_prices.loc[
            validated_prices["is_valid"] & validated_prices["ticker"].isin(config.tickers),
            ["date", "ticker", "adj_close"],
        ].copy()
    if price_history.empty:
        raise ValueError("No validated Phase 1 price history was available after bootstrap.")

    logger.info(
        "Starting Phase 2 parameter sweep",
        extra={
            "context": {
                "risk_budget_utilizations": config.phase2_sweep.risk_budget_utilizations,
                "corr_floors": config.phase2_sweep.corr_floors,
                "density_floors": config.phase2_sweep.density_floors,
                "base_results_path": str(config.paths.processed_dir / "phase2_sweep_results.csv"),
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
    rows: list[dict[str, object]] = []
    previous_results_path = config.paths.processed_dir / "phase2_sweep_results.csv"
    if not previous_results_path.exists():
        raise ValueError(
            f"Previous sweep results were not found at '{previous_results_path}'. "
            "Run the previous sweep before starting the constrained v2 sweep."
        )

    previous_results = pd.read_csv(previous_results_path)
    base_configurations = build_base_phase2_configs(
        previous_results,
        config.phase2,
        top_n=TOP_BASE_CONFIGURATION_COUNT,
    )

    candidate_grid = product(
        base_configurations,
        config.phase2_sweep.corr_floors,
        config.phase2_sweep.density_floors,
        config.phase2_sweep.risk_budget_utilizations,
    )

    for (
        (base_rank, base_config),
        corr_floor,
        density_floor,
        risk_budget_utilization,
    ) in candidate_grid:
        candidate_config = replace(
            base_config,
            risk_budget_utilization=risk_budget_utilization,
            corr_floor=corr_floor,
            density_floor=density_floor,
        )
        candidate_signals = compute_graph_signals(price_history, config.tickers, candidate_config)
        scaling_result = scale_signals_to_risk_budget(candidate_signals, candidate_config)
        backtest_result = run_walk_forward_backtest(
            scaling_result.scaled_signals,
            candidate_config,
            run_id=f"sweep-{uuid4()}",
        )
        summary_metrics = backtest_result.summary_metrics.copy()
        summary_metrics["base_rank"] = base_rank
        summary_metrics["baseline_max_drawdown"] = scaling_result.baseline_max_drawdown
        summary_metrics["target_max_drawdown"] = scaling_result.target_max_drawdown
        summary_metrics["position_scale_factor"] = scaling_result.scale_factor
        summary_metrics["mean_graph_density"] = float(candidate_signals["graph_density"].mean())
        summary_metrics["median_graph_density"] = float(candidate_signals["graph_density"].median())
        summary_metrics["mean_avg_pairwise_corr"] = float(candidate_signals["avg_pairwise_corr"].mean())
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
    results_path = processed_dir / "phase2_sweep_results_v3.csv"
    best_path = processed_dir / "phase2_sweep_best_v3.json"

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


def build_base_phase2_configs(
    previous_results: pd.DataFrame,
    default_config: Phase2Config,
    top_n: int = TOP_BASE_CONFIGURATION_COUNT,
) -> list[tuple[int, Phase2Config]]:
    required_columns = [
        "lookback_window",
        "diffusion_alpha",
        "diffusion_steps",
        "sigma_scale",
        "min_weight",
        "zscore_lookback",
        "signal_threshold",
    ]
    missing_columns = [column for column in required_columns if column not in previous_results.columns]
    if missing_columns:
        raise ValueError(
            "Previous sweep results are missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    unique_bases = previous_results.drop_duplicates(subset=required_columns).head(top_n)
    if len(unique_bases) < top_n:
        raise ValueError(
            f"Previous sweep results only contained {len(unique_bases)} unique base configurations; "
            f"{top_n} are required."
        )

    configurations: list[tuple[int, Phase2Config]] = []
    for base_rank, row in enumerate(unique_bases.itertuples(index=False), start=1):
        configurations.append(
            (
                base_rank,
                replace(
                    default_config,
                    lookback_window=int(row.lookback_window),
                    diffusion_alpha=float(row.diffusion_alpha),
                    diffusion_steps=int(row.diffusion_steps),
                    sigma_scale=float(row.sigma_scale),
                    min_weight=float(row.min_weight),
                    zscore_lookback=int(row.zscore_lookback),
                    signal_threshold=float(row.signal_threshold),
                ),
            )
        )

    return configurations
