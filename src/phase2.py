from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .backtest import BacktestResult, run_walk_forward_backtest, scale_signals_to_risk_budget
from .config_loader import config_to_dict, load_config
from .database import Phase2RunSummary, PostgresStore
from .graph_engine import compute_graph_signals
from .logging_utils import setup_logger
from .storage import ensure_output_directories, load_validated_price_data


@dataclass(frozen=True)
class Phase2Result:
    run_id: str
    summary_metrics: dict[str, object]
    output_paths: dict[str, Path]


@dataclass(frozen=True)
class Phase2VerificationResult:
    latest_run: Phase2RunSummary


def run_phase2_pipeline(config_path: str | Path) -> Phase2Result:
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
        if price_history.empty:
            validated_prices = load_validated_price_data(config, dataset="sector", logger=logger)
            price_history = validated_prices.loc[
                validated_prices["is_valid"] & validated_prices["ticker"].isin(config.tickers),
                ["date", "ticker", "adj_close"],
            ].copy()
        if price_history.empty:
            raise ValueError("No validated Phase 1 price history was available after bootstrap.")

        logger.info(
            "Starting Phase 2 graph engine",
            extra={
                "context": {
                "tickers": config.tickers,
                "lookback_window": config.phase2.lookback_window,
                "diffusion_alpha": config.phase2.diffusion_alpha,
                "diffusion_steps": config.phase2.diffusion_steps,
                "sigma_scale": config.phase2.sigma_scale,
                "min_weight": config.phase2.min_weight,
            }
        },
    )

        daily_signals = compute_graph_signals(price_history, config.tickers, config.phase2)
        scaling_result = scale_signals_to_risk_budget(daily_signals, config.phase2)
        scaled_signals = scaling_result.scaled_signals
        backtest_result = run_walk_forward_backtest(scaled_signals, config.phase2, run_id)
        backtest_result.summary_metrics.update(
            {
                "baseline_max_drawdown": scaling_result.baseline_max_drawdown,
                "target_max_drawdown": scaling_result.target_max_drawdown,
                "position_scale_factor": scaling_result.scale_factor,
            }
        )
        output_paths = _save_phase2_outputs(config.paths.processed_dir, run_id, scaled_signals, backtest_result)

        completed_at = datetime.now(timezone.utc)
        store.persist_phase2_run(
            run_id=run_id,
            config_snapshot=config_to_dict(config),
            summary_metrics=backtest_result.summary_metrics,
            daily_signals=scaled_signals,
            trade_log=backtest_result.trade_log,
            monthly_results=backtest_result.monthly_results,
            started_at=started_at,
            completed_at=completed_at,
        )
    finally:
        store.stop()

    logger.info(
        "Completed Phase 2 graph engine",
        extra={
            "context": {
                "run_id": run_id,
                "gate_passed": backtest_result.summary_metrics["gate_passed"],
                "sharpe_ratio": backtest_result.summary_metrics["sharpe_ratio"],
                "max_drawdown": backtest_result.summary_metrics["max_drawdown"],
                "profitable_month_fraction": backtest_result.summary_metrics["profitable_month_fraction"],
                "output_paths": {name: str(path) for name, path in output_paths.items()},
            }
        },
    )

    return Phase2Result(
        run_id=run_id,
        summary_metrics=backtest_result.summary_metrics,
        output_paths=output_paths,
    )


def verify_phase2_gate(config_path: str | Path) -> Phase2VerificationResult:
    config = load_config(config_path)
    logger = setup_logger(config.paths.pipeline_log_file, task="verify-phase2", phase="phase2")
    store = PostgresStore(config.database, config.paths, logger)
    try:
        store.initialize()
        store.ensure_phase2_schema()
        latest_run = store.fetch_latest_phase2_run_summary()
    finally:
        store.stop()

    if latest_run is None:
        raise ValueError("No Phase 2 run found in PostgreSQL.")

    return Phase2VerificationResult(latest_run=latest_run)


def _save_phase2_outputs(
    processed_dir: Path,
    run_id: str,
    daily_signals,
    backtest_result: BacktestResult,
) -> dict[str, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    signals_path = processed_dir / "phase2_daily_signals.csv"
    trades_path = processed_dir / "phase2_trade_log.csv"
    monthly_path = processed_dir / "phase2_monthly_results.csv"
    summary_path = processed_dir / "phase2_summary.json"

    daily_signals.to_csv(signals_path, index=False)
    backtest_result.trade_log.to_csv(trades_path, index=False)
    backtest_result.monthly_results.to_csv(monthly_path, index=False)
    summary_payload = {"run_id": run_id, **backtest_result.summary_metrics}
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")

    return {
        "signals": signals_path,
        "trades": trades_path,
        "monthly_results": monthly_path,
        "summary": summary_path,
    }
