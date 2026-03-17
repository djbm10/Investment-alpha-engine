from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config_loader import config_to_dict, load_config
from .database import DatabaseInitResult, DatabaseVerificationResult, PostgresStore
from .ingestion import download_universe_data
from .logging_utils import setup_logger
from .storage import ensure_output_directories, save_pipeline_outputs
from .validation import build_issue_report, build_quality_report, validate_prices


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    output_paths: dict[str, Path]
    raw_rows: int
    valid_rows: int
    issue_rows: int
    db_init_result: DatabaseInitResult


@dataclass(frozen=True)
class Phase1GateResult:
    can_backfill_full_history: bool
    automated_validation_reports: bool
    postgres_storage_indexed: bool
    externalized_config: bool
    clean_directory_structure: bool
    db_verification: DatabaseVerificationResult


def run_phase1_pipeline(config_path: str | Path) -> PipelineResult:
    config = load_config(config_path)
    run_id = str(uuid4())
    ensure_output_directories(config.paths)
    logger = setup_logger(config.paths.pipeline_log_file, run_id=run_id)
    started_at = datetime.now(timezone.utc)

    logger.info(
        "Starting Phase 1 pipeline",
        extra={"context": {"config_path": str(config_path), "tickers": config.tickers}},
    )

    raw_prices = download_universe_data(
        tickers=config.tickers,
        start_date=config.start_date,
        end_date=config.end_date,
        cache_dir=config.paths.cache_dir,
        logger=logger,
    )
    validated_prices = validate_prices(raw_prices, config.validation)
    quality_report = build_quality_report(validated_prices)
    issue_report = build_issue_report(validated_prices)
    output_paths = save_pipeline_outputs(
        raw_prices=raw_prices,
        validated_prices=validated_prices,
        quality_report=quality_report,
        issue_report=issue_report,
        paths=config.paths,
    )

    store = PostgresStore(config.database, config.paths, logger)
    db_init_result = store.initialize()
    try:
        completed_at = datetime.now(timezone.utc)
        store.persist_pipeline_run(
            run_id=run_id,
            source_name=config.price_source,
            start_date=config.start_date,
            end_date=config.end_date,
            config_snapshot=config_to_dict(config),
            raw_prices=raw_prices,
            validated_prices=validated_prices,
            quality_report=quality_report,
            issue_report=issue_report,
            started_at=started_at,
            completed_at=completed_at,
        )
    finally:
        store.stop()

    logger.info(
        "Completed Phase 1 pipeline",
        extra={
            "context": {
                "raw_rows": len(raw_prices),
                "valid_rows": int(validated_prices["is_valid"].sum()),
                "issue_rows": len(issue_report),
                "db_dsn": db_init_result.dsn,
                "timescaledb_enabled": db_init_result.timescaledb_enabled,
                "output_paths": {name: str(path) for name, path in output_paths.items()},
            }
        },
    )

    return PipelineResult(
        run_id=run_id,
        output_paths=output_paths,
        raw_rows=len(raw_prices),
        valid_rows=int(validated_prices["is_valid"].sum()),
        issue_rows=len(issue_report),
        db_init_result=db_init_result,
    )


def initialize_database(config_path: str | Path) -> DatabaseInitResult:
    config = load_config(config_path)
    logger = setup_logger(config.paths.pipeline_log_file, task="init-db")
    store = PostgresStore(config.database, config.paths, logger)
    try:
        return store.initialize()
    finally:
        store.stop()


def verify_phase1_gate(config_path: str | Path) -> Phase1GateResult:
    config = load_config(config_path)
    logger = setup_logger(config.paths.pipeline_log_file, task="verify-phase1")
    store = PostgresStore(config.database, config.paths, logger)
    try:
        store.initialize()
        db_verification = store.verify_phase1_gate()
    finally:
        store.stop()

    can_backfill_full_history = (
        db_verification.daily_prices_row_count > 0
        and db_verification.distinct_tickers == len(config.tickers)
        and db_verification.earliest_trade_date is not None
        and db_verification.latest_trade_date is not None
        and (
            (
                datetime.fromisoformat(db_verification.latest_trade_date)
                - datetime.fromisoformat(db_verification.earliest_trade_date)
            ).days
            >= 365 * 5
        )
    )
    automated_validation_reports = db_verification.quality_report_rows >= len(config.tickers)
    postgres_storage_indexed = db_verification.indexes_present
    externalized_config = str(config_path).endswith((".yaml", ".yml"))
    clean_directory_structure = all(
        path.exists()
        for path in (
            config.paths.raw_dir,
            config.paths.processed_dir,
            config.paths.log_dir,
            config.paths.project_root / "src",
            config.paths.project_root / "docs",
            config.paths.project_root / "config",
        )
    )

    return Phase1GateResult(
        can_backfill_full_history=can_backfill_full_history,
        automated_validation_reports=automated_validation_reports,
        postgres_storage_indexed=postgres_storage_indexed,
        externalized_config=externalized_config,
        clean_directory_structure=clean_directory_structure,
        db_verification=db_verification,
    )
