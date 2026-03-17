from __future__ import annotations

import argparse
from pathlib import Path

from .config_loader import load_config
from .diagnostics.asset_contribution import diagnose_asset_contribution
from .diagnostics.monthly_analysis import diagnose_monthly_performance
from .diagnostics.regime_validation import validate_regime_detector
from .pipeline import initialize_database, run_phase1_pipeline, verify_phase1_gate
from .phase2 import run_phase2_pipeline, verify_phase2_gate
from .phase3 import run_phase3_pipeline, verify_phase3_gate
from .phase2_sweep import run_phase2_sweep
from .phase3_sweep import run_phase3_sweep
from .scheduler import next_run_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trading system build pipeline.")
    parser.add_argument(
        "--config",
        default="config/phase1.yaml",
        help="Path to the Phase 1 configuration file.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("run-pipeline", help="Download, validate, store, and report Phase 1 data.")
    subparsers.add_parser("run-phase2", help="Run the Phase 2 graph engine and walk-forward backtest.")
    subparsers.add_parser("run-phase2-sweep", help="Sweep Phase 2 graph parameters across the configured search grid.")
    subparsers.add_parser("run-phase3", help="Run the combined Phase 2 graph engine with the Phase 3 TDA regime overlay.")
    subparsers.add_parser("run-phase3-sweep", help="Sweep the constrained Phase 3 TDA overlay parameters.")
    subparsers.add_parser("diagnose-monthly", help="Analyze the monthly P&L distribution for the current best Phase 2 run.")
    subparsers.add_parser("diagnose-assets", help="Analyze per-asset trade contribution for the latest Phase 2 run.")
    subparsers.add_parser("validate-regime-detector", help="Validate the Phase 3 TDA regime detector against known crisis windows.")
    subparsers.add_parser("init-db", help="Initialize the local PostgreSQL cluster and schema.")
    subparsers.add_parser("verify-phase1", help="Verify the Phase 1 validation gate against stored data.")
    subparsers.add_parser("verify-phase2", help="Verify the latest stored Phase 2 backtest run.")
    subparsers.add_parser("verify-phase3", help="Verify the Phase 3 combined-system gate against the frozen Phase 2 baseline.")
    subparsers.add_parser("show-schedule", help="Print the next scheduled Phase 1 run time.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = args.command or "run-pipeline"
    config_path = _resolve_config_path(args.config, command)

    if command == "run-pipeline":
        result = run_phase1_pipeline(config_path)
        print("Phase 1 pipeline completed.")
        print(f"Run ID: {result.run_id}")
        print(f"Raw rows: {result.raw_rows}")
        print(f"Valid rows: {result.valid_rows}")
        print(f"Issue rows: {result.issue_rows}")
        print(f"Database DSN: {result.db_init_result.dsn}")
        print(f"TimescaleDB enabled: {result.db_init_result.timescaledb_enabled}")
        for name, path in result.output_paths.items():
            print(f"{name}: {path}")
        return 0

    if command == "run-phase2":
        result = run_phase2_pipeline(config_path)
        print("Phase 2 graph engine completed.")
        print(f"Run ID: {result.run_id}")
        for key, value in result.summary_metrics.items():
            print(f"{key}: {value}")
        for name, path in result.output_paths.items():
            print(f"{name}: {path}")
        return 0

    if command == "run-phase2-sweep":
        result = run_phase2_sweep(config_path)
        print("Phase 2 parameter sweep completed.")
        print(f"Candidates evaluated: {result.total_candidates}")
        for key, value in result.best_metrics.items():
            print(f"{key}: {value}")
        for name, path in result.output_paths.items():
            print(f"{name}: {path}")
        return 0

    if command == "run-phase3-sweep":
        output_paths = run_phase3_sweep(config_path)
        print("Phase 3 parameter sweep completed.")
        for name, path in output_paths.items():
            print(f"{name}: {path}")
        return 0

    if command == "run-phase3":
        result = run_phase3_pipeline(config_path)
        print("Phase 3 combined system completed.")
        print(f"Run ID: {result.run_id}")
        for key, value in result.summary_metrics.items():
            print(f"{key}: {value}")
        for name, path in result.output_paths.items():
            print(f"{name}: {path}")
        return 0

    if command == "diagnose-monthly":
        result = diagnose_monthly_performance(config_path)
        print("Phase 2 monthly diagnostics completed.")
        print(f"Run ID: {result.run_id}")
        print(f"Breakdown: {result.output_path}")
        return 0

    if command == "diagnose-assets":
        result = diagnose_asset_contribution(config_path)
        print("Phase 2 asset diagnostics completed.")
        print(f"Run ID: {result.run_id}")
        print(f"Breakdown: {result.output_path}")
        return 0

    if command == "validate-regime-detector":
        result = validate_regime_detector(config_path)
        print("Phase 3 regime validation completed.")
        print(f"Hit rate: {result.hit_rate:.2%}")
        print(f"False positive rate: {result.false_positive_rate:.2%}")
        print(f"Report: {result.output_path}")
        return 0

    if command == "init-db":
        result = initialize_database(config_path)
        print("PostgreSQL initialized.")
        print(f"Database DSN: {result.dsn}")
        print(f"TimescaleDB enabled: {result.timescaledb_enabled}")
        return 0

    if command == "verify-phase1":
        result = verify_phase1_gate(config_path)
        print("Phase 1 verification:")
        print(f"can_backfill_full_history: {result.can_backfill_full_history}")
        print(f"automated_validation_reports: {result.automated_validation_reports}")
        print(f"postgres_storage_indexed: {result.postgres_storage_indexed}")
        print(f"externalized_config: {result.externalized_config}")
        print(f"clean_directory_structure: {result.clean_directory_structure}")
        print(f"daily_prices_row_count: {result.db_verification.daily_prices_row_count}")
        print(f"distinct_tickers: {result.db_verification.distinct_tickers}")
        print(f"earliest_trade_date: {result.db_verification.earliest_trade_date}")
        print(f"latest_trade_date: {result.db_verification.latest_trade_date}")
        print(f"quality_report_rows: {result.db_verification.quality_report_rows}")
        return 0

    if command == "verify-phase2":
        result = verify_phase2_gate(config_path)
        print("Phase 2 verification:")
        print(f"run_id: {result.latest_run.run_id}")
        print(f"sharpe_ratio: {result.latest_run.sharpe_ratio}")
        print(f"max_drawdown: {result.latest_run.max_drawdown}")
        print(f"win_rate: {result.latest_run.win_rate}")
        print(f"profit_factor: {result.latest_run.profit_factor}")
        print(f"avg_holding_days: {result.latest_run.avg_holding_days}")
        print(f"annual_turnover: {result.latest_run.annual_turnover}")
        print(f"profitable_month_fraction: {result.latest_run.profitable_month_fraction}")
        print(f"out_of_sample_months: {result.latest_run.out_of_sample_months}")
        print(f"gate_passed: {result.latest_run.gate_passed}")
        return 0

    if command == "verify-phase3":
        result = verify_phase3_gate(config_path)
        print("Phase 3 verification:")
        for key, value in result.summary_metrics.items():
            print(f"{key}: {value}")
        return 0

    if command == "show-schedule":
        config = load_config(config_path)
        upcoming = next_run_time(config)
        print(f"Next scheduled run: {upcoming.isoformat()}")
        return 0

    return 0


def _resolve_config_path(config_path: str, command: str) -> str:
    default_config = "config/phase1.yaml"
    phase3_commands = {"run-phase3", "run-phase3-sweep", "verify-phase3", "validate-regime-detector"}
    if config_path == default_config and command in phase3_commands:
        phase3_config = Path("config/phase3.yaml")
        if phase3_config.exists():
            return str(phase3_config)
    return config_path


if __name__ == "__main__":
    raise SystemExit(main())
