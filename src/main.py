from __future__ import annotations

import argparse
from pathlib import Path

from .config_loader import load_config
from .diagnostics.asset_contribution import diagnose_asset_contribution
from .diagnostics.monthly_analysis import diagnose_monthly_performance
from .diagnostics.regime_false_positives import diagnose_regime_false_positives
from .diagnostics.regime_validation import validate_regime_detector
from .deployment import check_live_readiness, deploy_live_mode
from .learning.bayesian_optimizer import run_bayesian_update
from .learning.learning_validation import validate_learning_loops
from .learning.mistake_analyzer import run_mistake_analysis
from .performance_tracker import build_performance_summary, generate_performance_report
from .pipeline import (
    initialize_database,
    run_daily_pipeline,
    run_daily_summary,
    run_phase1_pipeline,
    verify_phase1_gate,
)
from .phase2 import run_phase2_pipeline, verify_phase2_gate
from .phase3 import run_phase3_pipeline, verify_phase3_gate
from .phase4 import run_phase4_pipeline, train_tcn_ensemble, verify_phase4_gate
from .phase5 import run_phase5_pipeline, verify_phase5_gate
from .phase2_sweep import run_phase2_sweep
from .phase3_sweep import run_phase3_sweep
from .simulation import run_simulation
from .trend_strategy import run_trend_strategy_pipeline
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
    subparsers.add_parser("train-tcn", help="Train the latest Phase 4 TCN ensemble split and persist the checkpoints.")
    subparsers.add_parser("run-phase4", help="Run the Phase 4 walk-forward backtest with the TCN veto overlay.")
    subparsers.add_parser("run-trend-strategy", help="Run the standalone Phase 5 cross-asset trend-following strategy.")
    subparsers.add_parser("run-phase5", help="Run the combined Phase 5 dynamic allocation backtest.")
    bayesian_parser = subparsers.add_parser("run-bayesian-update", help="Run the Phase 6 Bayesian parameter update on Strategy A.")
    bayesian_parser.add_argument("--eval-date", required=True, help="Evaluation end date in YYYY-MM-DD format.")
    mistakes_parser = subparsers.add_parser("analyze-mistakes", help="Run the Phase 6 mistake pattern analyzer over a date range.")
    mistakes_parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    mistakes_parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    subparsers.add_parser("validate-learning", help="Run the Phase 6 historical learning-loop validation.")
    run_daily_parser = subparsers.add_parser("run-daily", help="Run the Phase 7 daily orchestration pipeline.")
    run_daily_parser.add_argument("--date", help="Trading date in YYYY-MM-DD format. Defaults to the latest available date.")
    run_daily_parser.add_argument("--mode", help="Override the configured broker mode for this run.")
    run_daily_summary_parser = subparsers.add_parser("run-daily-summary", help="Print the latest Phase 7 daily summary.")
    run_daily_summary_parser.add_argument("--mode", help="Override the configured broker mode for this run.")
    simulation_parser = subparsers.add_parser("run-simulation", help="Run the Phase 7 historical daily-pipeline simulation.")
    simulation_parser.add_argument("--days", type=int, default=30, help="Number of trading days to replay in simulation mode.")
    performance_report_parser = subparsers.add_parser("performance-report", help="Generate the Phase 8 performance report.")
    performance_report_parser.add_argument("--start", help="Optional start date in YYYY-MM-DD format.")
    performance_report_parser.add_argument("--end", help="Optional end date in YYYY-MM-DD format.")
    subparsers.add_parser("performance-summary", help="Print a one-line Phase 8 performance summary.")
    subparsers.add_parser("check-live-readiness", help="Validate whether the system is ready to switch from paper to live.")
    subparsers.add_parser("deploy-live", help="Run the readiness checks and switch deployment mode to live if all checks pass.")
    subparsers.add_parser("diagnose-monthly", help="Analyze the monthly P&L distribution for the current best Phase 2 run.")
    subparsers.add_parser("diagnose-assets", help="Analyze per-asset trade contribution for the latest Phase 2 run.")
    subparsers.add_parser("diagnose-fp", help="Analyze false positive regime flags for the latest Phase 3 run.")
    subparsers.add_parser("validate-regime-detector", help="Validate the Phase 3 TDA regime detector against known crisis windows.")
    subparsers.add_parser("init-db", help="Initialize the local PostgreSQL cluster and schema.")
    subparsers.add_parser("verify-phase1", help="Verify the Phase 1 validation gate against stored data.")
    subparsers.add_parser("verify-phase2", help="Verify the latest stored Phase 2 backtest run.")
    subparsers.add_parser("verify-phase3", help="Verify the Phase 3 combined-system gate against the frozen Phase 2 baseline.")
    subparsers.add_parser("verify-phase4", help="Verify the Phase 4 TCN gate against the frozen Phase 2 baseline.")
    subparsers.add_parser("verify-phase5", help="Verify the Phase 5 dynamic allocation gate.")
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

    if command == "train-tcn":
        result = train_tcn_ensemble(config_path)
        print("Phase 4 TCN training completed.")
        print(f"Latest window: {result.latest_window}")
        print(f"Validation losses: {result.validation_losses}")
        print(f"Model: {result.model_path}")
        print(f"Summary: {result.summary_path}")
        return 0

    if command == "run-phase4":
        result = run_phase4_pipeline(config_path)
        print("Phase 4 combined system completed.")
        print(f"Run ID: {result.run_id}")
        for key, value in result.summary_metrics.items():
            print(f"{key}: {value}")
        for name, path in result.output_paths.items():
            print(f"{name}: {path}")
        return 0

    if command == "run-trend-strategy":
        result = run_trend_strategy_pipeline(config_path)
        print("Trend strategy completed.")
        for key, value in result.summary_metrics.items():
            print(f"{key}: {value}")
        for name, path in result.output_paths.items():
            print(f"{name}: {path}")
        return 0

    if command == "run-phase5":
        result = run_phase5_pipeline(config_path)
        print("Phase 5 combined system completed.")
        print(f"Run ID: {result.run_id}")
        for key, value in result.summary_metrics.items():
            print(f"{key}: {value}")
        for name, path in result.output_paths.items():
            print(f"{name}: {path}")
        return 0

    if command == "run-bayesian-update":
        result = run_bayesian_update(config_path, args.eval_date)
        print("Bayesian parameter update completed.")
        print(f"Updated params: {result.updated_params}")
        print(f"Report: {result.output_path}")
        return 0

    if command == "analyze-mistakes":
        result = run_mistake_analysis(config_path, args.start, args.end)
        print("Mistake analysis completed.")
        print(f"Categorized rate: {result.categorized_rate:.2%}")
        print(f"Report: {result.output_path}")
        return 0

    if command == "validate-learning":
        result = validate_learning_loops(config_path)
        print("Learning validation completed.")
        for key, value in result.summary_metrics.items():
            print(f"{key}: {value}")
        print(f"Report: {result.output_path}")
        return 0

    if command == "run-daily":
        result = run_daily_pipeline(config_path, date=args.date, mode=args.mode)
        print("Phase 7 daily pipeline completed.")
        print(f"Date: {result.date.date().isoformat()}")
        print(f"Aborted: {result.aborted}")
        print(f"Manual review: {result.manual_review}")
        print(f"Allocations: {result.allocations}")
        print(f"Approved orders: {len(result.approved_orders)}")
        print(f"Rejected orders: {len(result.rejected_orders)}")
        print(f"Fills: {len(result.fills)}")
        print(f"Alerts: {len(result.alerts)}")
        for key, value in result.daily_summary.items():
            print(f"{key}: {value}")
        return 0

    if command == "run-daily-summary":
        summary = run_daily_summary(config_path, mode=args.mode)
        print("Phase 7 daily summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
        return 0

    if command == "run-simulation":
        result = run_simulation(config_path, days=args.days)
        print("Phase 7 simulation completed.")
        for key, value in result.summary_metrics.items():
            print(f"{key}: {value}")
        print(f"Report: {result.output_path}")
        return 0

    if command == "performance-report":
        stats, output_path = generate_performance_report(config_path, start_date=args.start, end_date=args.end)
        print("Phase 8 performance report generated.")
        print(f"sharpe_ratio: {stats['sharpe_ratio']}")
        print(f"max_drawdown: {stats['max_drawdown']}")
        print(f"cumulative_return: {stats['cumulative_return']}")
        print(f"Report: {output_path}")
        return 0

    if command == "performance-summary":
        stats = build_performance_summary(config_path)
        print(
            "performance_summary:"
            f" sharpe={stats['sharpe_ratio']:.4f}"
            f" max_dd={stats['max_drawdown']:.2%}"
            f" cumulative_return={stats['cumulative_return']:.2%}"
            f" rows={stats['row_count']}"
        )
        return 0

    if command == "check-live-readiness":
        ready, issues = check_live_readiness(config_path)
        print(f"ready: {ready}")
        for issue in issues:
            print(f"issue: {issue}")
        return 0 if ready else 1

    if command == "deploy-live":
        ready, issues = deploy_live_mode(config_path)
        print(f"deployed: {ready}")
        for issue in issues:
            print(f"issue: {issue}")
        return 0 if ready else 1

    if command == "diagnose-monthly":
        result = diagnose_monthly_performance(config_path)
        print("Monthly diagnostics completed.")
        print(f"Run ID: {result.run_id}")
        print(f"Breakdown: {result.output_path}")
        return 0

    if command == "diagnose-assets":
        result = diagnose_asset_contribution(config_path)
        print("Phase 2 asset diagnostics completed.")
        print(f"Run ID: {result.run_id}")
        print(f"Breakdown: {result.output_path}")
        return 0

    if command == "diagnose-fp":
        result = diagnose_regime_false_positives(config_path)
        print("Phase 3 false-positive diagnostics completed.")
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

    if command == "verify-phase4":
        result = verify_phase4_gate(config_path)
        print("Phase 4 verification:")
        for key, value in result.summary_metrics.items():
            print(f"{key}: {value}")
        return 0

    if command == "verify-phase5":
        result = verify_phase5_gate(config_path)
        print("Phase 5 verification:")
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
    phase3_commands = {"run-phase3", "run-phase3-sweep", "verify-phase3", "validate-regime-detector", "diagnose-fp"}
    phase4_commands = {"train-tcn", "run-phase4", "verify-phase4"}
    phase5_commands = {"run-trend-strategy", "run-phase5", "verify-phase5"}
    phase6_commands = {"run-bayesian-update", "analyze-mistakes", "validate-learning"}
    phase8_commands = {
        "run-daily",
        "run-daily-summary",
        "run-simulation",
        "performance-report",
        "performance-summary",
        "check-live-readiness",
        "deploy-live",
    }
    if config_path == default_config and command in phase3_commands:
        phase3_config = Path("config/phase3.yaml")
        if phase3_config.exists():
            return str(phase3_config)
    if config_path == default_config and command in phase4_commands:
        phase4_config = Path("config/phase4.yaml")
        if phase4_config.exists():
            return str(phase4_config)
    if config_path == default_config and command in phase5_commands:
        phase5_config = Path("config/phase5.yaml")
        if phase5_config.exists():
            return str(phase5_config)
    if config_path == default_config and command in phase6_commands:
        phase6_config = Path("config/phase6.yaml")
        if phase6_config.exists():
            return str(phase6_config)
    if config_path == default_config and command in phase8_commands:
        phase8_config = Path("config/phase8.yaml")
        if phase8_config.exists():
            return str(phase8_config)
        phase7_config = Path("config/phase7.yaml")
        if phase7_config.exists():
            return str(phase7_config)
    return config_path


if __name__ == "__main__":
    raise SystemExit(main())
