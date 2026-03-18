from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .backtest import scale_signals_to_risk_budget
from .capital_scaler import CapitalScaler
from .config_loader import config_to_dict, load_config
from .database import DatabaseInitResult, DatabaseVerificationResult, PostgresStore
from .deployment import DeploymentManager
from .graph_engine import compute_graph_signals
from .ingestion import download_universe_data
from .learning.bayesian_optimizer import BayesianParameterOptimizer
from .learning.kill_switch import StrategyKillSwitch
from .learning.mistake_analyzer import MistakeAnalyzer
from .logging_utils import setup_logger
from .monitoring import Monitor
from .order_manager import OrderManager
from .performance_tracker import PerformanceTracker
from .phase5 import _run_combined_backtest
from .portfolio_allocator import DynamicAllocator
from .risk_manager import RiskManager
from .storage import ensure_output_directories, save_pipeline_outputs
from .trade_journal import TradeJournal
from .trend_strategy import (
    backtest_trend_strategy,
    load_or_fetch_trend_price_history,
    load_phase2_baseline_backtest,
)
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


@dataclass(frozen=True)
class DailyPipelineResult:
    date: pd.Timestamp
    allocations: dict[str, float]
    approved_orders: list[dict[str, Any]]
    rejected_orders: list[dict[str, Any]]
    fills: list[dict[str, Any]]
    discrepancies: list[dict[str, Any]]
    alerts: list[str]
    daily_summary: dict[str, Any]
    manual_review: bool
    aborted: bool
    learning_actions: list[str]
    health_status: str = "HEALTHY"
    report_path: str | None = None


class DailyPipeline:
    def __init__(
        self,
        config_path: str | Path,
        *,
        broker_client: Any | None = None,
        mode_override: str | None = None,
        state_path: str | Path | None = None,
        journal_path: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.mode = mode_override or self.config.deployment.mode
        self.state_path = Path(state_path) if state_path is not None else self.config.paths.processed_dir / "phase7_state.json"
        self.logger = setup_logger(self.config.paths.pipeline_log_file, task="phase7-daily", mode=self.mode)
        ensure_output_directories(self.config.paths)

        self.trade_journal = TradeJournal(
            journal_path if journal_path is not None else self.config.paths.project_root / self.config.learning.trade_journal_path
        )
        self.allocator = DynamicAllocator(self.config)
        self.risk_manager = RiskManager(self.config)
        self.kill_switch = StrategyKillSwitch(self.config)
        self.mistake_analyzer = MistakeAnalyzer(self.config)
        self.bayesian_optimizer = BayesianParameterOptimizer(self.config)
        self.monitor = Monitor(self.config)
        self.performance_tracker = PerformanceTracker(self.config.paths.project_root / "data" / "performance.db", config=self.config)
        self.broker_client = broker_client or self._build_default_broker_client()
        self.order_manager = OrderManager(self.config, self.broker_client)
        self.state = self._load_state()
        self._load_strategy_context()

    def close(self) -> None:
        self.performance_tracker.close()
        self.trade_journal.close()

    def _build_default_broker_client(self) -> Any:
        deployment_config = replace(
            self.config,
            deployment=replace(self.config.deployment, mode=self.mode),
        )
        return DeploymentManager(deployment_config).get_broker_client()

    def run_daily(self, date: str | pd.Timestamp | None = None) -> DailyPipelineResult:
        alerts: list[str] = []
        learning_actions: list[str] = []
        manual_review = False
        approved_orders: list[dict[str, Any]] = []
        rejected_orders: list[dict[str, Any]] = []
        fills: list[dict[str, Any]] = []
        discrepancies: list[dict[str, Any]] = []

        trading_date = self._resolve_trading_date(date)
        if any(record["date"] == trading_date.date().isoformat() for record in self.state["daily_records"]):
            raise ValueError(f"Trading date {trading_date.date().isoformat()} already exists in the Phase 7 state.")

        try:
            self._mark_broker_to_market(trading_date)
            current_positions = self.broker_client.get_positions()
            pre_trade_portfolio_value = float(self.broker_client.get_portfolio_value())
            daily_pnl, weekly_pnl, monthly_pnl = self._compute_mark_to_market_pnl(
                trading_date=trading_date,
                current_portfolio_value=pre_trade_portfolio_value,
            )
            allocation_weights, strategy_statuses, corr_result = self._compute_allocations(trading_date)
            if corr_result.warning:
                alerts.append(
                    f"SPY correlation warning on {trading_date.date().isoformat()}: {corr_result.correlation:.3f}"
                )
            capital_scale_factor = self._current_capital_scale_factor(trading_date)
            target_positions = self._build_target_positions(
                trading_date=trading_date,
                portfolio_value=pre_trade_portfolio_value,
                allocation_weights=allocation_weights,
                position_scale=corr_result.reduction_scale,
                capital_scale_factor=capital_scale_factor,
            )
            proposed_orders = self.order_manager.generate_orders(target_positions, current_positions)
        except Exception as exc:
            alerts.append(f"Pre-trade pipeline failure: {exc}")
            self.logger.error("Phase 7 pre-trade failure", extra={"context": {"traceback": traceback.format_exc()}})
            summary = self._build_daily_summary(
                trading_date=trading_date,
                current_positions=self.broker_client.get_positions(),
                allocations=self.state.get("allocations", {"strategy_a": 0.5, "strategy_b": 0.5}),
                alerts=alerts,
                daily_pnl=0.0,
                weekly_pnl=0.0,
                monthly_pnl=0.0,
                capital_scale_factor=self._current_capital_scale_factor(trading_date),
                tracking_error_pct=0.0,
            )
            result = DailyPipelineResult(
                date=trading_date,
                allocations=self.state.get("allocations", {"strategy_a": 0.5, "strategy_b": 0.5}),
                approved_orders=[],
                rejected_orders=[],
                fills=[],
                discrepancies=[],
                alerts=alerts,
                daily_summary=summary,
                manual_review=True,
                aborted=True,
                learning_actions=[],
            )
            return self._finalize_result(result)

        try:
            approved_order_objs, rejected_orders, rejection_reasons = self.risk_manager.check_pre_trade(
                proposed_orders=[_order_to_risk_payload(order) for order in proposed_orders],
                current_positions=current_positions,
                portfolio_value=pre_trade_portfolio_value,
                daily_pnl=daily_pnl,
                weekly_pnl=weekly_pnl,
                monthly_pnl=monthly_pnl,
            )
            approved_lookup = {(_order_key(order)): order for order in proposed_orders}
            approved_order_list = [approved_lookup[_order_key_from_payload(payload)] for payload in approved_order_objs]
            approved_orders = [payload.copy() for payload in approved_order_objs]
            for order_ref, reasons in rejection_reasons.items():
                alerts.append(f"Rejected order {order_ref}: {'; '.join(reasons)}")

            submissions = self.order_manager.submit_orders(approved_order_list)
            if submissions:
                fills = self.order_manager.check_fills(
                    [submission["order_id"] for submission in submissions],
                    timeout_seconds=self.config.phase7.fill_timeout_seconds,
                )
            broker_positions = self.broker_client.get_positions()
            expected_positions = _expected_positions_after_orders(current_positions, approved_order_list)
            discrepancies = self.order_manager.reconcile(expected_positions, broker_positions)
            discrepancies.extend(
                self.risk_manager.check_post_trade(
                    current_positions=broker_positions,
                    portfolio_value=float(self.broker_client.get_portfolio_value()),
                )
            )
            if discrepancies:
                manual_review = True
                alerts.append("Position reconciliation mismatch detected; manual review required.")
        except Exception as exc:
            alerts.append(f"Execution failure: {exc}")
            manual_review = True
            self.logger.error("Phase 7 execution failure", extra={"context": {"traceback": traceback.format_exc()}})
            discrepancies = self.order_manager.reconcile(
                target_positions,
                self.broker_client.get_positions(),
            )
            summary = self._build_daily_summary(
                trading_date=trading_date,
                current_positions=self.broker_client.get_positions(),
                allocations=allocation_weights,
                alerts=alerts,
                daily_pnl=daily_pnl,
                weekly_pnl=weekly_pnl,
                monthly_pnl=monthly_pnl,
                discrepancies=discrepancies,
                strategy_statuses=strategy_statuses,
                capital_scale_factor=capital_scale_factor,
                tracking_error_pct=0.0,
            )
            result = DailyPipelineResult(
                date=trading_date,
                allocations=allocation_weights,
                approved_orders=approved_orders,
                rejected_orders=rejected_orders,
                fills=fills,
                discrepancies=discrepancies,
                alerts=alerts,
                daily_summary=summary,
                manual_review=True,
                aborted=True,
                learning_actions=[],
            )
            return self._finalize_result(result)

        try:
            self._update_open_trades(
                trading_date=trading_date,
                pre_trade_positions=current_positions,
                post_trade_positions=self.broker_client.get_positions(),
                fills=fills,
            )
        except Exception as exc:
            alerts.append(f"Trade journal update failed: {exc}")
            manual_review = True
            self.logger.error("Phase 7 journal failure", extra={"context": {"traceback": traceback.format_exc()}})

        end_portfolio_value = float(self.broker_client.get_portfolio_value())
        post_daily_pnl, post_weekly_pnl, post_monthly_pnl = self._compute_mark_to_market_pnl(
            trading_date=trading_date,
            current_portfolio_value=end_portfolio_value,
        )
        tracking_error_pct = self._tracking_error_pct(
            trading_date=trading_date,
            daily_pnl=post_daily_pnl,
            end_portfolio_value=end_portfolio_value,
        )
        circuit_action, circuit_reason = self.risk_manager.check_circuit_breakers(
            daily_pnl=post_daily_pnl,
            weekly_pnl=post_weekly_pnl,
            monthly_pnl=post_monthly_pnl,
            portfolio_value=end_portfolio_value,
        )
        if circuit_action != "CONTINUE":
            alerts.append(f"Circuit breaker: {circuit_reason}")
            manual_review = True

        record = {
            "date": trading_date.date().isoformat(),
            "portfolio_value": end_portfolio_value,
            "day_pnl": post_daily_pnl,
            "week_pnl": post_weekly_pnl,
            "month_pnl": post_monthly_pnl,
            "ytd_pnl": self._year_to_date_pnl(trading_date, post_daily_pnl),
            "allocation_strategy_a": float(allocation_weights["strategy_a"]),
            "allocation_strategy_b": float(allocation_weights["strategy_b"]),
            "capital_scale_factor": float(capital_scale_factor),
            "tracking_error_pct": float(tracking_error_pct),
            "strategy_a_status": strategy_statuses["strategy_a"],
            "strategy_b_status": strategy_statuses["strategy_b"],
            "rejected_order_count": len(rejected_orders),
            "approved_order_count": len(approved_orders),
            "circuit_action": circuit_action,
            "manual_review": manual_review or bool(discrepancies),
        }
        self.state["daily_records"].append(record)
        self.state["last_rebalance_date"] = (
            trading_date.date().isoformat() if self._rebalanced_on(trading_date) else self.state.get("last_rebalance_date")
        )
        self.state["allocations"] = allocation_weights
        self.state["last_positions"] = self.broker_client.get_positions()
        self._save_state()

        if self._is_week_end(trading_date):
            try:
                start_date = trading_date - pd.Timedelta(days=6)
                result = self.mistake_analyzer.analyze_period(self.trade_journal, start_date, trading_date)
                report_path = self.mistake_analyzer.generate_report(
                    result,
                    start_date=start_date,
                    end_date=trading_date,
                )
                learning_actions.append(f"weekly_mistake_analysis:{report_path}")
            except Exception as exc:
                alerts.append(f"Weekly mistake analysis failed: {exc}")
                manual_review = True

        if self._is_month_end(trading_date):
            try:
                bayes_result = self.bayesian_optimizer.run_optimization(
                    self.trade_journal,
                    {
                        "alpha": float(self.config.phase2.diffusion_alpha),
                        "J": float(self.config.phase2.diffusion_steps),
                        "sigma_scale": float(self.config.phase2.sigma_scale),
                        "zscore_threshold": float(self.config.phase2.signal_threshold),
                    },
                    trading_date,
                )
                learning_actions.append(f"monthly_bayesian_update:{bayes_result.output_path}")
            except Exception as exc:
                alerts.append(f"Monthly Bayesian update failed: {exc}")
                manual_review = True

        summary = self._build_daily_summary(
            trading_date=trading_date,
            current_positions=self.broker_client.get_positions(),
            allocations=allocation_weights,
            alerts=alerts,
            daily_pnl=post_daily_pnl,
            weekly_pnl=post_weekly_pnl,
            monthly_pnl=post_monthly_pnl,
            discrepancies=discrepancies,
            strategy_statuses=strategy_statuses,
            learning_actions=learning_actions,
            circuit_action=circuit_action,
            capital_scale_factor=capital_scale_factor,
            tracking_error_pct=tracking_error_pct,
        )
        result = DailyPipelineResult(
            date=trading_date,
            allocations=allocation_weights,
            approved_orders=approved_orders,
            rejected_orders=rejected_orders,
            fills=fills,
            discrepancies=discrepancies,
            alerts=alerts,
            daily_summary=summary,
            manual_review=manual_review or bool(discrepancies),
            aborted=False,
            learning_actions=learning_actions,
        )
        return self._finalize_result(result)

    def run_daily_summary(self) -> dict[str, Any]:
        if not self.state["daily_records"]:
            return {
                "message": "No daily pipeline history available yet.",
                "positions": self.state.get("last_positions", {}) or self.broker_client.get_positions(),
                "capital_scale_factor": 1.0 if self.mode != "live" else self._current_capital_scale_factor(self._resolve_trading_date(None)),
                "health_status": "UNKNOWN",
                "tracking_error_pct": 0.0,
            }
        latest = self.state["daily_records"][-1]
        date = pd.Timestamp(latest["date"])
        summary = self._build_daily_summary(
            trading_date=date,
            current_positions=self.state.get("last_positions", {}) or self.broker_client.get_positions(),
            allocations={
                "strategy_a": float(latest["allocation_strategy_a"]),
                "strategy_b": float(latest["allocation_strategy_b"]),
            },
            alerts=[],
            daily_pnl=float(latest["day_pnl"]),
            weekly_pnl=float(latest["week_pnl"]),
            monthly_pnl=float(latest["month_pnl"]),
            strategy_statuses={
                "strategy_a": str(latest.get("strategy_a_status", "ACTIVE")),
                "strategy_b": str(latest.get("strategy_b_status", "ACTIVE")),
            },
            circuit_action=str(latest.get("circuit_action", "CONTINUE")),
            learning_actions=[],
            capital_scale_factor=float(latest.get("capital_scale_factor", 1.0)),
            tracking_error_pct=float(latest.get("tracking_error_pct", 0.0)),
            health_status=str(latest.get("health_status", "UNKNOWN")),
        )
        return summary

    def _load_strategy_context(self) -> None:
        strategy_a = load_phase2_baseline_backtest(self.config)
        trend_prices = load_or_fetch_trend_price_history(self.config)
        strategy_b = backtest_trend_strategy(
            config=self.config,
            trend_prices=trend_prices,
            strategy_a_returns=strategy_a.daily_results.set_index("date")["net_portfolio_return"],
        )
        combined_results, _ = _run_combined_backtest(self.config, strategy_a, strategy_b)

        sector_prices = pd.read_csv(
            self.config.paths.processed_dir / "sector_etf_prices_validated.csv",
            parse_dates=["date"],
        )
        sector_prices = sector_prices.loc[
            sector_prices["is_valid"] & sector_prices["ticker"].isin(self.config.tickers),
            ["date", "ticker", "adj_close"],
        ].copy()
        signals = scale_signals_to_risk_budget(
            compute_graph_signals(sector_prices, self.config.tickers, self.config.phase2),
            self.config.phase2,
        ).scaled_signals
        self.strategy_a_signals = signals.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.strategy_a_signal_lookup = self.strategy_a_signals.set_index(["date", "ticker"]).sort_index()
        self.strategy_a_weights = (
            self.strategy_a_signals.pivot(index="date", columns="ticker", values="target_position")
            .sort_index()
            .reindex(columns=self.config.tickers)
            .fillna(0.0)
        )
        self.strategy_b_weights = (
            strategy_b.daily_positions.pivot(index="date", columns="ticker", values="target_weight")
            .sort_index()
            .reindex(columns=self.config.phase5.trend_tickers)
            .fillna(0.0)
        )
        self.strategy_a_returns = strategy_a.daily_results.set_index("date")["net_portfolio_return"].sort_index()
        self.strategy_b_returns = strategy_b.daily_results.set_index("date")["net_portfolio_return"].sort_index()
        self.strategy_a_costs = strategy_a.daily_results.set_index("date")["transaction_cost"].sort_index()
        self.strategy_b_costs = strategy_b.daily_results.set_index("date")["transaction_cost"].sort_index()
        self.combined_returns = combined_results.set_index("date")["net_portfolio_return"].sort_index()

        trend_price_matrix = (
            trend_prices.pivot(index="date", columns="ticker", values="adj_close")
            .sort_index()
            .reindex(columns=self.config.phase5.trend_tickers)
        )
        sector_price_matrix = (
            sector_prices.pivot(index="date", columns="ticker", values="adj_close")
            .sort_index()
            .reindex(columns=self.config.tickers)
        )
        self.price_matrix = sector_price_matrix.join(trend_price_matrix, how="outer").sort_index()
        self.spy_returns = trend_price_matrix["SPY"].pct_change().fillna(0.0)
        available_from_positions = set(self.strategy_a_weights.index) | set(self.strategy_b_weights.index)
        available_from_prices = set(self.price_matrix.dropna(how="all").index)
        self.available_dates = sorted(available_from_positions & available_from_prices)

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            state.setdefault("live_start_date", None)
            return state
        return {
            "last_rebalance_date": None,
            "allocations": {"strategy_a": 0.5, "strategy_b": 0.5},
            "open_trades": {},
            "daily_records": [],
            "last_positions": {},
            "live_start_date": None,
        }

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")

    def _resolve_trading_date(self, date: str | pd.Timestamp | None) -> pd.Timestamp:
        if not self.available_dates:
            raise ValueError("No overlapping Phase 5 execution dates are available.")
        if date is None:
            return pd.Timestamp(self.available_dates[-1])
        requested = pd.Timestamp(date)
        eligible = [value for value in self.available_dates if pd.Timestamp(value) <= requested]
        if not eligible:
            raise ValueError(f"No trading dates are available on or before {requested.date().isoformat()}.")
        return pd.Timestamp(eligible[-1])

    def _mark_broker_to_market(self, trading_date: pd.Timestamp) -> None:
        price_map = self._price_map_for_date(trading_date)
        if hasattr(self.broker_client, "set_market_prices"):
            self.broker_client.set_market_prices(price_map)

    def _compute_allocations(
        self,
        trading_date: pd.Timestamp,
    ) -> tuple[dict[str, float], dict[str, str], Any]:
        status_a = self.kill_switch.evaluate("A", self.strategy_a_returns, trading_date).status
        status_b = self.kill_switch.evaluate("B", self.strategy_b_returns, trading_date).status
        strategy_statuses = {
            "strategy_a": "ACTIVE" if status_a == "REACTIVATE" else status_a,
            "strategy_b": "ACTIVE" if status_b == "REACTIVATE" else status_b,
        }

        last_rebalance_date = (
            None if self.state.get("last_rebalance_date") is None else pd.Timestamp(self.state["last_rebalance_date"])
        )
        allocations = dict(self.state.get("allocations", {"strategy_a": 0.5, "strategy_b": 0.5}))
        rebalanced = False
        if self.allocator.should_rebalance(trading_date, last_rebalance_date):
            utilities = {
                "strategy_a": self.allocator.compute_utility(
                    "strategy_a",
                    trading_date,
                    self.strategy_a_returns,
                    self.strategy_a_costs,
                ),
                "strategy_b": self.allocator.compute_utility(
                    "strategy_b",
                    trading_date,
                    self.strategy_b_returns,
                    self.strategy_b_costs,
                ),
            }
            allocations = self.allocator.compute_allocations(
                trading_date,
                utilities,
                strategy_statuses=strategy_statuses,
            )
            self.state["pending_rebalance_date"] = trading_date.date().isoformat()
            rebalanced = True
        else:
            self.state["pending_rebalance_date"] = None

        portfolio_returns = self._portfolio_returns_from_state()
        corr_result = self.risk_manager.check_spy_correlation(portfolio_returns, self.spy_returns, trading_date)
        self.state["rebalanced_today"] = rebalanced
        return allocations, strategy_statuses, corr_result

    def _build_target_positions(
        self,
        *,
        trading_date: pd.Timestamp,
        portfolio_value: float,
        allocation_weights: dict[str, float],
        position_scale: float,
        capital_scale_factor: float,
    ) -> dict[str, dict[str, float]]:
        price_map = self._price_map_for_date(trading_date)
        target_positions: dict[str, dict[str, float]] = {}

        strategy_a_weights = (
            self.strategy_a_weights.loc[trading_date]
            if trading_date in self.strategy_a_weights.index
            else pd.Series(0.0, index=self.config.tickers, dtype=float)
        )
        strategy_b_weights = (
            self.strategy_b_weights.loc[trading_date]
            if trading_date in self.strategy_b_weights.index
            else pd.Series(0.0, index=self.config.phase5.trend_tickers, dtype=float)
        )
        combined_weights: dict[str, float] = {}
        for asset, weight in strategy_a_weights.items():
            combined_weights[str(asset)] = float(allocation_weights["strategy_a"] * float(weight) * position_scale)
        for asset, weight in strategy_b_weights.items():
            combined_weights[str(asset)] = float(allocation_weights["strategy_b"] * float(weight) * position_scale)

        for asset, weight in combined_weights.items():
            price = float(price_map.get(asset, 0.0))
            quantity = 0.0 if price <= 0.0 else (portfolio_value * weight) / price
            target_positions[asset] = {
                "quantity": float(quantity),
                "price": price,
                "market_value": float(quantity * price),
            }
        if self.mode != "live":
            return target_positions

        scaled_positions: dict[str, dict[str, float]] = {}
        for asset, payload in target_positions.items():
            scaled_quantity = float(payload["quantity"]) * capital_scale_factor
            scaled_positions[asset] = {
                **payload,
                "quantity": float(scaled_quantity),
                "market_value": float(scaled_quantity * float(payload["price"])),
            }
        return scaled_positions

    def _compute_mark_to_market_pnl(
        self,
        *,
        trading_date: pd.Timestamp,
        current_portfolio_value: float,
    ) -> tuple[float, float, float]:
        history = self.state["daily_records"]
        previous_value = float(history[-1]["portfolio_value"]) if history else current_portfolio_value
        daily_pnl = float(current_portfolio_value - previous_value)
        weekly_pnl = daily_pnl
        monthly_pnl = daily_pnl
        for record in history:
            record_date = pd.Timestamp(record["date"])
            if record_date.isocalendar()[:2] == trading_date.isocalendar()[:2]:
                weekly_pnl += float(record["day_pnl"])
            if (record_date.year, record_date.month) == (trading_date.year, trading_date.month):
                monthly_pnl += float(record["day_pnl"])
        return daily_pnl, weekly_pnl, monthly_pnl

    def _year_to_date_pnl(self, trading_date: pd.Timestamp, current_day_pnl: float) -> float:
        total = current_day_pnl
        for record in self.state["daily_records"]:
            record_date = pd.Timestamp(record["date"])
            if record_date.year == trading_date.year:
                total += float(record["day_pnl"])
        return float(total)

    def _update_open_trades(
        self,
        *,
        trading_date: pd.Timestamp,
        pre_trade_positions: dict[str, Any],
        post_trade_positions: dict[str, Any],
        fills: list[dict[str, Any]],
    ) -> None:
        price_map = self._price_map_for_date(trading_date)
        fill_lookup = {str(fill["asset"]): fill for fill in fills}
        previous_quantities = {
            asset: _position_quantity(payload)
            for asset, payload in pre_trade_positions.items()
        }
        new_quantities = {
            asset: _position_quantity(payload)
            for asset, payload in post_trade_positions.items()
        }
        assets = sorted(set(previous_quantities) | set(new_quantities) | set(self.state["open_trades"]))

        for asset in assets:
            previous_qty = float(previous_quantities.get(asset, 0.0))
            new_qty = float(new_quantities.get(asset, 0.0))
            previous_direction = _quantity_direction(previous_qty)
            new_direction = _quantity_direction(new_qty)
            open_trade = self.state["open_trades"].get(asset)

            if previous_direction != 0 and (new_direction == 0 or new_direction != previous_direction):
                if open_trade is not None:
                    self.trade_journal.log_trade(
                        self._close_trade_payload(
                            asset=asset,
                            open_trade=open_trade,
                            exit_date=trading_date,
                            exit_price=float(fill_lookup.get(asset, {}).get("fill_price", price_map.get(asset, 0.0))),
                            exit_reason=self._infer_exit_reason(asset, open_trade, trading_date, price_map),
                            exit_transaction_cost=self._fill_cost_ratio(fill_lookup.get(asset), open_trade["entry_price"]),
                        )
                    )
                    self.state["open_trades"].pop(asset, None)

            if new_direction != 0 and (previous_direction == 0 or new_direction != previous_direction):
                fill = fill_lookup.get(asset)
                entry_price = float(fill["fill_price"]) if fill is not None else float(price_map.get(asset, 0.0))
                self.state["open_trades"][asset] = self._open_trade_payload(
                    asset=asset,
                    direction=new_direction,
                    trading_date=trading_date,
                    entry_price=entry_price,
                    entry_transaction_cost=self._fill_cost_ratio(fill, entry_price),
                    quantity=new_qty,
                )

    def _open_trade_payload(
        self,
        *,
        asset: str,
        direction: int,
        trading_date: pd.Timestamp,
        entry_price: float,
        entry_transaction_cost: float,
        quantity: float,
    ) -> dict[str, Any]:
        strategy = "A" if asset in self.config.tickers else "B"
        signal_snapshot = self._signal_snapshot(trading_date, asset)
        return {
            "strategy": strategy,
            "direction": "long" if direction > 0 else "short",
            "entry_date": trading_date.date().isoformat(),
            "entry_price": float(entry_price),
            "entry_zscore": _optional_float(signal_snapshot.get("zscore")),
            "entry_node_corr": _optional_float(signal_snapshot.get("node_avg_corr")),
            "entry_regime": str(signal_snapshot.get("graph_regime", "TRADEABLE")),
            "predicted_residual": _optional_float(signal_snapshot.get("predicted_residual_mean")),
            "actual_residual": _optional_float(signal_snapshot.get("residual")),
            "portfolio_exposure_at_entry": abs(float(quantity) * float(entry_price)),
            "concurrent_positions": int(
                sum(1 for payload in self.broker_client.get_positions().values() if abs(_position_quantity(payload)) > 1e-9)
            ),
            "entry_transaction_cost": float(entry_transaction_cost),
        }

    def _close_trade_payload(
        self,
        *,
        asset: str,
        open_trade: dict[str, Any],
        exit_date: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
        exit_transaction_cost: float,
    ) -> dict[str, Any]:
        entry_date = pd.Timestamp(open_trade["entry_date"])
        direction = 1 if open_trade["direction"] == "long" else -1
        holding_days = max(len(pd.bdate_range(entry_date, exit_date)) - 1, 0)
        gross_return = direction * ((float(exit_price) / float(open_trade["entry_price"])) - 1.0)
        transaction_cost = float(open_trade["entry_transaction_cost"]) + float(exit_transaction_cost)
        actual_residual = _optional_float(self._signal_snapshot(exit_date, asset).get("residual"))
        predicted_residual = _optional_float(open_trade.get("predicted_residual"))
        return {
            "trade_id": f"phase7:{asset}:{entry_date.date()}:{exit_date.date()}:{open_trade['strategy']}",
            "strategy": open_trade["strategy"],
            "asset": asset,
            "direction": open_trade["direction"],
            "entry_date": entry_date,
            "exit_date": exit_date,
            "holding_days": holding_days,
            "entry_price": float(open_trade["entry_price"]),
            "exit_price": float(exit_price),
            "entry_zscore": _optional_float(open_trade.get("entry_zscore")),
            "entry_node_corr": _optional_float(open_trade.get("entry_node_corr")),
            "entry_regime": str(open_trade.get("entry_regime", "TRADEABLE")),
            "exit_reason": exit_reason,
            "gross_pnl": gross_return,
            "transaction_cost": transaction_cost,
            "net_pnl": gross_return - transaction_cost,
            "predicted_residual": predicted_residual,
            "actual_residual": actual_residual,
            "prediction_error": (
                None
                if predicted_residual is None or actual_residual is None
                else float(actual_residual - predicted_residual)
            ),
            "portfolio_exposure_at_entry": float(open_trade.get("portfolio_exposure_at_entry", 0.0)),
            "concurrent_positions": int(open_trade.get("concurrent_positions", 0)),
        }

    def _infer_exit_reason(
        self,
        asset: str,
        open_trade: dict[str, Any],
        trading_date: pd.Timestamp,
        price_map: dict[str, float],
    ) -> str:
        if open_trade["strategy"] == "B":
            return "signal_flip"

        snapshot = self._signal_snapshot(trading_date, asset)
        entry_date = pd.Timestamp(open_trade["entry_date"])
        holding_days = max(len(pd.bdate_range(entry_date, trading_date)) - 1, 0)
        entry_price = float(open_trade["entry_price"])
        exit_price = float(price_map.get(asset, entry_price))
        direction = 1 if open_trade["direction"] == "long" else -1
        gross_return = direction * ((exit_price / entry_price) - 1.0)
        zscore = snapshot.get("zscore")
        if zscore is not None and not pd.isna(zscore):
            if (direction == 1 and float(zscore) >= 0.0) or (direction == -1 and float(zscore) <= 0.0):
                return "reversion"
        if gross_return <= -self.config.phase2.stop_loss:
            return "stop_loss"
        if holding_days >= self.config.phase2.max_holding_days:
            return "max_hold"
        return "signal_flip"

    def _signal_snapshot(self, trading_date: pd.Timestamp, asset: str) -> dict[str, Any]:
        if (trading_date, asset) not in self.strategy_a_signal_lookup.index:
            return {}
        snapshot = self.strategy_a_signal_lookup.loc[(trading_date, asset)]
        if isinstance(snapshot, pd.DataFrame):
            snapshot = snapshot.iloc[-1]
        return snapshot.to_dict()

    def _fill_cost_ratio(self, fill: dict[str, Any] | None, entry_price: float) -> float:
        if fill is None or entry_price == 0.0:
            return 0.0
        quantity = float(fill["fill_quantity"])
        if quantity == 0.0:
            return 0.0
        expected_price = float(fill["expected_price"])
        fill_price = float(fill["fill_price"])
        cost = abs(fill_price - expected_price) * quantity
        notional = abs(entry_price * quantity)
        return 0.0 if notional == 0.0 else float(cost / notional)

    def _price_map_for_date(self, trading_date: pd.Timestamp) -> dict[str, float]:
        if trading_date not in self.price_matrix.index:
            raise ValueError(f"Price history missing for {trading_date.date().isoformat()}.")
        row = self.price_matrix.loc[trading_date].dropna()
        return {str(asset): float(price) for asset, price in row.items()}

    def _portfolio_returns_from_state(self) -> pd.Series:
        if not self.state["daily_records"]:
            return pd.Series(dtype=float)
        frame = pd.DataFrame(self.state["daily_records"])
        return pd.Series(
            frame["day_pnl"].astype(float).to_numpy() / frame["portfolio_value"].astype(float).shift(1).fillna(frame["portfolio_value"].astype(float)),
            index=pd.to_datetime(frame["date"]),
        ).fillna(0.0)

    def _build_daily_summary(
        self,
        *,
        trading_date: pd.Timestamp,
        current_positions: dict[str, Any],
        allocations: dict[str, float],
        alerts: list[str],
        daily_pnl: float,
        weekly_pnl: float,
        monthly_pnl: float,
        discrepancies: list[dict[str, Any]] | None = None,
        strategy_statuses: dict[str, str] | None = None,
        learning_actions: list[str] | None = None,
        circuit_action: str = "CONTINUE",
        capital_scale_factor: float = 1.0,
        tracking_error_pct: float = 0.0,
        health_status: str = "UNKNOWN",
    ) -> dict[str, Any]:
        portfolio_value = float(self.broker_client.get_portfolio_value())
        ytd_pnl = self._year_to_date_pnl(trading_date, daily_pnl)
        limits = self.config.phase7.risk_limits
        risk_headroom = {
            "daily_loss_pct_remaining": float(limits.max_daily_loss_pct - max(0.0, -daily_pnl / max(portfolio_value, 1e-9))),
            "weekly_loss_pct_remaining": float(limits.max_weekly_loss_pct - max(0.0, -weekly_pnl / max(portfolio_value, 1e-9))),
            "monthly_loss_pct_remaining": float(limits.max_monthly_loss_pct - max(0.0, -monthly_pnl / max(portfolio_value, 1e-9))),
        }
        active_a = self.strategy_a_signals.loc[self.strategy_a_signals["date"] == trading_date]
        regime_state = "TRADEABLE"
        if not active_a.empty and (active_a["graph_regime"] == "REDUCED").any():
            regime_state = "REDUCED"

        return {
            "date": trading_date.date().isoformat(),
            "positions": current_positions,
            "portfolio_value": portfolio_value,
            "day_pnl": daily_pnl,
            "week_pnl": weekly_pnl,
            "month_pnl": monthly_pnl,
            "ytd_pnl": ytd_pnl,
            "risk_limit_headroom": risk_headroom,
            "regime_state": regime_state,
            "allocation_weights": allocations,
            "capital_scale_factor": float(capital_scale_factor),
            "tracking_error_pct": float(tracking_error_pct),
            "health_status": health_status,
            "strategy_statuses": strategy_statuses or {"strategy_a": "ACTIVE", "strategy_b": "ACTIVE"},
            "alerts": alerts,
            "discrepancies": discrepancies or [],
            "learning_actions": learning_actions or [],
            "circuit_action": circuit_action,
        }

    def _live_start_date(self, trading_date: pd.Timestamp) -> pd.Timestamp:
        if self.state.get("live_start_date") is None:
            self.state["live_start_date"] = trading_date.date().isoformat()
        return pd.Timestamp(self.state["live_start_date"])

    def _current_capital_scale_factor(self, trading_date: pd.Timestamp) -> float:
        if self.mode != "live":
            return 1.0
        scaler = CapitalScaler(self.config, self._live_start_date(trading_date))
        return scaler.get_scale_factor(trading_date)

    def _tracking_error_pct(
        self,
        *,
        trading_date: pd.Timestamp,
        daily_pnl: float,
        end_portfolio_value: float,
    ) -> float:
        previous_value = max(end_portfolio_value - daily_pnl, 1e-9)
        actual_return = float(daily_pnl / previous_value)
        expected_return = float(self.combined_returns.get(trading_date, 0.0))
        return float(abs(actual_return - expected_return))

    def _finalize_result(self, result: DailyPipelineResult) -> DailyPipelineResult:
        health = self.monitor.daily_health_check(result)
        summary = dict(result.daily_summary)
        summary["health_status"] = health.status
        summary["tracking_error_pct"] = health.tracking_error_pct
        updated = replace(
            result,
            daily_summary=summary,
            health_status=health.status,
        )
        report_path = self.monitor.generate_daily_email(updated, health)
        severity = "INFO" if health.status == "HEALTHY" else "WARNING" if health.status == "DEGRADED" else "CRITICAL"
        self.monitor.send_alert(
            severity,
            f"Daily pipeline {health.status} for {result.date.date().isoformat()}",
            {
                "issues": health.issues,
                "tracking_error_pct": health.tracking_error_pct,
                "alerts": result.alerts,
            },
        )
        updated = replace(updated, report_path=str(report_path))
        summary = updated.daily_summary
        self.performance_tracker.record_daily(
            date=str(summary["date"]),
            portfolio_value=float(summary["portfolio_value"]),
            daily_pnl=float(summary["day_pnl"]),
            positions=dict(summary.get("positions", {})),
            allocation_weights=dict(summary.get("allocation_weights", {})),
            regime_state=str(summary.get("regime_state", "UNKNOWN")),
            risk_headroom=dict(summary.get("risk_limit_headroom", {})),
            capital_scale_factor=float(summary.get("capital_scale_factor", 1.0)),
            health_status=health.status,
            tracking_error_pct=health.tracking_error_pct,
            spy_return=float(self.spy_returns.get(result.date, 0.0)),
        )
        if self.state["daily_records"] and self.state["daily_records"][-1]["date"] == result.date.date().isoformat():
            self.state["daily_records"][-1]["health_status"] = health.status
            self.state["daily_records"][-1]["tracking_error_pct"] = health.tracking_error_pct
            self.state["daily_records"][-1]["report_path"] = str(report_path)
            self._save_state()
        return updated

    def _is_week_end(self, trading_date: pd.Timestamp) -> bool:
        return self._period_boundary(trading_date, "week")

    def _is_month_end(self, trading_date: pd.Timestamp) -> bool:
        return self._period_boundary(trading_date, "month")

    def _period_boundary(self, trading_date: pd.Timestamp, period: str) -> bool:
        dates = [pd.Timestamp(value) for value in self.available_dates]
        index = dates.index(trading_date)
        if index == len(dates) - 1:
            return True
        next_date = dates[index + 1]
        if period == "week":
            return trading_date.isocalendar()[:2] != next_date.isocalendar()[:2]
        return (trading_date.year, trading_date.month) != (next_date.year, next_date.month)

    def _rebalanced_on(self, trading_date: pd.Timestamp) -> bool:
        return self.state.get("pending_rebalance_date") == trading_date.date().isoformat()


def run_daily_pipeline(
    config_path: str | Path,
    *,
    date: str | pd.Timestamp | None = None,
    mode: str | None = None,
) -> DailyPipelineResult:
    pipeline = DailyPipeline(config_path, mode_override=mode)
    try:
        return pipeline.run_daily(date=date)
    finally:
        pipeline.close()


def run_daily_summary(config_path: str | Path, *, mode: str | None = None) -> dict[str, Any]:
    pipeline = DailyPipeline(config_path, mode_override=mode)
    try:
        return pipeline.run_daily_summary()
    finally:
        pipeline.close()


def _order_to_risk_payload(order: Any) -> dict[str, Any]:
    return {
        "asset": order.asset,
        "side": order.side,
        "quantity": order.quantity,
        "price": order.expected_price,
        "notional": order.notional,
    }


def _order_key(order: Any) -> tuple[str, str, float, float]:
    return (str(order.asset), str(order.side), float(order.quantity), float(order.expected_price))


def _order_key_from_payload(payload: dict[str, Any]) -> tuple[str, str, float, float]:
    return (
        str(payload["asset"]),
        str(payload["side"]),
        float(payload["quantity"]),
        float(payload["price"]),
    )


def _position_quantity(payload: Any) -> float:
    if payload is None:
        return 0.0
    if isinstance(payload, dict):
        if "quantity" in payload:
            return float(payload.get("quantity", 0.0))
        market_value = float(payload.get("market_value", 0.0))
        price = float(payload.get("price", payload.get("expected_price", 0.0)))
        return 0.0 if price == 0.0 else float(market_value / price)
    return float(payload)


def _expected_positions_after_orders(
    current_positions: dict[str, Any],
    approved_orders: list[Any],
) -> dict[str, dict[str, float]]:
    expected: dict[str, dict[str, float]] = {}
    for asset, payload in current_positions.items():
        quantity = _position_quantity(payload)
        price = float(payload.get("price", payload.get("expected_price", 0.0))) if isinstance(payload, dict) else 0.0
        expected[str(asset)] = {
            "quantity": float(quantity),
            "price": price,
        }
    for order in approved_orders:
        current = expected.get(order.asset, {"quantity": 0.0, "price": float(order.expected_price)})
        signed_quantity = float(order.quantity) if order.side == "buy" else -float(order.quantity)
        current["quantity"] = float(current["quantity"] + signed_quantity)
        current["price"] = float(order.expected_price)
        expected[order.asset] = current
    return expected


def _quantity_direction(quantity: float) -> int:
    if quantity > 1e-9:
        return 1
    if quantity < -1e-9:
        return -1
    return 0


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
