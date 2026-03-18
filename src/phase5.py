from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .config_loader import load_config
from .portfolio_allocator import DynamicAllocator
from .trade_journal import TradeJournal
from .trend_strategy import (
    TrendStrategyBacktest,
    backtest_trend_strategy,
    build_generic_monthly_results,
    build_generic_summary_metrics,
    load_or_fetch_trend_price_history,
    load_phase2_baseline_backtest,
)


@dataclass(frozen=True)
class Phase5Result:
    run_id: str
    summary_metrics: dict[str, object]
    output_paths: dict[str, Path]


@dataclass(frozen=True)
class Phase5VerificationResult:
    gate_passed: bool
    summary_metrics: dict[str, object]
    output_path: Path


def run_phase5_pipeline(config_path: str | Path) -> Phase5Result:
    config = load_config(config_path)
    journal = TradeJournal(config.paths.project_root / "data/trade_journal.db")
    try:
        strategy_a = load_phase2_baseline_backtest(config, trade_journal=journal)
        trend_prices = load_or_fetch_trend_price_history(config)
        strategy_b = backtest_trend_strategy(
            config=config,
            trend_prices=trend_prices,
            strategy_a_returns=strategy_a.daily_results.set_index("date")["net_portfolio_return"],
            trade_journal=journal,
        )
    finally:
        journal.close()

    combined_daily_results, allocation_history = _run_combined_backtest(config, strategy_a, strategy_b)
    trade_log = _combine_trade_logs(strategy_a.trade_log, strategy_b.trade_log)
    monthly_results = build_generic_monthly_results(combined_daily_results, config.phase2.min_training_months)
    summary_metrics = build_generic_summary_metrics(
        daily_results=combined_daily_results,
        trade_log=trade_log,
        monthly_results=monthly_results,
        annualization_days=config.phase2.annualization_days,
    )

    strategy_a_returns = strategy_a.daily_results.set_index("date")["net_portfolio_return"]
    strategy_b_returns = strategy_b.daily_results.set_index("date")["net_portfolio_return"]
    aligned_returns = pd.concat(
        {"strategy_a": strategy_a_returns, "strategy_b": strategy_b_returns},
        axis=1,
        join="inner",
    ).fillna(0.0)
    strategy_correlation = aligned_returns["strategy_a"].corr(aligned_returns["strategy_b"])
    allocation_responsiveness, corr_a, corr_b = _allocator_responsiveness(
        allocation_history=allocation_history,
        strategy_a_returns=strategy_a_returns,
        strategy_b_returns=strategy_b_returns,
        lookback=config.phase5.performance_lookback,
    )

    summary_metrics.update(
        {
            "run_id": str(uuid4()),
            "strategy_a_sharpe": strategy_a.summary_metrics["sharpe_ratio"],
            "strategy_b_sharpe": strategy_b.summary_metrics["sharpe_ratio"],
            "strategy_a_b_correlation": 0.0 if pd.isna(strategy_correlation) else float(strategy_correlation),
            "allocator_responsiveness": allocation_responsiveness,
            "allocator_corr_strategy_a": corr_a,
            "allocator_corr_strategy_b": corr_b,
            "average_allocation_strategy_a": float(allocation_history["allocation_strategy_a"].mean()),
            "average_allocation_strategy_b": float(allocation_history["allocation_strategy_b"].mean()),
            "strategy_b_correlation_with_spy": strategy_b.summary_metrics["correlation_with_spy"],
            "strategy_b_correlation_with_strategy_a": strategy_b.summary_metrics["correlation_with_strategy_a"],
            "profitable_months": int(monthly_results["profitable"].sum()) if not monthly_results.empty else 0,
        }
    )
    gate_passed = (
        summary_metrics["sharpe_ratio"] > 0.85
        and summary_metrics["max_drawdown"] < 0.08
        and summary_metrics["allocator_responsiveness"] > 0.2
    )
    summary_metrics["gate_passed"] = bool(gate_passed)

    run_id = str(summary_metrics["run_id"])
    output_paths = _save_phase5_outputs(
        processed_dir=config.paths.processed_dir,
        run_id=run_id,
        daily_results=combined_daily_results,
        trade_log=trade_log,
        monthly_results=monthly_results,
        allocation_history=allocation_history,
        strategy_a_daily=strategy_a.daily_results,
        strategy_b_daily=strategy_b.daily_results,
        summary_metrics=summary_metrics,
    )
    return Phase5Result(run_id=run_id, summary_metrics=summary_metrics, output_paths=output_paths)


def verify_phase5_gate(config_path: str | Path) -> Phase5VerificationResult:
    config = load_config(config_path)
    summary_path = config.paths.processed_dir / "phase5_summary.json"
    if not summary_path.exists():
        raise ValueError("Phase 5 summary was not found. Run `python3 -m src.main run-phase5` first.")
    summary_metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    gate_passed = (
        float(summary_metrics["sharpe_ratio"]) > 0.85
        and float(summary_metrics["max_drawdown"]) < 0.08
        and float(summary_metrics["allocator_responsiveness"]) > 0.2
    )
    summary_metrics["gate_passed"] = gate_passed
    summary_path.write_text(json.dumps(summary_metrics, indent=2, default=str), encoding="utf-8")
    return Phase5VerificationResult(gate_passed=gate_passed, summary_metrics=summary_metrics, output_path=summary_path)


def _run_combined_backtest(
    config,
    strategy_a: TrendStrategyBacktest,
    strategy_b: TrendStrategyBacktest,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allocator = DynamicAllocator(config)
    strategy_a_daily = strategy_a.daily_results.set_index("date").sort_index()
    strategy_b_daily = strategy_b.daily_results.set_index("date").sort_index()
    aligned = strategy_a_daily.join(strategy_b_daily, how="inner", lsuffix="_a", rsuffix="_b")
    if aligned.empty:
        raise ValueError("Strategy A and Strategy B did not have overlapping daily history.")

    last_rebalance_date: pd.Timestamp | None = None
    current_weights = {"strategy_a": 0.5, "strategy_b": 0.5}
    daily_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []

    strategy_a_returns = strategy_a_daily["net_portfolio_return"]
    strategy_b_returns = strategy_b_daily["net_portfolio_return"]
    strategy_a_costs = strategy_a_daily["transaction_cost"]
    strategy_b_costs = strategy_b_daily["transaction_cost"]

    for date in aligned.index:
        rebalanced = False
        if allocator.should_rebalance(date, last_rebalance_date):
            utilities = {
                "strategy_a": allocator.compute_utility("strategy_a", date, strategy_a_returns, strategy_a_costs),
                "strategy_b": allocator.compute_utility("strategy_b", date, strategy_b_returns, strategy_b_costs),
            }
            if strategy_a_returns.loc[strategy_a_returns.index < date].tail(config.phase5.performance_lookback).empty:
                current_weights = {"strategy_a": 0.5, "strategy_b": 0.5}
            else:
                current_weights = allocator.compute_allocations(date, utilities)
            last_rebalance_date = pd.Timestamp(date)
            rebalanced = True

        gross_return = (
            current_weights["strategy_a"] * float(aligned.at[date, "gross_portfolio_return_a"])
            + current_weights["strategy_b"] * float(aligned.at[date, "gross_portfolio_return_b"])
        )
        net_return = (
            current_weights["strategy_a"] * float(aligned.at[date, "net_portfolio_return_a"])
            + current_weights["strategy_b"] * float(aligned.at[date, "net_portfolio_return_b"])
        )
        if net_return < -config.phase5.daily_loss_limit:
            scale = config.phase5.daily_loss_limit / abs(net_return)
            gross_return *= scale
            net_return = -config.phase5.daily_loss_limit
        turnover = (
            current_weights["strategy_a"] * float(aligned.at[date, "turnover_a"])
            + current_weights["strategy_b"] * float(aligned.at[date, "turnover_b"])
        )
        transaction_cost = (
            current_weights["strategy_a"] * float(aligned.at[date, "transaction_cost_a"])
            + current_weights["strategy_b"] * float(aligned.at[date, "transaction_cost_b"])
        )
        gross_exposure = (
            current_weights["strategy_a"] * float(aligned.at[date, "gross_exposure_a"])
            + current_weights["strategy_b"] * float(aligned.at[date, "gross_exposure_b"])
        )
        daily_rows.append(
            {
                "date": pd.Timestamp(date),
                "gross_portfolio_return": gross_return,
                "net_portfolio_return": net_return,
                "portfolio_return": net_return,
                "gross_exposure": gross_exposure,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "strategy_a_contribution": current_weights["strategy_a"] * float(aligned.at[date, "net_portfolio_return_a"]),
                "strategy_b_contribution": current_weights["strategy_b"] * float(aligned.at[date, "net_portfolio_return_b"]),
                "allocation_strategy_a": current_weights["strategy_a"],
                "allocation_strategy_b": current_weights["strategy_b"],
                "rebalanced": rebalanced,
            }
        )
        allocation_rows.append(
            {
                "date": pd.Timestamp(date),
                "allocation_strategy_a": current_weights["strategy_a"],
                "allocation_strategy_b": current_weights["strategy_b"],
                "rebalanced": rebalanced,
            }
        )

    return pd.DataFrame(daily_rows), pd.DataFrame(allocation_rows)


def _combine_trade_logs(strategy_a_trades: pd.DataFrame, strategy_b_trades: pd.DataFrame) -> pd.DataFrame:
    a_trades = strategy_a_trades.copy()
    if not a_trades.empty:
        a_trades["strategy_id"] = "strategy_a"
    b_trades = strategy_b_trades.copy()
    if not b_trades.empty:
        b_trades["strategy_id"] = "strategy_b"
    combined = pd.concat([a_trades, b_trades], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=["trade_id", "strategy_id", "entry_date", "exit_date", "holding_days", "net_return"])
    return combined.sort_values(["entry_date", "strategy_id", "ticker"]).reset_index(drop=True)


def _allocator_responsiveness(
    *,
    allocation_history: pd.DataFrame,
    strategy_a_returns: pd.Series,
    strategy_b_returns: pd.Series,
    lookback: int,
) -> tuple[float, float, float]:
    weights = allocation_history.set_index("date").sort_index()
    rolling_a = _rolling_sharpe(strategy_a_returns.reindex(weights.index).fillna(0.0), lookback)
    rolling_b = _rolling_sharpe(strategy_b_returns.reindex(weights.index).fillna(0.0), lookback)
    performance_spread = rolling_b - rolling_a
    corr_a = weights["allocation_strategy_a"].corr(-performance_spread)
    corr_b = weights["allocation_strategy_b"].corr(performance_spread)
    corr_a_value = 0.0 if pd.isna(corr_a) else float(corr_a)
    corr_b_value = 0.0 if pd.isna(corr_b) else float(corr_b)
    responsiveness = float(np.mean([corr_a_value, corr_b_value]))
    return responsiveness, corr_a_value, corr_b_value


def _rolling_sharpe(returns: pd.Series, lookback: int) -> pd.Series:
    rolling_mean = returns.rolling(lookback, min_periods=lookback).mean()
    rolling_std = returns.rolling(lookback, min_periods=lookback).std(ddof=0).replace(0.0, np.nan)
    return (rolling_mean / rolling_std).fillna(0.0)


def _save_phase5_outputs(
    *,
    processed_dir: Path,
    run_id: str,
    daily_results: pd.DataFrame,
    trade_log: pd.DataFrame,
    monthly_results: pd.DataFrame,
    allocation_history: pd.DataFrame,
    strategy_a_daily: pd.DataFrame,
    strategy_b_daily: pd.DataFrame,
    summary_metrics: dict[str, object],
) -> dict[str, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    daily_path = processed_dir / "phase5_daily_results.csv"
    trades_path = processed_dir / "phase5_trade_log.csv"
    monthly_path = processed_dir / "phase5_monthly_results.csv"
    allocation_path = processed_dir / "phase5_allocation_history.csv"
    strategy_a_path = processed_dir / "phase5_strategy_a_daily.csv"
    strategy_b_path = processed_dir / "phase5_strategy_b_daily.csv"
    summary_path = processed_dir / "phase5_summary.json"

    daily_results.to_csv(daily_path, index=False)
    trade_log.to_csv(trades_path, index=False)
    monthly_results.to_csv(monthly_path, index=False)
    allocation_history.to_csv(allocation_path, index=False)
    strategy_a_daily.to_csv(strategy_a_path, index=False)
    strategy_b_daily.to_csv(strategy_b_path, index=False)
    summary_path.write_text(json.dumps({"run_id": run_id, **summary_metrics}, indent=2, default=str), encoding="utf-8")

    return {
        "daily_results": daily_path,
        "trades": trades_path,
        "monthly_results": monthly_path,
        "allocations": allocation_path,
        "strategy_a_daily": strategy_a_path,
        "strategy_b_daily": strategy_b_path,
        "summary": summary_path,
    }
