from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_walk_forward_backtest, scale_signals_to_risk_budget
from .config_loader import PipelineConfig, load_config
from .graph_engine import compute_graph_signals
from .ingestion import download_universe_data
from .logging_utils import setup_logger
from .storage import ensure_output_directories
from .validation import build_issue_report, build_quality_report, validate_prices


@dataclass(frozen=True)
class TrendStrategyBacktest:
    daily_positions: pd.DataFrame
    daily_results: pd.DataFrame
    trade_log: pd.DataFrame
    monthly_results: pd.DataFrame
    summary_metrics: dict[str, object]


@dataclass(frozen=True)
class TrendStrategyResult:
    summary_metrics: dict[str, object]
    output_paths: dict[str, Path]


def run_trend_strategy_pipeline(config_path: str | Path) -> TrendStrategyResult:
    config = load_config(config_path)
    trend_prices = load_or_fetch_trend_price_history(config)
    strategy_a = load_phase2_baseline_backtest(config)
    trend_result = backtest_trend_strategy(
        config=config,
        trend_prices=trend_prices,
        strategy_a_returns=strategy_a.daily_results.set_index("date")["net_portfolio_return"],
    )
    output_paths = _save_trend_outputs(config.paths.processed_dir, trend_result)
    return TrendStrategyResult(summary_metrics=trend_result.summary_metrics, output_paths=output_paths)


def load_or_fetch_trend_price_history(config: PipelineConfig) -> pd.DataFrame:
    ensure_output_directories(config.paths)
    validated_path = config.paths.processed_dir / "trend_universe_prices_validated.csv"
    if validated_path.exists():
        validated = pd.read_csv(validated_path, parse_dates=["date"])
        available = sorted(validated["ticker"].unique().tolist())
        if set(config.phase5.trend_tickers).issubset(available):
            valid_rows = validated.loc[
                validated["is_valid"] & validated["ticker"].isin(config.phase5.trend_tickers),
                ["date", "ticker", "adj_close", "volume"],
            ].copy()
            valid_rows["volume"] = valid_rows["volume"].fillna(0.0)
            return valid_rows.sort_values(["date", "ticker"]).reset_index(drop=True)

    logger = setup_logger(config.paths.pipeline_log_file, task="trend-data", phase="phase5")
    raw_prices = download_universe_data(
        tickers=config.phase5.trend_tickers,
        start_date=config.start_date,
        end_date=config.end_date,
        cache_dir=config.paths.cache_dir,
        logger=logger,
    )
    validated = validate_prices(raw_prices, config.validation)
    quality_report = build_quality_report(validated)
    issue_report = build_issue_report(validated)

    raw_path = config.paths.raw_dir / "trend_universe_prices_raw.csv"
    quality_path = config.paths.processed_dir / "trend_universe_quality_report.csv"
    issue_path = config.paths.processed_dir / "trend_universe_validation_issues.csv"
    raw_prices.to_csv(raw_path, index=False)
    validated.to_csv(validated_path, index=False)
    quality_report.to_csv(quality_path, index=False)
    issue_report.to_csv(issue_path, index=False)

    valid_rows = validated.loc[
        validated["is_valid"] & validated["ticker"].isin(config.phase5.trend_tickers),
        ["date", "ticker", "adj_close", "volume"],
    ].copy()
    valid_rows["volume"] = valid_rows["volume"].fillna(0.0)
    return valid_rows.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_phase2_baseline_backtest(config: PipelineConfig) -> TrendStrategyBacktest:
    validated_path = config.paths.processed_dir / "sector_etf_prices_validated.csv"
    if not validated_path.exists():
        raise ValueError(f"Validated Phase 2 price history was not found at '{validated_path}'.")
    price_history = pd.read_csv(validated_path, parse_dates=["date"])
    price_history = price_history.loc[
        price_history["is_valid"] & price_history["ticker"].isin(config.tickers),
        ["date", "ticker", "adj_close"],
    ].copy()
    daily_signals = compute_graph_signals(price_history, config.tickers, config.phase2)
    scaled_signals = scale_signals_to_risk_budget(daily_signals, config.phase2).scaled_signals
    backtest = run_walk_forward_backtest(scaled_signals, config.phase2, run_id="phase5-strategy-a")
    return TrendStrategyBacktest(
        daily_positions=scaled_signals.loc[:, ["date", "ticker", "target_position"]].copy(),
        daily_results=backtest.daily_results.copy(),
        trade_log=backtest.trade_log.copy(),
        monthly_results=backtest.monthly_results.copy(),
        summary_metrics=dict(backtest.summary_metrics),
    )


def backtest_trend_strategy(
    *,
    config: PipelineConfig,
    trend_prices: pd.DataFrame,
    strategy_a_returns: pd.Series | None = None,
) -> TrendStrategyBacktest:
    tickers = list(config.phase5.trend_tickers)
    price_matrix = (
        trend_prices.pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
        .reindex(columns=tickers)
        .dropna(how="any")
    )
    if len(price_matrix) < config.phase5.trend_long_ma + 2:
        raise ValueError("Not enough trend price history to build the moving-average strategy.")

    target_weights, signal_frame, forward_returns = compute_trend_target_weights(
        price_matrix=price_matrix,
        short_window=config.phase5.trend_short_ma,
        long_window=config.phase5.trend_long_ma,
        vol_lookback=config.phase5.trend_vol_lookback,
    )
    cash_return_daily = config.phase5.cash_return_annual / config.phase2.annualization_days
    daily_results, trade_log = _backtest_weight_matrix(
        target_weights=target_weights,
        forward_returns=forward_returns,
        trading_cost_bps=config.phase5.trend_cost_bps,
        cash_return_daily=cash_return_daily,
    )
    monthly_results = build_generic_monthly_results(daily_results, config.phase2.min_training_months)
    summary_metrics = build_generic_summary_metrics(
        daily_results=daily_results,
        trade_log=trade_log,
        monthly_results=monthly_results,
        annualization_days=config.phase2.annualization_days,
    )

    dated_portfolio_returns = daily_results.set_index("date")["net_portfolio_return"]
    aligned_spy = forward_returns["SPY"].reindex(dated_portfolio_returns.index).fillna(0.0)
    summary_metrics["correlation_with_spy"] = float(
        dated_portfolio_returns.corr(aligned_spy)
    ) if not daily_results.empty else 0.0
    if strategy_a_returns is not None and not strategy_a_returns.empty:
        aligned_a = strategy_a_returns.reindex(dated_portfolio_returns.index).fillna(0.0)
        correlation_with_a = dated_portfolio_returns.corr(aligned_a)
        summary_metrics["correlation_with_strategy_a"] = 0.0 if pd.isna(correlation_with_a) else float(correlation_with_a)
    else:
        summary_metrics["correlation_with_strategy_a"] = 0.0

    position_rows: list[dict[str, object]] = []
    for date in target_weights.index:
        for ticker in tickers:
            position_rows.append(
                {
                    "date": pd.Timestamp(date),
                    "ticker": ticker,
                    "target_weight": float(target_weights.at[date, ticker]),
                    "signal": bool(signal_frame.at[date, ticker]),
                }
            )
    daily_positions = pd.DataFrame(position_rows)
    return TrendStrategyBacktest(
        daily_positions=daily_positions,
        daily_results=daily_results,
        trade_log=trade_log,
        monthly_results=monthly_results,
        summary_metrics=summary_metrics,
    )


def compute_trend_target_weights(
    *,
    price_matrix: pd.DataFrame,
    short_window: int,
    long_window: int,
    vol_lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_returns = price_matrix.pct_change()
    forward_returns = price_matrix.pct_change().shift(-1)
    short_ma = price_matrix.rolling(short_window, min_periods=short_window).mean()
    long_ma = price_matrix.rolling(long_window, min_periods=long_window).mean()
    long_signal = (short_ma > long_ma).fillna(False)
    vol_20d = daily_returns.rolling(vol_lookback, min_periods=vol_lookback).std(ddof=0)
    inv_vol = (1.0 / vol_20d.replace(0.0, np.nan)).where(long_signal)
    target_weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0.0)
    target_weights = target_weights.reindex(index=forward_returns.dropna(how="all").index).fillna(0.0)
    signal_frame = long_signal.reindex(index=target_weights.index).fillna(False)
    forward_returns = forward_returns.reindex(index=target_weights.index).fillna(0.0)
    return target_weights, signal_frame, forward_returns


def build_generic_monthly_results(
    daily_results: pd.DataFrame,
    min_training_months: int,
) -> pd.DataFrame:
    if daily_results.empty:
        return pd.DataFrame(columns=["test_month", "training_end_date", "monthly_return", "profitable", "active_month"])

    monthly_returns = (
        daily_results.assign(test_month=daily_results["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("test_month")["net_portfolio_return"]
        .apply(lambda values: float((1.0 + values).prod() - 1.0))
        .rename("monthly_return")
        .reset_index()
    )
    monthly_activity = (
        daily_results.assign(test_month=daily_results["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("test_month")
        .agg(monthly_turnover=("turnover", "sum"), gross_exposure_sum=("gross_exposure", "sum"))
        .reset_index()
    )
    monthly = monthly_returns.merge(monthly_activity, on="test_month", how="left")

    rows: list[dict[str, object]] = []
    for month_idx in range(min_training_months, len(monthly)):
        month = pd.Timestamp(monthly.loc[month_idx, "test_month"])
        monthly_return = float(monthly.loc[month_idx, "monthly_return"])
        active_month = (
            float(monthly.loc[month_idx, "monthly_turnover"]) > 0
            or float(monthly.loc[month_idx, "gross_exposure_sum"]) > 0
        )
        rows.append(
            {
                "test_month": month,
                "training_end_date": month - pd.Timedelta(days=1),
                "monthly_return": monthly_return,
                "profitable": active_month and monthly_return > 0,
                "active_month": active_month,
            }
        )
    return pd.DataFrame(rows)


def build_generic_summary_metrics(
    *,
    daily_results: pd.DataFrame,
    trade_log: pd.DataFrame,
    monthly_results: pd.DataFrame,
    annualization_days: int,
) -> dict[str, object]:
    if daily_results.empty:
        raise ValueError("Daily results were empty; cannot build summary metrics.")

    net_returns = daily_results["net_portfolio_return"].fillna(0.0)
    gross_returns = daily_results["gross_portfolio_return"].fillna(0.0)
    equity_curve = (1.0 + net_returns).cumprod()
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    avg_return = float(net_returns.mean())
    std_return = float(net_returns.std(ddof=0))
    sharpe = 0.0 if std_return == 0.0 else float((avg_return / std_return) * np.sqrt(annualization_days))
    annualized_return = float(equity_curve.iloc[-1] ** (annualization_days / len(equity_curve)) - 1.0)
    trade_returns = trade_log["net_return"].astype(float) if not trade_log.empty else pd.Series(dtype=float)
    gross_turnover = float(daily_results["turnover"].sum() * annualization_days / max(len(daily_results), 1))
    active_months = int(monthly_results["active_month"].sum()) if not monthly_results.empty else 0
    profitable_fraction = (
        float(monthly_results.loc[monthly_results["active_month"], "profitable"].mean())
        if active_months
        else 0.0
    )
    profit_factor = 0.0
    if not trade_returns.empty:
        gross_profit = float(trade_returns.loc[trade_returns > 0].sum())
        gross_loss = float(abs(trade_returns.loc[trade_returns < 0].sum()))
        profit_factor = 0.0 if gross_loss == 0 else gross_profit / gross_loss

    return {
        "total_trades": int(len(trade_log)),
        "sharpe_ratio": sharpe,
        "annualized_return": annualized_return,
        "max_drawdown": float(abs(drawdown.min())) if not drawdown.empty else 0.0,
        "win_rate": float((trade_returns > 0).mean()) if not trade_returns.empty else 0.0,
        "profit_factor": profit_factor,
        "avg_holding_days": float(trade_log["holding_days"].mean()) if not trade_log.empty else 0.0,
        "annual_turnover": gross_turnover,
        "profitable_month_fraction": profitable_fraction,
        "out_of_sample_months": int(len(monthly_results)),
        "active_out_of_sample_months": active_months,
        "gross_return_mean": float(gross_returns.mean()),
    }


def _backtest_weight_matrix(
    *,
    target_weights: pd.DataFrame,
    forward_returns: pd.DataFrame,
    trading_cost_bps: float,
    cash_return_daily: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    previous_weights = pd.Series(0.0, index=target_weights.columns, dtype=float)
    states = {ticker: _new_position_state() for ticker in target_weights.columns}
    daily_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    cost_rate = trading_cost_bps / 10000.0

    for date in target_weights.index:
        weights = target_weights.loc[date].fillna(0.0).astype(float)
        turnover = float((weights - previous_weights).abs().sum())
        gross_exposure = float(weights.sum())
        cash_weight = max(0.0, 1.0 - gross_exposure)
        gross_return = float((weights * forward_returns.loc[date]).sum()) + cash_weight * cash_return_daily
        transaction_cost = turnover * cost_rate
        net_return = gross_return - transaction_cost

        for ticker in target_weights.columns:
            state = states[ticker]
            desired_weight = float(weights[ticker])
            if state["is_open"] and desired_weight == 0.0:
                trade_rows.append(_close_trade_row(ticker, date, state, cost_rate))
                states[ticker] = _new_position_state()
            if not state["is_open"] and desired_weight > 0.0:
                states[ticker] = {
                    "is_open": True,
                    "entry_date": date,
                    "entry_weight": desired_weight,
                    "current_weight": desired_weight,
                    "gross_return": 0.0,
                    "holding_days": 0,
                }
            elif state["is_open"]:
                state["current_weight"] = desired_weight

        for ticker in target_weights.columns:
            state = states[ticker]
            if state["is_open"]:
                ticker_return = float(forward_returns.at[date, ticker])
                state["gross_return"] = (1.0 + state["gross_return"]) * (1.0 + ticker_return) - 1.0
                state["holding_days"] += 1

        daily_rows.append(
            {
                "date": pd.Timestamp(date),
                "gross_portfolio_return": gross_return,
                "net_portfolio_return": net_return,
                "portfolio_return": net_return,
                "gross_exposure": gross_exposure,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
            }
        )
        previous_weights = weights.copy()

    if len(target_weights.index) > 0:
        final_date = pd.Timestamp(target_weights.index[-1])
        for ticker, state in states.items():
            if state["is_open"]:
                trade_rows.append(_close_trade_row(ticker, final_date, state, cost_rate))

    return pd.DataFrame(daily_rows), pd.DataFrame(trade_rows)


def _close_trade_row(
    ticker: str,
    exit_date: pd.Timestamp,
    state: dict[str, object],
    cost_rate: float,
) -> dict[str, object]:
    gross_return = float(state["gross_return"])
    net_return = gross_return - (2.0 * cost_rate)
    return {
        "trade_id": f"trend:{ticker}:{pd.Timestamp(state['entry_date']).date()}:{pd.Timestamp(exit_date).date()}",
        "ticker": ticker,
        "entry_date": state["entry_date"],
        "exit_date": exit_date,
        "position_direction": 1,
        "entry_zscore": np.nan,
        "exit_zscore": np.nan,
        "holding_days": int(state["holding_days"]),
        "entry_weight": float(state["entry_weight"]),
        "gross_return": gross_return,
        "net_return": net_return,
    }


def _new_position_state() -> dict[str, object]:
    return {
        "is_open": False,
        "entry_date": None,
        "entry_weight": 0.0,
        "current_weight": 0.0,
        "gross_return": 0.0,
        "holding_days": 0,
    }


def _save_trend_outputs(processed_dir: Path, result: TrendStrategyBacktest) -> dict[str, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    positions_path = processed_dir / "trend_strategy_daily_positions.csv"
    daily_path = processed_dir / "trend_strategy_daily_results.csv"
    trades_path = processed_dir / "trend_strategy_trade_log.csv"
    monthly_path = processed_dir / "trend_strategy_monthly_results.csv"
    summary_path = processed_dir / "trend_strategy_summary.json"

    result.daily_positions.to_csv(positions_path, index=False)
    result.daily_results.to_csv(daily_path, index=False)
    result.trade_log.to_csv(trades_path, index=False)
    result.monthly_results.to_csv(monthly_path, index=False)
    summary_path.write_text(json.dumps(result.summary_metrics, indent=2, default=str), encoding="utf-8")

    return {
        "positions": positions_path,
        "daily_results": daily_path,
        "trades": trades_path,
        "monthly_results": monthly_path,
        "summary": summary_path,
    }
