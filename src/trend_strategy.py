from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import apply_phase2_geo_overlay, run_walk_forward_backtest, scale_signals_to_risk_budget
from .config_loader import PipelineConfig, load_config
from .geo.storage import GeoStore
from .graph_engine import compute_graph_signals
from .logging_utils import setup_logger
from .storage import load_validated_price_data
from .trade_journal import TradeJournal


@dataclass(frozen=True)
class TrendStrategyBacktest:
    daily_positions: pd.DataFrame
    daily_results: pd.DataFrame
    trade_log: pd.DataFrame
    monthly_results: pd.DataFrame
    summary_metrics: dict[str, object]
    daily_signals: pd.DataFrame | None = None


@dataclass(frozen=True)
class TrendStrategyResult:
    summary_metrics: dict[str, object]
    output_paths: dict[str, Path]


def run_trend_strategy_pipeline(config_path: str | Path) -> TrendStrategyResult:
    config = load_config(config_path)
    trend_prices = load_or_fetch_trend_price_history(config)
    strategy_a = load_phase2_baseline_backtest(config)
    journal = TradeJournal(config.paths.project_root / "data/trade_journal.db")
    try:
        trend_result = backtest_trend_strategy(
            config=config,
            trend_prices=trend_prices,
            strategy_a_returns=strategy_a.daily_results.set_index("date")["net_portfolio_return"],
            trade_journal=journal,
        )
    finally:
        journal.close()
    output_paths = _save_trend_outputs(config.paths.processed_dir, trend_result)
    return TrendStrategyResult(summary_metrics=trend_result.summary_metrics, output_paths=output_paths)


def load_or_fetch_trend_price_history(config: PipelineConfig) -> pd.DataFrame:
    validated = load_validated_price_data(config, dataset="trend")
    valid_rows = validated.loc[
        validated["is_valid"] & validated["ticker"].isin(config.phase5.trend_tickers),
        ["date", "ticker", "adj_close", "volume"],
    ].copy()
    valid_rows["volume"] = valid_rows["volume"].fillna(0.0)
    return valid_rows.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_phase2_baseline_backtest(
    config: PipelineConfig,
    trade_journal: TradeJournal | None = None,
    geo_store: GeoStore | None = None,
    geo_snapshot: pd.DataFrame | None = None,
) -> TrendStrategyBacktest:
    price_history = load_validated_price_data(config, dataset="sector")
    price_history = price_history.loc[
        price_history["is_valid"] & price_history["ticker"].isin(config.tickers),
        ["date", "ticker", "adj_close"],
    ].copy()
    scaled_signals = build_strategy_a_signal_history(
        config=config,
        price_history=price_history,
        geo_store=geo_store,
        geo_snapshot=geo_snapshot,
    )
    backtest = run_walk_forward_backtest(
        scaled_signals,
        config.phase2,
        run_id="phase5-strategy-a",
        trade_journal=trade_journal,
        strategy_label="A",
    )
    return TrendStrategyBacktest(
        daily_positions=scaled_signals.loc[:, ["date", "ticker", "target_position"]].copy(),
        daily_results=backtest.daily_results.copy(),
        trade_log=backtest.trade_log.copy(),
        monthly_results=backtest.monthly_results.copy(),
        summary_metrics=dict(backtest.summary_metrics),
        daily_signals=scaled_signals.copy(),
    )


def build_strategy_a_signal_history(
    *,
    config: PipelineConfig,
    price_history: pd.DataFrame,
    geo_store: GeoStore | None = None,
    geo_snapshot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    daily_signals = compute_graph_signals(price_history, config.tickers, config.phase2)
    active_geo_snapshot = geo_snapshot
    if active_geo_snapshot is None and config.geo.enabled:
        active_geo_snapshot = _load_geo_snapshot_for_signal_window(
            config=config,
            signal_dates=daily_signals["date"],
            geo_store=geo_store,
        )
    overlaid_signals = apply_phase2_geo_overlay(
        daily_signals,
        config.phase2,
        config.geo,
        geo_snapshot=active_geo_snapshot,
    )
    return scale_signals_to_risk_budget(overlaid_signals, config.phase2).scaled_signals


def backtest_trend_strategy(
    *,
    config: PipelineConfig,
    trend_prices: pd.DataFrame,
    strategy_a_returns: pd.Series | None = None,
    trade_journal: TradeJournal | None = None,
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
        price_matrix=price_matrix.reindex(index=target_weights.index),
        forward_returns=forward_returns,
        trading_cost_bps=config.phase5.trend_cost_bps,
        cash_return_daily=cash_return_daily,
        trade_journal=trade_journal,
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
        daily_signals=None,
    )


def _load_geo_snapshot_for_signal_window(
    *,
    config: PipelineConfig,
    signal_dates: pd.Series,
    geo_store: GeoStore | None,
) -> pd.DataFrame:
    if signal_dates.empty or not config.geo.enabled:
        return pd.DataFrame()

    store = geo_store or GeoStore(config.database, config.paths, setup_logger(config.paths.pipeline_log_file, task="geo-history", phase="phase2"))
    start_date = pd.to_datetime(signal_dates.min()).date()
    end_date = pd.to_datetime(signal_dates.max()).date()
    return store.fetch_feature_snapshot_range(start_date=start_date, end_date=end_date)


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
    price_matrix: pd.DataFrame,
    forward_returns: pd.DataFrame,
    trading_cost_bps: float,
    cash_return_daily: float,
    trade_journal: TradeJournal | None = None,
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
        concurrent_positions = int((weights > 0.0).sum())
        cash_weight = max(0.0, 1.0 - gross_exposure)
        gross_return = float((weights * forward_returns.loc[date]).sum()) + cash_weight * cash_return_daily
        transaction_cost = turnover * cost_rate
        net_return = gross_return - transaction_cost

        for ticker in target_weights.columns:
            state = states[ticker]
            desired_weight = float(weights[ticker])
            if state["is_open"] and desired_weight == 0.0:
                trade_row = _close_trade_row(
                    ticker=ticker,
                    exit_date=date,
                    state=state,
                    exit_price=float(price_matrix.at[date, ticker]),
                    cost_rate=cost_rate,
                )
                trade_rows.append(trade_row)
                if trade_journal is not None:
                    trade_journal.log_trade(_trade_row_to_journal_payload(trade_row))
                states[ticker] = _new_position_state()
            if not state["is_open"] and desired_weight > 0.0:
                states[ticker] = {
                    "is_open": True,
                    "entry_date": date,
                    "entry_price": float(price_matrix.at[date, ticker]),
                    "entry_weight": desired_weight,
                    "portfolio_exposure_at_entry": gross_exposure,
                    "concurrent_positions": concurrent_positions,
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
                trade_row = _close_trade_row(
                    ticker=ticker,
                    exit_date=final_date,
                    state=state,
                    exit_price=float(price_matrix.at[final_date, ticker]),
                    cost_rate=cost_rate,
                )
                trade_rows.append(trade_row)
                if trade_journal is not None:
                    trade_journal.log_trade(_trade_row_to_journal_payload(trade_row))

    return pd.DataFrame(daily_rows), pd.DataFrame(trade_rows)


def _close_trade_row(
    ticker: str,
    exit_date: pd.Timestamp,
    state: dict[str, object],
    exit_price: float,
    cost_rate: float,
) -> dict[str, object]:
    gross_return = float(state["gross_return"])
    transaction_cost = 2.0 * cost_rate
    net_return = gross_return - transaction_cost
    return {
        "trade_id": f"trend:{ticker}:{pd.Timestamp(state['entry_date']).date()}:{pd.Timestamp(exit_date).date()}",
        "ticker": ticker,
        "entry_date": state["entry_date"],
        "exit_date": exit_date,
        "position_direction": 1,
        "direction": "long",
        "entry_zscore": np.nan,
        "exit_zscore": np.nan,
        "holding_days": int(state["holding_days"]),
        "entry_weight": float(state["entry_weight"]),
        "entry_price": float(state["entry_price"]),
        "exit_price": float(exit_price),
        "entry_node_corr": np.nan,
        "entry_regime": "TRADEABLE",
        "exit_reason": "signal_flip",
        "gross_return": gross_return,
        "transaction_cost": transaction_cost,
        "net_return": net_return,
        "gross_pnl": gross_return,
        "net_pnl": net_return,
        "predicted_residual": np.nan,
        "actual_residual": np.nan,
        "prediction_error": np.nan,
        "portfolio_exposure_at_entry": float(state["portfolio_exposure_at_entry"]),
        "concurrent_positions": int(state["concurrent_positions"]),
    }


def _new_position_state() -> dict[str, object]:
    return {
        "is_open": False,
        "entry_date": None,
        "entry_price": 0.0,
        "entry_weight": 0.0,
        "portfolio_exposure_at_entry": 0.0,
        "concurrent_positions": 0,
        "current_weight": 0.0,
        "gross_return": 0.0,
        "holding_days": 0,
    }


def _trade_row_to_journal_payload(trade_row: dict[str, object]) -> dict[str, object]:
    return {
        "trade_id": trade_row["trade_id"],
        "strategy": "B",
        "asset": trade_row["ticker"],
        "direction": trade_row["direction"],
        "entry_date": trade_row["entry_date"],
        "exit_date": trade_row["exit_date"],
        "holding_days": trade_row["holding_days"],
        "entry_price": trade_row["entry_price"],
        "exit_price": trade_row["exit_price"],
        "entry_zscore": trade_row["entry_zscore"],
        "entry_node_corr": trade_row["entry_node_corr"],
        "entry_regime": trade_row["entry_regime"],
        "exit_reason": trade_row["exit_reason"],
        "gross_pnl": trade_row["gross_pnl"],
        "transaction_cost": trade_row["transaction_cost"],
        "net_pnl": trade_row["net_pnl"],
        "predicted_residual": trade_row["predicted_residual"],
        "actual_residual": trade_row["actual_residual"],
        "prediction_error": trade_row["prediction_error"],
        "portfolio_exposure_at_entry": trade_row["portfolio_exposure_at_entry"],
        "concurrent_positions": trade_row["concurrent_positions"],
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
