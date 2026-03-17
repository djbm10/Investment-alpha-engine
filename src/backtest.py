from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np
import pandas as pd

from .config_loader import Phase2Config, Phase3Config
from .graph_engine import apply_signal_rules


@dataclass(frozen=True)
class BacktestResult:
    daily_results: pd.DataFrame
    trade_log: pd.DataFrame
    monthly_results: pd.DataFrame
    summary_metrics: dict[str, object]


@dataclass(frozen=True)
class RiskBudgetScalingResult:
    scaled_signals: pd.DataFrame
    baseline_max_drawdown: float | None
    target_max_drawdown: float
    scale_factor: float


def apply_phase3_regime_overlay(
    daily_signals: pd.DataFrame,
    phase2_config: Phase2Config,
    phase3_config: Phase3Config,
    regime_states: dict[pd.Timestamp, str],
    freeze_dates: set[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    if daily_signals.empty:
        return daily_signals.copy()

    freeze_dates = freeze_dates or set()
    overlaid = daily_signals.copy()
    overlaid["phase3_regime"] = overlaid["date"].map(
        lambda value: regime_states.get(pd.Timestamp(value), "STABLE")
    )
    if "regime_threshold_multiplier" not in overlaid.columns:
        overlaid["regime_threshold_multiplier"] = 1.0
    if "regime_position_scale" not in overlaid.columns:
        overlaid["regime_position_scale"] = 1.0
    if "allow_new_entries" not in overlaid.columns:
        overlaid["allow_new_entries"] = True

    transition_mask = overlaid["phase3_regime"] == "TRANSITIONING"
    new_regime_mask = overlaid["phase3_regime"] == "NEW_REGIME"

    overlaid.loc[transition_mask, "regime_threshold_multiplier"] = (
        overlaid.loc[transition_mask, "regime_threshold_multiplier"].astype(float)
        * phase3_config.transition_threshold_mult
    )
    overlaid.loc[transition_mask, "regime_position_scale"] = (
        overlaid.loc[transition_mask, "regime_position_scale"].astype(float)
        * phase3_config.transition_position_scale
    )
    overlaid.loc[new_regime_mask, "regime_threshold_multiplier"] = (
        overlaid.loc[new_regime_mask, "regime_threshold_multiplier"].astype(float)
        * phase3_config.new_regime_threshold_mult
    )
    overlaid.loc[new_regime_mask, "regime_position_scale"] = (
        overlaid.loc[new_regime_mask, "regime_position_scale"].astype(float)
        * phase3_config.new_regime_position_scale
    )
    return apply_signal_rules(overlaid, phase2_config)


def apply_phase4_tcn_filter(
    daily_signals: pd.DataFrame,
    predictions: pd.DataFrame,
    reversion_confirm_threshold: float,
) -> pd.DataFrame:
    if daily_signals.empty:
        return daily_signals.copy()
    if predictions.empty:
        filtered = daily_signals.copy()
        filtered["tcn_prediction_available"] = False
        filtered["predicted_residual_mean"] = np.nan
        filtered["predicted_residual_std"] = np.nan
        filtered["actual_next_residual"] = np.nan
        filtered["tcn_veto"] = False
        filtered["tcn_uncertainty_scale"] = 1.0
        return filtered

    filtered = daily_signals.copy().merge(
        predictions.loc[:, ["signal_date", "ticker", "predicted_residual_mean", "predicted_residual_std", "actual_next_residual"]],
        left_on=["date", "ticker"],
        right_on=["signal_date", "ticker"],
        how="left",
    )
    filtered.drop(columns=["signal_date"], inplace=True)
    filtered["tcn_prediction_available"] = (
        filtered["predicted_residual_mean"].notna() & filtered["predicted_residual_std"].notna()
    )

    active_signal = filtered["signal_direction"] != 0
    same_sign_persistence = (
        np.sign(filtered["zscore"].fillna(0.0))
        * np.sign(filtered["predicted_residual_mean"].fillna(0.0))
    ) > 0
    magnitude_persistence = (
        filtered["predicted_residual_mean"].abs().fillna(0.0)
        > filtered["residual"].abs().fillna(0.0) * reversion_confirm_threshold
    )
    filtered["tcn_veto"] = (
        active_signal
        & filtered["tcn_prediction_available"]
        & same_sign_persistence
        & magnitude_persistence
    )

    filtered["tcn_uncertainty_scale"] = 1.0
    accepted_mask = active_signal & filtered["tcn_prediction_available"] & ~filtered["tcn_veto"]
    if accepted_mask.any():
        inverse_uncertainty = 1.0 / filtered.loc[accepted_mask, "predicted_residual_std"].clip(lower=1e-6)
        normalization = float(inverse_uncertainty.mean())
        if normalization > 0:
            filtered.loc[accepted_mask, "tcn_uncertainty_scale"] = inverse_uncertainty / normalization

    filtered["target_position"] = (
        filtered["target_position"].astype(float) * filtered["tcn_uncertainty_scale"].astype(float)
    )
    filtered.loc[filtered["tcn_veto"], "target_position"] = 0.0
    return filtered


def run_walk_forward_backtest(
    daily_signals: pd.DataFrame,
    config: Phase2Config,
    run_id: str,
) -> BacktestResult:
    if daily_signals.empty:
        raise ValueError("No graph signals available for backtesting.")

    tickers = sorted(daily_signals["ticker"].unique().tolist())
    target_frame = _pivot_frame(daily_signals, "target_position", tickers)
    zscore_frame = _pivot_frame(daily_signals, "zscore", tickers)
    forward_return_frame = _pivot_frame(daily_signals, "forward_return", tickers)

    valid_dates = forward_return_frame.dropna(how="all").index
    target_frame = target_frame.loc[valid_dates].fillna(0.0)
    zscore_frame = zscore_frame.loc[valid_dates]
    forward_return_frame = forward_return_frame.loc[valid_dates].fillna(0.0)

    total_cost_rate = (
        config.commission_bps
        + config.bid_ask_bps
        + config.market_impact_bps
        + config.slippage_bps
    ) / 10000.0

    states = {ticker: _new_position_state() for ticker in tickers}
    previous_weights = pd.Series(0.0, index=tickers, dtype=float)
    trade_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []

    for date in valid_dates:
        desired_weights = pd.Series(0.0, index=tickers, dtype=float)

        for ticker in tickers:
            state = states[ticker]
            zscore = zscore_frame.at[date, ticker]
            if state["is_open"]:
                if _should_exit(state, zscore, config):
                    desired_weights[ticker] = 0.0
                else:
                    desired_weights[ticker] = state["direction"] * state["current_weight"]
            else:
                desired_weights[ticker] = float(target_frame.at[date, ticker])

        if config.enforce_dollar_neutral:
            desired_weights = _neutralize_relative_weights(desired_weights)
        gross_exposure = desired_weights.abs().sum()
        if gross_exposure > 1.0:
            desired_weights = desired_weights / gross_exposure
            gross_exposure = 1.0

        for ticker in tickers:
            state = states[ticker]
            desired_direction = int(np.sign(desired_weights[ticker]))
            if state["is_open"] and desired_direction == 0:
                trade_rows.append(_close_trade_row(run_id, ticker, date, zscore_frame.at[date, ticker], state, total_cost_rate))
                states[ticker] = _new_position_state()

        for ticker in tickers:
            state = states[ticker]
            desired_weight = float(abs(desired_weights[ticker]))
            desired_direction = int(np.sign(desired_weights[ticker]))
            if not state["is_open"] and desired_direction != 0:
                states[ticker] = {
                    "is_open": True,
                    "direction": desired_direction,
                    "entry_date": date,
                    "entry_zscore": _optional_float(zscore_frame.at[date, ticker]),
                    "entry_weight": desired_weight,
                    "current_weight": desired_weight,
                    "gross_return": 0.0,
                    "holding_days": 0,
                }
            elif state["is_open"]:
                state["current_weight"] = desired_weight

        turnover = float((desired_weights - previous_weights).abs().sum())
        gross_portfolio_return = float((desired_weights * forward_return_frame.loc[date]).sum())
        transaction_cost = turnover * total_cost_rate
        portfolio_return = gross_portfolio_return - transaction_cost

        for ticker in tickers:
            state = states[ticker]
            if state["is_open"]:
                signed_return = state["direction"] * float(forward_return_frame.at[date, ticker])
                state["gross_return"] = (1.0 + state["gross_return"]) * (1.0 + signed_return) - 1.0
                state["holding_days"] += 1

        daily_rows.append(
            {
                "date": date,
                "gross_portfolio_return": gross_portfolio_return,
                "net_portfolio_return": portfolio_return,
                "portfolio_return": portfolio_return,
                "gross_exposure": gross_exposure,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
            }
        )
        previous_weights = desired_weights.copy()

    if len(valid_dates) > 0:
        final_date = valid_dates[-1]
        for ticker, state in states.items():
            if state["is_open"]:
                trade_rows.append(_close_trade_row(run_id, ticker, final_date, zscore_frame.at[final_date, ticker], state, total_cost_rate))

    daily_results = pd.DataFrame(daily_rows)
    trade_log = pd.DataFrame(trade_rows)
    monthly_results = _build_monthly_results(daily_results, config)
    summary_metrics = _build_summary_metrics(
        daily_signals=daily_signals,
        daily_results=daily_results,
        trade_log=trade_log,
        monthly_results=monthly_results,
        config=config,
    )
    return BacktestResult(
        daily_results=daily_results,
        trade_log=trade_log,
        monthly_results=monthly_results,
        summary_metrics=summary_metrics,
    )


def scale_signals_to_risk_budget(
    daily_signals: pd.DataFrame,
    config: Phase2Config,
) -> RiskBudgetScalingResult:
    preview_result = run_walk_forward_backtest(daily_signals, config, run_id="risk-budget-preview")
    observed_drawdown = preview_result.summary_metrics.get("max_drawdown")
    target_drawdown = config.risk_budget_utilization * config.max_drawdown_limit

    if observed_drawdown is None or observed_drawdown <= 0:
        return RiskBudgetScalingResult(
            scaled_signals=daily_signals.copy(),
            baseline_max_drawdown=observed_drawdown,
            target_max_drawdown=target_drawdown,
            scale_factor=1.0,
        )

    scale_factor = target_drawdown / observed_drawdown
    scaled_signals = daily_signals.copy()
    scaled_signals["target_position"] = scaled_signals["target_position"] * scale_factor
    return RiskBudgetScalingResult(
        scaled_signals=scaled_signals,
        baseline_max_drawdown=float(observed_drawdown),
        target_max_drawdown=target_drawdown,
        scale_factor=float(scale_factor),
    )


def _pivot_frame(frame: pd.DataFrame, value_column: str, tickers: list[str]) -> pd.DataFrame:
    return (
        frame.pivot(index="date", columns="ticker", values=value_column)
        .sort_index()
        .reindex(columns=tickers)
    )


def _new_position_state() -> dict[str, object]:
    return {
        "is_open": False,
        "direction": 0,
        "entry_date": None,
        "entry_zscore": None,
        "entry_weight": 0.0,
        "current_weight": 0.0,
        "gross_return": 0.0,
        "holding_days": 0,
    }


def _neutralize_relative_weights(weights: pd.Series) -> pd.Series:
    adjusted = weights.copy()
    long_gross = float(adjusted[adjusted > 0].sum())
    short_gross = float(abs(adjusted[adjusted < 0].sum()))

    if long_gross == 0 or short_gross == 0:
        return adjusted * 0.0

    if long_gross > short_gross:
        adjusted[adjusted > 0] *= short_gross / long_gross
    elif short_gross > long_gross:
        adjusted[adjusted < 0] *= long_gross / short_gross

    return adjusted


def _should_exit(state: dict[str, object], zscore: float | None, config: Phase2Config) -> bool:
    direction = int(state["direction"])
    hit_reversion = False
    if zscore is not None and not pd.isna(zscore):
        hit_reversion = (direction == 1 and zscore >= 0) or (direction == -1 and zscore <= 0)

    return (
        hit_reversion
        or float(state["gross_return"]) <= -config.stop_loss
        or int(state["holding_days"]) >= config.max_holding_days
    )


def _close_trade_row(
    run_id: str,
    ticker: str,
    exit_date: pd.Timestamp,
    exit_zscore: float | None,
    state: dict[str, object],
    total_cost_rate: float,
) -> dict[str, object]:
    trade_id = f"{run_id}:{ticker}:{pd.Timestamp(state['entry_date']).date()}:{uuid4()}"
    gross_return = float(state["gross_return"])
    net_return = gross_return - (2.0 * total_cost_rate)
    return {
        "trade_id": trade_id,
        "run_id": run_id,
        "ticker": ticker,
        "entry_date": state["entry_date"],
        "exit_date": exit_date,
        "position_direction": int(state["direction"]),
        "entry_zscore": _optional_float(state["entry_zscore"]),
        "exit_zscore": _optional_float(exit_zscore),
        "holding_days": int(state["holding_days"]),
        "entry_weight": float(state["entry_weight"]),
        "gross_return": gross_return,
        "net_return": net_return,
    }


def _build_monthly_results(daily_results: pd.DataFrame, config: Phase2Config) -> pd.DataFrame:
    if daily_results.empty:
        return pd.DataFrame(
            columns=["test_month", "training_end_date", "monthly_return", "profitable", "active_month"]
        )

    monthly_returns = (
        daily_results.assign(test_month=daily_results["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("test_month")["portfolio_return"]
        .apply(lambda series: float((1.0 + series).prod() - 1.0))
        .rename("monthly_return")
        .reset_index()
    )
    monthly_activity = (
        daily_results.assign(test_month=daily_results["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("test_month")
        .agg(
            monthly_turnover=("turnover", "sum"),
            gross_exposure_sum=("gross_exposure", "sum"),
        )
        .reset_index()
    )
    monthly_stats = monthly_returns.merge(monthly_activity, on="test_month", how="left")

    rows: list[dict[str, object]] = []
    for month_idx in range(config.min_training_months, len(monthly_stats)):
        test_month = pd.Timestamp(monthly_stats.loc[month_idx, "test_month"])
        monthly_return = float(monthly_stats.loc[month_idx, "monthly_return"])
        active_month = (
            float(monthly_stats.loc[month_idx, "monthly_turnover"]) > 0
            or float(monthly_stats.loc[month_idx, "gross_exposure_sum"]) > 0
        )
        rows.append(
            {
                "test_month": test_month,
                "training_end_date": test_month - pd.Timedelta(days=1),
                "monthly_return": monthly_return,
                "profitable": active_month and monthly_return > 0,
                "active_month": active_month,
            }
        )

    return pd.DataFrame(rows)


def _build_summary_metrics(
    *,
    daily_signals: pd.DataFrame,
    daily_results: pd.DataFrame,
    trade_log: pd.DataFrame,
    monthly_results: pd.DataFrame,
    config: Phase2Config,
) -> dict[str, object]:
    if not daily_results.empty:
        daily_results = daily_results.copy()
        daily_results["test_month"] = daily_results["date"].dt.to_period("M").dt.to_timestamp()
    oos_months = set(monthly_results["test_month"].tolist()) if not monthly_results.empty else set()
    oos_daily_results = (
        daily_results.loc[daily_results["test_month"].isin(oos_months)].copy()
        if oos_months
        else pd.DataFrame(columns=daily_results.columns)
    )

    sharpe_ratio = _annualized_sharpe(oos_daily_results["portfolio_return"], config.annualization_days)
    annualized_return = _annualized_return(oos_daily_results["portfolio_return"], config.annualization_days)
    max_drawdown = _max_drawdown(oos_daily_results["portfolio_return"])
    win_rate = _win_rate(trade_log)
    profit_factor = _profit_factor(trade_log)
    avg_holding_days = _average_holding_days(trade_log)
    annual_turnover = _annual_turnover(oos_daily_results, config.annualization_days)
    active_monthly_results = (
        monthly_results.loc[monthly_results["active_month"]].copy()
        if not monthly_results.empty and "active_month" in monthly_results.columns
        else monthly_results.copy()
    )
    profitable_month_fraction = (
        float(active_monthly_results["profitable"].mean()) if not active_monthly_results.empty else 0.0
    )

    gate_passed = (
        sharpe_ratio is not None
        and sharpe_ratio > 0.7
        and profitable_month_fraction >= 0.6
        and max_drawdown is not None
        and max_drawdown <= 0.20
    )

    return {
        "lookback_window": config.lookback_window,
        "diffusion_alpha": config.diffusion_alpha,
        "diffusion_steps": config.diffusion_steps,
        "sigma_scale": config.sigma_scale,
        "min_weight": config.min_weight,
        "zscore_lookback": config.zscore_lookback,
        "signal_threshold": config.signal_threshold,
        "tier2_enabled": config.tier2_enabled,
        "tier2_fraction": config.tier2_fraction,
        "tier2_size_fraction": config.tier2_size_fraction,
        "max_position_size": config.max_position_size,
        "risk_budget_utilization": config.risk_budget_utilization,
        "max_drawdown_limit": config.max_drawdown_limit,
        "enforce_dollar_neutral": config.enforce_dollar_neutral,
        "corr_floor": config.corr_floor,
        "density_floor": config.density_floor,
        "total_signals": int((daily_signals["signal_direction"] != 0).sum()),
        "total_trades": int(len(trade_log)),
        "sharpe_ratio": sharpe_ratio,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_holding_days": avg_holding_days,
        "annual_turnover": annual_turnover,
        "profitable_month_fraction": profitable_month_fraction,
        "out_of_sample_months": int(len(monthly_results)),
        "active_out_of_sample_months": int(len(active_monthly_results)),
        "gate_passed": gate_passed,
    }


def _annualized_sharpe(returns: pd.Series, annualization_days: int) -> float | None:
    if returns.empty:
        return None
    volatility = returns.std(ddof=0)
    if volatility == 0 or pd.isna(volatility):
        return None
    return float((returns.mean() / volatility) * np.sqrt(annualization_days))


def _annualized_return(returns: pd.Series, annualization_days: int) -> float | None:
    if returns.empty:
        return None
    periods = len(returns)
    cumulative_return = float((1.0 + returns).prod())
    if periods <= 0 or cumulative_return <= 0:
        return None
    return float(cumulative_return ** (annualization_days / periods) - 1.0)


def _max_drawdown(returns: pd.Series) -> float | None:
    if returns.empty:
        return None
    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return float(abs(drawdown.min()))


def _win_rate(trade_log: pd.DataFrame) -> float | None:
    if trade_log.empty:
        return None
    return float((trade_log["net_return"] > 0).mean())


def _profit_factor(trade_log: pd.DataFrame) -> float | None:
    if trade_log.empty:
        return None
    gross_profit = float(trade_log.loc[trade_log["net_return"] > 0, "net_return"].sum())
    gross_loss = float(abs(trade_log.loc[trade_log["net_return"] < 0, "net_return"].sum()))
    if gross_loss == 0:
        return 999.0 if gross_profit > 0 else None
    return gross_profit / gross_loss


def _average_holding_days(trade_log: pd.DataFrame) -> float | None:
    if trade_log.empty:
        return None
    return float(trade_log["holding_days"].mean())


def _annual_turnover(daily_results: pd.DataFrame, annualization_days: int) -> float | None:
    if daily_results.empty:
        return None
    return float(daily_results["turnover"].mean() * annualization_days * 100.0)


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
