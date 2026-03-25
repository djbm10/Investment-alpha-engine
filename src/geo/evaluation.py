from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..backtest import _annual_turnover, _annualized_sharpe, _max_drawdown
from ..config_loader import PipelineConfig, load_config
from ..trend_strategy import TrendStrategyBacktest, load_phase2_baseline_backtest

BOOTSTRAP_RANDOM_SEED = 1729
BOOTSTRAP_SAMPLES = 250
BOOTSTRAP_P25_TOLERANCE = 0.05
MIN_GEO_DAYS_FRACTION = 0.05
MIN_GEO_MEDIAN_COVERAGE = 0.50
MIN_GEO_ASSET_COVERAGE = 0.50
MIN_GEO_ASSETS_PER_DAY = 2
HIGH_GEO_BLOCK_MIN_LENGTH = 3
MAX_REMOVED_TRADE_FRACTION = 0.20
MAX_STALE_MEDIAN_FRESHNESS_MINUTES = 240
STALE_WINDOW_CONSECUTIVE_DAYS = 3
PARITY_NUMERIC_TOLERANCE = 1e-12
EXTREME_RETURN_ABS_THRESHOLD = 0.50
REPORT_VERSION = "geo_overlay_eval_v4"
PRICE_RETURN_PARITY_COLUMNS = ("adj_close", "current_return", "forward_return")
TRADABILITY_PARITY_COLUMNS = ("allow_new_entries", "node_tradeable")
REQUIRED_OVERLAY_SIGNAL_COLUMNS = frozenset(
    {
        "date",
        "ticker",
        "final_signal_direction",
        "final_target_position",
        "geo_entry_blocked",
        "position_scale",
        "geo_net_score",
        "geo_structural_score",
        "coverage_score",
        "data_freshness_minutes",
    }
)


@dataclass(frozen=True)
class GeoOverlayEvaluationResult:
    report: dict[str, object]
    output_path: Path


def run_geo_overlay_evaluation(
    config_path: str | Path,
    *,
    geo_snapshot: pd.DataFrame | None = None,
) -> GeoOverlayEvaluationResult:
    config = load_config(config_path)
    baseline_config = replace(config, geo=replace(config.geo, enabled=False))
    geo_enabled_config = replace(config, geo=replace(config.geo, enabled=True))
    baseline_cost_model_hash = _cost_model_hash_from_phase2(baseline_config)
    geo_cost_model_hash = _cost_model_hash_from_phase2(geo_enabled_config)
    if baseline_cost_model_hash != geo_cost_model_hash:
        raise ValueError("Cost model hash mismatch between baseline and geo-enabled runs")

    baseline_result = load_phase2_baseline_backtest(baseline_config)
    geo_enabled_result = load_phase2_baseline_backtest(geo_enabled_config, geo_snapshot=geo_snapshot)

    report = build_geo_overlay_evaluation_report(
        baseline_result=baseline_result,
        geo_enabled_result=geo_enabled_result,
        annualization_days=config.phase2.annualization_days,
        random_seed=BOOTSTRAP_RANDOM_SEED,
    )
    report["metadata"] = _build_metadata(config, baseline_result, geo_enabled_result, random_seed=BOOTSTRAP_RANDOM_SEED)
    report["alignment_check"] = (
        _validate_overlay_alignment(geo_enabled_result.daily_signals, geo_snapshot)
        if geo_snapshot is not None
        else {"performed": False, "passed": False, "checked_date": "", "checked_rows": 0}
    )
    report["snapshot_timing_check"] = (
        _validate_snapshot_timeliness(geo_snapshot, cutoff_time_et=config.geo.cutoff_time_et)
        if geo_snapshot is not None
        else {"performed": False, "enforceable": False, "passed": False, "checked_rows": 0, "violations": 0}
    )
    report["report_version"] = REPORT_VERSION
    report["report_schema_hash"] = _schema_hash(report)
    output_path = config.paths.processed_dir / "geo_overlay_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return GeoOverlayEvaluationResult(report=report, output_path=output_path)


def build_geo_overlay_evaluation_report(
    *,
    baseline_result: TrendStrategyBacktest,
    geo_enabled_result: TrendStrategyBacktest,
    annualization_days: int,
    random_seed: int = BOOTSTRAP_RANDOM_SEED,
) -> dict[str, object]:
    _validate_run_parity(baseline_result, geo_enabled_result)
    _validate_trade_log_contradiction_contract(baseline_result.trade_log)
    _validate_trade_log_contradiction_contract(geo_enabled_result.trade_log)
    _validate_overlay_signal_contract(geo_enabled_result.daily_signals)
    _validate_overlay_signal_contract(baseline_result.daily_signals)
    input_parity = _build_input_parity(
        baseline_result.daily_signals,
        geo_enabled_result.daily_signals,
    )

    baseline_metrics = _extract_summary_metrics(baseline_result, annualization_days=annualization_days)
    geo_enabled_metrics = _extract_summary_metrics(geo_enabled_result, annualization_days=annualization_days)

    geo_stress = _build_geo_stress_frame(geo_enabled_result)
    eligible_geo_stress = geo_stress.loc[
        pd.to_numeric(geo_stress.get("median_data_freshness_minutes", pd.Series(dtype=float)), errors="coerce").fillna(float("inf"))
        <= MAX_STALE_MEDIAN_FRESHNESS_MINUTES
    ].copy()
    geo_stress_series = (
        eligible_geo_stress.set_index("date")["geo_stress"]
        if not geo_stress.empty
        else pd.Series(dtype=float)
    )
    top_decile_dates, normal_dates = _split_geo_stress_dates(geo_stress_series)
    high_geo_block_dates, high_geo_block_count = _extract_high_geo_blocks(geo_stress_series, top_decile_dates)

    slices = {
        "global": {
            "date_count": int(len(geo_stress)),
            "median_coverage": _slice_median_coverage(geo_stress, set(pd.to_datetime(geo_stress["date"])) if not geo_stress.empty else set()),
            "baseline": _build_slice_metrics(
                baseline_result,
                annualization_days=annualization_days,
                slice_dates=set(pd.to_datetime(geo_stress["date"])) if not geo_stress.empty else set(),
            ),
            "geo_enabled": _build_slice_metrics(
                geo_enabled_result,
                annualization_days=annualization_days,
                slice_dates=set(pd.to_datetime(geo_stress["date"])) if not geo_stress.empty else set(),
            ),
        },
        "top_decile_geo_stress": {
            "date_count": int(len(top_decile_dates)),
            "median_coverage": _slice_median_coverage(geo_stress, top_decile_dates),
            "baseline": _build_slice_metrics(
                baseline_result,
                annualization_days=annualization_days,
                slice_dates=top_decile_dates,
            ),
            "geo_enabled": _build_slice_metrics(
                geo_enabled_result,
                annualization_days=annualization_days,
                slice_dates=top_decile_dates,
            ),
        },
        "normal_periods": {
            "date_count": int(len(normal_dates)),
            "median_coverage": _slice_median_coverage(geo_stress, normal_dates),
            "baseline": _build_slice_metrics(
                baseline_result,
                annualization_days=annualization_days,
                slice_dates=normal_dates,
            ),
            "geo_enabled": _build_slice_metrics(
                geo_enabled_result,
                annualization_days=annualization_days,
                slice_dates=normal_dates,
            ),
        },
        "high_geo_blocks": {
            "block_count": int(high_geo_block_count),
            "date_count": int(len(high_geo_block_dates)),
            "median_coverage": _slice_median_coverage(geo_stress, high_geo_block_dates),
            "baseline": _build_slice_metrics(
                baseline_result,
                annualization_days=annualization_days,
                slice_dates=high_geo_block_dates,
            ),
            "geo_enabled": _build_slice_metrics(
                geo_enabled_result,
                annualization_days=annualization_days,
                slice_dates=high_geo_block_dates,
            ),
        },
    }

    comparison = {
        metric: {
            "baseline": baseline_metrics[metric],
            "geo_enabled": geo_enabled_metrics[metric],
            "delta": _delta(geo_enabled_metrics[metric], baseline_metrics[metric]),
        }
        for metric in ("sharpe_ratio", "max_drawdown", "annual_turnover", "contradiction_loss_rate")
    }
    filtering_impact = _build_filtering_impact(
        baseline_result.trade_log,
        geo_enabled_result.trade_log,
    )
    bootstrap = _build_bootstrap_report(
        baseline_result.daily_results,
        geo_enabled_result.daily_results,
        annualization_days=annualization_days,
        random_seed=random_seed,
    )

    report = {
        "baseline": baseline_metrics,
        "geo_enabled": geo_enabled_metrics,
        "comparison": comparison,
        "geo_stress": {
            "available": _geo_stress_available(geo_stress),
            "date_count": int(len(geo_stress)),
            "days_with_geo_signal": int((geo_stress["geo_stress"] > 0.0).sum()) if not geo_stress.empty else 0,
            "min_days_required": int(max(1, math.ceil(len(geo_stress) * MIN_GEO_DAYS_FRACTION))) if len(geo_stress) else 1,
            "variance": float(geo_stress["geo_stress"].var(ddof=0)) if not geo_stress.empty else 0.0,
            "median_coverage_on_geo_days": _median_coverage_on_geo_days(geo_stress),
            "stale_window_detected": _detect_stale_window(geo_stress),
            "per_asset_coverage_floor": MIN_GEO_ASSET_COVERAGE,
            "top_decile_date_count": int(len(top_decile_dates)),
            "top_decile_avg_coverage": _slice_avg_coverage(geo_stress, top_decile_dates),
            "normal_date_count": int(len(normal_dates)),
            "top_decile_threshold": (
                None
                if geo_stress.empty or not top_decile_dates
                else float(geo_stress.loc[geo_stress["date"].isin(list(top_decile_dates)), "geo_stress"].min())
            ),
        },
        "slices": slices,
        "trade_diagnostics": {
            "baseline": _build_trade_diagnostics(baseline_result.trade_log),
            "geo_enabled": _build_trade_diagnostics(geo_enabled_result.trade_log),
        },
        "filtering_impact": filtering_impact,
        "overlay_action_diagnostics": _build_overlay_action_diagnostics(
            baseline_result.trade_log,
            geo_enabled_result.trade_log,
            geo_enabled_result.daily_signals,
        ),
        "input_parity": input_parity,
        "data_sanity": _build_data_sanity(
            baseline_result.daily_signals,
            geo_enabled_result.daily_signals,
        ),
        "bootstrap": bootstrap,
        "curves": _build_curves(baseline_result.daily_results, geo_enabled_result.daily_results),
        "geo_stress_series": _build_geo_stress_series(geo_stress, high_geo_block_dates),
        "per_asset_delta_pnl": _build_per_asset_delta_pnl(
            baseline_result.trade_log,
            geo_enabled_result.trade_log,
        ),
    }
    report["acceptance"] = _build_acceptance(report)
    report["report_version"] = REPORT_VERSION
    report["report_schema_hash"] = _schema_hash(report)
    return report


def _extract_summary_metrics(
    result: TrendStrategyBacktest,
    *,
    annualization_days: int,
) -> dict[str, float | int | None]:
    daily_results = _sorted_daily_results(result.daily_results)
    trade_log = _sorted_trade_log(result.trade_log)
    summary = dict(result.summary_metrics)
    portfolio_returns = _portfolio_returns(daily_results)
    net_portfolio_returns = _net_portfolio_returns(daily_results)

    sharpe_ratio = _float_or_none(summary.get("sharpe_ratio"))
    if sharpe_ratio is None:
        sharpe_ratio = _float_or_none(_annualized_sharpe(portfolio_returns, annualization_days))

    max_drawdown = _float_or_none(summary.get("max_drawdown"))
    if max_drawdown is None:
        max_drawdown = _float_or_none(_max_drawdown(portfolio_returns))

    annual_turnover = _float_or_none(summary.get("annual_turnover"))
    if annual_turnover is None:
        annual_turnover = _float_or_none(_annual_turnover(daily_results, annualization_days))

    contradiction_loss_rate = _float_or_none(summary.get("contradiction_loss_rate"))
    if contradiction_loss_rate is None:
        contradiction_loss_rate = _contradiction_loss_rate(trade_log)

    return {
        "sharpe_ratio": sharpe_ratio,
        "net_sharpe_after_costs": _float_or_none(_annualized_sharpe(net_portfolio_returns, annualization_days)),
        "net_cvar_5": _net_cvar_5(net_portfolio_returns),
        "max_drawdown": max_drawdown,
        "annual_turnover": annual_turnover,
        "contradiction_loss_rate": contradiction_loss_rate,
        "avg_net_pnl_per_trade": _mean_net_return(trade_log),
        "total_trades": int(len(trade_log)),
    }


def _build_geo_stress_frame(result: TrendStrategyBacktest) -> pd.DataFrame:
    if result.daily_signals is None or result.daily_signals.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "geo_stress",
                "median_coverage_score",
                "median_data_freshness_minutes",
                "qualified_asset_count",
            ]
        )
    if "date" not in result.daily_signals.columns or "geo_net_score" not in result.daily_signals.columns:
        return pd.DataFrame(
            columns=[
                "date",
                "geo_stress",
                "median_coverage_score",
                "median_data_freshness_minutes",
                "qualified_asset_count",
            ]
        )
    if result.daily_results.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "geo_stress",
                "median_coverage_score",
                "median_data_freshness_minutes",
                "qualified_asset_count",
            ]
        )

    signal_frame = result.daily_signals.copy()
    signal_frame["date"] = pd.to_datetime(signal_frame["date"])
    signal_frame = signal_frame.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    signal_frame["geo_net_score"] = pd.to_numeric(signal_frame["geo_net_score"], errors="coerce").fillna(0.0)
    signal_frame["coverage_score"] = pd.to_numeric(
        signal_frame.get("coverage_score", pd.Series(0.0, index=signal_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    signal_frame["data_freshness_minutes"] = pd.to_numeric(
        signal_frame.get("data_freshness_minutes", pd.Series(float("inf"), index=signal_frame.index)),
        errors="coerce",
    ).fillna(float("inf"))
    signal_frame["coverage_qualified"] = signal_frame["coverage_score"] >= MIN_GEO_ASSET_COVERAGE
    result_dates = pd.Index(pd.to_datetime(result.daily_results["date"]).sort_values().unique())
    grouped = signal_frame.groupby("date").agg(
        geo_stress=("geo_net_score", lambda values: 0.0),
        median_coverage_score=("coverage_score", "median"),
        median_data_freshness_minutes=("data_freshness_minutes", "median"),
        qualified_asset_count=("coverage_qualified", "sum"),
    )
    qualifying_stress = (
        signal_frame.loc[signal_frame["coverage_qualified"]]
        .groupby("date")["geo_net_score"]
        .apply(lambda values: float(values.abs().mean()))
    )
    grouped["geo_stress"] = qualifying_stress.reindex(grouped.index, fill_value=0.0)
    grouped.loc[grouped["qualified_asset_count"] < MIN_GEO_ASSETS_PER_DAY, "geo_stress"] = 0.0
    grouped = grouped.reindex(result_dates, fill_value=0.0)
    grouped = grouped.reset_index().rename(columns={"index": "date"})
    grouped["date"] = pd.to_datetime(grouped["date"])
    return grouped.sort_values("date", kind="mergesort").reset_index(drop=True)


def _split_geo_stress_dates(geo_stress: pd.Series) -> tuple[set[pd.Timestamp], set[pd.Timestamp]]:
    if geo_stress.empty or float(geo_stress.abs().max()) <= 0.0:
        return set(), set(pd.to_datetime(geo_stress.index))

    top_count = max(1, int(math.ceil(len(geo_stress) * 0.10)))
    ordered = geo_stress.sort_values(ascending=False, kind="mergesort")
    top_dates = set(pd.to_datetime(ordered.head(top_count).index))
    normal_dates = set(pd.to_datetime(geo_stress.index)) - top_dates
    return top_dates, normal_dates


def _build_slice_metrics(
    result: TrendStrategyBacktest,
    *,
    annualization_days: int,
    slice_dates: set[pd.Timestamp],
) -> dict[str, float | None]:
    daily_results = _sorted_daily_results(result.daily_results)
    trade_log = _sorted_trade_log(result.trade_log)
    if not slice_dates:
        return {
            "sharpe_ratio": None,
            "net_sharpe_after_costs": None,
            "net_cvar_5": None,
            "max_drawdown": None,
            "annual_turnover": None,
            "contradiction_loss_rate": None,
            "trade_count": 0,
        }

    slice_index = pd.Index(pd.to_datetime(sorted(slice_dates)))
    sliced_daily = daily_results.loc[daily_results["date"].isin(slice_index)].copy()
    sliced_trades = trade_log.loc[pd.to_datetime(trade_log["entry_date"]).isin(slice_index)].copy()
    portfolio_returns = _portfolio_returns(sliced_daily)
    net_portfolio_returns = _net_portfolio_returns(sliced_daily)

    return {
        "sharpe_ratio": _float_or_none(_annualized_sharpe(portfolio_returns, annualization_days)),
        "net_sharpe_after_costs": _float_or_none(_annualized_sharpe(net_portfolio_returns, annualization_days)),
        "net_cvar_5": _net_cvar_5(net_portfolio_returns),
        "max_drawdown": _float_or_none(_max_drawdown(portfolio_returns)),
        "annual_turnover": _float_or_none(_annual_turnover(sliced_daily, annualization_days)),
        "contradiction_loss_rate": _contradiction_loss_rate(sliced_trades),
        "trade_count": int(len(sliced_trades)),
    }


def _build_trade_diagnostics(trade_log: pd.DataFrame) -> dict[str, float | int]:
    sorted_log = _sorted_trade_log(trade_log)
    contradictory = _contradictory_trades(sorted_log)
    non_contradictory = sorted_log.loc[~sorted_log.index.isin(contradictory.index)].copy()
    return {
        "contradictory_trade_count": int(len(contradictory)),
        "avg_pnl_contradictory": _mean_net_return(contradictory),
        "avg_pnl_non_contradictory": _mean_net_return(non_contradictory),
    }


def _build_filtering_impact(
    baseline_trade_log: pd.DataFrame,
    geo_enabled_trade_log: pd.DataFrame,
) -> dict[str, float | int]:
    baseline = _sorted_trade_log(baseline_trade_log).copy()
    geo_enabled = _sorted_trade_log(geo_enabled_trade_log).copy()
    baseline["_entry_key"] = _trade_entry_keys(baseline)
    geo_enabled_keys = set(_trade_entry_keys(geo_enabled))
    removed = baseline.loc[~baseline["_entry_key"].isin(geo_enabled_keys)].copy()
    kept = baseline.loc[baseline["_entry_key"].isin(geo_enabled_keys)].copy()
    return {
        "removed_trade_count": int(len(removed)),
        "removed_trade_fraction": (
            float(len(removed)) / float(len(baseline))
            if len(baseline)
            else 0.0
        ),
        "avg_removed_trade_pnl": _mean_net_return(removed),
        "avg_all_trade_pnl": _mean_net_return(baseline),
        "avg_pnl_kept_winners": _mean_net_return(kept.loc[pd.to_numeric(kept["net_return"], errors="coerce").fillna(0.0) > 0.0]) if not kept.empty else 0.0,
        "avg_pnl_removed_losers": _mean_net_return(removed.loc[pd.to_numeric(removed["net_return"], errors="coerce").fillna(0.0) < 0.0]) if not removed.empty else 0.0,
        "fraction_winners_removed": _fraction_winners_removed(baseline, removed),
    }


def _build_overlay_action_diagnostics(
    baseline_trade_log: pd.DataFrame,
    geo_enabled_trade_log: pd.DataFrame,
    geo_daily_signals: pd.DataFrame | None,
) -> dict[str, dict[str, float | int | None]]:
    if geo_daily_signals is None or geo_daily_signals.empty:
        return {
            bucket: {
                "trade_count": 0,
                "trade_fraction": 0.0,
                "avg_baseline_pnl": 0.0,
                "avg_geo_enabled_pnl": 0.0,
            }
            for bucket in ("blocked", "scaled", "untouched")
        }

    baseline = _sorted_trade_log(baseline_trade_log).copy()
    if baseline.empty:
        return {
            bucket: {
                "trade_count": 0,
                "trade_fraction": 0.0,
                "avg_baseline_pnl": 0.0,
                "avg_geo_enabled_pnl": 0.0,
            }
            for bucket in ("blocked", "scaled", "untouched")
        }

    action_frame = _sorted_signal_frame(geo_daily_signals).copy()
    action_frame["geo_entry_blocked"] = action_frame["geo_entry_blocked"].astype(bool)
    action_frame["position_scale"] = pd.to_numeric(action_frame["position_scale"], errors="coerce").fillna(1.0)
    action_frame = action_frame.loc[:, ["date", "ticker", "geo_entry_blocked", "position_scale"]].copy()
    merged = baseline.merge(
        action_frame,
        how="left",
        left_on=["entry_date", "ticker"],
        right_on=["date", "ticker"],
    )
    merged["geo_entry_blocked"] = merged["geo_entry_blocked"].fillna(False).astype(bool)
    merged["position_scale"] = pd.to_numeric(merged["position_scale"], errors="coerce").fillna(1.0)
    merged["_entry_key"] = _trade_entry_keys(merged)
    geo_lookup = _sorted_trade_log(geo_enabled_trade_log).copy()
    geo_lookup["_entry_key"] = _trade_entry_keys(geo_lookup)
    geo_lookup = geo_lookup.loc[:, ["_entry_key", "net_return"]].rename(columns={"net_return": "geo_enabled_net_return"})
    merged = merged.merge(geo_lookup, how="left", on="_entry_key")
    merged["action_bucket"] = np.select(
        [
            merged["geo_entry_blocked"],
            (~merged["geo_entry_blocked"]) & (merged["position_scale"] < 1.0),
        ],
        ["blocked", "scaled"],
        default="untouched",
    )

    diagnostics: dict[str, dict[str, float | int | None]] = {}
    total_trades = len(merged)
    for bucket in ("blocked", "scaled", "untouched"):
        bucket_frame = merged.loc[merged["action_bucket"] == bucket].copy()
        diagnostics[bucket] = {
            "trade_count": int(len(bucket_frame)),
            "trade_fraction": float(len(bucket_frame) / total_trades) if total_trades else 0.0,
            "avg_baseline_pnl": _mean_net_return(bucket_frame),
            "avg_geo_enabled_pnl": _mean_series(
                bucket_frame["geo_enabled_net_return"]
                if "geo_enabled_net_return" in bucket_frame.columns
                else pd.Series(dtype=float)
            ),
        }
        diagnostics[bucket]["delta_avg_pnl"] = float(
            diagnostics[bucket]["avg_geo_enabled_pnl"] - diagnostics[bucket]["avg_baseline_pnl"]
        )
    return diagnostics


def _build_acceptance(report: dict[str, object]) -> dict[str, object]:
    baseline = report["baseline"]
    geo_enabled = report["geo_enabled"]
    slices = report["slices"]
    geo_stress = report["geo_stress"]
    filtering_impact = report["filtering_impact"]

    top_baseline = slices["top_decile_geo_stress"]["baseline"]
    top_geo = slices["top_decile_geo_stress"]["geo_enabled"]
    block_baseline = slices["high_geo_blocks"]["baseline"]
    block_geo = slices["high_geo_blocks"]["geo_enabled"]
    normal_baseline = slices["normal_periods"]["baseline"]
    normal_geo = slices["normal_periods"]["geo_enabled"]
    drawdown_improved_materially = _improved_by_ratio(
        baseline["max_drawdown"],
        geo_enabled["max_drawdown"],
        ratio=0.90,
    )
    contradiction_improved = _less_than_or_equal(
        top_geo["contradiction_loss_rate"],
        top_baseline["contradiction_loss_rate"],
    )
    sharpe_tolerance = 0.10 if drawdown_improved_materially and contradiction_improved else 0.05

    criteria = {
        "geo_stress_days_sufficient": int(geo_stress["days_with_geo_signal"]) >= int(geo_stress["min_days_required"]),
        "geo_stress_variance_non_zero": float(geo_stress["variance"]) > 0.0,
        "geo_stress_coverage_sufficient": float(geo_stress["median_coverage_on_geo_days"]) >= MIN_GEO_MEDIAN_COVERAGE,
        "geo_windows_not_stale": not bool(geo_stress["stale_window_detected"]),
        "overall_net_sharpe_after_costs_not_materially_worse": _greater_than_or_within_delta(
            geo_enabled["net_sharpe_after_costs"],
            baseline["net_sharpe_after_costs"],
            tolerance=sharpe_tolerance,
        ),
        "overall_net_cvar_5_not_materially_worse": _greater_than_or_within_delta(
            geo_enabled["net_cvar_5"],
            baseline["net_cvar_5"],
            tolerance=0.001,
        ),
        "overall_max_drawdown_not_materially_worse": _less_than_or_within_ratio(
            geo_enabled["max_drawdown"],
            baseline["max_drawdown"],
            ratio=1.05,
        ),
        "overall_turnover_not_excessive": _less_than_or_within_ratio(
            geo_enabled["annual_turnover"],
            baseline["annual_turnover"],
            ratio=1.10,
        ),
        "top_decile_contradiction_loss_reduced": _less_than_or_equal(
            top_geo["contradiction_loss_rate"],
            top_baseline["contradiction_loss_rate"],
        ),
        "high_geo_blocks_contradiction_loss_reduced": (
            _strictly_less_than(
                block_geo["contradiction_loss_rate"],
                block_baseline["contradiction_loss_rate"],
            )
            if slices["high_geo_blocks"]["date_count"] > 0
            else True
        ),
        "high_geo_blocks_drawdown_not_worse": (
            _less_than_or_equal(
                block_geo["max_drawdown"],
                block_baseline["max_drawdown"],
            )
            if slices["high_geo_blocks"]["date_count"] > 0
            else True
        ),
        "normal_period_sharpe_not_materially_worse": _greater_than_or_within_delta(
            normal_geo["net_sharpe_after_costs"],
            normal_baseline["net_sharpe_after_costs"],
            tolerance=0.05,
        ),
        "trade_removal_not_excessive_or_value_additive": (
            float(filtering_impact["removed_trade_fraction"]) <= MAX_REMOVED_TRADE_FRACTION
            and float(filtering_impact["avg_removed_trade_pnl"]) <= float(filtering_impact["avg_all_trade_pnl"])
        ),
        "bootstrap_delta_sharpe_median_non_negative": float(report["bootstrap"]["delta_sharpe"]["p50"]) >= 0.0,
        "bootstrap_delta_sharpe_p25_within_tolerance": (
            float(report["bootstrap"]["delta_sharpe"]["p25"]) >= -BOOTSTRAP_P25_TOLERANCE
        ),
    }
    return {
        "passed": bool(all(criteria.values())),
        "sharpe_tolerance_used": sharpe_tolerance,
        "criteria": criteria,
    }


def _sorted_daily_results(daily_results: pd.DataFrame) -> pd.DataFrame:
    if daily_results.empty:
        return daily_results.copy()
    sorted_results = daily_results.copy()
    sorted_results["date"] = pd.to_datetime(sorted_results["date"])
    return sorted_results.sort_values("date", kind="mergesort").reset_index(drop=True)


def _sorted_trade_log(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log.empty:
        return trade_log.copy()
    sorted_log = trade_log.copy()
    sorted_log["entry_date"] = pd.to_datetime(sorted_log["entry_date"])
    sorted_log["exit_date"] = pd.to_datetime(sorted_log["exit_date"])
    sort_columns = [column for column in ("entry_date", "ticker", "position_direction", "exit_date") if column in sorted_log.columns]
    return sorted_log.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _portfolio_returns(daily_results: pd.DataFrame) -> pd.Series:
    if "portfolio_return" in daily_results.columns:
        return pd.to_numeric(daily_results["portfolio_return"], errors="coerce").fillna(0.0)
    return pd.to_numeric(daily_results["net_portfolio_return"], errors="coerce").fillna(0.0)


def _net_portfolio_returns(daily_results: pd.DataFrame) -> pd.Series:
    if "net_portfolio_return" in daily_results.columns:
        return pd.to_numeric(daily_results["net_portfolio_return"], errors="coerce").fillna(0.0)
    return _portfolio_returns(daily_results)


def _net_cvar_5(returns: pd.Series) -> float | None:
    cleaned = pd.to_numeric(returns, errors="coerce").dropna()
    if cleaned.empty:
        return None
    threshold = float(cleaned.quantile(0.05))
    tail = cleaned.loc[cleaned <= threshold]
    if tail.empty:
        tail = pd.Series([threshold], dtype=float)
    return float(tail.mean())


def _contradictory_trades(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log.empty or "entry_contradiction" not in trade_log.columns:
        return trade_log.iloc[0:0].copy()
    contradiction = pd.to_numeric(trade_log["entry_contradiction"], errors="coerce").fillna(0.0)
    return trade_log.loc[contradiction > 0.0].copy()


def _contradiction_loss_rate(trade_log: pd.DataFrame) -> float | None:
    contradictory = _contradictory_trades(trade_log)
    if contradictory.empty:
        return 0.0
    losses = pd.to_numeric(contradictory["net_return"], errors="coerce").fillna(0.0) < 0.0
    return float(losses.mean())


def _mean_net_return(trade_log: pd.DataFrame) -> float:
    if trade_log.empty:
        return 0.0
    net_returns = pd.to_numeric(trade_log["net_return"], errors="coerce").dropna()
    if net_returns.empty:
        return 0.0
    return float(net_returns.mean())


def _mean_series(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(numeric.mean())


def _fraction_winners_removed(baseline_trade_log: pd.DataFrame, removed_trade_log: pd.DataFrame) -> float:
    baseline_winners = pd.to_numeric(baseline_trade_log.get("net_return"), errors="coerce").fillna(0.0) > 0.0
    removed_winners = pd.to_numeric(removed_trade_log.get("net_return"), errors="coerce").fillna(0.0) > 0.0
    winner_count = int(baseline_winners.sum())
    if winner_count == 0:
        return 0.0
    return float(int(removed_winners.sum()) / winner_count)


def _trade_entry_keys(trade_log: pd.DataFrame) -> pd.Series:
    if trade_log.empty:
        return pd.Series(dtype=str)
    direction = (
        pd.to_numeric(trade_log["position_direction"], errors="coerce").fillna(0).astype(int)
        if "position_direction" in trade_log.columns
        else trade_log.get("direction", pd.Series(index=trade_log.index, dtype=object)).map({"long": 1, "short": -1}).fillna(0).astype(int)
    )
    entry_dates = pd.to_datetime(trade_log["entry_date"]).dt.strftime("%Y-%m-%d")
    tickers = trade_log["ticker"].astype(str)
    return tickers + "|" + entry_dates + "|" + direction.astype(str)


def _delta(candidate: float | int | None, baseline: float | int | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate) - float(baseline)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def _less_than_or_equal(candidate: float | None, baseline: float | None) -> bool:
    if candidate is None or baseline is None:
        return False
    return float(candidate) <= float(baseline)


def _strictly_less_than(candidate: float | None, baseline: float | None) -> bool:
    if candidate is None or baseline is None:
        return False
    return float(candidate) < float(baseline)


def _less_than_or_within_ratio(candidate: float | None, baseline: float | None, *, ratio: float) -> bool:
    if candidate is None or baseline is None:
        return False
    if float(baseline) == 0.0:
        return float(candidate) == 0.0
    return float(candidate) <= float(baseline) * ratio


def _greater_than_or_within_delta(candidate: float | None, baseline: float | None, *, tolerance: float) -> bool:
    if candidate is None or baseline is None:
        return False
    return float(candidate) >= float(baseline) - tolerance


def _improved_by_ratio(baseline: float | None, candidate: float | None, *, ratio: float) -> bool:
    if baseline is None or candidate is None:
        return False
    return float(candidate) <= float(baseline) * ratio


def _median_coverage_on_geo_days(geo_stress: pd.DataFrame) -> float:
    if geo_stress.empty:
        return 0.0
    geo_days = geo_stress.loc[geo_stress["geo_stress"] > 0.0, "median_coverage_score"]
    if geo_days.empty:
        return 0.0
    return float(pd.to_numeric(geo_days, errors="coerce").fillna(0.0).median())


def _slice_avg_coverage(geo_stress: pd.DataFrame, slice_dates: set[pd.Timestamp]) -> float:
    if geo_stress.empty or not slice_dates:
        return 0.0
    coverage = geo_stress.loc[
        geo_stress["date"].isin(pd.Index(pd.to_datetime(sorted(slice_dates)))),
        "median_coverage_score",
    ]
    if coverage.empty:
        return 0.0
    return float(pd.to_numeric(coverage, errors="coerce").fillna(0.0).mean())


def _slice_median_coverage(geo_stress: pd.DataFrame, slice_dates: set[pd.Timestamp]) -> float:
    if geo_stress.empty or not slice_dates:
        return 0.0
    coverage = geo_stress.loc[
        geo_stress["date"].isin(pd.Index(pd.to_datetime(sorted(slice_dates)))),
        "median_coverage_score",
    ]
    if coverage.empty:
        return 0.0
    return float(pd.to_numeric(coverage, errors="coerce").fillna(0.0).median())


def _geo_stress_available(geo_stress: pd.DataFrame) -> bool:
    if geo_stress.empty:
        return False
    days_with_geo = int((geo_stress["geo_stress"] > 0.0).sum())
    min_days_required = max(1, int(math.ceil(len(geo_stress) * MIN_GEO_DAYS_FRACTION)))
    variance = float(geo_stress["geo_stress"].var(ddof=0))
    median_coverage = _median_coverage_on_geo_days(geo_stress)
    return (
        days_with_geo >= min_days_required
        and variance > 0.0
        and median_coverage >= MIN_GEO_MEDIAN_COVERAGE
    )


def _detect_stale_window(geo_stress: pd.DataFrame) -> bool:
    if geo_stress.empty or "median_data_freshness_minutes" not in geo_stress.columns:
        return False
    stale_days = (
        pd.to_numeric(geo_stress["median_data_freshness_minutes"], errors="coerce").fillna(float("inf"))
        > MAX_STALE_MEDIAN_FRESHNESS_MINUTES
    ) & (pd.to_numeric(geo_stress["geo_stress"], errors="coerce").fillna(0.0) > 0.0)
    streak = 0
    for is_stale in stale_days.tolist():
        if is_stale:
            streak += 1
            if streak >= STALE_WINDOW_CONSECUTIVE_DAYS:
                return True
        else:
            streak = 0
    return False


def _extract_high_geo_blocks(
    geo_stress: pd.Series,
    top_decile_dates: set[pd.Timestamp],
) -> tuple[set[pd.Timestamp], int]:
    if geo_stress.empty or not top_decile_dates:
        return set(), 0

    ordered_dates = list(pd.to_datetime(geo_stress.index))
    qualifying = [date for date in ordered_dates if pd.Timestamp(date) in top_decile_dates]
    blocks: list[list[pd.Timestamp]] = []
    current_block: list[pd.Timestamp] = []
    for date in qualifying:
        ts_date = pd.Timestamp(date)
        if not current_block:
            current_block = [ts_date]
            continue
        previous_index = ordered_dates.index(current_block[-1])
        current_index = ordered_dates.index(ts_date)
        if current_index == previous_index + 1:
            current_block.append(ts_date)
        else:
            if len(current_block) >= HIGH_GEO_BLOCK_MIN_LENGTH:
                blocks.append(current_block.copy())
            current_block = [ts_date]
    if len(current_block) >= HIGH_GEO_BLOCK_MIN_LENGTH:
        blocks.append(current_block.copy())

    flattened = {date for block in blocks for date in block}
    return flattened, len(blocks)


def _build_bootstrap_report(
    baseline_daily_results: pd.DataFrame,
    geo_enabled_daily_results: pd.DataFrame,
    *,
    annualization_days: int,
    random_seed: int,
) -> dict[str, object]:
    baseline_returns = _net_portfolio_returns(_sorted_daily_results(baseline_daily_results)).to_numpy(dtype=float)
    geo_returns = _net_portfolio_returns(_sorted_daily_results(geo_enabled_daily_results)).to_numpy(dtype=float)
    sample_count = min(len(baseline_returns), len(geo_returns))
    if sample_count == 0:
        return {
            "random_seed": random_seed,
            "samples": 0,
            "sample_indices": [],
            "delta_sharpe": {"mean": None, "p05": None, "p25": None, "p50": None, "p95": None},
            "delta_max_drawdown": {"mean": None, "p05": None, "p25": None, "p50": None, "p95": None},
        }

    baseline_returns = baseline_returns[:sample_count]
    geo_returns = geo_returns[:sample_count]
    rng = np.random.default_rng(random_seed)
    delta_sharpes: list[float] = []
    delta_drawdowns: list[float] = []
    sample_indices: list[list[int]] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample_idx = rng.integers(0, sample_count, size=sample_count)
        sample_indices.append(sample_idx.astype(int).tolist())
        baseline_sample = pd.Series(baseline_returns[sample_idx])
        geo_sample = pd.Series(geo_returns[sample_idx])
        baseline_sharpe = _annualized_sharpe(baseline_sample, annualization_days) or 0.0
        geo_sharpe = _annualized_sharpe(geo_sample, annualization_days) or 0.0
        baseline_drawdown = _max_drawdown(baseline_sample) or 0.0
        geo_drawdown = _max_drawdown(geo_sample) or 0.0
        delta_sharpes.append(float(geo_sharpe - baseline_sharpe))
        delta_drawdowns.append(float(geo_drawdown - baseline_drawdown))

    return {
        "random_seed": random_seed,
        "samples": BOOTSTRAP_SAMPLES,
        "sample_indices": sample_indices,
        "delta_sharpe": _distribution_summary(delta_sharpes),
        "delta_max_drawdown": _distribution_summary(delta_drawdowns),
    }


def _distribution_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p05": None, "p25": None, "p50": None, "p95": None}
    series = pd.Series(values, dtype=float)
    return {
        "mean": float(series.mean()),
        "p05": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p95": float(series.quantile(0.95)),
    }


def _validate_overlay_signal_contract(daily_signals: pd.DataFrame | None) -> None:
    if daily_signals is None or daily_signals.empty:
        raise ValueError("overlay signal contract requires non-empty daily_signals")
    normalized_signals = _sorted_signal_frame(daily_signals)
    missing = REQUIRED_OVERLAY_SIGNAL_COLUMNS.difference(normalized_signals.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"overlay signal contract missing required columns: {missing_list}")
    if normalized_signals.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("overlay signal contract requires unique normalized (date, ticker) keys")
    geo_net = pd.to_numeric(normalized_signals["geo_net_score"], errors="coerce").fillna(0.0)
    if ((geo_net < -1.0) | (geo_net > 1.0)).any():
        raise ValueError("overlay signal contract requires geo_net_score in [-1, 1]")
    coverage = pd.to_numeric(normalized_signals["coverage_score"], errors="coerce").fillna(0.0)
    if ((coverage < 0.0) | (coverage > 1.0)).any():
        raise ValueError("overlay signal contract requires coverage_score in [0, 1]")


def _build_input_parity(
    baseline_daily_signals: pd.DataFrame | None,
    geo_enabled_daily_signals: pd.DataFrame | None,
) -> dict[str, object]:
    baseline = _sorted_signal_frame(baseline_daily_signals)
    geo_enabled = _sorted_signal_frame(geo_enabled_daily_signals)
    market_columns = [
        column for column in PRICE_RETURN_PARITY_COLUMNS
        if column in baseline.columns and column in geo_enabled.columns
    ]
    tradability_columns = [
        column for column in TRADABILITY_PARITY_COLUMNS
        if column in baseline.columns and column in geo_enabled.columns
    ]

    market_hashes_match = True
    tradability_hashes_match = True
    market_hash_digest = ""
    tradability_hash_digest = ""

    if market_columns:
        _assert_parity_values_match(baseline, geo_enabled, market_columns, label="Market data parity")
        baseline_market_hashes = _per_day_signal_hashes(baseline, market_columns)
        geo_market_hashes = _per_day_signal_hashes(geo_enabled, market_columns)
        market_hashes_match = baseline_market_hashes == geo_market_hashes
        market_hash_digest = _hash_payload(baseline_market_hashes)
        if not market_hashes_match:
            raise ValueError("Market data parity failed: baseline and geo-enabled runs used different price/return inputs")

    if tradability_columns:
        _assert_parity_values_match(baseline, geo_enabled, tradability_columns, label="Tradability parity")
        baseline_tradability_hashes = _per_day_signal_hashes(baseline, tradability_columns)
        geo_tradability_hashes = _per_day_signal_hashes(geo_enabled, tradability_columns)
        tradability_hashes_match = baseline_tradability_hashes == geo_tradability_hashes
        tradability_hash_digest = _hash_payload(baseline_tradability_hashes)
        if not tradability_hashes_match:
            raise ValueError("Tradability parity failed: baseline and geo-enabled runs used different tradability inputs")

    return {
        "passed": bool(market_hashes_match and tradability_hashes_match),
        "market_data_columns_checked": market_columns,
        "market_data_hash_digest": market_hash_digest,
        "market_data_hashes_match": market_hashes_match,
        "tradability_columns_checked": tradability_columns,
        "tradability_hash_digest": tradability_hash_digest,
        "tradability_hashes_match": tradability_hashes_match,
    }


def _validate_run_parity(
    baseline_result: TrendStrategyBacktest,
    geo_enabled_result: TrendStrategyBacktest,
) -> None:
    baseline_signals = _sorted_signal_frame(baseline_result.daily_signals)
    geo_signals = _sorted_signal_frame(geo_enabled_result.daily_signals)
    baseline_pairs = baseline_signals.loc[:, ["date", "ticker"]].reset_index(drop=True)
    geo_pairs = geo_signals.loc[:, ["date", "ticker"]].reset_index(drop=True)
    if not baseline_pairs.equals(geo_pairs):
        raise ValueError("Universe parity failed: baseline and geo-enabled runs used different (date, ticker) pairs")

    baseline_counts = baseline_signals.groupby("date")["ticker"].count()
    geo_counts = geo_signals.groupby("date")["ticker"].count()
    if not baseline_counts.equals(geo_counts):
        raise ValueError("Universe parity failed: baseline and geo-enabled runs used different per-day asset counts")

    baseline_dates = _sorted_daily_results(baseline_result.daily_results)["date"].reset_index(drop=True)
    geo_dates = _sorted_daily_results(geo_enabled_result.daily_results)["date"].reset_index(drop=True)
    if not baseline_dates.equals(geo_dates):
        raise ValueError("Calendar parity failed: baseline and geo-enabled runs used different trading dates")


def _sorted_signal_frame(daily_signals: pd.DataFrame | None) -> pd.DataFrame:
    if daily_signals is None or daily_signals.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    signal_frame = daily_signals.copy()
    signal_frame["date"] = pd.to_datetime(signal_frame["date"])
    signal_frame["ticker"] = _normalize_identifier_series(signal_frame["ticker"])
    if signal_frame.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("overlay signal contract requires unique normalized (date, ticker) keys")
    return signal_frame.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)


def _validate_trade_log_contradiction_contract(trade_log: pd.DataFrame) -> None:
    contradiction_columns = {column for column in trade_log.columns if "contradiction" in column}
    if contradiction_columns.difference({"entry_contradiction"}):
        invalid = ", ".join(sorted(contradiction_columns.difference({"entry_contradiction"})))
        raise ValueError(f"evaluation harness requires entry-time contradiction only; found: {invalid}")
    if not trade_log.empty and "entry_contradiction" not in trade_log.columns:
        raise ValueError("evaluation harness requires entry-time contradiction only via 'entry_contradiction'")


def _build_metadata(
    config: PipelineConfig,
    baseline_result: TrendStrategyBacktest,
    geo_enabled_result: TrendStrategyBacktest,
    *,
    random_seed: int,
) -> dict[str, object]:
    daily_dates = pd.concat(
        [
            _sorted_daily_results(baseline_result.daily_results)["date"],
            _sorted_daily_results(geo_enabled_result.daily_results)["date"],
        ],
        axis=0,
    ).dropna()
    date_range = {
        "start_date": pd.Timestamp(daily_dates.min()).date().isoformat() if not daily_dates.empty else None,
        "end_date": pd.Timestamp(daily_dates.max()).date().isoformat() if not daily_dates.empty else None,
    }
    return {
        "config_hash": _hash_payload({"phase2": _normalize_for_hash(asdict(config.phase2)), "geo": _normalize_for_hash(asdict(config.geo))}),
        "cost_model_hash": _cost_model_hash_from_phase2(config),
        "execution_model_hash": _execution_model_hash(),
        "code_version": _resolve_code_version(),
        "data_range": date_range,
        "universe_hash": _hash_payload(sorted(config.tickers)),
        "random_seed": random_seed,
    }


def _cost_model_hash_from_phase2(config: PipelineConfig) -> str:
    return _hash_payload(
        {
            "commission_bps": config.phase2.commission_bps,
            "bid_ask_bps": config.phase2.bid_ask_bps,
            "market_impact_bps": config.phase2.market_impact_bps,
            "slippage_bps": config.phase2.slippage_bps,
        }
    )


def _execution_model_hash() -> str:
    return _hash_payload(
        {
            "engine": "run_walk_forward_backtest_v1",
            "daily_transaction_cost": "turnover * (commission_bps + bid_ask_bps + market_impact_bps + slippage_bps)",
            "trade_exit_cost": "2.0 * (commission_bps + bid_ask_bps + market_impact_bps + slippage_bps)",
        }
    )


def _normalize_for_hash(payload: object) -> object:
    if isinstance(payload, dict):
        return {key: _normalize_for_hash(value) for key, value in sorted(payload.items())}
    if isinstance(payload, (list, tuple)):
        return [_normalize_for_hash(value) for value in payload]
    if isinstance(payload, Path):
        return str(payload)
    return payload


def _hash_payload(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _resolve_code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNKNOWN"
    return result.stdout.strip() or "UNKNOWN"


def _validate_overlay_alignment(
    daily_signals: pd.DataFrame | None,
    geo_snapshot: pd.DataFrame,
) -> dict[str, object]:
    if daily_signals is None or daily_signals.empty or geo_snapshot.empty:
        return {"performed": False, "passed": False, "checked_date": "", "checked_rows": 0}
    if "date" not in daily_signals.columns or "trade_date" not in geo_snapshot.columns or "asset" not in geo_snapshot.columns:
        return {"performed": False, "passed": False, "checked_date": "", "checked_rows": 0}
    signal_frame = _sorted_signal_frame(daily_signals).copy()
    signal_frame["date"] = pd.to_datetime(signal_frame["date"]).dt.date
    snapshot_frame = _sorted_snapshot_frame(geo_snapshot).copy()
    snapshot_frame["trade_date"] = pd.to_datetime(snapshot_frame["trade_date"]).dt.date
    common_dates = sorted(set(signal_frame["date"]).intersection(set(snapshot_frame["trade_date"])))
    if not common_dates:
        return {"performed": False, "passed": False, "checked_date": "", "checked_rows": 0}

    merged = signal_frame.merge(
        snapshot_frame,
        how="left",
        left_on=["date", "ticker"],
        right_on=["trade_date", "asset"],
        suffixes=("_signal", "_snapshot"),
        validate="1:1",
    )
    if len(merged) != len(signal_frame):
        raise ValueError("alignment merge changed row count")
    if merged.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("alignment merge violated 1:1 cardinality")

    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    sample_size = min(5, len(common_dates))
    checked_dates = sorted(rng.choice(common_dates, size=sample_size, replace=False).tolist()) if sample_size else []
    sample_rows = merged.loc[merged["date"].isin(checked_dates)].copy()
    sample_rows["geo_net_score_signal"] = pd.to_numeric(sample_rows["geo_net_score_signal"], errors="coerce").fillna(0.0)
    sample_rows["geo_net_score_snapshot"] = pd.to_numeric(sample_rows["geo_net_score_snapshot"], errors="coerce").fillna(0.0)
    passed = bool(
        not sample_rows.empty
        and (sample_rows["date"] == sample_rows["trade_date"]).all()
        and (sample_rows["geo_net_score_signal"] == sample_rows["geo_net_score_snapshot"]).all()
    )
    return {
        "performed": True,
        "passed": passed,
        "checked_date": str(checked_dates[0]) if checked_dates else "",
        "checked_rows": int(len(sample_rows)),
    }


def _validate_snapshot_timeliness(
    geo_snapshot: pd.DataFrame,
    *,
    cutoff_time_et: str,
) -> dict[str, object]:
    if geo_snapshot.empty:
        return {"performed": False, "enforceable": False, "passed": False, "checked_rows": 0, "violations": 0}
    if "trade_date" not in geo_snapshot.columns:
        return {"performed": False, "enforceable": False, "passed": False, "checked_rows": int(len(geo_snapshot)), "violations": 0}
    if "available_at" not in geo_snapshot.columns:
        return {"performed": True, "enforceable": False, "passed": False, "checked_rows": int(len(geo_snapshot)), "violations": 0}

    hours, minutes = [int(part) for part in cutoff_time_et.split(":", maxsplit=1)]
    snapshot_frame = geo_snapshot.copy()
    snapshot_frame["trade_date"] = pd.to_datetime(snapshot_frame["trade_date"]).dt.date
    snapshot_frame["available_at"] = pd.to_datetime(snapshot_frame["available_at"], errors="coerce")
    available_at = snapshot_frame["available_at"]
    if getattr(available_at.dt, "tz", None) is not None:
        cutoff = pd.to_datetime(snapshot_frame["trade_date"]).dt.tz_localize(
            ZoneInfo("America/New_York")
        ) + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")
        cutoff = cutoff.dt.tz_convert(available_at.dt.tz)
    else:
        cutoff = pd.to_datetime(snapshot_frame["trade_date"]) + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")

    violations = int((available_at > cutoff).fillna(False).sum())
    return {
        "performed": True,
        "enforceable": True,
        "passed": violations == 0,
        "checked_rows": int(len(snapshot_frame)),
        "violations": violations,
    }


def _build_curves(
    baseline_daily_results: pd.DataFrame,
    geo_enabled_daily_results: pd.DataFrame,
) -> dict[str, dict[str, list[float]]]:
    return {
        "baseline": _curve_payload(baseline_daily_results),
        "geo_enabled": _curve_payload(geo_enabled_daily_results),
    }


def _curve_payload(daily_results: pd.DataFrame) -> dict[str, list[float]]:
    sorted_results = _sorted_daily_results(daily_results)
    returns = _net_portfolio_returns(sorted_results)
    if returns.empty:
        return {"equity": [], "drawdown": []}
    equity = (1.0 + returns).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return {
        "equity": [float(value) for value in equity.tolist()],
        "drawdown": [float(value) for value in drawdown.tolist()],
    }


def _build_geo_stress_series(
    geo_stress: pd.DataFrame,
    high_geo_block_dates: set[pd.Timestamp],
) -> list[dict[str, object]]:
    if geo_stress.empty:
        return []
    block_dates = {pd.Timestamp(value) for value in high_geo_block_dates}
    rows: list[dict[str, object]] = []
    for row in geo_stress.itertuples(index=False):
        date_value = pd.Timestamp(row.date)
        rows.append(
            {
                "date": date_value.date().isoformat(),
                "geo_stress": float(row.geo_stress),
                "high_geo_block": date_value in block_dates,
            }
        )
    return rows


def _build_per_asset_delta_pnl(
    baseline_trade_log: pd.DataFrame,
    geo_enabled_trade_log: pd.DataFrame,
) -> dict[str, list[dict[str, object]]]:
    baseline = _aggregate_asset_pnl(baseline_trade_log).rename("baseline_net_pnl")
    geo_enabled = _aggregate_asset_pnl(geo_enabled_trade_log).rename("geo_enabled_net_pnl")
    joined = pd.concat([baseline, geo_enabled], axis=1).fillna(0.0)
    joined["delta_net_pnl"] = joined["geo_enabled_net_pnl"] - joined["baseline_net_pnl"]
    ranked = joined.reset_index(names="ticker").sort_values("delta_net_pnl", ascending=False, kind="mergesort")
    records = [
        {
            "ticker": str(row["ticker"]),
            "baseline_net_pnl": float(row["baseline_net_pnl"]),
            "geo_enabled_net_pnl": float(row["geo_enabled_net_pnl"]),
            "delta_net_pnl": float(row["delta_net_pnl"]),
        }
        for _, row in ranked.iterrows()
    ]
    return {
        "top_positive": records[:5],
        "top_negative": list(reversed(records[-5:])) if records else [],
    }


def _build_data_sanity(
    baseline_daily_signals: pd.DataFrame | None,
    geo_enabled_daily_signals: pd.DataFrame | None,
) -> dict[str, object]:
    if geo_enabled_daily_signals is None or geo_enabled_daily_signals.empty:
        return {
            "duplicate_date_ticker": False,
            "geo_net_score_bounds_ok": False,
            "coverage_score_bounds_ok": False,
            "coverage_degenerate": True,
            "extreme_return_days": [],
            "extreme_return_treated_identically": False,
        }
    frame = _sorted_signal_frame(geo_enabled_daily_signals)
    baseline_frame = _sorted_signal_frame(baseline_daily_signals)
    geo_net = pd.to_numeric(frame["geo_net_score"], errors="coerce").fillna(0.0)
    coverage = pd.to_numeric(frame["coverage_score"], errors="coerce").fillna(0.0)
    extreme_days = _extreme_return_days(frame)
    return {
        "duplicate_date_ticker": bool(frame.duplicated(subset=["date", "ticker"]).any()),
        "geo_net_score_bounds_ok": bool((geo_net.between(-1.0, 1.0)).all()),
        "coverage_score_bounds_ok": bool((coverage.between(0.0, 1.0)).all()),
        "coverage_degenerate": bool(coverage.nunique(dropna=False) <= 1 and (coverage.iloc[0] in (0.0, 1.0))),
        "extreme_return_days": [value.date().isoformat() for value in extreme_days],
        "extreme_return_treated_identically": _extreme_return_days(baseline_frame) == extreme_days,
    }


def _per_day_signal_hashes(
    signal_frame: pd.DataFrame,
    columns: list[str],
) -> dict[str, str]:
    if signal_frame.empty or not columns:
        return {}
    hashes: dict[str, str] = {}
    for date_value, date_frame in signal_frame.groupby("date", sort=True):
        ordered = date_frame.loc[:, ["ticker", *columns]].copy()
        ordered = ordered.sort_values("ticker", kind="mergesort").reset_index(drop=True)
        for column in columns:
            if pd.api.types.is_numeric_dtype(ordered[column]):
                ordered[column] = pd.to_numeric(ordered[column], errors="coerce").round(12)
        hashes[pd.Timestamp(date_value).date().isoformat()] = _hash_payload(_normalize_for_hash(ordered.to_dict(orient="records")))
    return hashes


def _assert_parity_values_match(
    baseline_signal_frame: pd.DataFrame,
    geo_enabled_signal_frame: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> None:
    if not columns:
        return
    baseline = baseline_signal_frame.loc[:, ["date", "ticker", *columns]].reset_index(drop=True)
    geo_enabled = geo_enabled_signal_frame.loc[:, ["date", "ticker", *columns]].reset_index(drop=True)
    if len(baseline) != len(geo_enabled):
        raise ValueError(f"{label} failed: row count mismatch")
    for column in columns:
        baseline_values = baseline[column]
        geo_values = geo_enabled[column]
        if pd.api.types.is_numeric_dtype(baseline_values) or pd.api.types.is_numeric_dtype(geo_values):
            baseline_numeric = pd.to_numeric(baseline_values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            geo_numeric = pd.to_numeric(geo_values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if not np.isclose(
                baseline_numeric,
                geo_numeric,
                rtol=PARITY_NUMERIC_TOLERANCE,
                atol=PARITY_NUMERIC_TOLERANCE,
            ).all():
                raise ValueError(f"{label} failed: {column} diverged beyond tolerance")
            continue
        if not baseline_values.astype(str).equals(geo_values.astype(str)):
            raise ValueError(f"{label} failed: {column} diverged")


def _normalize_identifier_series(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.upper()
    if (normalized == "").any():
        raise ValueError("overlay signal contract requires non-empty normalized identifiers")
    return normalized


def _sorted_snapshot_frame(geo_snapshot: pd.DataFrame) -> pd.DataFrame:
    snapshot_frame = geo_snapshot.copy()
    snapshot_frame["trade_date"] = pd.to_datetime(snapshot_frame["trade_date"])
    snapshot_frame["asset"] = _normalize_identifier_series(snapshot_frame["asset"])
    if snapshot_frame.duplicated(subset=["trade_date", "asset"]).any():
        raise ValueError("duplicate normalized snapshot keys")
    return snapshot_frame.sort_values(["trade_date", "asset"], kind="mergesort").reset_index(drop=True)


def _extreme_return_days(signal_frame: pd.DataFrame) -> set[pd.Timestamp]:
    if signal_frame.empty:
        return set()
    extreme_mask = pd.Series(False, index=signal_frame.index)
    for column in ("current_return", "forward_return"):
        if column in signal_frame.columns:
            values = pd.to_numeric(signal_frame[column], errors="coerce").fillna(0.0)
            extreme_mask = extreme_mask | (values.abs() > EXTREME_RETURN_ABS_THRESHOLD)
    if not extreme_mask.any():
        return set()
    return {pd.Timestamp(value) for value in pd.to_datetime(signal_frame.loc[extreme_mask, "date"]).unique()}


def _aggregate_asset_pnl(trade_log: pd.DataFrame) -> pd.Series:
    sorted_log = _sorted_trade_log(trade_log)
    if sorted_log.empty:
        return pd.Series(dtype=float)
    return (
        sorted_log.groupby("ticker")["net_return"]
        .apply(lambda values: float(pd.to_numeric(values, errors="coerce").fillna(0.0).sum()))
        .sort_index()
    )


def _schema_hash(report: dict[str, object]) -> str:
    return _hash_payload(_schema_signature(report))


def _schema_signature(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _schema_signature(nested)
            for key, nested in sorted(value.items())
            if key != "report_schema_hash"
        }
    if isinstance(value, list):
        if not value:
            return {"type": "list", "items": "empty"}
        return {"type": "list", "items": _schema_signature(value[0])}
    return type(value).__name__
