from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest import _annual_turnover, _annualized_sharpe, _max_drawdown, run_walk_forward_backtest, scale_signals_to_risk_budget
from ..config_loader import Phase2Config, PipelineConfig, load_config
from ..graph_engine import apply_signal_rules
from ..trend_strategy import TrendStrategyBacktest, load_phase2_baseline_backtest
from .regime_labels import (
    GEO_STRESS_REGIME,
    NORMAL_REGIME,
    PLACEBO_LAG_DAYS,
    PLACEBO_RANDOM_SEED,
    REGIME_ORDER,
    STRUCTURAL_GEO_REGIME,
    GeoRegimeDefinition,
    build_geo_regime_labels,
    build_placebo_regime_frames,
    load_geo_snapshot_frame,
)

REPORT_VERSION = "geo_regime_stage1_v1"
BOOTSTRAP_RANDOM_SEED = 1729
BOOTSTRAP_SAMPLES = 250
BASELINE_DAILY_ARTIFACT = "phase5_strategy_a_daily.csv"
BASELINE_TRADE_ARTIFACT = "phase2_trade_log.csv"
DEFAULT_OUTPUT_DIR_NAME = "geo_regime_experiment_stage1"
DEFAULT_SNAPSHOT_FILE = "geo_feature_snapshot_minimal.csv"
DAILY_PARITY_COLUMNS = [
    "date",
    "gross_portfolio_return",
    "net_portfolio_return",
    "portfolio_return",
    "gross_exposure",
    "turnover",
    "transaction_cost",
]
TRADE_PARITY_COLUMNS = [
    "ticker",
    "entry_date",
    "exit_date",
    "position_direction",
    "entry_zscore",
    "exit_zscore",
    "holding_days",
    "entry_weight",
    "gross_return",
    "net_return",
]


@dataclass(frozen=True)
class GeoRegimePolicy:
    name: str
    stress_threshold_multiplier: float = 1.0
    structural_threshold_multiplier: float = 1.0
    stress_position_scale: float = 1.0
    structural_position_scale: float = 1.0

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class GeoRegimePolicyRun:
    variant: str
    label_source: str
    daily_results: pd.DataFrame
    trade_log: pd.DataFrame
    daily_signals: pd.DataFrame
    metrics: dict[str, object]


@dataclass(frozen=True)
class GeoRegimeExperimentResult:
    report: dict[str, object]
    output_path: Path


DEFAULT_POLICIES: tuple[GeoRegimePolicy, ...] = (
    GeoRegimePolicy(
        name="threshold_only",
        stress_threshold_multiplier=1.10,
        structural_threshold_multiplier=1.25,
    ),
    GeoRegimePolicy(
        name="sizing_only",
        stress_position_scale=0.75,
        structural_position_scale=0.50,
    ),
)


def run_geo_regime_stage1_experiment(
    config_path: str | Path,
    *,
    snapshot_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> GeoRegimeExperimentResult:
    config = load_config(config_path)
    active_output_dir = Path(output_dir) if output_dir is not None else config.paths.processed_dir / DEFAULT_OUTPUT_DIR_NAME
    active_output_dir.mkdir(parents=True, exist_ok=True)

    baseline_config = replace(config, geo=replace(config.geo, enabled=False))
    baseline_result = load_phase2_baseline_backtest(baseline_config)
    baseline_parity = _assert_baseline_parity(config, baseline_result)

    active_snapshot_path = Path(snapshot_path) if snapshot_path is not None else config.paths.processed_dir / DEFAULT_SNAPSHOT_FILE
    snapshot_frame = load_geo_snapshot_frame(active_snapshot_path)
    regime_definition = GeoRegimeDefinition()
    regime_frame = build_geo_regime_labels(
        snapshot_frame,
        min_asset_count=len(config.tickers),
        definition=regime_definition,
    )

    baseline_by_regime = _build_baseline_by_regime_frame(
        baseline_result=baseline_result,
        regime_frame=regime_frame,
        annualization_days=config.phase2.annualization_days,
    )
    baseline_lookup = _baseline_regime_lookup(baseline_by_regime)
    baseline_metrics = _build_policy_metrics_row(
        variant="baseline",
        label_source="REAL",
        daily_results=baseline_result.daily_results,
        trade_log=baseline_result.trade_log,
        baseline_daily_results=baseline_result.daily_results,
        baseline_trade_log=baseline_result.trade_log,
        regime_frame=regime_frame,
        baseline_lookup=baseline_lookup,
        annualization_days=config.phase2.annualization_days,
    )
    baseline_run = GeoRegimePolicyRun(
        variant="baseline",
        label_source="REAL",
        daily_results=baseline_result.daily_results.copy(),
        trade_log=baseline_result.trade_log.copy(),
        daily_signals=baseline_result.daily_signals.copy() if baseline_result.daily_signals is not None else pd.DataFrame(),
        metrics=baseline_metrics,
    )

    real_policy_runs = _evaluate_real_policy_variants(
        baseline_result=baseline_result,
        regime_frame=regime_frame,
        phase2_config=config.phase2,
        annualization_days=config.phase2.annualization_days,
        baseline_lookup=baseline_lookup,
    )
    policy_runs = [baseline_run, *real_policy_runs]
    best_real_variant = _select_best_variant(real_policy_runs)

    placebo_runs = _evaluate_placebo_variants(
        baseline_result=baseline_result,
        regime_frame=regime_frame,
        selected_variant=best_real_variant,
        phase2_config=config.phase2,
        annualization_days=config.phase2.annualization_days,
        baseline_lookup=baseline_lookup,
    )
    bootstrap = _build_bootstrap_report(
        baseline_daily_results=baseline_result.daily_results,
        variant_daily_results=best_real_variant.daily_results if best_real_variant is not None else baseline_result.daily_results,
        annualization_days=config.phase2.annualization_days,
        random_seed=BOOTSTRAP_RANDOM_SEED,
    )
    conclusions = _build_conclusions(
        baseline_by_regime=baseline_by_regime,
        best_real_variant=best_real_variant,
        placebo_runs=placebo_runs,
        bootstrap=bootstrap,
    )

    labels_path = active_output_dir / "geo_regime_daily_labels.csv"
    baseline_slice_path = active_output_dir / "geo_regime_baseline_by_regime.csv"
    policy_results_path = active_output_dir / "geo_regime_policy_results.csv"
    placebo_results_path = active_output_dir / "geo_regime_placebo_results.csv"
    output_path = active_output_dir / "geo_regime_experiment.json"

    regime_frame.to_csv(labels_path, index=False)
    baseline_by_regime.to_csv(baseline_slice_path, index=False)
    pd.DataFrame([run.metrics for run in policy_runs]).to_csv(policy_results_path, index=False)
    pd.DataFrame([run.metrics for run in placebo_runs]).to_csv(placebo_results_path, index=False)

    report = {
        "report_version": REPORT_VERSION,
        "config_path": str(Path(config_path)),
        "snapshot_path": str(active_snapshot_path),
        "output_dir": str(active_output_dir),
        "baseline_parity": baseline_parity,
        "regime_definition": {
            **regime_definition.to_dict(),
            "min_asset_count": int(len(config.tickers)),
            "placebo_lag_days": PLACEBO_LAG_DAYS,
            "placebo_random_seed": PLACEBO_RANDOM_SEED,
        },
        "regime_counts": _build_regime_counts(regime_frame, baseline_result),
        "baseline_by_regime": baseline_by_regime.to_dict(orient="records"),
        "policy_results": [run.metrics for run in policy_runs],
        "best_real_variant": (
            {
                "variant": best_real_variant.variant,
                "label_source": best_real_variant.label_source,
                "metrics": best_real_variant.metrics,
            }
            if best_real_variant is not None
            else None
        ),
        "placebo_results": [run.metrics for run in placebo_runs],
        "bootstrap": bootstrap,
        "conclusions": conclusions,
        "artifacts": {
            "labels": str(labels_path),
            "baseline_by_regime": str(baseline_slice_path),
            "policy_results": str(policy_results_path),
            "placebo_results": str(placebo_results_path),
            "report": str(output_path),
        },
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return GeoRegimeExperimentResult(report=report, output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 1 geo regime-labeler research experiment.")
    parser.add_argument("--config", default="config/phase8.yaml", help="Pipeline config path.")
    parser.add_argument(
        "--snapshot-path",
        default=None,
        help="Optional geo snapshot history CSV. Defaults to data/processed/geo_feature_snapshot_minimal.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for Stage 1 regime-labeler artifacts. Defaults to data/processed/geo_regime_experiment_stage1.",
    )
    args = parser.parse_args()
    result = run_geo_regime_stage1_experiment(
        args.config,
        snapshot_path=args.snapshot_path,
        output_dir=args.output_dir,
    )
    print(str(result.output_path))


def _assert_baseline_parity(
    config: PipelineConfig,
    baseline_result: TrendStrategyBacktest,
) -> dict[str, object]:
    daily_artifact_path = config.paths.processed_dir / BASELINE_DAILY_ARTIFACT
    trade_artifact_path = config.paths.processed_dir / BASELINE_TRADE_ARTIFACT
    if not daily_artifact_path.exists():
        raise ValueError(f"Baseline parity hard-failed: missing Strategy A daily artifact at '{daily_artifact_path}'.")
    if not trade_artifact_path.exists():
        raise ValueError(f"Baseline parity hard-failed: missing Strategy A trade artifact at '{trade_artifact_path}'.")

    actual_daily = baseline_result.daily_results.loc[:, DAILY_PARITY_COLUMNS].copy()
    expected_daily = pd.read_csv(daily_artifact_path, parse_dates=["date"]).loc[:, DAILY_PARITY_COLUMNS].copy()
    actual_daily = actual_daily.sort_values("date", kind="mergesort").reset_index(drop=True)
    expected_daily = expected_daily.sort_values("date", kind="mergesort").reset_index(drop=True)

    actual_trade = baseline_result.trade_log.loc[:, TRADE_PARITY_COLUMNS].copy()
    expected_trade = pd.read_csv(
        trade_artifact_path,
        parse_dates=["entry_date", "exit_date"],
    ).loc[:, TRADE_PARITY_COLUMNS].copy()
    trade_sort = ["entry_date", "exit_date", "ticker", "position_direction"]
    actual_trade = actual_trade.sort_values(trade_sort, kind="mergesort").reset_index(drop=True)
    expected_trade = expected_trade.sort_values(trade_sort, kind="mergesort").reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            actual_daily,
            expected_daily,
            check_dtype=False,
            check_like=False,
            atol=1e-10,
            rtol=0.0,
        )
        pd.testing.assert_frame_equal(
            actual_trade,
            expected_trade,
            check_dtype=False,
            check_like=False,
            atol=1e-9,
            rtol=0.0,
        )
    except AssertionError as exc:
        raise ValueError(
            "Baseline parity hard-failed: rebuilt Strategy A baseline does not match persisted baseline artifacts."
        ) from exc

    return {
        "passed": True,
        "daily_artifact_path": str(daily_artifact_path),
        "trade_artifact_path": str(trade_artifact_path),
        "checked_daily_rows": int(len(expected_daily)),
        "checked_trade_rows": int(len(expected_trade)),
    }


def _build_regime_counts(
    regime_frame: pd.DataFrame,
    baseline_result: TrendStrategyBacktest,
) -> dict[str, object]:
    labeled_days = regime_frame["geo_regime"].value_counts().to_dict()
    signal_frame = baseline_result.daily_signals.copy() if baseline_result.daily_signals is not None else pd.DataFrame()
    signal_counts = {label: 0 for label in REGIME_ORDER}
    trade_counts = {label: 0 for label in REGIME_ORDER}
    if not signal_frame.empty:
        active_signal_dates = (
            signal_frame.assign(date=pd.to_datetime(signal_frame["date"]).dt.normalize())
            .groupby("date")["signal_direction"]
            .apply(lambda series: bool((pd.to_numeric(series, errors="coerce").fillna(0.0) != 0).any()))
            .reset_index(name="active_signal_day")
        )
        merged = active_signal_dates.merge(
            regime_frame.loc[:, ["date", "geo_regime"]],
            on="date",
            how="left",
        ).fillna({"geo_regime": NORMAL_REGIME})
        signal_counts = merged.loc[merged["active_signal_day"]].groupby("geo_regime")["date"].count().to_dict()
    if not baseline_result.trade_log.empty:
        trades = baseline_result.trade_log.copy()
        trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.normalize()
        merged_trades = trades.merge(
            regime_frame.loc[:, ["date", "geo_regime"]].rename(columns={"date": "entry_date"}),
            on="entry_date",
            how="left",
        ).fillna({"geo_regime": NORMAL_REGIME})
        trade_counts = merged_trades.groupby("geo_regime")["entry_date"].count().to_dict()
    return {
        "days": {label: int(labeled_days.get(label, 0)) for label in REGIME_ORDER},
        "active_signal_days": {label: int(signal_counts.get(label, 0)) for label in REGIME_ORDER},
        "entry_trade_counts": {label: int(trade_counts.get(label, 0)) for label in REGIME_ORDER},
    }


def _build_baseline_by_regime_frame(
    *,
    baseline_result: TrendStrategyBacktest,
    regime_frame: pd.DataFrame,
    annualization_days: int,
) -> pd.DataFrame:
    daily_results = _attach_regime_labels(baseline_result.daily_results, regime_frame, date_column="date")
    trade_log = _attach_regime_labels(baseline_result.trade_log, regime_frame, date_column="entry_date")
    signal_day_frame = _build_active_signal_day_frame(
        baseline_result.daily_signals.copy() if baseline_result.daily_signals is not None else pd.DataFrame(),
        regime_frame,
    )
    worst_day_count = min(25, len(daily_results))
    worst_dates = set(
        pd.to_datetime(
            daily_results.nsmallest(worst_day_count, "net_portfolio_return")["date"]
        ).dt.normalize()
    )
    total_day_count = int(len(daily_results))
    worst_day_denominator = max(1, worst_day_count)

    rows: list[dict[str, object]] = []
    for regime_name in REGIME_ORDER:
        regime_daily = daily_results.loc[daily_results["geo_regime"] == regime_name].copy()
        regime_trades = trade_log.loc[trade_log["geo_regime"] == regime_name].copy()
        signal_day_count = int(
            signal_day_frame.loc[
                (signal_day_frame["geo_regime"] == regime_name) & signal_day_frame["active_signal_day"],
                "date",
            ].nunique()
        )
        portfolio_metrics = _portfolio_trade_metrics(
            daily_results=regime_daily,
            trade_log=regime_trades,
            annualization_days=annualization_days,
        )
        regime_day_count = int(len(regime_daily))
        worst_day_hits = int(
            pd.to_datetime(regime_daily["date"]).dt.normalize().isin(worst_dates).sum()
        )
        day_share = float(regime_day_count / total_day_count) if total_day_count else 0.0
        worst_share = float(worst_day_hits / worst_day_denominator) if worst_day_denominator else 0.0
        rows.append(
            {
                "regime": regime_name,
                "date_count": regime_day_count,
                "active_signal_day_count": signal_day_count,
                "trade_count": int(len(regime_trades)),
                "net_sharpe_after_costs": portfolio_metrics["net_sharpe_after_costs"],
                "gross_sharpe": portfolio_metrics["gross_sharpe"],
                "max_drawdown": portfolio_metrics["max_drawdown"],
                "net_cvar_5": portfolio_metrics["net_cvar_5"],
                "avg_net_pnl_per_trade": portfolio_metrics["avg_net_pnl_per_trade"],
                "avg_holding_days": portfolio_metrics["avg_holding_days"],
                "annual_turnover": portfolio_metrics["annual_turnover"],
                "win_rate": portfolio_metrics["win_rate"],
                "abs_zscore_decay_per_day": _abs_zscore_decay_per_day(regime_trades),
                "reversion_exit_rate": _reversion_exit_rate(regime_trades),
                "worst_25_day_share": worst_share,
                "worst_25_day_ratio": 0.0 if day_share == 0.0 else float(worst_share / day_share),
            }
        )
    return pd.DataFrame(rows)


def _baseline_regime_lookup(baseline_by_regime: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {
        str(row["regime"]): row.to_dict()
        for _, row in baseline_by_regime.iterrows()
    }


def _evaluate_real_policy_variants(
    *,
    baseline_result: TrendStrategyBacktest,
    regime_frame: pd.DataFrame,
    phase2_config: Phase2Config,
    annualization_days: int,
    baseline_lookup: dict[str, dict[str, object]],
) -> list[GeoRegimePolicyRun]:
    return [
        _run_policy_variant(
            baseline_result=baseline_result,
            regime_frame=regime_frame,
            policy=policy,
            label_source="REAL",
            phase2_config=phase2_config,
            annualization_days=annualization_days,
            baseline_lookup=baseline_lookup,
        )
        for policy in DEFAULT_POLICIES
    ]


def _evaluate_placebo_variants(
    *,
    baseline_result: TrendStrategyBacktest,
    regime_frame: pd.DataFrame,
    selected_variant: GeoRegimePolicyRun | None,
    phase2_config: Phase2Config,
    annualization_days: int,
    baseline_lookup: dict[str, dict[str, object]],
) -> list[GeoRegimePolicyRun]:
    if selected_variant is None:
        return []
    policy = next((candidate for candidate in DEFAULT_POLICIES if candidate.name == selected_variant.variant), None)
    if policy is None:
        return []
    placebo_frames = build_placebo_regime_frames(regime_frame)
    runs: list[GeoRegimePolicyRun] = []
    for label_source, placebo_frame in placebo_frames.items():
        runs.append(
            _run_policy_variant(
                baseline_result=baseline_result,
                regime_frame=placebo_frame,
                policy=policy,
                label_source=label_source,
                phase2_config=phase2_config,
                annualization_days=annualization_days,
                baseline_lookup=baseline_lookup,
            )
        )
    return runs


def _run_policy_variant(
    *,
    baseline_result: TrendStrategyBacktest,
    regime_frame: pd.DataFrame,
    policy: GeoRegimePolicy,
    label_source: str,
    phase2_config: Phase2Config,
    annualization_days: int,
    baseline_lookup: dict[str, dict[str, object]],
) -> GeoRegimePolicyRun:
    if baseline_result.daily_signals is None or baseline_result.daily_signals.empty:
        raise ValueError("baseline_result.daily_signals must be populated for regime policy evaluation")

    variant_signals = _apply_policy_to_signals(
        baseline_result.daily_signals,
        regime_frame=regime_frame,
        policy=policy,
        phase2_config=phase2_config,
    )
    scaled_variant_signals = scale_signals_to_risk_budget(variant_signals, phase2_config).scaled_signals
    backtest = run_walk_forward_backtest(
        scaled_variant_signals,
        phase2_config,
        run_id=f"geo-regime-{label_source.lower()}-{policy.name}",
        strategy_label="A",
    )
    metrics = _build_policy_metrics_row(
        variant=policy.name,
        label_source=label_source,
        daily_results=backtest.daily_results,
        trade_log=backtest.trade_log,
        baseline_daily_results=baseline_result.daily_results,
        baseline_trade_log=baseline_result.trade_log,
        regime_frame=regime_frame,
        baseline_lookup=baseline_lookup,
        annualization_days=annualization_days,
    )
    return GeoRegimePolicyRun(
        variant=policy.name,
        label_source=label_source,
        daily_results=backtest.daily_results.copy(),
        trade_log=backtest.trade_log.copy(),
        daily_signals=scaled_variant_signals.copy(),
        metrics=metrics,
    )


def _apply_policy_to_signals(
    daily_signals: pd.DataFrame,
    *,
    regime_frame: pd.DataFrame,
    policy: GeoRegimePolicy,
    phase2_config: Phase2Config,
) -> pd.DataFrame:
    signals = daily_signals.copy()
    signals["date"] = pd.to_datetime(signals["date"]).dt.normalize()
    labeled = signals.merge(
        regime_frame.loc[:, ["date", "geo_regime"]],
        on="date",
        how="left",
    ).fillna({"geo_regime": NORMAL_REGIME})
    if "regime_threshold_multiplier" not in labeled.columns:
        labeled["regime_threshold_multiplier"] = 1.0
    if "regime_position_scale" not in labeled.columns:
        labeled["regime_position_scale"] = 1.0
    if "allow_new_entries" not in labeled.columns:
        labeled["allow_new_entries"] = True

    threshold_map = {
        NORMAL_REGIME: 1.0,
        GEO_STRESS_REGIME: float(policy.stress_threshold_multiplier),
        STRUCTURAL_GEO_REGIME: float(policy.structural_threshold_multiplier),
    }
    position_map = {
        NORMAL_REGIME: 1.0,
        GEO_STRESS_REGIME: float(policy.stress_position_scale),
        STRUCTURAL_GEO_REGIME: float(policy.structural_position_scale),
    }
    labeled["regime_threshold_multiplier"] = (
        pd.to_numeric(labeled["regime_threshold_multiplier"], errors="coerce").fillna(1.0)
        * labeled["geo_regime"].map(threshold_map).fillna(1.0).astype(float)
    )
    labeled["regime_position_scale"] = (
        pd.to_numeric(labeled["regime_position_scale"], errors="coerce").fillna(1.0)
        * labeled["geo_regime"].map(position_map).fillna(1.0).astype(float)
    )
    return apply_signal_rules(labeled, phase2_config)


def _build_policy_metrics_row(
    *,
    variant: str,
    label_source: str,
    daily_results: pd.DataFrame,
    trade_log: pd.DataFrame,
    baseline_daily_results: pd.DataFrame,
    baseline_trade_log: pd.DataFrame,
    regime_frame: pd.DataFrame,
    baseline_lookup: dict[str, dict[str, object]],
    annualization_days: int,
) -> dict[str, object]:
    metrics = _portfolio_trade_metrics(daily_results=daily_results, trade_log=trade_log, annualization_days=annualization_days)
    baseline_metrics = _portfolio_trade_metrics(
        daily_results=baseline_daily_results,
        trade_log=baseline_trade_log,
        annualization_days=annualization_days,
    )
    quiet_metrics = _slice_metrics_by_regime(
        daily_results=daily_results,
        trade_log=trade_log,
        regime_frame=regime_frame,
        regime_name=NORMAL_REGIME,
        annualization_days=annualization_days,
    )
    stress_metrics = _slice_metrics_by_regime(
        daily_results=daily_results,
        trade_log=trade_log,
        regime_frame=regime_frame,
        regime_name=GEO_STRESS_REGIME,
        annualization_days=annualization_days,
    )
    structural_metrics = _slice_metrics_by_regime(
        daily_results=daily_results,
        trade_log=trade_log,
        regime_frame=regime_frame,
        regime_name=STRUCTURAL_GEO_REGIME,
        annualization_days=annualization_days,
    )
    removal = _trade_removal_metrics(baseline_trade_log, trade_log)
    baseline_quiet_sharpe = float(baseline_lookup.get(NORMAL_REGIME, {}).get("net_sharpe_after_costs", 0.0) or 0.0)

    return {
        "variant": variant,
        "label_source": label_source,
        "net_sharpe_after_costs": metrics["net_sharpe_after_costs"],
        "delta_net_sharpe_after_costs": float(metrics["net_sharpe_after_costs"] - baseline_metrics["net_sharpe_after_costs"]),
        "gross_sharpe": metrics["gross_sharpe"],
        "delta_gross_sharpe": float(metrics["gross_sharpe"] - baseline_metrics["gross_sharpe"]),
        "max_drawdown": metrics["max_drawdown"],
        "delta_max_drawdown": float(metrics["max_drawdown"] - baseline_metrics["max_drawdown"]),
        "net_cvar_5": metrics["net_cvar_5"],
        "delta_net_cvar_5": float(metrics["net_cvar_5"] - baseline_metrics["net_cvar_5"]),
        "trade_count": int(metrics["trade_count"]),
        "trade_count_delta_pct": (
            0.0
            if baseline_metrics["trade_count"] == 0
            else float((metrics["trade_count"] - baseline_metrics["trade_count"]) / baseline_metrics["trade_count"])
        ),
        "avg_net_pnl_per_trade": metrics["avg_net_pnl_per_trade"],
        "avg_holding_days": metrics["avg_holding_days"],
        "annual_turnover": metrics["annual_turnover"],
        "win_rate": metrics["win_rate"],
        "quiet_net_sharpe_after_costs": quiet_metrics["net_sharpe_after_costs"],
        "quiet_delta_net_sharpe_after_costs": float(quiet_metrics["net_sharpe_after_costs"] - baseline_quiet_sharpe),
        "stress_net_sharpe_after_costs": stress_metrics["net_sharpe_after_costs"],
        "structural_net_sharpe_after_costs": structural_metrics["net_sharpe_after_costs"],
        "removed_trade_fraction": removal["removed_trade_fraction"],
        "avg_removed_trade_pnl": removal["avg_removed_trade_pnl"],
    }


def _portfolio_trade_metrics(
    *,
    daily_results: pd.DataFrame,
    trade_log: pd.DataFrame,
    annualization_days: int,
) -> dict[str, float | int]:
    sorted_daily = daily_results.sort_values("date", kind="mergesort").reset_index(drop=True)
    net_returns = pd.to_numeric(sorted_daily.get("net_portfolio_return", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    gross_returns = pd.to_numeric(sorted_daily.get("gross_portfolio_return", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    annual_turnover = _annual_turnover(sorted_daily, annualization_days) if not sorted_daily.empty else 0.0
    avg_holding_days = (
        float(pd.to_numeric(trade_log["holding_days"], errors="coerce").fillna(0.0).mean())
        if not trade_log.empty and "holding_days" in trade_log.columns
        else 0.0
    )
    trade_returns = pd.to_numeric(trade_log.get("net_return", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return {
        "net_sharpe_after_costs": float(_annualized_sharpe(net_returns, annualization_days) or 0.0),
        "gross_sharpe": float(_annualized_sharpe(gross_returns, annualization_days) or 0.0),
        "max_drawdown": float(_max_drawdown(net_returns) or 0.0),
        "net_cvar_5": float(_net_cvar_5(net_returns) or 0.0),
        "trade_count": int(len(trade_log)),
        "avg_net_pnl_per_trade": float(trade_returns.mean()) if not trade_returns.empty else 0.0,
        "avg_holding_days": avg_holding_days,
        "annual_turnover": float(annual_turnover or 0.0),
        "win_rate": float((trade_returns > 0.0).mean()) if not trade_returns.empty else 0.0,
    }


def _slice_metrics_by_regime(
    *,
    daily_results: pd.DataFrame,
    trade_log: pd.DataFrame,
    regime_frame: pd.DataFrame,
    regime_name: str,
    annualization_days: int,
) -> dict[str, float | int]:
    labeled_daily = _attach_regime_labels(daily_results, regime_frame, date_column="date")
    labeled_trades = _attach_regime_labels(trade_log, regime_frame, date_column="entry_date")
    return _portfolio_trade_metrics(
        daily_results=labeled_daily.loc[labeled_daily["geo_regime"] == regime_name].copy(),
        trade_log=labeled_trades.loc[labeled_trades["geo_regime"] == regime_name].copy(),
        annualization_days=annualization_days,
    )


def _build_active_signal_day_frame(
    daily_signals: pd.DataFrame,
    regime_frame: pd.DataFrame,
) -> pd.DataFrame:
    if daily_signals.empty:
        return pd.DataFrame(columns=["date", "active_signal_day", "geo_regime"])
    signal_days = (
        daily_signals.assign(date=pd.to_datetime(daily_signals["date"]).dt.normalize())
        .groupby("date")["signal_direction"]
        .apply(lambda series: bool((pd.to_numeric(series, errors="coerce").fillna(0.0) != 0).any()))
        .reset_index(name="active_signal_day")
    )
    return signal_days.merge(
        regime_frame.loc[:, ["date", "geo_regime"]],
        on="date",
        how="left",
    ).fillna({"geo_regime": NORMAL_REGIME})


def _attach_regime_labels(
    frame: pd.DataFrame,
    regime_frame: pd.DataFrame,
    *,
    date_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(geo_regime=pd.Series(dtype="object"))
    labeled = frame.copy()
    labeled[date_column] = pd.to_datetime(labeled[date_column]).dt.normalize()
    merged = labeled.merge(
        regime_frame.loc[:, ["date", "geo_regime"]].rename(columns={"date": date_column}),
        on=date_column,
        how="left",
    )
    merged["geo_regime"] = merged["geo_regime"].fillna(NORMAL_REGIME)
    return merged


def _abs_zscore_decay_per_day(trade_log: pd.DataFrame) -> float:
    if trade_log.empty or "entry_zscore" not in trade_log.columns or "exit_zscore" not in trade_log.columns:
        return 0.0
    frame = trade_log.copy()
    frame["entry_abs_zscore"] = pd.to_numeric(frame["entry_zscore"], errors="coerce").abs()
    frame["exit_abs_zscore"] = pd.to_numeric(frame["exit_zscore"], errors="coerce").abs()
    frame["holding_days"] = pd.to_numeric(frame["holding_days"], errors="coerce").clip(lower=1.0)
    frame = frame.dropna(subset=["entry_abs_zscore", "exit_abs_zscore", "holding_days"])
    if frame.empty:
        return 0.0
    decay = (frame["entry_abs_zscore"] - frame["exit_abs_zscore"]) / frame["holding_days"]
    return float(decay.mean())


def _reversion_exit_rate(trade_log: pd.DataFrame) -> float:
    if trade_log.empty or "exit_reason" not in trade_log.columns:
        return 0.0
    reasons = trade_log["exit_reason"].astype(str)
    return float((reasons == "reversion").mean())


def _trade_removal_metrics(
    baseline_trade_log: pd.DataFrame,
    variant_trade_log: pd.DataFrame,
) -> dict[str, float]:
    baseline = baseline_trade_log.copy()
    variant = variant_trade_log.copy()
    baseline_keys = _trade_entry_keys(baseline)
    variant_keys = set(_trade_entry_keys(variant))
    removed = baseline.loc[~baseline_keys.isin(variant_keys)].copy()
    removed_returns = pd.to_numeric(removed.get("net_return", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    return {
        "removed_trade_fraction": (
            0.0 if len(baseline) == 0 else float(len(removed) / len(baseline))
        ),
        "avg_removed_trade_pnl": float(removed_returns.mean()) if not removed_returns.empty else 0.0,
    }


def _trade_entry_keys(trade_log: pd.DataFrame) -> pd.Series:
    if trade_log.empty:
        return pd.Series(dtype="object")
    dates = pd.to_datetime(trade_log["entry_date"], errors="coerce").dt.normalize().dt.strftime("%Y-%m-%d")
    tickers = trade_log["ticker"].astype(str)
    directions = pd.to_numeric(trade_log["position_direction"], errors="coerce").fillna(0).astype(int).astype(str)
    return dates + "|" + tickers + "|" + directions


def _select_best_variant(policy_runs: list[GeoRegimePolicyRun]) -> GeoRegimePolicyRun | None:
    if not policy_runs:
        return None
    return max(
        policy_runs,
        key=lambda run: (
            float(run.metrics.get("delta_net_sharpe_after_costs", 0.0)),
            -float(run.metrics.get("delta_max_drawdown", 0.0)),
            float(run.metrics.get("delta_net_cvar_5", 0.0)),
            -float(run.metrics.get("removed_trade_fraction", 0.0)),
        ),
    )


def _build_conclusions(
    *,
    baseline_by_regime: pd.DataFrame,
    best_real_variant: GeoRegimePolicyRun | None,
    placebo_runs: list[GeoRegimePolicyRun],
    bootstrap: dict[str, object],
) -> dict[str, object]:
    baseline_lookup = {
        str(row["regime"]): row.to_dict()
        for _, row in baseline_by_regime.iterrows()
    }
    normal = baseline_lookup.get(NORMAL_REGIME, {})
    stress = baseline_lookup.get(GEO_STRESS_REGIME, {})
    structural = baseline_lookup.get(STRUCTURAL_GEO_REGIME, {})

    sample_sufficient = (
        int(stress.get("trade_count", 0) or 0) >= 8
        and int(structural.get("trade_count", 0) or 0) >= 10
        and (
            int(stress.get("date_count", 0) or 0)
            + int(structural.get("date_count", 0) or 0)
        ) >= 40
    )
    behavior_checks = {
        "structural_net_sharpe_lower": float(structural.get("net_sharpe_after_costs", 0.0) or 0.0)
        <= float(normal.get("net_sharpe_after_costs", 0.0) or 0.0) - 0.20,
        "structural_avg_pnl_lower": float(structural.get("avg_net_pnl_per_trade", 0.0) or 0.0)
        <= float(normal.get("avg_net_pnl_per_trade", 0.0) or 0.0) * 0.80,
        "structural_win_rate_lower": float(structural.get("win_rate", 0.0) or 0.0)
        <= float(normal.get("win_rate", 0.0) or 0.0) - 0.10,
        "structural_reversion_speed_lower": float(structural.get("abs_zscore_decay_per_day", 0.0) or 0.0)
        <= float(normal.get("abs_zscore_decay_per_day", 0.0) or 0.0) * 0.80,
    }
    behavior_differs = sum(bool(value) for value in behavior_checks.values()) >= 2

    best_metrics = best_real_variant.metrics if best_real_variant is not None else {}
    placebo_best_delta = max(
        (float(run.metrics.get("delta_net_sharpe_after_costs", float("-inf"))) for run in placebo_runs),
        default=float("-inf"),
    )
    placebo_margin = (
        float(best_metrics.get("delta_net_sharpe_after_costs", 0.0)) - placebo_best_delta
        if best_real_variant is not None and placebo_runs
        else float(best_metrics.get("delta_net_sharpe_after_costs", 0.0))
    )
    bootstrap_sharpe = bootstrap.get("delta_net_sharpe_after_costs", {})
    bootstrap_drawdown = bootstrap.get("delta_max_drawdown", {})
    bootstrap_cvar = bootstrap.get("delta_net_cvar_5", {})

    research_criteria = {
        "sample_sufficient": sample_sufficient,
        "behavior_differs": behavior_differs,
        "best_variant_positive_sharpe_delta": float(best_metrics.get("delta_net_sharpe_after_costs", 0.0)) > 0.0,
        "best_variant_beats_placebos": float(best_metrics.get("delta_net_sharpe_after_costs", 0.0)) > placebo_best_delta,
        "bootstrap_median_positive": float(bootstrap_sharpe.get("median", 0.0) or 0.0) > 0.0,
    }
    production_criteria = {
        "sample_sufficient": sample_sufficient,
        "behavior_differs": behavior_differs,
        "delta_net_sharpe_after_costs_ge_0_05": float(best_metrics.get("delta_net_sharpe_after_costs", 0.0)) >= 0.05,
        "delta_max_drawdown_le_0_005": float(best_metrics.get("delta_max_drawdown", 0.0)) <= 0.005,
        "delta_net_cvar_5_ge_neg_0_00025": float(best_metrics.get("delta_net_cvar_5", 0.0)) >= -0.00025,
        "quiet_period_not_materially_worse": float(best_metrics.get("quiet_delta_net_sharpe_after_costs", 0.0)) >= -0.05,
        "trade_removal_not_excessive": (
            float(best_metrics.get("removed_trade_fraction", 0.0)) <= 0.20
            or float(best_metrics.get("avg_removed_trade_pnl", 0.0)) <= 0.0
        ),
        "bootstrap_median_positive": float(bootstrap_sharpe.get("median", 0.0) or 0.0) > 0.0,
        "bootstrap_p25_non_negative": float(bootstrap_sharpe.get("p25", 0.0) or 0.0) >= 0.0,
        "bootstrap_drawdown_median_non_positive": float(bootstrap_drawdown.get("median", 0.0) or 0.0) <= 0.0,
        "bootstrap_cvar_median_non_negative": float(bootstrap_cvar.get("median", 0.0) or 0.0) >= 0.0,
        "placebo_margin_ge_0_02": float(placebo_margin) >= 0.02,
    }
    return {
        "research_signal_present": {
            "passed": bool(all(research_criteria.values())),
            "criteria": research_criteria,
            "best_variant": None if best_real_variant is None else best_real_variant.variant,
            "placebo_margin": float(placebo_margin),
        },
        "production_gate_pass": {
            "passed": bool(all(production_criteria.values())),
            "criteria": production_criteria,
            "best_variant": None if best_real_variant is None else best_real_variant.variant,
            "placebo_margin": float(placebo_margin),
        },
    }


def _build_bootstrap_report(
    *,
    baseline_daily_results: pd.DataFrame,
    variant_daily_results: pd.DataFrame,
    annualization_days: int,
    random_seed: int,
) -> dict[str, object]:
    merged = (
        baseline_daily_results.loc[:, ["date", "net_portfolio_return"]]
        .rename(columns={"net_portfolio_return": "baseline"})
        .merge(
            variant_daily_results.loc[:, ["date", "net_portfolio_return"]].rename(columns={"net_portfolio_return": "variant"}),
            on="date",
            how="inner",
        )
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    if merged.empty:
        return {
            "random_seed": random_seed,
            "samples": 0,
            "delta_net_sharpe_after_costs": {"median": 0.0, "p25": 0.0},
            "delta_max_drawdown": {"median": 0.0},
            "delta_net_cvar_5": {"median": 0.0},
        }

    baseline = pd.to_numeric(merged["baseline"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    variant = pd.to_numeric(merged["variant"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    delta_sharpes: list[float] = []
    delta_drawdowns: list[float] = []
    delta_cvars: list[float] = []

    for _ in range(BOOTSTRAP_SAMPLES):
        sample_idx = rng.integers(0, len(merged), len(merged))
        baseline_sample = pd.Series(baseline[sample_idx], dtype=float)
        variant_sample = pd.Series(variant[sample_idx], dtype=float)
        baseline_sharpe = float(_annualized_sharpe(baseline_sample, annualization_days) or 0.0)
        variant_sharpe = float(_annualized_sharpe(variant_sample, annualization_days) or 0.0)
        baseline_drawdown = float(_max_drawdown(baseline_sample) or 0.0)
        variant_drawdown = float(_max_drawdown(variant_sample) or 0.0)
        baseline_cvar = float(_net_cvar_5(baseline_sample) or 0.0)
        variant_cvar = float(_net_cvar_5(variant_sample) or 0.0)
        delta_sharpes.append(float(variant_sharpe - baseline_sharpe))
        delta_drawdowns.append(float(variant_drawdown - baseline_drawdown))
        delta_cvars.append(float(variant_cvar - baseline_cvar))

    return {
        "random_seed": random_seed,
        "samples": BOOTSTRAP_SAMPLES,
        "delta_net_sharpe_after_costs": {
            "median": float(np.median(delta_sharpes)),
            "p25": float(np.percentile(delta_sharpes, 25)),
        },
        "delta_max_drawdown": {
            "median": float(np.median(delta_drawdowns)),
        },
        "delta_net_cvar_5": {
            "median": float(np.median(delta_cvars)),
        },
    }


def _net_cvar_5(returns: pd.Series) -> float | None:
    cleaned = pd.to_numeric(returns, errors="coerce").dropna()
    if cleaned.empty:
        return None
    threshold = float(cleaned.quantile(0.05))
    tail = cleaned.loc[cleaned <= threshold]
    if tail.empty:
        tail = pd.Series([threshold], dtype=float)
    return float(tail.mean())


if __name__ == "__main__":
    main()
