from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import yaml

from .backtest import (
    BacktestResult,
    apply_phase4_tcn_filter,
    run_walk_forward_backtest,
    scale_signals_to_risk_budget,
)
from .config_loader import config_to_dict, load_config
from .features import FeatureBuilder
from .graph_engine import compute_daily_graph_matrices, compute_graph_signals
from .tcn_trainer import SequenceDatasetBundle, TCNTrainer


@dataclass(frozen=True)
class Phase4TrainingResult:
    model_path: Path
    summary_path: Path
    latest_window: dict[str, str]
    validation_losses: list[float]


@dataclass(frozen=True)
class Phase4Result:
    run_id: str
    summary_metrics: dict[str, object]
    output_paths: dict[str, Path]


@dataclass(frozen=True)
class Phase4VerificationResult:
    gate_passed: bool
    summary_metrics: dict[str, object]
    output_path: Path


def train_tcn_ensemble(config_path: str | Path) -> Phase4TrainingResult:
    config = load_config(config_path)
    phase4_context = _prepare_phase4_context(config)
    windows = _build_walk_forward_windows(phase4_context.dataset.signal_dates)
    if not windows:
        raise ValueError("No valid walk-forward windows were available for Phase 4 training.")

    latest_window = windows[-1]
    trainer = phase4_context.trainer
    ensemble = trainer.train_ensemble(
        phase4_context.features_by_date,
        phase4_context.residuals_by_date,
        train_end_date=latest_window["validation_end"],
        validation_start_date=latest_window["validation_start"],
    )

    processed_dir = config.paths.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_path = processed_dir / "phase4_latest_ensemble.pt"
    summary_path = processed_dir / "phase4_latest_training_summary.json"
    torch.save(
        {
            "state_dicts": [model.state_dict() for model in ensemble.models],
            "config": config_to_dict(config),
            "window": {key: value.isoformat() for key, value in latest_window.items()},
            "validation_losses": ensemble.validation_losses,
        },
        model_path,
    )
    summary_path.write_text(
        json.dumps(
            {
                "window": {key: value.isoformat() for key, value in latest_window.items()},
                "validation_losses": ensemble.validation_losses,
                "n_models": len(ensemble.models),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return Phase4TrainingResult(
        model_path=model_path,
        summary_path=summary_path,
        latest_window={key: value.isoformat() for key, value in latest_window.items()},
        validation_losses=ensemble.validation_losses,
    )


def run_phase4_pipeline(config_path: str | Path) -> Phase4Result:
    pipeline_start = time.time()
    config = load_config(config_path)
    phase4_context = _prepare_phase4_context(config)
    windows = _build_walk_forward_windows(phase4_context.dataset.signal_dates)
    trainer = phase4_context.trainer

    walk_forward_start = time.time()
    prediction_rows: list[dict[str, object]] = []
    window_profiles: list[dict[str, object]] = []
    for window in windows:
        window_profile: dict[str, object] = {
            "label": f"{window['test_start'].date()}:{window['test_end'].date()}",
        }
        _print_profile(f"Window {window_profile['label']} started")
        train_start = time.time()
        ensemble = trainer.train_ensemble(
            phase4_context.features_by_date,
            phase4_context.residuals_by_date,
            train_end_date=window["validation_end"],
            validation_start_date=window["validation_start"],
        )
        train_end = time.time()
        window_profile["train_seconds"] = train_end - train_start
        _print_profile(
            f"Window {window_profile['label']} training: {window_profile['train_seconds']:.1f}s"
        )

        inference_start = time.time()
        test_indices = [
            idx
            for idx, signal_date in enumerate(phase4_context.dataset.signal_dates)
            if window["test_start"] <= pd.Timestamp(signal_date) <= window["test_end"]
        ]
        window_profile["test_samples"] = len(test_indices)
        for index in test_indices:
            feature_sequence = phase4_context.dataset.inputs[index]
            predicted_mean, predicted_std = trainer.predict(ensemble.models, feature_sequence)
            signal_date = pd.Timestamp(phase4_context.dataset.signal_dates[index])
            target_date = pd.Timestamp(phase4_context.dataset.target_dates[index])
            actual_next_residual = phase4_context.dataset.targets[index]
            for asset_idx, ticker in enumerate(config.tickers):
                prediction_rows.append(
                    {
                        "signal_date": signal_date,
                        "target_date": target_date,
                        "ticker": ticker,
                        "predicted_residual_mean": float(predicted_mean[asset_idx]),
                        "predicted_residual_std": float(predicted_std[asset_idx]),
                        "actual_next_residual": float(actual_next_residual[asset_idx]),
                        "window_label": f"{window['test_start'].date()}:{window['test_end'].date()}",
                    }
                )
        inference_end = time.time()
        window_profile["inference_seconds"] = inference_end - inference_start
        _print_profile(
            f"Window {window_profile['label']} inference: {window_profile['inference_seconds']:.1f}s "
            f"for {window_profile['test_samples']} samples"
        )
        window_profiles.append(window_profile)

    walk_forward_end = time.time()
    _print_profile(f"Walk-forward total: {walk_forward_end - walk_forward_start:.1f}s")
    _print_profile("  Per-window breakdown:")
    for window_profile in window_profiles:
        _print_profile(
            "  "
            f"{window_profile['label']} "
            f"train={window_profile['train_seconds']:.1f}s "
            f"inference={window_profile['inference_seconds']:.1f}s "
            f"test_samples={window_profile['test_samples']}"
        )

    post_walk_start = time.time()
    predictions = pd.DataFrame(prediction_rows).sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    filtered_signals = apply_phase4_tcn_filter(
        phase4_context.daily_signals,
        predictions,
        config.phase4.reversion_confirm_threshold,
    )
    scaling_result = scale_signals_to_risk_budget(filtered_signals, config.phase2)
    scaled_signals = scaling_result.scaled_signals
    run_id = str(uuid4())

    backtest_start = time.time()
    backtest_result = run_walk_forward_backtest(scaled_signals, config.phase2, run_id)
    backtest_end = time.time()
    _print_profile(f"Post-walk-forward backtest: {backtest_end - backtest_start:.1f}s")

    baseline_start = time.time()
    baseline_result = _run_phase2_baseline_backtest(config)
    baseline_end = time.time()
    _print_profile(f"Phase 2 baseline backtest: {baseline_end - baseline_start:.1f}s")
    calibration_rate = _calibration_rate(predictions)
    veto_rate = _veto_rate(filtered_signals)
    veto_accuracy = _veto_accuracy(filtered_signals, baseline_result.trade_log)
    predicted_actual_correlation = _predicted_actual_correlation(predictions)

    backtest_result.summary_metrics.update(
        {
            "baseline_max_drawdown": scaling_result.baseline_max_drawdown,
            "target_max_drawdown": scaling_result.target_max_drawdown,
            "position_scale_factor": scaling_result.scale_factor,
            "phase2_baseline_sharpe": baseline_result.summary_metrics["sharpe_ratio"],
            "phase2_baseline_max_drawdown": baseline_result.summary_metrics["max_drawdown"],
            "phase2_baseline_annualized_return": baseline_result.summary_metrics["annualized_return"],
            "calibration_rate": calibration_rate,
            "calibration_error": abs(calibration_rate - 0.68),
            "tcn_veto_rate": veto_rate,
            "tcn_veto_accuracy": veto_accuracy,
            "predicted_actual_residual_correlation": predicted_actual_correlation,
            "walk_forward_windows": len(windows),
        }
    )
    output_paths = _save_phase4_outputs(
        processed_dir=config.paths.processed_dir,
        run_id=run_id,
        config_snapshot=config_to_dict(config),
        predictions=predictions,
        daily_signals=scaled_signals,
        daily_results=backtest_result.daily_results,
        trade_log=backtest_result.trade_log,
        monthly_results=backtest_result.monthly_results,
        summary_metrics=backtest_result.summary_metrics,
    )
    post_walk_end = time.time()
    _print_profile(f"Post-walk-forward total: {post_walk_end - post_walk_start:.1f}s")
    _print_profile(f"Total Phase 4 pipeline: {post_walk_end - pipeline_start:.1f}s")
    return Phase4Result(run_id=run_id, summary_metrics=backtest_result.summary_metrics, output_paths=output_paths)


def verify_phase4_gate(config_path: str | Path) -> Phase4VerificationResult:
    config = load_config(config_path)
    summary_path = config.paths.processed_dir / "phase4_summary.json"
    if not summary_path.exists():
        raise ValueError("Phase 4 summary was not found. Run `python3 -m src.main run-phase4` first.")

    summary_metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline_sharpe = float(summary_metrics["phase2_baseline_sharpe"])
    sharpe_ratio = float(summary_metrics["sharpe_ratio"])
    calibration_rate = float(summary_metrics["calibration_rate"])

    gate_passed = sharpe_ratio > (baseline_sharpe + 0.10) and 0.63 <= calibration_rate <= 0.73
    summary_metrics["gate_passed"] = gate_passed
    summary_path.write_text(json.dumps(summary_metrics, indent=2, default=str), encoding="utf-8")
    return Phase4VerificationResult(gate_passed=gate_passed, summary_metrics=summary_metrics, output_path=summary_path)


@dataclass(frozen=True)
class Phase4Context:
    daily_signals: pd.DataFrame
    feature_state: object
    features_by_date: dict[pd.Timestamp, np.ndarray]
    residuals_by_date: dict[pd.Timestamp, np.ndarray]
    dataset: SequenceDatasetBundle
    trainer: TCNTrainer


def _prepare_phase4_context(config) -> Phase4Context:
    graph_setup_start = time.time()
    price_history = _load_validated_price_history(config.paths.processed_dir, config.tickers)
    graph_matrices = compute_daily_graph_matrices(
        price_history.loc[:, ["date", "ticker", "adj_close"]],
        config.tickers,
        config.phase2.lookback_window,
    )
    daily_signals = compute_graph_signals(
        price_history.loc[:, ["date", "ticker", "adj_close"]],
        config.tickers,
        config.phase2,
        graph_matrices=graph_matrices,
    )
    graph_setup_end = time.time()
    _print_profile(f"Graph engine setup: {graph_setup_end - graph_setup_start:.1f}s")

    feature_build_start = time.time()
    builder = FeatureBuilder(config)
    feature_state = builder.prepare_graph_engine_state(
        price_history,
        config.tickers,
        signals=daily_signals,
        graph_matrices=graph_matrices,
    )
    features_by_date = builder.build_feature_history(feature_state)
    residuals_by_date = builder.build_residual_history(feature_state)
    trainer = TCNTrainer(config)
    dataset = trainer.prepare_dataset(features_by_date, residuals_by_date)
    feature_build_end = time.time()
    _print_profile(f"Feature building: {feature_build_end - feature_build_start:.1f}s")
    return Phase4Context(
        daily_signals=daily_signals,
        feature_state=feature_state,
        features_by_date=features_by_date,
        residuals_by_date=residuals_by_date,
        dataset=dataset,
        trainer=trainer,
    )


def _build_walk_forward_windows(signal_dates: list[pd.Timestamp]) -> list[dict[str, pd.Timestamp]]:
    if not signal_dates:
        return []
    signal_min = min(pd.Timestamp(date) for date in signal_dates)
    signal_max = max(pd.Timestamp(date) for date in signal_dates)
    test_start = pd.Timestamp("2022-07-01")
    windows: list[dict[str, pd.Timestamp]] = []
    while test_start <= signal_max:
        validation_start = test_start - pd.DateOffset(months=6)
        validation_end = test_start - pd.Timedelta(days=1)
        test_end = min(signal_max, test_start + pd.DateOffset(months=6) - pd.Timedelta(days=1))
        if validation_start > signal_min:
            windows.append(
                {
                    "validation_start": pd.Timestamp(validation_start),
                    "validation_end": pd.Timestamp(validation_end),
                    "test_start": pd.Timestamp(test_start),
                    "test_end": pd.Timestamp(test_end),
                }
            )
        test_start = test_start + pd.DateOffset(months=6)
    return windows


def _load_validated_price_history(processed_dir: Path, tickers: list[str]) -> pd.DataFrame:
    validated_path = processed_dir / "sector_etf_prices_validated.csv"
    if not validated_path.exists():
        raise ValueError(f"Validated price history was not found at '{validated_path}'.")
    price_history = pd.read_csv(validated_path, parse_dates=["date"])
    filtered = price_history.loc[
        price_history["is_valid"] & price_history["ticker"].isin(tickers),
        ["date", "ticker", "adj_close", "volume"],
    ].copy()
    filtered["volume"] = filtered["volume"].fillna(0.0)
    return filtered


def _run_phase2_baseline_backtest(config) -> BacktestResult:
    baseline_config_path = config.paths.project_root / "config/phase2_cleared.yaml"
    baseline_payload = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    baseline_payload["phase4"] = config_to_dict(config)["phase4"]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(yaml.safe_dump(baseline_payload, sort_keys=False))
        baseline_runtime_path = Path(handle.name)
    try:
        baseline_config = load_config(baseline_runtime_path)
    finally:
        baseline_runtime_path.unlink(missing_ok=True)
    price_history = _load_validated_price_history(baseline_config.paths.processed_dir, baseline_config.tickers)
    daily_signals = compute_graph_signals(price_history.loc[:, ["date", "ticker", "adj_close"]], baseline_config.tickers, baseline_config.phase2)
    scaled_signals = scale_signals_to_risk_budget(daily_signals, baseline_config.phase2).scaled_signals
    return run_walk_forward_backtest(scaled_signals, baseline_config.phase2, run_id="phase4-phase2-baseline")


def _calibration_rate(predictions: pd.DataFrame) -> float:
    if predictions.empty:
        return 0.0
    within_interval = (
        predictions["actual_next_residual"].between(
            predictions["predicted_residual_mean"] - predictions["predicted_residual_std"],
            predictions["predicted_residual_mean"] + predictions["predicted_residual_std"],
        )
    )
    return float(within_interval.mean())


def _veto_rate(filtered_signals: pd.DataFrame) -> float:
    candidate_mask = filtered_signals["signal_direction"].ne(0) & filtered_signals["tcn_prediction_available"].fillna(False)
    if not candidate_mask.any():
        return 0.0
    return float(filtered_signals.loc[candidate_mask, "tcn_veto"].mean())


def _veto_accuracy(filtered_signals: pd.DataFrame, baseline_trade_log: pd.DataFrame) -> float:
    if baseline_trade_log.empty:
        return 0.0
    entry_lookup = baseline_trade_log.copy()
    entry_lookup["signal_direction"] = entry_lookup["position_direction"].astype(int)
    entry_lookup["entry_key"] = list(zip(pd.to_datetime(entry_lookup["entry_date"]), entry_lookup["ticker"], entry_lookup["signal_direction"]))
    entry_returns = entry_lookup.set_index("entry_key")["net_return"].to_dict()

    vetoed = filtered_signals.loc[filtered_signals["tcn_veto"]].copy()
    if vetoed.empty:
        return 0.0
    vetoed["entry_key"] = list(zip(pd.to_datetime(vetoed["date"]), vetoed["ticker"], vetoed["signal_direction"]))
    mapped_returns = vetoed["entry_key"].map(entry_returns)
    matched = mapped_returns.dropna()
    if matched.empty:
        return 0.0
    return float((matched <= 0).mean())


def _predicted_actual_correlation(predictions: pd.DataFrame) -> float:
    if predictions.empty:
        return 0.0
    correlation = predictions["predicted_residual_mean"].corr(predictions["actual_next_residual"])
    if pd.isna(correlation):
        return 0.0
    return float(correlation)


def _print_profile(message: str) -> None:
    print(f"[PROFILE] {message}", flush=True)


def _save_phase4_outputs(
    *,
    processed_dir: Path,
    run_id: str,
    config_snapshot: dict[str, object],
    predictions: pd.DataFrame,
    daily_signals: pd.DataFrame,
    daily_results: pd.DataFrame,
    trade_log: pd.DataFrame,
    monthly_results: pd.DataFrame,
    summary_metrics: dict[str, object],
) -> dict[str, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = processed_dir / "phase4_predictions.csv"
    signals_path = processed_dir / "phase4_daily_signals.csv"
    daily_results_path = processed_dir / "phase4_daily_results.csv"
    trades_path = processed_dir / "phase4_trade_log.csv"
    monthly_path = processed_dir / "phase4_monthly_results.csv"
    summary_path = processed_dir / "phase4_summary.json"

    predictions.to_csv(predictions_path, index=False)
    daily_signals.to_csv(signals_path, index=False)
    daily_results.to_csv(daily_results_path, index=False)
    trade_log.to_csv(trades_path, index=False)
    monthly_results.to_csv(monthly_path, index=False)
    summary_path.write_text(
        json.dumps({"run_id": run_id, **summary_metrics, "config_snapshot": config_snapshot}, indent=2, default=str),
        encoding="utf-8",
    )
    return {
        "predictions": predictions_path,
        "signals": signals_path,
        "daily_results": daily_results_path,
        "trades": trades_path,
        "monthly_results": monthly_path,
        "summary": summary_path,
    }
