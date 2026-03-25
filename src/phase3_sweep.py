from __future__ import annotations

import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd

from .backtest import run_walk_forward_backtest, scale_signals_to_risk_budget
from .config_loader import load_config
from .diagnostics.regime_validation import (
    KNOWN_CRISIS_WINDOWS,
    compute_false_positive_rate,
    evaluate_crisis_detection,
)
from .phase3 import (
    _build_freeze_dates,
    _build_phase3_signal_frame,
    _build_signal_variant_schedule,
    _compute_phase2_baseline_metrics,
    _compute_regime_observations,
    _load_validated_price_history,
)


def run_phase3_sweep(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    price_history = _load_validated_price_history(config)
    baseline_metrics = _compute_phase2_baseline_metrics(config.paths.project_root / "config/phase2_cleared.yaml")
    rows: list[dict[str, object]] = []

    grid = product([1.0, 1.5, 2.0], [2.0, 2.5, 3.0], [0.3, 0.5])
    for transition_sigma, new_regime_sigma, transition_position_scale in grid:
        candidate_phase3 = replace(
            config.phase3,
            transition_threshold_sigma=transition_sigma,
            new_regime_threshold_sigma=new_regime_sigma,
            transition_position_scale=transition_position_scale,
        )
        candidate_config = replace(config, phase3=candidate_phase3)
        regime_observations = _compute_regime_observations(price_history, candidate_config)
        regime_states = {date: observation.state.value for date, observation in regime_observations.items()}
        freeze_dates = _build_freeze_dates(regime_observations, candidate_phase3)
        signal_variant_by_date = _build_signal_variant_schedule(regime_observations, candidate_phase3)
        phase3_signals = _build_phase3_signal_frame(
            price_history,
            candidate_config,
            signal_variant_by_date,
            regime_states,
            freeze_dates,
        )
        scaling_result = scale_signals_to_risk_budget(phase3_signals, candidate_config.phase2)
        backtest_result = run_walk_forward_backtest(
            scaling_result.scaled_signals,
            candidate_config.phase2,
            run_id=f"sweep-{transition_sigma}-{new_regime_sigma}-{transition_position_scale}",
        )
        _, hit_rate = evaluate_crisis_detection(regime_observations, KNOWN_CRISIS_WINDOWS)
        false_positive_rate = compute_false_positive_rate(regime_observations, KNOWN_CRISIS_WINDOWS)
        drawdown_reduction = (
            (baseline_metrics["max_drawdown"] - backtest_result.summary_metrics["max_drawdown"])
            / baseline_metrics["max_drawdown"]
        )
        annualized_return_ratio = (
            backtest_result.summary_metrics["annualized_return"] / baseline_metrics["annualized_return"]
        )
        gate_passed = (
            hit_rate >= 0.70
            and drawdown_reduction >= 0.20
            and annualized_return_ratio >= 0.90
            and backtest_result.summary_metrics["sharpe_ratio"] > 0.80
        )
        rows.append(
            {
                **backtest_result.summary_metrics,
                "transition_threshold_sigma": transition_sigma,
                "new_regime_threshold_sigma": new_regime_sigma,
                "transition_position_scale": transition_position_scale,
                "regime_hit_rate": hit_rate,
                "regime_false_positive_rate": false_positive_rate,
                "drawdown_reduction_relative": drawdown_reduction,
                "annualized_return_ratio": annualized_return_ratio,
                "gate_passed": gate_passed,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        by=[
            "gate_passed",
            "sharpe_ratio",
            "drawdown_reduction_relative",
            "annualized_return_ratio",
            "profitable_month_fraction",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    processed_dir = config.paths.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_path = processed_dir / "phase3_sweep_results.csv"
    best_path = processed_dir / "phase3_sweep_best.json"
    results.to_csv(results_path, index=False)
    best_path.write_text(
        json.dumps(
            {
                "candidate_count": int(len(results)),
                "gate_pass_count": int(results["gate_passed"].sum()),
                "best_candidate": results.iloc[0].to_dict() if not results.empty else {},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return {"results": results_path, "best": best_path}
