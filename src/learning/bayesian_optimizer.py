from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest import run_walk_forward_backtest, scale_signals_to_risk_budget
from ..config_loader import PipelineConfig, load_config
from ..graph_engine import apply_signal_rules, compute_graph_signals
from ..trade_journal import TradeJournal


@dataclass(frozen=True)
class BayesianOptimizationResult:
    updated_params: dict[str, float]
    report: dict[str, object]
    output_path: Path


@dataclass(frozen=True)
class ParameterSpec:
    prior_mean: float
    prior_std: float
    valid_range: tuple[float, float]
    discrete_values: list[int] | None = None


class BayesianParameterOptimizer:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.evaluation_window = config.learning.bayesian.evaluation_window
        self.update_smoothing = config.learning.bayesian.update_smoothing
        self.grid_resolution = config.learning.bayesian.grid_resolution
        self.sharpe_scaling = config.learning.bayesian.sharpe_scaling
        self.parameter_specs = {
            "alpha": ParameterSpec(0.05, 0.03, (0.01, 0.15)),
            "J": ParameterSpec(3.0, 1.0, (1.0, 5.0), discrete_values=[1, 2, 3, 4, 5]),
            "sigma_scale": ParameterSpec(1.0, 0.3, (0.5, 2.0)),
            "zscore_threshold": ParameterSpec(1.5, 0.3, (0.8, 2.5)),
        }
        self._signals: pd.DataFrame | None = None
        self._position_scale_factor: float | None = None

    def run_optimization(
        self,
        trade_journal: TradeJournal,
        current_params: dict[str, float],
        eval_end_date: str | pd.Timestamp,
    ) -> BayesianOptimizationResult:
        eval_end = pd.Timestamp(eval_end_date)
        eval_start = eval_end - pd.tseries.offsets.BDay(self.evaluation_window - 1)
        strategy_a_trades = trade_journal.get_trades(
            start_date=eval_start,
            end_date=eval_end,
            strategy="A",
        )

        updated_params = dict(current_params)
        report: dict[str, object] = {
            "eval_start_date": eval_start.date().isoformat(),
            "eval_end_date": eval_end.date().isoformat(),
            "strategy_a_trade_count": int(len(strategy_a_trades)),
            "parameter_updates": {},
        }

        best_grid_sharpe = -np.inf
        for parameter_name, spec in self.parameter_specs.items():
            candidate_rows: list[dict[str, float]] = []
            for candidate_value in self._parameter_grid(spec):
                candidate_params = dict(updated_params)
                candidate_params[parameter_name] = float(candidate_value)
                sharpe = self.evaluate_parameter_set(candidate_params, eval_start, eval_end)
                prior = self._prior_weight(parameter_name, candidate_value)
                likelihood = math.exp(
                    np.clip((0.0 if sharpe is None else sharpe) * self.sharpe_scaling, -20.0, 20.0)
                )
                posterior_weight = prior * likelihood
                candidate_rows.append(
                    {
                        "value": float(candidate_value),
                        "sharpe": 0.0 if sharpe is None else float(sharpe),
                        "prior": float(prior),
                        "likelihood": float(likelihood),
                        "posterior_weight": float(posterior_weight),
                    }
                )
                best_grid_sharpe = max(best_grid_sharpe, 0.0 if sharpe is None else float(sharpe))

            candidate_frame = pd.DataFrame(candidate_rows)
            posterior_sum = float(candidate_frame["posterior_weight"].sum())
            if posterior_sum <= 0:
                posterior_mean = float(current_params[parameter_name])
            else:
                posterior_mean = float(
                    (candidate_frame["value"] * candidate_frame["posterior_weight"]).sum() / posterior_sum
                )
            smoothed_update = (
                self.update_smoothing * float(current_params[parameter_name])
                + (1.0 - self.update_smoothing) * posterior_mean
            )
            if parameter_name == "J":
                smoothed_update = float(np.clip(round(smoothed_update), 1, 5))

            updated_params[parameter_name] = float(smoothed_update)
            report["parameter_updates"][parameter_name] = {
                "prior_mean": float(spec.prior_mean),
                "posterior_mean": float(posterior_mean),
                "smoothed_update": float(smoothed_update),
                "movement": float(smoothed_update - current_params[parameter_name]),
                "candidates": candidate_rows,
            }

        report["best_grid_point_sharpe"] = float(best_grid_sharpe if np.isfinite(best_grid_sharpe) else 0.0)
        report["updated_params"] = updated_params
        output_path = self.config.paths.processed_dir / "bayesian_update_report.json"
        output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return BayesianOptimizationResult(
            updated_params=updated_params,
            report=report,
            output_path=output_path,
        )

    def evaluate_parameter_set(
        self,
        params: dict[str, float],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> float:
        signal_window = self._load_signal_history()
        mask = (
            (signal_window["date"] >= pd.Timestamp(start_date))
            & (signal_window["date"] <= pd.Timestamp(end_date))
        )
        candidate = signal_window.loc[mask].copy()
        if candidate.empty:
            return 0.0

        adjusted = self._apply_parameter_surface(candidate, params)
        if adjusted["target_position"].abs().sum() == 0:
            return 0.0
        result = run_walk_forward_backtest(adjusted, self.config.phase2, run_id="bayes-replay")
        sharpe = result.summary_metrics.get("sharpe_ratio")
        return 0.0 if sharpe is None else float(sharpe)

    def _load_signal_history(self) -> pd.DataFrame:
        if self._signals is not None:
            return self._signals

        validated_path = self.config.paths.processed_dir / "sector_etf_prices_validated.csv"
        price_history = pd.read_csv(validated_path, parse_dates=["date"])
        filtered = price_history.loc[
            price_history["is_valid"] & price_history["ticker"].isin(self.config.tickers),
            ["date", "ticker", "adj_close"],
        ].copy()
        raw_signals = compute_graph_signals(filtered, self.config.tickers, self.config.phase2)
        scaling_result = scale_signals_to_risk_budget(raw_signals, self.config.phase2)
        self._signals = scaling_result.scaled_signals.sort_values(["date", "ticker"]).reset_index(drop=True)
        target_abs_max = float(self._signals["target_position"].abs().max())
        self._position_scale_factor = (
            target_abs_max / self.config.phase2.max_position_size
            if self.config.phase2.max_position_size > 0
            else 1.0
        )
        return self._signals

    def _apply_parameter_surface(
        self,
        signals: pd.DataFrame,
        params: dict[str, float],
    ) -> pd.DataFrame:
        adjusted = signals.copy()
        base_alpha = max(self.config.phase2.diffusion_alpha, 1e-6)
        base_steps = max(self.config.phase2.diffusion_steps, 1)
        base_sigma_scale = max(self.config.phase2.sigma_scale, 1e-6)

        diffusion_multiplier = float(params["alpha"] / base_alpha) * math.sqrt(float(params["J"]) / base_steps)
        sigma_multiplier = base_sigma_scale / max(float(params["sigma_scale"]), 1e-6)
        adjusted["zscore"] = adjusted["zscore"].fillna(0.0) * diffusion_multiplier * sigma_multiplier

        scaled_max_position = self.config.phase2.max_position_size * float(self._position_scale_factor or 1.0)
        replay_config = replace(
            self.config.phase2,
            signal_threshold=float(params["zscore_threshold"]),
            max_position_size=scaled_max_position,
        )
        adjusted = apply_signal_rules(adjusted, replay_config)
        return adjusted

    def _parameter_grid(self, spec: ParameterSpec) -> list[float]:
        if spec.discrete_values is not None:
            return [float(value) for value in spec.discrete_values]
        low, high = spec.valid_range
        return [float(value) for value in np.linspace(low, high, self.grid_resolution)]

    def _prior_weight(self, parameter_name: str, value: float) -> float:
        spec = self.parameter_specs[parameter_name]
        if spec.discrete_values is not None:
            return 1.0 / len(spec.discrete_values)
        variance = max(spec.prior_std**2, 1e-6)
        exponent = -((float(value) - spec.prior_mean) ** 2) / (2.0 * variance)
        return float(math.exp(exponent))


def run_bayesian_update(config_path: str | Path, eval_end_date: str | pd.Timestamp) -> BayesianOptimizationResult:
    config = load_config(config_path)
    journal = TradeJournal(config.paths.project_root / config.learning.trade_journal_path)
    try:
        optimizer = BayesianParameterOptimizer(config)
        current_params = {
            "alpha": float(config.phase2.diffusion_alpha),
            "J": float(config.phase2.diffusion_steps),
            "sigma_scale": float(config.phase2.sigma_scale),
            "zscore_threshold": float(config.phase2.signal_threshold),
        }
        return optimizer.run_optimization(journal, current_params, eval_end_date)
    finally:
        journal.close()
