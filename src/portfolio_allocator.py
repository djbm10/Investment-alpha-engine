from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config_loader import Phase5Config, PipelineConfig


@dataclass(frozen=True)
class AllocationSnapshot:
    date: pd.Timestamp
    weights: dict[str, float]
    utilities: dict[str, float]
    rebalanced: bool


class DynamicAllocator:
    def __init__(self, config: PipelineConfig | Phase5Config) -> None:
        self.config = config.phase5 if isinstance(config, PipelineConfig) else config

    def compute_utility(
        self,
        strategy_id: str,
        date: pd.Timestamp | str,
        strategy_returns: pd.Series,
        strategy_costs: pd.Series,
    ) -> float:
        timestamp = pd.Timestamp(date)
        returns_window = strategy_returns.loc[strategy_returns.index < timestamp].tail(
            self.config.performance_lookback
        )
        costs_window = strategy_costs.loc[strategy_costs.index < timestamp].tail(
            self.config.performance_lookback
        )
        if returns_window.empty:
            return 0.0
        predicted_return = float(returns_window.mean())
        uncertainty = float(returns_window.std(ddof=0))
        cost = float(costs_window.mean()) if not costs_window.empty else 0.0
        return predicted_return - self.config.utility_lambda_uncertainty * uncertainty - self.config.utility_lambda_cost * cost

    def compute_allocations(
        self,
        date: pd.Timestamp | str,
        strategy_utilities: dict[str, float],
    ) -> dict[str, float]:
        if not strategy_utilities:
            raise ValueError("At least one strategy utility is required.")
        strategy_ids = list(strategy_utilities)
        utility_values = np.array([strategy_utilities[strategy_id] for strategy_id in strategy_ids], dtype=float)
        scaled = utility_values / max(self.config.softmax_temperature, 1e-6)
        stabilized = scaled - scaled.max()
        exp_values = np.exp(stabilized)
        weights = exp_values / exp_values.sum()
        clipped = np.clip(weights, self.config.min_allocation, self.config.max_allocation)
        normalized = clipped / clipped.sum()
        return {strategy_id: float(normalized[idx]) for idx, strategy_id in enumerate(strategy_ids)}

    def should_rebalance(
        self,
        date: pd.Timestamp | str,
        last_rebalance_date: pd.Timestamp | None,
    ) -> bool:
        if last_rebalance_date is None:
            return True
        current = pd.Timestamp(date)
        elapsed_days = len(pd.bdate_range(last_rebalance_date, current)) - 1
        return elapsed_days >= self.config.rebalance_frequency_days
