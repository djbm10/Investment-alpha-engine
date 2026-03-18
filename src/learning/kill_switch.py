from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config_loader import KillSwitchConfig, PipelineConfig


@dataclass(frozen=True)
class KillSwitchEvaluation:
    status: str
    rolling_60d_sharpe: float
    rolling_120d_sharpe: float
    rolling_reactivation_sharpe: float
    reactivation_streak: int


class StrategyKillSwitch:
    def __init__(self, config: PipelineConfig | KillSwitchConfig) -> None:
        self.config = config.learning.kill_switch if isinstance(config, PipelineConfig) else config

    def evaluate(
        self,
        strategy_id: str,
        daily_returns: pd.Series,
        current_date: pd.Timestamp | str,
    ) -> KillSwitchEvaluation:
        del strategy_id
        current = pd.Timestamp(current_date)
        returns = daily_returns.sort_index()
        historical = returns.loc[returns.index <= current].fillna(0.0)

        rolling_60 = _rolling_sharpe_window(historical, self.config.reduction_lookback_days)
        rolling_120 = _rolling_sharpe_window(historical, self.config.quarantine_lookback_days)
        rolling_reactivation = _rolling_sharpe_window(historical, self.config.reactivation_lookback_days)
        reactivation_streak = _positive_streak(
            historical,
            self.config.reactivation_lookback_days,
            self.config.reactivation_threshold,
        )
        was_quarantined = _has_quarantine_history(
            historical,
            self.config.quarantine_lookback_days,
            self.config.quarantine_threshold,
        )

        if (
            was_quarantined
            and reactivation_streak >= self.config.reactivation_days
            and rolling_reactivation > self.config.reactivation_threshold
        ):
            status = "REACTIVATE"
        elif rolling_120 < self.config.quarantine_threshold:
            status = "QUARANTINED"
        elif rolling_60 < self.config.reduction_threshold:
            status = "REDUCED"
        else:
            status = "ACTIVE"

        return KillSwitchEvaluation(
            status=status,
            rolling_60d_sharpe=rolling_60,
            rolling_120d_sharpe=rolling_120,
            rolling_reactivation_sharpe=rolling_reactivation,
            reactivation_streak=reactivation_streak,
        )


def _rolling_sharpe_window(returns: pd.Series, window: int) -> float:
    if len(returns) < window:
        return 0.0
    sample = returns.tail(window)
    volatility = float(sample.std(ddof=0))
    if volatility == 0.0:
        mean_return = float(sample.mean())
        if mean_return > 0:
            return 1_000_000.0
        if mean_return < 0:
            return -1_000_000.0
        return 0.0
    return float((sample.mean() / volatility) * np.sqrt(252))


def _positive_streak(returns: pd.Series, window: int, threshold: float) -> int:
    if len(returns) < window:
        return 0
    streak = 0
    for index in range(window, len(returns) + 1):
        sample = returns.iloc[index - window : index]
        sharpe = _rolling_sharpe_window(sample, window)
        if sharpe > threshold:
            streak += 1
        else:
            streak = 0
    return streak


def _has_quarantine_history(returns: pd.Series, window: int, threshold: float) -> bool:
    if len(returns) < window:
        return False
    for index in range(window, len(returns) + 1):
        sample = returns.iloc[index - window : index]
        if _rolling_sharpe_window(sample, window) < threshold:
            return True
    return False
