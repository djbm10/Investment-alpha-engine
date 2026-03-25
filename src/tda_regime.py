from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

_DISABLE_TDA = os.getenv("DISABLE_TDA") == "true"
_TDA_IMPORT_ERROR: Exception | None = None

if not _DISABLE_TDA:
    try:
        from persim import wasserstein
        from ripser import ripser
    except Exception as exc:
        wasserstein = None
        ripser = None
        _TDA_IMPORT_ERROR = exc
else:
    wasserstein = None
    ripser = None

from .config_loader import Phase3Config, PipelineConfig


class RegimeState(str, Enum):
    STABLE = "STABLE"
    TRANSITIONING = "TRANSITIONING"
    NEW_REGIME = "NEW_REGIME"


@dataclass(frozen=True)
class RegimeObservation:
    state: RegimeState
    total_distance: float | None
    h0_distance: float | None
    h1_distance: float | None
    rolling_mean: float | None
    rolling_std: float | None


def tda_enabled() -> bool:
    return not _DISABLE_TDA and ripser is not None and wasserstein is not None


def stable_regime_observations(dates: Iterable[object]) -> dict[pd.Timestamp, RegimeObservation]:
    observations: dict[pd.Timestamp, RegimeObservation] = {}
    for date in sorted({pd.Timestamp(value) for value in dates}):
        observations[date] = RegimeObservation(
            state=RegimeState.STABLE,
            total_distance=None,
            h0_distance=None,
            h1_distance=None,
            rolling_mean=None,
            rolling_std=None,
        )
    return observations


def stable_regime_observations_from_price_history(
    price_history: pd.DataFrame,
) -> dict[pd.Timestamp, RegimeObservation]:
    if price_history.empty or "date" not in price_history:
        return {}
    return stable_regime_observations(pd.to_datetime(price_history["date"]))


class TDARegimeDetector:
    def __init__(self, config: PipelineConfig | Phase3Config):
        self.config = config.phase3 if isinstance(config, PipelineConfig) else config
        self.enabled = tda_enabled()

    def compute_persistence_diagram(self, distance_matrix: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return np.empty((0, 3), dtype=float)
        ripser_result = ripser(distance_matrix, distance_matrix=True, maxdim=1)
        diagrams = [
            np.column_stack([diagram, np.full(len(diagram), dimension)])
            for dimension, diagram in enumerate(ripser_result["dgms"][:2])
            if len(diagram) > 0
        ]
        if not diagrams:
            return np.empty((0, 3), dtype=float)
        return np.vstack(diagrams).astype(float)

    def compute_daily_regime(
        self,
        distance_matrices: dict[pd.Timestamp, np.ndarray],
    ) -> dict[pd.Timestamp, RegimeObservation]:
        if not self.enabled:
            return stable_regime_observations(distance_matrices.keys())

        observations: dict[pd.Timestamp, RegimeObservation] = {}
        previous_h0: np.ndarray | None = None
        previous_h1: np.ndarray | None = None
        prior_total_distances: list[float] = []
        transition_exceedances: list[bool] = []
        new_regime_exceedances: list[bool] = []

        for date in sorted(distance_matrices):
            diagram = self.compute_persistence_diagram(distance_matrices[date])
            h0_diagram, h1_diagram = _split_diagram_by_dimension(diagram)

            if previous_h0 is None or previous_h1 is None:
                observations[pd.Timestamp(date)] = RegimeObservation(
                    state=RegimeState.STABLE,
                    total_distance=None,
                    h0_distance=None,
                    h1_distance=None,
                    rolling_mean=None,
                    rolling_std=None,
                )
            else:
                h0_distance = _wasserstein_distance(previous_h0, h0_diagram)
                h1_distance = _wasserstein_distance(previous_h1, h1_diagram)
                total_distance = 0.5 * h0_distance + 0.5 * h1_distance

                if len(prior_total_distances) < self.config.wasserstein_lookback:
                    observations[pd.Timestamp(date)] = RegimeObservation(
                        state=RegimeState.STABLE,
                        total_distance=total_distance,
                        h0_distance=h0_distance,
                        h1_distance=h1_distance,
                        rolling_mean=None,
                        rolling_std=None,
                    )
                else:
                    trailing = np.asarray(prior_total_distances[-self.config.wasserstein_lookback :], dtype=float)
                    rolling_mean = float(trailing.mean())
                    rolling_std = float(trailing.std(ddof=0))
                    transition_threshold = rolling_mean + self.config.transition_threshold_sigma * rolling_std
                    new_regime_threshold = rolling_mean + self.config.new_regime_threshold_sigma * rolling_std
                    transition_exceeded = total_distance > transition_threshold
                    new_regime_exceeded = total_distance > new_regime_threshold
                    transition_exceedances.append(transition_exceeded)
                    new_regime_exceedances.append(new_regime_exceeded)
                    recent_transition_count = _recent_true_count(
                        transition_exceedances,
                        self.config.confirmation_window,
                    )
                    recent_new_regime_count = _recent_true_count(
                        new_regime_exceedances,
                        self.config.confirmation_window,
                    )

                    if recent_new_regime_count >= self.config.confirmation_required_new_regime:
                        state = RegimeState.NEW_REGIME
                    elif recent_transition_count >= self.config.confirmation_required_transition:
                        state = RegimeState.TRANSITIONING
                    else:
                        state = RegimeState.STABLE

                    observations[pd.Timestamp(date)] = RegimeObservation(
                        state=state,
                        total_distance=total_distance,
                        h0_distance=h0_distance,
                        h1_distance=h1_distance,
                        rolling_mean=rolling_mean,
                        rolling_std=rolling_std,
                    )

                prior_total_distances.append(total_distance)

            previous_h0 = h0_diagram
            previous_h1 = h1_diagram

        return observations


def _split_diagram_by_dimension(diagram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if diagram.size == 0:
        return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float)
    h0 = diagram[diagram[:, 2] == 0][:, :2]
    h1 = diagram[diagram[:, 2] == 1][:, :2]
    return h0.astype(float), h1.astype(float)


def _wasserstein_distance(previous: np.ndarray, current: np.ndarray) -> float:
    previous = _finite_diagram(previous)
    current = _finite_diagram(current)
    if len(previous) == 0 and len(current) == 0:
        return 0.0
    if len(previous) == 0:
        previous = np.empty((0, 2), dtype=float)
    if len(current) == 0:
        current = np.empty((0, 2), dtype=float)
    return float(wasserstein(previous, current, matching=False))


def _finite_diagram(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.empty((0, 2), dtype=float)
    finite_mask = np.isfinite(diagram[:, 1])
    return diagram[finite_mask].astype(float)


def _recent_true_count(values: list[bool], window: int) -> int:
    if not values or window <= 0:
        return 0
    return int(sum(bool(value) for value in values[-window:]))
