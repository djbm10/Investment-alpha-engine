from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config_loader import Phase3Config
from src.tda_regime import RegimeState, TDARegimeDetector


def _distance_matrix(level: float) -> np.ndarray:
    matrix = np.full((8, 8), level, dtype=float)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def test_compute_persistence_diagram_returns_dimension_column() -> None:
    detector = TDARegimeDetector(
        Phase3Config(
            rolling_window=60,
            wasserstein_lookback=3,
            transition_threshold_sigma=1.5,
            new_regime_threshold_sigma=2.5,
            confirmation_window=5,
            confirmation_required_transition=3,
            confirmation_required_new_regime=4,
            transition_position_scale=0.5,
            transition_threshold_mult=1.25,
            transition_lookback_cap=30,
            new_regime_freeze_days=0,
            emergency_recalib_days=5,
            emergency_lookback=20,
        )
    )

    diagram = detector.compute_persistence_diagram(_distance_matrix(0.5))

    assert diagram.ndim == 2
    assert diagram.shape[1] == 3
    assert set(np.unique(diagram[:, 2])).issubset({0.0, 1.0})


def test_compute_daily_regime_requires_confirmed_large_topology_shift() -> None:
    detector = TDARegimeDetector(
        Phase3Config(
            rolling_window=60,
            wasserstein_lookback=2,
            transition_threshold_sigma=1.0,
            new_regime_threshold_sigma=1.5,
            confirmation_window=3,
            confirmation_required_transition=2,
            confirmation_required_new_regime=2,
            transition_position_scale=0.5,
            transition_threshold_mult=1.25,
            transition_lookback_cap=30,
            new_regime_freeze_days=0,
            emergency_recalib_days=5,
            emergency_lookback=20,
        )
    )
    dates = pd.bdate_range("2024-01-02", periods=8)
    matrices = {date: _distance_matrix(0.3) for date in dates}
    total_distances = [0.05, 0.05, 4.0, 0.05, 8.0, 12.0, 12.0]
    mocked_distances = [value for distance in total_distances for value in (distance, distance)]

    with patch("src.tda_regime._wasserstein_distance", side_effect=mocked_distances):
        observations = detector.compute_daily_regime(matrices)

    assert observations[dates[0]].state == RegimeState.STABLE
    assert observations[dates[3]].state == RegimeState.STABLE
    assert observations[dates[6]].state in {RegimeState.TRANSITIONING, RegimeState.NEW_REGIME}
