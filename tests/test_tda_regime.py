import builtins
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.config_loader import Phase3Config
from src.phase3 import _compute_regime_observations
from src.tda_regime import RegimeState, TDARegimeDetector


def _phase3_config(**overrides) -> Phase3Config:
    values = {
        "rolling_window": 60,
        "wasserstein_lookback": 3,
        "transition_threshold_sigma": 1.5,
        "new_regime_threshold_sigma": 2.5,
        "confirmation_window": 5,
        "confirmation_required_transition": 3,
        "confirmation_required_new_regime": 4,
        "transition_position_scale": 0.5,
        "transition_threshold_mult": 1.25,
        "new_regime_position_scale": 0.5,
        "new_regime_threshold_mult": 1.25,
        "transition_lookback_cap": 30,
        "new_regime_freeze_days": 0,
        "emergency_recalib_days": 5,
        "emergency_lookback": 20,
    }
    values.update(overrides)
    return Phase3Config(**values)


def _distance_matrix(level: float) -> np.ndarray:
    matrix = np.full((8, 8), level, dtype=float)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def test_compute_persistence_diagram_returns_dimension_column() -> None:
    detector = TDARegimeDetector(_phase3_config())

    if not detector.enabled:
        pytest.skip("TDA dependencies are unavailable in this environment.")

    diagram = detector.compute_persistence_diagram(_distance_matrix(0.5))

    assert diagram.ndim == 2
    assert diagram.shape[1] == 3
    assert set(np.unique(diagram[:, 2])).issubset({0.0, 1.0})


def test_compute_daily_regime_requires_confirmed_large_topology_shift() -> None:
    detector = TDARegimeDetector(
        _phase3_config(
            wasserstein_lookback=2,
            transition_threshold_sigma=1.0,
            new_regime_threshold_sigma=1.5,
            confirmation_window=3,
            confirmation_required_transition=2,
            confirmation_required_new_regime=2,
        )
    )
    if not detector.enabled:
        pytest.skip("TDA dependencies are unavailable in this environment.")

    dates = pd.bdate_range("2024-01-02", periods=8)
    matrices = {date: _distance_matrix(0.3) for date in dates}
    total_distances = [0.05, 0.05, 4.0, 0.05, 8.0, 12.0, 12.0]
    mocked_distances = [value for distance in total_distances for value in (distance, distance)]

    with patch("src.tda_regime._wasserstein_distance", side_effect=mocked_distances):
        observations = detector.compute_daily_regime(matrices)

    assert observations[dates[0]].state == RegimeState.STABLE
    assert observations[dates[3]].state == RegimeState.STABLE
    assert observations[dates[6]].state in {RegimeState.TRANSITIONING, RegimeState.NEW_REGIME}


def test_disable_tda_skips_optional_imports_and_returns_stable_observations() -> None:
    import src.tda_regime as tda_regime

    attempted_imports: list[str] = []
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"persim", "ripser"}:
            attempted_imports.append(name)
            raise AssertionError(f"Unexpected import attempted for {name}.")
        return original_import(name, globals, locals, fromlist, level)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("DISABLE_TDA", "true")
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        tda_regime = importlib.reload(tda_regime)

        detector = tda_regime.TDARegimeDetector(_phase3_config())
        dates = pd.bdate_range("2024-01-02", periods=3)
        observations = detector.compute_daily_regime({date: _distance_matrix(0.2) for date in dates})

        assert detector.enabled is False
        assert attempted_imports == []
        assert list(observations) == list(dates)
        assert all(observation.state == tda_regime.RegimeState.STABLE for observation in observations.values())
        assert all(observation.total_distance is None for observation in observations.values())

    importlib.reload(tda_regime)


def test_phase3_regime_observations_skip_graph_matrices_when_tda_disabled() -> None:
    class DisabledDetector:
        def __init__(self, config) -> None:
            self.enabled = False

    price_history = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "ticker": "XLF", "adj_close": 10.0},
            {"date": pd.Timestamp("2024-01-02"), "ticker": "XLK", "adj_close": 20.0},
            {"date": pd.Timestamp("2024-01-03"), "ticker": "XLF", "adj_close": 10.5},
            {"date": pd.Timestamp("2024-01-03"), "ticker": "XLK", "adj_close": 20.5},
        ]
    )
    config = SimpleNamespace(tickers=["XLF", "XLK"], phase3=_phase3_config())

    with patch("src.phase3.TDARegimeDetector", DisabledDetector):
        with patch("src.phase3.compute_daily_graph_matrices", side_effect=AssertionError("Should not run.")):
            observations = _compute_regime_observations(price_history, config)

    assert list(observations) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert all(observation.state == RegimeState.STABLE for observation in observations.values())
