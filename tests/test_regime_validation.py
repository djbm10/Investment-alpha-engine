import pandas as pd

from src.diagnostics.regime_validation import (
    CrisisWindow,
    compute_false_positive_rate,
    evaluate_crisis_detection,
)
from src.tda_regime import RegimeObservation, RegimeState


def test_evaluate_crisis_detection_scores_hits_by_window() -> None:
    observations = {
        pd.Timestamp("2020-02-20"): RegimeObservation(RegimeState.TRANSITIONING, 1.0, 1.0, 1.0, 0.1, 0.1),
        pd.Timestamp("2022-08-01"): RegimeObservation(RegimeState.TRANSITIONING, 1.0, 1.0, 1.0, 0.1, 0.1),
    }
    windows = [
        CrisisWindow("A", pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23")),
        CrisisWindow("B", pd.Timestamp("2022-01-03"), pd.Timestamp("2022-06-16")),
    ]

    records, hit_rate = evaluate_crisis_detection(observations, windows)

    assert len(records) == 2
    assert records[0].detected_within_window is True
    assert records[1].detected_within_window is False
    assert hit_rate == 0.5


def test_compute_false_positive_rate_uses_crisis_buffer() -> None:
    observations = {
        pd.Timestamp("2020-02-18"): RegimeObservation(RegimeState.TRANSITIONING, 1.0, 1.0, 1.0, 0.1, 0.1),
        pd.Timestamp("2020-03-01"): RegimeObservation(RegimeState.NEW_REGIME, 1.0, 1.0, 1.0, 0.1, 0.1),
        pd.Timestamp("2020-05-01"): RegimeObservation(RegimeState.TRANSITIONING, 1.0, 1.0, 1.0, 0.1, 0.1),
    }
    windows = [CrisisWindow("A", pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"))]

    false_positive_rate = compute_false_positive_rate(observations, windows)

    assert false_positive_rate == 1 / 3
