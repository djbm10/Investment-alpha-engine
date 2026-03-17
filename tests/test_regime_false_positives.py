import pandas as pd

from src.diagnostics.regime_false_positives import (
    build_false_positive_breakdown,
    build_false_positive_summary_text,
    compute_drawdown_start_dates,
)
from src.diagnostics.regime_validation import CrisisWindow


def test_build_false_positive_breakdown_classifies_isolated_cluster_and_pre_drawdown() -> None:
    regime_history = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "state": "STABLE", "total_distance": 0.10},
            {"date": pd.Timestamp("2024-01-03"), "state": "TRANSITIONING", "total_distance": 1.00},
            {"date": pd.Timestamp("2024-01-04"), "state": "STABLE", "total_distance": 0.11},
            {"date": pd.Timestamp("2024-01-05"), "state": "TRANSITIONING", "total_distance": 2.00},
            {"date": pd.Timestamp("2024-01-08"), "state": "NEW_REGIME", "total_distance": 2.20},
            {"date": pd.Timestamp("2024-01-09"), "state": "TRANSITIONING", "total_distance": 3.00},
            {"date": pd.Timestamp("2024-01-10"), "state": "STABLE", "total_distance": 0.09},
            {"date": pd.Timestamp("2024-01-18"), "state": "NEW_REGIME", "total_distance": 9.00},
        ]
    )
    crisis_windows = [CrisisWindow("Crisis", pd.Timestamp("2024-01-18"), pd.Timestamp("2024-01-18"))]
    drawdown_start_dates = [pd.Timestamp("2024-01-19")]

    breakdown = build_false_positive_breakdown(regime_history, crisis_windows, drawdown_start_dates)

    january_3 = breakdown.loc[breakdown["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    january_5 = breakdown.loc[breakdown["date"] == pd.Timestamp("2024-01-05")].iloc[0]
    january_8 = breakdown.loc[breakdown["date"] == pd.Timestamp("2024-01-08")].iloc[0]
    january_9 = breakdown.loc[breakdown["date"] == pd.Timestamp("2024-01-09")].iloc[0]

    assert january_3["classification"] == "ISOLATED"
    assert january_5["classification"] == "CLUSTER"
    assert january_8["classification"] == "CLUSTER"
    assert january_9["classification"] == "PRE_DRAWDOWN"
    assert january_5["cluster_size"] == 3
    assert january_9["days_to_drawdown_start"] == 10


def test_false_positive_summary_reports_breakdown_and_distance_separation() -> None:
    regime_history = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "state": "TRANSITIONING", "total_distance": 1.00},
            {"date": pd.Timestamp("2024-01-03"), "state": "TRANSITIONING", "total_distance": 2.00},
            {"date": pd.Timestamp("2024-01-18"), "state": "NEW_REGIME", "total_distance": 8.00},
        ]
    )
    crisis_windows = [CrisisWindow("Crisis", pd.Timestamp("2024-01-18"), pd.Timestamp("2024-01-18"))]
    breakdown = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "state": "TRANSITIONING", "total_distance": 1.00, "classification": "ISOLATED", "days_to_drawdown_start": None, "cluster_size": 1},
            {"date": pd.Timestamp("2024-01-03"), "state": "TRANSITIONING", "total_distance": 2.00, "classification": "PRE_DRAWDOWN", "days_to_drawdown_start": 2, "cluster_size": 1},
        ]
    )

    summary = build_false_positive_summary_text(regime_history, breakdown, crisis_windows)

    assert "2 of 3 flagged regime days were outside the crisis buffers." in summary
    assert "1 were isolated spikes, 0 were clustered flags, and 1 preceded a >1% equal-weight universe drawdown" in summary
    assert "8.0000" in summary
    assert "1.5000" in summary


def test_compute_drawdown_start_dates_flags_threshold_crossings() -> None:
    price_history = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "ticker": "XLK", "adj_close": 100.0},
            {"date": pd.Timestamp("2024-01-02"), "ticker": "XLV", "adj_close": 100.0},
            {"date": pd.Timestamp("2024-01-03"), "ticker": "XLK", "adj_close": 100.0},
            {"date": pd.Timestamp("2024-01-03"), "ticker": "XLV", "adj_close": 100.0},
            {"date": pd.Timestamp("2024-01-04"), "ticker": "XLK", "adj_close": 98.0},
            {"date": pd.Timestamp("2024-01-04"), "ticker": "XLV", "adj_close": 98.0},
            {"date": pd.Timestamp("2024-01-05"), "ticker": "XLK", "adj_close": 97.0},
            {"date": pd.Timestamp("2024-01-05"), "ticker": "XLV", "adj_close": 97.0},
            {"date": pd.Timestamp("2024-01-08"), "ticker": "XLK", "adj_close": 101.0},
            {"date": pd.Timestamp("2024-01-08"), "ticker": "XLV", "adj_close": 101.0},
            {"date": pd.Timestamp("2024-01-09"), "ticker": "XLK", "adj_close": 98.5},
            {"date": pd.Timestamp("2024-01-09"), "ticker": "XLV", "adj_close": 98.5},
        ]
    )

    drawdown_start_dates = compute_drawdown_start_dates(price_history, threshold=0.01)

    assert drawdown_start_dates == [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-09")]
