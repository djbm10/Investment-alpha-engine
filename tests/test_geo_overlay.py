import math

import pandas as pd
import pytest

from src.geo.overlay import (
    OVERLAY_OUTPUT_COLUMNS,
    REQUIRED_BASE_SIGNAL_COLUMNS,
    REQUIRED_SIGNAL_COLUMNS,
    apply_geo_overlay,
    compute_adjusted_entry_threshold,
    compute_adjusted_position_size,
    compute_contradiction,
    compute_geo_break_risk,
    compute_geo_net_score,
    compute_hard_override,
    compute_position_scale,
)


def test_compute_geo_net_score_prefers_precomputed_score_and_clips() -> None:
    snapshot = pd.DataFrame(
        {
            "asset": ["XLK", "XLE"],
            "geo_net_score": [1.5, -2.0],
            "geo_net_raw": [0.1, -0.1],
        }
    )

    scores = compute_geo_net_score(snapshot)

    assert scores.tolist() == [1.0, -1.0]


def test_compute_geo_net_score_falls_back_to_tanh_raw_when_score_missing() -> None:
    snapshot = pd.DataFrame(
        {
            "asset": ["XLK", "XLE"],
            "geo_net_raw": [2.0, -2.0],
        }
    )

    scores = compute_geo_net_score(snapshot)

    assert scores.iloc[0] == math.tanh(2.0)
    assert scores.iloc[1] == math.tanh(-2.0)


def test_compute_contradiction_only_penalizes_conflicting_signals() -> None:
    signal_direction = pd.Series([1, -1, 1, 0], dtype=float)
    geo_net_score = pd.Series([-0.6, -0.4, 0.5, 0.9], dtype=float)

    contradiction = compute_contradiction(signal_direction, geo_net_score)

    assert contradiction.tolist() == [0.6, 0.0, 0.0, 0.0]


def test_compute_geo_break_risk_uses_absolute_structural_score_and_clips() -> None:
    structural_score = pd.Series([-2.0, -0.4, 0.0, 0.7], dtype=float)

    break_risk = compute_geo_break_risk(structural_score)

    assert break_risk.tolist() == [1.0, 0.4, 0.0, 0.7]


def test_compute_adjusted_entry_threshold_applies_contradiction_and_break_terms() -> None:
    adjusted = compute_adjusted_entry_threshold(
        pd.Series([2.5], dtype=float),
        pd.Series([0.6], dtype=float),
        pd.Series([0.8], dtype=float),
        gamma=0.75,
    )

    assert adjusted.iloc[0] == 4.375


def test_compute_position_scale_and_adjusted_position_size_clip_extremes() -> None:
    scale = compute_position_scale(
        pd.Series([0.6, 1.0], dtype=float),
        pd.Series([0.8, 1.0], dtype=float),
    )
    adjusted_position = compute_adjusted_position_size(
        pd.Series([0.2, 0.2], dtype=float),
        scale,
    )

    assert scale.tolist() == pytest.approx([0.5, 0.25])
    assert adjusted_position.tolist() == pytest.approx([0.1, 0.05])


def test_compute_hard_override_requires_contradiction_confidence_coverage_and_freshness() -> None:
    override = compute_hard_override(
        pd.Series([1, 1, 1, -1], dtype=float),
        pd.Series([-0.8, -0.8, -0.8, -0.8], dtype=float),
        pd.Series([0.8, 0.69, 0.8, 0.8], dtype=float),
        pd.Series([0.8, 0.8, 0.69, 0.8], dtype=float),
        pd.Series([30, 30, 30, 30], dtype=float),
        hard_override_threshold=0.80,
        min_mapping_confidence=0.70,
        min_coverage_score=0.70,
        freshness_cutoff_minutes=120,
    )

    assert override.tolist() == [True, False, False, False]


def test_apply_geo_overlay_requires_signal_contract_columns() -> None:
    signals = pd.DataFrame(
        {
            "ticker": ["XLK"],
            "target_position": [0.20],
            "zscore": [-2.5],
        }
    )

    with pytest.raises(ValueError, match="signal_direction"):
        apply_geo_overlay(signals, None, base_signal_threshold=2.0)


def test_apply_geo_overlay_requires_base_signal_column() -> None:
    signals = pd.DataFrame(
        {
            "ticker": ["XLK"],
            "signal_direction": [1],
            "target_position": [0.20],
        }
    )

    with pytest.raises(ValueError, match="zscore|residual"):
        apply_geo_overlay(signals, None, base_signal_threshold=2.0)


def test_apply_geo_overlay_adjusts_threshold_position_and_override() -> None:
    signals = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-03-23")],
            "ticker": ["XLK"],
            "zscore": [-2.8],
            "signal_direction": [1],
            "target_position": [0.20],
            "regime_threshold_multiplier": [1.25],
        }
    )
    snapshot = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-03-23").date()],
            "asset": ["XLK"],
            "geo_net_score": [-0.60],
            "geo_structural_score": [-0.80],
            "avg_mapping_confidence": [0.90],
            "coverage_score": [0.90],
            "data_freshness_minutes": [30],
        }
    )

    adjusted = apply_geo_overlay(
        signals,
        snapshot,
        base_signal_threshold=2.0,
        gamma=0.75,
        hard_override_threshold=0.80,
    )

    row = adjusted.iloc[0]
    assert set(OVERLAY_OUTPUT_COLUMNS).issubset(adjusted.columns)
    assert row["geo_net_score"] == -0.60
    assert row["effective_geo_net_score"] == -0.60
    assert row["contradiction"] == 0.60
    assert row["geo_break_risk"] == 0.80
    assert row["base_entry_threshold"] == 2.5
    assert row["adjusted_entry_threshold"] == 4.375
    assert row["position_scale"] == pytest.approx(0.50)
    assert row["adjusted_target_position"] == pytest.approx(0.10)
    assert bool(row["hard_override"]) is True


def test_apply_geo_overlay_returns_neutral_outputs_when_disabled() -> None:
    signals = pd.DataFrame(
        {
            "ticker": ["XLK"],
            "zscore": [-2.5],
            "signal_direction": [1],
            "target_position": [0.20],
        }
    )
    snapshot = pd.DataFrame(
        {
            "asset": ["XLK"],
            "geo_net_score": [-1.0],
            "geo_structural_score": [-1.0],
            "avg_mapping_confidence": [1.0],
            "coverage_score": [1.0],
            "data_freshness_minutes": [0],
        }
    )

    adjusted = apply_geo_overlay(
        signals,
        snapshot,
        base_signal_threshold=2.0,
        enabled=False,
    )

    row = adjusted.iloc[0]
    assert set(REQUIRED_SIGNAL_COLUMNS).issubset(signals.columns)
    assert REQUIRED_BASE_SIGNAL_COLUMNS.intersection(signals.columns)
    assert set(OVERLAY_OUTPUT_COLUMNS).issubset(adjusted.columns)
    assert row["geo_net_score"] == 0.0
    assert row["contradiction"] == 0.0
    assert row["geo_break_risk"] == 0.0
    assert row["adjusted_entry_threshold"] == 2.0
    assert row["adjusted_target_position"] == 0.20
    assert bool(row["hard_override"]) is False


def test_apply_geo_overlay_handles_missing_snapshot_rows() -> None:
    signals = pd.DataFrame(
        {
            "ticker": ["XLK", "XLE"],
            "zscore": [-2.5, 2.4],
            "signal_direction": [1, -1],
            "target_position": [0.20, -0.15],
        }
    )
    snapshot = pd.DataFrame(
        {
            "asset": ["XLK"],
            "geo_net_score": [-0.50],
            "geo_structural_score": [-0.50],
            "avg_mapping_confidence": [0.90],
            "coverage_score": [0.90],
            "data_freshness_minutes": [30],
        }
    )

    adjusted = apply_geo_overlay(signals, snapshot, base_signal_threshold=2.0)

    xle_row = adjusted.loc[adjusted["ticker"] == "XLE"].iloc[0]
    assert xle_row["geo_net_score"] == 0.0
    assert xle_row["adjusted_entry_threshold"] == 2.0
    assert xle_row["adjusted_target_position"] == -0.15
    assert bool(xle_row["hard_override"]) is False


def test_apply_geo_overlay_clips_extreme_geo_values() -> None:
    signals = pd.DataFrame(
        {
            "ticker": ["XLK"],
            "zscore": [-3.0],
            "signal_direction": [1],
            "target_position": [0.20],
        }
    )
    snapshot = pd.DataFrame(
        {
            "asset": ["XLK"],
            "geo_net_score": [-5.0],
            "geo_structural_score": [-10.0],
            "avg_mapping_confidence": [0.95],
            "coverage_score": [0.95],
            "data_freshness_minutes": [10],
        }
    )

    adjusted = apply_geo_overlay(signals, snapshot, base_signal_threshold=2.0, gamma=0.75)

    row = adjusted.iloc[0]
    assert row["geo_net_score"] == -1.0
    assert row["geo_structural_score"] == -1.0
    assert row["contradiction"] == 1.0
    assert row["geo_break_risk"] == 1.0
    assert row["position_scale"] == pytest.approx(0.25)
    assert row["adjusted_target_position"] == pytest.approx(0.05)


def test_apply_geo_overlay_blocks_override_when_confidence_coverage_or_freshness_fail() -> None:
    signals = pd.DataFrame(
        {
            "ticker": ["XLK", "XLE", "XLF"],
            "zscore": [-2.5, -2.5, -2.5],
            "signal_direction": [1, 1, 1],
            "target_position": [0.20, 0.20, 0.20],
        }
    )
    snapshot = pd.DataFrame(
        {
            "asset": ["XLK", "XLE", "XLF"],
            "geo_net_score": [-0.9, -0.9, -0.9],
            "geo_structural_score": [-0.9, -0.9, -0.9],
            "avg_mapping_confidence": [0.69, 0.90, 0.90],
            "coverage_score": [0.90, 0.69, 0.90],
            "data_freshness_minutes": [30, 30, 121],
        }
    )

    adjusted = apply_geo_overlay(
        signals,
        snapshot,
        base_signal_threshold=2.0,
        hard_override_threshold=0.80,
        min_mapping_confidence=0.70,
        min_coverage_score=0.70,
        freshness_cutoff_minutes=120,
    )

    assert adjusted["hard_override"].tolist() == [False, False, False]


def test_apply_geo_overlay_aligns_by_ticker_not_row_order() -> None:
    signals = pd.DataFrame(
        {
            "ticker": ["XLK", "XLE"],
            "zscore": [-2.5, -2.5],
            "signal_direction": [1, 1],
            "target_position": [0.20, 0.20],
        }
    )
    snapshot = pd.DataFrame(
        {
            "asset": ["XLE", "XLK"],
            "geo_net_score": [-0.2, -0.8],
            "geo_structural_score": [-0.2, -0.8],
            "avg_mapping_confidence": [0.9, 0.9],
            "coverage_score": [0.9, 0.9],
            "data_freshness_minutes": [30, 30],
        }
    )

    adjusted = apply_geo_overlay(signals, snapshot, base_signal_threshold=2.0)

    xlk_row = adjusted.loc[adjusted["ticker"] == "XLK"].iloc[0]
    xle_row = adjusted.loc[adjusted["ticker"] == "XLE"].iloc[0]
    assert xlk_row["geo_net_score"] == -0.8
    assert xle_row["geo_net_score"] == -0.2


def test_apply_geo_overlay_low_coverage_is_near_neutral_not_confident() -> None:
    signals = pd.DataFrame(
        {
            "ticker": ["XLK"],
            "zscore": [-2.5],
            "signal_direction": [1],
            "target_position": [0.20],
        }
    )
    snapshot = pd.DataFrame(
        {
            "asset": ["XLK"],
            "geo_net_score": [-1.0],
            "geo_structural_score": [-1.0],
            "avg_mapping_confidence": [0.95],
            "coverage_score": [0.10],
            "data_freshness_minutes": [30],
        }
    )

    adjusted = apply_geo_overlay(
        signals,
        snapshot,
        base_signal_threshold=2.0,
        min_coverage_score=0.70,
    )

    row = adjusted.iloc[0]
    assert row["geo_net_score"] == -1.0
    assert row["geo_reliability_weight"] == pytest.approx(0.10 / 0.70)
    assert row["contradiction"] < 0.2
    assert row["adjusted_entry_threshold"] < 2.5
    assert row["adjusted_target_position"] > 0.15
    assert bool(row["hard_override"]) is False


def test_apply_geo_overlay_monotonic_in_contradictory_geo_direction() -> None:
    signals = pd.DataFrame(
        {
            "ticker": ["XLK", "XLE", "XLF"],
            "zscore": [-2.5, -2.5, -2.5],
            "signal_direction": [1, 1, 1],
            "target_position": [0.20, 0.20, 0.20],
        }
    )
    snapshot = pd.DataFrame(
        {
            "asset": ["XLK", "XLE", "XLF"],
            "geo_net_score": [-0.1, -0.5, -0.9],
            "geo_structural_score": [-0.1, -0.5, -0.9],
            "avg_mapping_confidence": [0.9, 0.9, 0.9],
            "coverage_score": [0.9, 0.9, 0.9],
            "data_freshness_minutes": [30, 30, 30],
        }
    )

    adjusted = apply_geo_overlay(signals, snapshot, base_signal_threshold=2.0)
    thresholds = adjusted["adjusted_entry_threshold"].tolist()
    positions = adjusted["adjusted_target_position"].tolist()

    assert thresholds[0] < thresholds[1] < thresholds[2]
    assert positions[0] > positions[1] > positions[2]


def test_apply_geo_overlay_missing_structural_score_stays_non_structural() -> None:
    signals = pd.DataFrame(
        {
            "ticker": ["XLK"],
            "zscore": [-2.5],
            "signal_direction": [1],
            "target_position": [0.20],
        }
    )
    snapshot = pd.DataFrame(
        {
            "asset": ["XLK"],
            "geo_net_score": [-0.9],
            "avg_mapping_confidence": [0.9],
            "coverage_score": [0.9],
            "data_freshness_minutes": [30],
        }
    )

    adjusted = apply_geo_overlay(signals, snapshot, base_signal_threshold=2.0)

    row = adjusted.iloc[0]
    assert row["geo_net_score"] == -0.9
    assert row["geo_structural_score"] == 0.0
    assert row["geo_break_risk"] == 0.0
    assert bool(row["hard_override"]) is False
