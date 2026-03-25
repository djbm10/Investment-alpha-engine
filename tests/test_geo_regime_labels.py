from __future__ import annotations

import pandas as pd

from src.geo.regime_labels import (
    GEO_STRESS_REGIME,
    NORMAL_REGIME,
    STRUCTURAL_GEO_REGIME,
    GeoRegimeDefinition,
    build_geo_regime_labels,
    build_placebo_regime_frames,
)


def _build_snapshot_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=7, freq="B")
    rows: list[dict[str, object]] = []
    for date in dates:
        day = date.strftime("%Y-%m-%d")
        if day == "2024-01-02":
            net_score = 0.0
            structural_score = 0.0
            coverage = 0.56
        elif day in {"2024-01-03", "2024-01-04"}:
            net_score = 0.02
            structural_score = 0.0
            coverage = 0.57
        elif day in {"2024-01-05", "2024-01-08", "2024-01-09"}:
            net_score = 0.03
            structural_score = 0.02
            coverage = 0.60
        else:
            net_score = 0.0
            structural_score = 0.0
            coverage = 0.60
        for asset in ("XLE", "XLK"):
            rows.append(
                {
                    "trade_date": date,
                    "asset": asset,
                    "geo_net_score": net_score,
                    "geo_structural_score": structural_score,
                    "coverage_score": coverage,
                }
            )
    return pd.DataFrame(rows)


def test_build_geo_regime_labels_applies_eligibility_persistence_and_precedence() -> None:
    regime_frame = build_geo_regime_labels(
        _build_snapshot_frame(),
        min_asset_count=2,
        definition=GeoRegimeDefinition(coverage_floor=0.57),
    )

    labels = regime_frame.set_index("date")["geo_regime"].to_dict()

    assert labels[pd.Timestamp("2024-01-02")] == NORMAL_REGIME
    assert labels[pd.Timestamp("2024-01-03")] == NORMAL_REGIME
    assert labels[pd.Timestamp("2024-01-04")] == GEO_STRESS_REGIME
    assert labels[pd.Timestamp("2024-01-05")] == GEO_STRESS_REGIME
    assert labels[pd.Timestamp("2024-01-08")] == GEO_STRESS_REGIME
    assert labels[pd.Timestamp("2024-01-09")] == STRUCTURAL_GEO_REGIME
    assert labels[pd.Timestamp("2024-01-10")] == NORMAL_REGIME

    day_two = regime_frame.loc[regime_frame["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert bool(day_two["regime_eligible"]) is False

    structural_day = regime_frame.loc[regime_frame["date"] == pd.Timestamp("2024-01-09")].iloc[0]
    assert int(structural_day["structural_persistence_count"]) == 3


def test_build_placebo_regime_frames_preserves_label_counts_and_eligibility() -> None:
    regime_frame = build_geo_regime_labels(
        _build_snapshot_frame(),
        min_asset_count=2,
        definition=GeoRegimeDefinition(coverage_floor=0.57),
    )

    placebos = build_placebo_regime_frames(regime_frame, seed=1729, lag_days=2)

    assert set(placebos) == {
        "shuffled_counts_preserved",
        "lag_broken_63d",
        "constant_geo_stress",
        "constant_structural_geo",
    }

    original_counts = regime_frame["geo_regime"].value_counts().to_dict()
    shuffled_counts = placebos["shuffled_counts_preserved"]["geo_regime"].value_counts().to_dict()
    assert shuffled_counts == original_counts

    ineligible_dates = regime_frame.loc[~regime_frame["regime_eligible"], "date"]
    lag_broken = placebos["lag_broken_63d"].set_index("date")
    for date in ineligible_dates:
        assert lag_broken.at[pd.Timestamp(date), "geo_regime"] == NORMAL_REGIME

    constant_structural = placebos["constant_structural_geo"]
    eligible_mask = constant_structural["regime_eligible"].astype(bool)
    assert (
        constant_structural.loc[eligible_mask, "geo_regime"] == STRUCTURAL_GEO_REGIME
    ).all()
    assert (
        constant_structural.loc[~eligible_mask, "geo_regime"] == NORMAL_REGIME
    ).all()
