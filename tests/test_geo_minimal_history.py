from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.geo.minimal_history import (
    MINIMAL_GEO_SNAPSHOT_COLUMNS,
    build_minimal_geo_feature_snapshot,
    write_minimal_geo_feature_snapshot,
)


def test_build_minimal_geo_feature_snapshot_is_deterministic() -> None:
    frame_one = build_minimal_geo_feature_snapshot(
        tickers=["XLK", "XLE"],
        start_date="2022-02-22",
        end_date="2022-02-28",
    )
    frame_two = build_minimal_geo_feature_snapshot(
        tickers=["XLK", "XLE"],
        start_date="2022-02-22",
        end_date="2022-02-28",
    )

    pd.testing.assert_frame_equal(frame_one, frame_two)


def test_build_minimal_geo_feature_snapshot_emits_required_columns_and_bounds() -> None:
    frame = build_minimal_geo_feature_snapshot(
        tickers=["XLK", "XLE", "XLV"],
        start_date="2024-01-02",
        end_date="2024-01-10",
    )

    assert list(frame.columns) == list(MINIMAL_GEO_SNAPSHOT_COLUMNS)
    assert len(frame) == len(pd.bdate_range("2024-01-02", "2024-01-10")) * 3
    assert frame["geo_net_score"].between(-1.0, 1.0).all()
    assert frame["geo_structural_score"].between(-1.0, 1.0).all()
    assert frame["coverage_score"].between(0.0, 1.0).all()
    assert (frame["data_freshness_minutes"] >= 0).all()
    assert frame["contributing_event_ids"].map(lambda value: isinstance(json.loads(value), list)).all()


def test_build_minimal_geo_feature_snapshot_is_time_consistent_around_known_event() -> None:
    frame = build_minimal_geo_feature_snapshot(
        tickers=["XLK", "XLE"],
        start_date="2022-02-23",
        end_date="2022-02-28",
    )
    lookup = frame.set_index(["trade_date", "asset"])

    pre_event_xle = float(lookup.loc[("2022-02-23", "XLE"), "geo_net_score"])
    invasion_day_xle = float(lookup.loc[("2022-02-24", "XLE"), "geo_net_score"])
    invasion_day_xlk = float(lookup.loc[("2022-02-24", "XLK"), "geo_structural_score"])

    assert abs(pre_event_xle) < 0.01
    assert invasion_day_xle > 0.30
    assert invasion_day_xlk > -0.10


def test_build_minimal_geo_feature_snapshot_gives_structural_weight_to_export_controls() -> None:
    frame = build_minimal_geo_feature_snapshot(
        tickers=["XLK", "XLE"],
        start_date="2022-10-07",
        end_date="2022-10-14",
    )
    lookup = frame.set_index(["trade_date", "asset"])

    assert float(lookup.loc[("2022-10-07", "XLK"), "geo_structural_score"]) < -0.40
    assert abs(float(lookup.loc[("2022-10-07", "XLE"), "geo_structural_score"])) < 0.05


def test_build_minimal_geo_feature_snapshot_shock_events_decay_quickly_after_active_window() -> None:
    frame = build_minimal_geo_feature_snapshot(
        tickers=["XLE"],
        start_date="2024-04-15",
        end_date="2024-04-25",
    )
    lookup = frame.set_index(["trade_date", "asset"])

    initial = float(lookup.loc[("2024-04-15", "XLE"), "geo_net_score"])
    later = float(lookup.loc[("2024-04-25", "XLE"), "geo_net_score"])

    assert initial > later
    assert later < initial * 0.60


def test_profile_a_heavily_damps_shock_events_but_preserves_structural_events() -> None:
    base = build_minimal_geo_feature_snapshot(
        tickers=["XLE", "XLK"],
        start_date="2022-02-24",
        end_date="2022-10-07",
        profile="base",
    ).set_index(["trade_date", "asset"])
    variant_a = build_minimal_geo_feature_snapshot(
        tickers=["XLE", "XLK"],
        start_date="2022-02-24",
        end_date="2022-10-07",
        profile="A",
    ).set_index(["trade_date", "asset"])

    assert abs(float(variant_a.loc[("2022-02-24", "XLE"), "geo_net_score"])) < abs(
        float(base.loc[("2022-02-24", "XLE"), "geo_net_score"])
    )
    assert float(variant_a.loc[("2022-10-07", "XLK"), "geo_structural_score"]) == pytest.approx(
        float(base.loc[("2022-10-07", "XLK"), "geo_structural_score"])
    )


def test_profile_b_shortens_shock_decay_relative_to_profile_a() -> None:
    variant_a = build_minimal_geo_feature_snapshot(
        tickers=["XLE"],
        start_date="2022-02-24",
        end_date="2022-03-10",
        profile="A",
    ).set_index(["trade_date", "asset"])
    variant_b = build_minimal_geo_feature_snapshot(
        tickers=["XLE"],
        start_date="2022-02-24",
        end_date="2022-03-10",
        profile="B",
    ).set_index(["trade_date", "asset"])

    assert abs(float(variant_b.loc[("2022-03-10", "XLE"), "geo_net_score"])) < abs(
        float(variant_a.loc[("2022-03-10", "XLE"), "geo_net_score"])
    )


def test_profile_c_makes_structural_mapping_more_conservative_than_profile_b() -> None:
    variant_b = build_minimal_geo_feature_snapshot(
        tickers=["XLK"],
        start_date="2022-10-07",
        end_date="2022-10-07",
        profile="B",
    ).set_index(["trade_date", "asset"])
    variant_c = build_minimal_geo_feature_snapshot(
        tickers=["XLK"],
        start_date="2022-10-07",
        end_date="2022-10-07",
        profile="C",
    ).set_index(["trade_date", "asset"])

    assert abs(float(variant_c.loc[("2022-10-07", "XLK"), "geo_net_score"])) < abs(
        float(variant_b.loc[("2022-10-07", "XLK"), "geo_net_score"])
    )
    assert abs(float(variant_c.loc[("2022-10-07", "XLK"), "geo_structural_score"])) < abs(
        float(variant_b.loc[("2022-10-07", "XLK"), "geo_structural_score"])
    )
    assert float(variant_c.loc[("2022-10-07", "XLK"), "coverage_score"]) <= float(
        variant_b.loc[("2022-10-07", "XLK"), "coverage_score"]
    )


def test_profile_d_restricts_conflict_mapping_to_more_direct_assets() -> None:
    base = build_minimal_geo_feature_snapshot(
        tickers=["XLE", "XLY", "XLK"],
        start_date="2022-02-24",
        end_date="2022-02-24",
        profile="base",
    ).set_index(["trade_date", "asset"])
    variant_d = build_minimal_geo_feature_snapshot(
        tickers=["XLE", "XLY", "XLK"],
        start_date="2022-02-24",
        end_date="2022-02-24",
        profile="D",
    ).set_index(["trade_date", "asset"])

    assert abs(float(variant_d.loc[("2022-02-24", "XLY"), "geo_net_score"])) < abs(
        float(base.loc[("2022-02-24", "XLY"), "geo_net_score"])
    )
    assert abs(float(variant_d.loc[("2022-02-24", "XLK"), "geo_net_score"])) < abs(
        float(base.loc[("2022-02-24", "XLK"), "geo_net_score"])
    )
    assert float(variant_d.loc[("2022-02-24", "XLE"), "geo_net_score"]) > 0.0


def test_profile_e_is_more_conservative_than_d_on_broad_mapping() -> None:
    variant_d = build_minimal_geo_feature_snapshot(
        tickers=["XLK", "XLB", "XLY"],
        start_date="2022-10-07",
        end_date="2022-10-07",
        profile="D",
    ).set_index(["trade_date", "asset"])
    variant_e = build_minimal_geo_feature_snapshot(
        tickers=["XLK", "XLB", "XLY"],
        start_date="2022-10-07",
        end_date="2022-10-07",
        profile="E",
    ).set_index(["trade_date", "asset"])

    assert abs(float(variant_e.loc[("2022-10-07", "XLY"), "geo_net_score"])) <= abs(
        float(variant_d.loc[("2022-10-07", "XLY"), "geo_net_score"])
    )
    assert abs(float(variant_e.loc[("2022-10-07", "XLB"), "geo_net_score"])) <= abs(
        float(variant_d.loc[("2022-10-07", "XLB"), "geo_net_score"])
    )
    assert abs(float(variant_e.loc[("2022-10-07", "XLK"), "geo_structural_score"])) < abs(
        float(variant_d.loc[("2022-10-07", "XLK"), "geo_structural_score"])
    )


def test_profile_f_neutralizes_shock_days_and_lowers_non_structural_coverage() -> None:
    variant_f = build_minimal_geo_feature_snapshot(
        tickers=["XLE", "XLK", "XLY"],
        start_date="2022-02-24",
        end_date="2022-02-24",
        profile="F",
    ).set_index(["trade_date", "asset"])

    for asset in ("XLE", "XLK", "XLY"):
        assert abs(float(variant_f.loc[("2022-02-24", asset), "geo_net_score"])) < 0.01
        assert abs(float(variant_f.loc[("2022-02-24", asset), "geo_structural_score"])) < 0.01
        assert float(variant_f.loc[("2022-02-24", asset), "coverage_score"]) <= 0.50


def test_profile_f_preserves_direct_structural_export_control_signal() -> None:
    variant_f = build_minimal_geo_feature_snapshot(
        tickers=["XLK", "XLB", "XLY"],
        start_date="2022-10-07",
        end_date="2022-10-07",
        profile="F",
    ).set_index(["trade_date", "asset"])

    assert float(variant_f.loc[("2022-10-07", "XLK"), "geo_net_score"]) < -0.45
    assert float(variant_f.loc[("2022-10-07", "XLK"), "geo_structural_score"]) < -0.40
    assert float(variant_f.loc[("2022-10-07", "XLK"), "coverage_score"]) >= 0.80
    assert abs(float(variant_f.loc[("2022-10-07", "XLY"), "geo_net_score"])) < 0.01
    assert float(variant_f.loc[("2022-10-07", "XLB"), "geo_net_score"]) < -0.02
    assert abs(float(variant_f.loc[("2022-10-07", "XLB"), "geo_structural_score"])) < 0.02
    assert float(variant_f.loc[("2022-10-07", "XLB"), "coverage_score"]) >= 0.80


def test_profile_f_expanded_structural_history_reaches_other_export_control_dates() -> None:
    variant_f = build_minimal_geo_feature_snapshot(
        tickers=["XLK", "XLB"],
        start_date="2023-10-17",
        end_date="2023-10-17",
        profile="F",
    ).set_index(["trade_date", "asset"])

    assert float(variant_f.loc[("2023-10-17", "XLK"), "geo_net_score"]) < -0.30
    assert float(variant_f.loc[("2023-10-17", "XLB"), "geo_net_score"]) < -0.01
    assert float(variant_f.loc[("2023-10-17", "XLK"), "coverage_score"]) >= 0.80
    assert float(variant_f.loc[("2023-10-17", "XLB"), "coverage_score"]) >= 0.80


def test_write_minimal_geo_feature_snapshot_writes_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "geo_feature_snapshot_minimal.csv"

    written = write_minimal_geo_feature_snapshot(
        output_path=output_path,
        tickers=["XLK", "XLE"],
        start_date="2024-04-10",
        end_date="2024-04-16",
    )

    assert written == output_path
    assert written.exists()

    frame = pd.read_csv(written)
    assert list(frame.columns) == list(MINIMAL_GEO_SNAPSHOT_COLUMNS)
    assert len(frame) == len(pd.bdate_range("2024-04-10", "2024-04-16")) * 2
