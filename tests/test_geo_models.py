from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime

import pytest

from src.geo.models import (
    AssetImpactRecord,
    GeoDataHealth,
    GeoFeatureSnapshot,
    NormalizedGeoEvent,
    RawWorldMonitorEvent,
)


def test_raw_worldmonitor_event_constructs_with_required_fields() -> None:
    event = RawWorldMonitorEvent(
        source_system="worldmonitor",
        source_endpoint="/api/example",
        source_record_id="abc123",
        pull_started_at=datetime(2026, 3, 22, 20, 0, tzinfo=UTC),
        pull_completed_at=datetime(2026, 3, 22, 20, 1, tzinfo=UTC),
        first_seen_at=datetime(2026, 3, 22, 20, 1, tzinfo=UTC),
        payload_sha256="a" * 64,
        raw_payload={"id": "abc123"},
    )

    assert event.source_system == "worldmonitor"
    assert event.payload_sha256 == "a" * 64


def test_normalized_geo_event_defaults_and_optional_region_code() -> None:
    event = NormalizedGeoEvent(
        event_id="evt-1",
        source_system="worldmonitor",
        source_endpoint="/api/example",
        source_record_id="abc123",
        payload_sha256="b" * 64,
        dedupe_group_id="group-1",
        event_type="SANCTION",
        event_subtype="export_control",
        country_iso3=None,
        region_code=None,
        latitude=None,
        longitude=None,
        occurred_at=None,
        published_at=None,
        first_seen_at=datetime(2026, 3, 22, 20, 1, tzinfo=UTC),
        available_at=datetime(2026, 3, 22, 20, 1, tzinfo=UTC),
        effective_start_at=datetime(2026, 3, 22, 20, 1, tzinfo=UTC),
        severity_norm=0.8,
        confidence_norm=0.9,
        normalization_confidence=0.7,
        fatalities_estimate=None,
        infrastructure_class=None,
        actor_1=None,
        actor_2=None,
    )

    assert event.region_code is None
    assert event.status == "ACTIVE"
    assert event.normalization_version == "geo_norm_v1"


def test_geo_model_mutable_defaults_are_isolated() -> None:
    left = AssetImpactRecord(
        event_id="evt-1",
        asset="XLK",
        mapping_version="geo_map_v1",
        impact_direction=1,
        directness_score=1.0,
        region_exposure=0.5,
        sector_exposure=1.0,
        infra_exposure=0.0,
        mapping_confidence=0.9,
        impact_score=0.4,
    )
    right = AssetImpactRecord(
        event_id="evt-2",
        asset="XLE",
        mapping_version="geo_map_v1",
        impact_direction=-1,
        directness_score=0.5,
        region_exposure=1.0,
        sector_exposure=0.0,
        infra_exposure=0.0,
        mapping_confidence=0.8,
        impact_score=-0.2,
    )

    assert left.impact_components == {}
    assert right.impact_components == {}
    assert left.impact_components is not right.impact_components


def test_geo_feature_snapshot_supports_tuple_event_ids() -> None:
    snapshot = GeoFeatureSnapshot(
        trade_date=date(2026, 3, 22),
        asset="XLK",
        snapshot_cutoff_at=datetime(2026, 3, 22, 20, 10, tzinfo=UTC),
        geo_net_raw=0.25,
        geo_net_score=0.24,
        geo_structural_score=0.24,
        geo_harm_score=0.0,
        geo_break_risk=0.1,
        geo_velocity_3d=0.05,
        geo_cluster_72h=1.0,
        region_stress=0.2,
        sector_disruption=0.3,
        infra_disruption=0.0,
        sanctions_score=0.4,
        avg_mapping_confidence=0.8,
        coverage_score=1.0,
        data_freshness_minutes=15,
        hard_override=False,
        contributing_event_ids=("evt-1", "evt-2"),
    )

    assert snapshot.contributing_event_ids == ("evt-1", "evt-2")


def test_geo_models_are_frozen() -> None:
    event = GeoDataHealth(
        source_system="worldmonitor",
        checked_at=datetime(2026, 3, 22, 20, 5, tzinfo=UTC),
        source_endpoint="/api/example",
        status="HEALTHY",
        freshness_minutes=5,
        max_allowed_staleness_minutes=60,
        records_seen_24h=10,
        expected_records_24h=12,
        coverage_score=0.83,
        last_success_at=datetime(2026, 3, 22, 20, 5, tzinfo=UTC),
    )

    with pytest.raises(FrozenInstanceError):
        event.status = "FAILED"
