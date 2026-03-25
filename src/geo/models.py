from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

EventType = Literal["SANCTION", "CONFLICT_ESCALATION", "INFRA_DISRUPTION", "UNREST"]
EventStatus = Literal["ACTIVE", "RESOLVED", "RETRACTED", "DUPLICATE"]
HealthStatus = Literal["HEALTHY", "PARTIAL", "STALE", "FAILED"]


@dataclass(frozen=True)
class RawWorldMonitorEvent:
    source_system: str
    source_endpoint: str
    source_record_id: str
    pull_started_at: datetime
    pull_completed_at: datetime
    first_seen_at: datetime
    payload_sha256: str
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class NormalizedGeoEvent:
    event_id: str
    source_system: str
    source_endpoint: str
    source_record_id: str
    payload_sha256: str
    dedupe_group_id: str
    event_type: EventType
    event_subtype: str
    country_iso3: str | None
    region_code: str | None
    latitude: float | None
    longitude: float | None
    occurred_at: datetime | None
    published_at: datetime | None
    first_seen_at: datetime
    available_at: datetime
    effective_start_at: datetime
    severity_norm: float
    confidence_norm: float
    normalization_confidence: float
    fatalities_estimate: int | None
    infrastructure_class: str | None
    actor_1: str | None
    actor_2: str | None
    tags: tuple[str, ...] = ()
    raw_payload: dict[str, Any] = field(default_factory=dict)
    status: EventStatus = "ACTIVE"
    normalization_version: str = "geo_norm_v1"


@dataclass(frozen=True)
class AssetImpactRecord:
    event_id: str
    asset: str
    mapping_version: str
    impact_direction: Literal[-1, 0, 1]
    directness_score: float
    region_exposure: float
    sector_exposure: float
    infra_exposure: float
    mapping_confidence: float
    impact_score: float
    impact_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GeoFeatureSnapshot:
    trade_date: date
    asset: str
    snapshot_cutoff_at: datetime
    geo_net_raw: float
    geo_net_score: float
    geo_structural_score: float
    geo_harm_score: float
    geo_break_risk: float
    geo_velocity_3d: float
    geo_cluster_72h: float
    region_stress: float
    sector_disruption: float
    infra_disruption: float
    sanctions_score: float
    avg_mapping_confidence: float
    coverage_score: float
    data_freshness_minutes: int
    hard_override: bool
    contributing_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GeoDataHealth:
    source_system: str
    checked_at: datetime
    source_endpoint: str
    status: HealthStatus
    freshness_minutes: int
    max_allowed_staleness_minutes: int
    records_seen_24h: int
    expected_records_24h: int
    coverage_score: float
    last_success_at: datetime | None
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
