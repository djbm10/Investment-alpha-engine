from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any

import pandas as pd

if os.getenv("DISABLE_DB") != "true":
    try:
        import psycopg
    except Exception:
        psycopg = None
else:
    psycopg = None

from ..config_loader import DatabaseConfig, PathsConfig

try:
    from ..database import PostgresStore
except Exception:
    PostgresStore = None
from .models import AssetImpactRecord, GeoDataHealth, GeoFeatureSnapshot, NormalizedGeoEvent


class GeoStore:
    def __init__(self, config: DatabaseConfig, paths: PathsConfig, logger) -> None:
        self.config = config
        self.paths = paths
        self.logger = logger
        self._postgres_store = PostgresStore(config, paths, logger) if PostgresStore is not None else None
        self._disabled = self._postgres_store is None or psycopg is None or bool(getattr(self._postgres_store, "disabled", False))

    def ensure_schema(self) -> None:
        if self._disabled:
            return None
        self._postgres_store.ensure_geo_schema()

    def upsert_geo_events(self, events: list[NormalizedGeoEvent]) -> int:
        if self._disabled:
            return 0
        if not events:
            return 0

        rows = [
            (
                event.event_id,
                event.source_system,
                event.source_endpoint,
                event.source_record_id,
                event.payload_sha256,
                event.dedupe_group_id,
                event.event_type,
                event.event_subtype,
                event.country_iso3,
                event.region_code,
                _optional_float(event.latitude),
                _optional_float(event.longitude),
                event.occurred_at,
                event.published_at,
                event.first_seen_at,
                event.available_at,
                event.effective_start_at,
                float(event.severity_norm),
                float(event.confidence_norm),
                float(event.normalization_confidence),
                _optional_int(event.fatalities_estimate),
                event.infrastructure_class,
                event.actor_1,
                event.actor_2,
                json.dumps(list(event.tags)),
                json.dumps(event.raw_payload),
                event.status,
                event.normalization_version,
            )
            for event in events
        ]

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO geo_event (
                        event_id, source_system, source_endpoint, source_record_id,
                        payload_sha256, dedupe_group_id, event_type, event_subtype,
                        country_iso3, region_code, latitude, longitude,
                        occurred_at, published_at, first_seen_at, available_at,
                        effective_start_at, severity_norm, confidence_norm,
                        normalization_confidence, fatalities_estimate, infrastructure_class,
                        actor_1, actor_2, tags_json, raw_payload, status, normalization_version
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s::jsonb, %s::jsonb, %s, %s
                    )
                    ON CONFLICT (source_system, source_record_id) DO UPDATE SET
                        source_endpoint = EXCLUDED.source_endpoint,
                        payload_sha256 = EXCLUDED.payload_sha256,
                        dedupe_group_id = EXCLUDED.dedupe_group_id,
                        event_type = EXCLUDED.event_type,
                        event_subtype = EXCLUDED.event_subtype,
                        country_iso3 = EXCLUDED.country_iso3,
                        region_code = EXCLUDED.region_code,
                        latitude = EXCLUDED.latitude,
                        longitude = EXCLUDED.longitude,
                        occurred_at = EXCLUDED.occurred_at,
                        published_at = EXCLUDED.published_at,
                        first_seen_at = EXCLUDED.first_seen_at,
                        available_at = EXCLUDED.available_at,
                        effective_start_at = EXCLUDED.effective_start_at,
                        severity_norm = EXCLUDED.severity_norm,
                        confidence_norm = EXCLUDED.confidence_norm,
                        normalization_confidence = EXCLUDED.normalization_confidence,
                        fatalities_estimate = EXCLUDED.fatalities_estimate,
                        infrastructure_class = EXCLUDED.infrastructure_class,
                        actor_1 = EXCLUDED.actor_1,
                        actor_2 = EXCLUDED.actor_2,
                        tags_json = EXCLUDED.tags_json,
                        raw_payload = EXCLUDED.raw_payload,
                        status = EXCLUDED.status,
                        normalization_version = EXCLUDED.normalization_version,
                        updated_at = NOW()
                    """,
                    rows,
                )
            conn.commit()
        return len(rows)

    def replace_event_asset_impacts(self, event_id: str, impacts: list[AssetImpactRecord]) -> None:
        if self._disabled:
            return None
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM geo_event_asset_impact WHERE event_id = %s", (event_id,))
                if impacts:
                    cur.executemany(
                        """
                        INSERT INTO geo_event_asset_impact (
                            event_id, asset, mapping_version, impact_direction,
                            directness_score, region_exposure, sector_exposure,
                            infra_exposure, mapping_confidence, impact_score, impact_components_json
                        ) VALUES (
                            %s, %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, %s, %s::jsonb
                        )
                        """,
                        [
                            (
                                impact.event_id,
                                impact.asset,
                                impact.mapping_version,
                                int(impact.impact_direction),
                                float(impact.directness_score),
                                float(impact.region_exposure),
                                float(impact.sector_exposure),
                                float(impact.infra_exposure),
                                float(impact.mapping_confidence),
                                float(impact.impact_score),
                                json.dumps(impact.impact_components),
                            )
                            for impact in impacts
                        ],
                    )
            conn.commit()

    def upsert_feature_snapshots(self, snapshots: list[GeoFeatureSnapshot]) -> int:
        if self._disabled:
            return 0
        if not snapshots:
            return 0

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO geo_feature_snapshot (
                        trade_date, asset, snapshot_cutoff_at, geo_net_raw, geo_net_score,
                        geo_structural_score, geo_harm_score, geo_break_risk, geo_velocity_3d,
                        geo_cluster_72h, region_stress, sector_disruption, infra_disruption,
                        sanctions_score, avg_mapping_confidence, coverage_score,
                        data_freshness_minutes, hard_override, contributing_event_ids
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s::jsonb
                    )
                    ON CONFLICT (trade_date, asset) DO UPDATE SET
                        snapshot_cutoff_at = EXCLUDED.snapshot_cutoff_at,
                        geo_net_raw = EXCLUDED.geo_net_raw,
                        geo_net_score = EXCLUDED.geo_net_score,
                        geo_structural_score = EXCLUDED.geo_structural_score,
                        geo_harm_score = EXCLUDED.geo_harm_score,
                        geo_break_risk = EXCLUDED.geo_break_risk,
                        geo_velocity_3d = EXCLUDED.geo_velocity_3d,
                        geo_cluster_72h = EXCLUDED.geo_cluster_72h,
                        region_stress = EXCLUDED.region_stress,
                        sector_disruption = EXCLUDED.sector_disruption,
                        infra_disruption = EXCLUDED.infra_disruption,
                        sanctions_score = EXCLUDED.sanctions_score,
                        avg_mapping_confidence = EXCLUDED.avg_mapping_confidence,
                        coverage_score = EXCLUDED.coverage_score,
                        data_freshness_minutes = EXCLUDED.data_freshness_minutes,
                        hard_override = EXCLUDED.hard_override,
                        contributing_event_ids = EXCLUDED.contributing_event_ids,
                        updated_at = NOW()
                    """,
                    [
                        (
                            snapshot.trade_date,
                            snapshot.asset,
                            snapshot.snapshot_cutoff_at,
                            float(snapshot.geo_net_raw),
                            float(snapshot.geo_net_score),
                            float(snapshot.geo_structural_score),
                            float(snapshot.geo_harm_score),
                            float(snapshot.geo_break_risk),
                            float(snapshot.geo_velocity_3d),
                            float(snapshot.geo_cluster_72h),
                            float(snapshot.region_stress),
                            float(snapshot.sector_disruption),
                            float(snapshot.infra_disruption),
                            float(snapshot.sanctions_score),
                            float(snapshot.avg_mapping_confidence),
                            float(snapshot.coverage_score),
                            int(snapshot.data_freshness_minutes),
                            bool(snapshot.hard_override),
                            json.dumps(list(snapshot.contributing_event_ids)),
                        )
                        for snapshot in snapshots
                    ],
                )
            conn.commit()
        return len(snapshots)

    def insert_data_health(self, rows: list[GeoDataHealth]) -> int:
        if self._disabled:
            return 0
        if not rows:
            return 0

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO geo_data_health (
                        source_system, source_endpoint, checked_at, status,
                        freshness_minutes, max_allowed_staleness_minutes,
                        records_seen_24h, expected_records_24h, coverage_score,
                        last_success_at, error_message, details_json
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s, %s::jsonb
                    )
                    ON CONFLICT (source_system, source_endpoint, checked_at) DO UPDATE SET
                        status = EXCLUDED.status,
                        freshness_minutes = EXCLUDED.freshness_minutes,
                        max_allowed_staleness_minutes = EXCLUDED.max_allowed_staleness_minutes,
                        records_seen_24h = EXCLUDED.records_seen_24h,
                        expected_records_24h = EXCLUDED.expected_records_24h,
                        coverage_score = EXCLUDED.coverage_score,
                        last_success_at = EXCLUDED.last_success_at,
                        error_message = EXCLUDED.error_message,
                        details_json = EXCLUDED.details_json
                    """,
                    [
                        (
                            row.source_system,
                            row.source_endpoint,
                            row.checked_at,
                            row.status,
                            int(row.freshness_minutes),
                            int(row.max_allowed_staleness_minutes),
                            int(row.records_seen_24h),
                            int(row.expected_records_24h),
                            float(row.coverage_score),
                            row.last_success_at,
                            row.error_message,
                            json.dumps(row.details),
                        )
                        for row in rows
                    ],
                )
            conn.commit()
        return len(rows)

    def fetch_active_events(self, *, available_at_lte: datetime) -> pd.DataFrame:
        if self._disabled:
            return pd.DataFrame()
        # "Active" in storage means a record is still marked ACTIVE and was causally
        # available by the supplied cutoff. Decay and recency windows are applied later.
        return self._fetch_dataframe(
            """
            SELECT *
            FROM geo_event
            WHERE status = 'ACTIVE'
              AND available_at <= %s
            ORDER BY available_at, event_id
            """,
            (available_at_lte,),
        )

    def fetch_feature_snapshot(self, *, trade_date: date) -> pd.DataFrame:
        if self._disabled:
            return pd.DataFrame()
        return self._fetch_dataframe(
            """
            SELECT *
            FROM geo_feature_snapshot
            WHERE trade_date = %s
            ORDER BY asset
            """,
            (trade_date,),
        )

    def fetch_feature_snapshot_range(self, *, start_date: date, end_date: date) -> pd.DataFrame:
        if self._disabled:
            return pd.DataFrame()
        return self._fetch_dataframe(
            """
            SELECT *
            FROM geo_feature_snapshot
            WHERE trade_date BETWEEN %s AND %s
            ORDER BY trade_date, asset
            """,
            (start_date, end_date),
        )

    def fetch_latest_health(self) -> pd.DataFrame:
        if self._disabled:
            return pd.DataFrame()
        return self._fetch_dataframe(
            """
            SELECT DISTINCT ON (source_system, source_endpoint) *
            FROM geo_data_health
            ORDER BY source_system, source_endpoint, checked_at DESC
            """
        )

    def fetch_health_as_of(self, *, as_of: datetime) -> pd.DataFrame:
        if self._disabled:
            return pd.DataFrame()
        return self._fetch_dataframe(
            """
            SELECT DISTINCT ON (source_system, source_endpoint) *
            FROM geo_data_health
            WHERE checked_at <= %s
            ORDER BY source_system, source_endpoint, checked_at DESC
            """,
            (as_of,),
        )

    def _connect(self) -> psycopg.Connection[Any]:
        if self._disabled or psycopg is None or self._postgres_store is None:
            raise RuntimeError("Database access is disabled.")
        return psycopg.connect(self._postgres_store.dsn(self.config.database_name))

    def _fetch_dataframe(self, query: str, params: tuple[object, ...] = ()) -> pd.DataFrame:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
        frame = pd.DataFrame(rows, columns=columns)
        if frame.empty:
            return frame

        for column in ("occurred_at", "published_at", "first_seen_at", "available_at", "effective_start_at", "created_at", "updated_at", "snapshot_cutoff_at", "checked_at", "last_success_at"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column])
        if "trade_date" in frame.columns:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        return frame


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value)
