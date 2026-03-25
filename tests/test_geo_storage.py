import logging
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import psycopg
import pytest
import yaml

from src.config_loader import load_config
from src.database import PostgresStore
from src.geo.models import (
    AssetImpactRecord,
    GeoDataHealth,
    GeoFeatureSnapshot,
    NormalizedGeoEvent,
)
from src.geo.storage import GeoStore

def _write_phase1_config(tmp_path: Path) -> Path:
    payload = yaml.safe_load(Path("config/phase1.yaml").read_text(encoding="utf-8"))
    payload["paths"]["raw_dir"] = str(tmp_path / "data" / "raw")
    payload["paths"]["processed_dir"] = str(tmp_path / "data" / "processed")
    payload["paths"]["log_dir"] = str(tmp_path / "logs")
    payload["paths"]["pipeline_log_file"] = str(tmp_path / "logs" / "phase1_pipeline.jsonl")
    payload["paths"]["postgres_log_file"] = str(tmp_path / "logs" / "postgres.log")
    payload["paths"]["cache_dir"] = str(tmp_path / "data" / ".cache" / "yfinance")
    payload["paths"]["postgres_dir"] = str(tmp_path / "data" / "postgres" / "phase1")
    payload["database"]["port"] = 55439
    payload["database"]["database_name"] = "investment_alpha_engine_geo_test"

    config_path = tmp_path / "config" / "phase1.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _insert_assets(dsn: str, tickers: list[str]) -> None:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO assets (ticker, asset_type, source_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (ticker) DO NOTHING
                """,
                [(ticker, "sector_etf", "test") for ticker in tickers],
            )
        conn.commit()


@pytest.fixture
def geo_db(tmp_path: Path):
    config_path = _write_phase1_config(tmp_path)
    config = load_config(config_path)
    logger = logging.getLogger(f"geo-storage-test-{config.database.port}")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    postgres_store = PostgresStore(config.database, config.paths, logger)
    postgres_store.initialize()
    geo_store = GeoStore(config.database, config.paths, logger)
    geo_store.ensure_schema()

    try:
        yield config, postgres_store, geo_store
    finally:
        postgres_store.stop()


def test_geo_schema_creation_is_idempotent_and_creates_expected_tables(geo_db) -> None:
    config, postgres_store, geo_store = geo_db

    postgres_store.ensure_geo_schema()
    geo_store.ensure_schema()

    with psycopg.connect(postgres_store.dsn(config.database.database_name)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                  AND tablename = ANY(%s)
                ORDER BY tablename
                """,
                (
                    [
                        "assets",
                        "geo_data_health",
                        "geo_event",
                        "geo_event_asset_impact",
                        "geo_feature_snapshot",
                    ],
                ),
            )
            table_names = {row[0] for row in cur.fetchall()}

            cur.execute(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = 'public'
                  AND tablename = 'geo_event'
                """
            )
            geo_event_indexes = {row[0] for row in cur.fetchall()}

    assert table_names == {
        "assets",
        "geo_data_health",
        "geo_event",
        "geo_event_asset_impact",
        "geo_feature_snapshot",
    }
    assert {
        "geo_event_pkey",
        "idx_geo_event_available_at",
        "idx_geo_event_country_iso3",
        "idx_geo_event_dedupe_group",
        "idx_geo_event_published_at",
        "idx_geo_event_status",
        "idx_geo_event_tags_gin",
        "idx_geo_event_type_available",
    }.issubset(geo_event_indexes)


def test_geo_store_roundtrips_events_impacts_and_feature_snapshots(geo_db) -> None:
    config, postgres_store, geo_store = geo_db
    _insert_assets(postgres_store.dsn(config.database.database_name), ["XLK", "XLE"])

    first_seen = datetime(2026, 3, 22, 20, 5, tzinfo=UTC)
    event = NormalizedGeoEvent(
        event_id="evt-1",
        source_system="worldmonitor",
        source_endpoint="/api/example",
        source_record_id="record-1",
        payload_sha256="a" * 64,
        dedupe_group_id="group-1",
        event_type="SANCTION",
        event_subtype="export_control",
        country_iso3="CHN",
        region_code="APAC",
        latitude=31.2,
        longitude=121.5,
        occurred_at=first_seen - timedelta(hours=2),
        published_at=first_seen - timedelta(hours=1),
        first_seen_at=first_seen,
        available_at=first_seen,
        effective_start_at=first_seen,
        severity_norm=0.70,
        confidence_norm=0.80,
        normalization_confidence=0.90,
        fatalities_estimate=0,
        infrastructure_class=None,
        actor_1="CN",
        actor_2="US",
        tags=("tech", "sanctions"),
        raw_payload={"headline": "test"},
    )

    assert geo_store.upsert_geo_events([event]) == 1
    assert geo_store.upsert_geo_events(
        [
            replace(
                event,
                event_id="evt-1-reingest",
                payload_sha256="b" * 64,
                severity_norm=0.95,
                raw_payload={"headline": "updated"},
            )
        ]
    ) == 1

    with psycopg.connect(postgres_store.dsn(config.database.database_name)) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM geo_event")
            assert cur.fetchone()[0] == 1

    active_events = geo_store.fetch_active_events(available_at_lte=first_seen + timedelta(minutes=1))
    assert len(active_events) == 1
    assert active_events.loc[0, "event_id"] == "evt-1"
    assert active_events.loc[0, "severity_norm"] == pytest.approx(0.95)
    assert active_events.loc[0, "raw_payload"]["headline"] == "updated"

    geo_store.replace_event_asset_impacts(
        event.event_id,
        [
            AssetImpactRecord(
                event_id=event.event_id,
                asset="XLK",
                mapping_version="geo_map_v1",
                impact_direction=-1,
                directness_score=1.0,
                region_exposure=0.6,
                sector_exposure=0.9,
                infra_exposure=0.0,
                mapping_confidence=0.95,
                impact_score=-0.5,
                impact_components={"region_term": -0.2, "sector_term": -0.3},
            ),
            AssetImpactRecord(
                event_id=event.event_id,
                asset="XLE",
                mapping_version="geo_map_v1",
                impact_direction=0,
                directness_score=0.25,
                region_exposure=0.1,
                sector_exposure=0.0,
                infra_exposure=0.0,
                mapping_confidence=0.5,
                impact_score=0.0,
            ),
        ],
    )

    with psycopg.connect(postgres_store.dsn(config.database.database_name)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM geo_event_asset_impact WHERE event_id = %s",
                (event.event_id,),
            )
            assert cur.fetchone()[0] == 2

    geo_store.replace_event_asset_impacts(
        event.event_id,
        [
            AssetImpactRecord(
                event_id=event.event_id,
                asset="XLK",
                mapping_version="geo_map_v1",
                impact_direction=-1,
                directness_score=1.0,
                region_exposure=0.7,
                sector_exposure=0.8,
                infra_exposure=0.0,
                mapping_confidence=0.90,
                impact_score=-0.55,
            )
        ],
    )

    with psycopg.connect(postgres_store.dsn(config.database.database_name)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT asset, impact_score
                FROM geo_event_asset_impact
                WHERE event_id = %s
                """,
                (event.event_id,),
            )
            rows = cur.fetchall()

    assert rows == [("XLK", -0.55)]

    snapshot_date = date(2026, 3, 23)
    assert geo_store.upsert_feature_snapshots(
        [
            GeoFeatureSnapshot(
                trade_date=snapshot_date,
                asset="XLK",
                snapshot_cutoff_at=datetime(2026, 3, 23, 20, 10, tzinfo=UTC),
                geo_net_raw=-0.55,
                geo_net_score=-0.50,
                geo_structural_score=-0.50,
                geo_harm_score=0.50,
                geo_break_risk=0.25,
                geo_velocity_3d=-0.20,
                geo_cluster_72h=1.0,
                region_stress=0.4,
                sector_disruption=0.6,
                infra_disruption=0.0,
                sanctions_score=0.8,
                avg_mapping_confidence=0.90,
                coverage_score=0.95,
                data_freshness_minutes=15,
                hard_override=False,
                contributing_event_ids=("evt-1",),
            )
        ]
    ) == 1

    assert geo_store.upsert_feature_snapshots(
        [
            GeoFeatureSnapshot(
                trade_date=snapshot_date,
                asset="XLK",
                snapshot_cutoff_at=datetime(2026, 3, 23, 20, 10, tzinfo=UTC),
                geo_net_raw=-0.70,
                geo_net_score=-0.60,
                geo_structural_score=-0.60,
                geo_harm_score=0.60,
                geo_break_risk=0.30,
                geo_velocity_3d=-0.25,
                geo_cluster_72h=2.0,
                region_stress=0.5,
                sector_disruption=0.7,
                infra_disruption=0.0,
                sanctions_score=0.9,
                avg_mapping_confidence=0.92,
                coverage_score=0.96,
                data_freshness_minutes=10,
                hard_override=True,
                contributing_event_ids=("evt-1",),
            )
        ]
    ) == 1

    snapshot = geo_store.fetch_feature_snapshot(trade_date=snapshot_date)
    assert len(snapshot) == 1
    assert snapshot.loc[0, "asset"] == "XLK"
    assert snapshot.loc[0, "geo_net_score"] == pytest.approx(-0.60)
    assert bool(snapshot.loc[0, "hard_override"]) is True

    with psycopg.connect(postgres_store.dsn(config.database.database_name)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM geo_feature_snapshot
                WHERE trade_date = %s AND asset = %s
                """,
                (snapshot_date, "XLK"),
            )
            assert cur.fetchone()[0] == 1

    snapshot_range = geo_store.fetch_feature_snapshot_range(
        start_date=date(2026, 3, 22),
        end_date=date(2026, 3, 24),
    )
    assert len(snapshot_range) == 1
    assert snapshot_range.loc[0, "trade_date"] == snapshot_date

    empty_snapshot_range = geo_store.fetch_feature_snapshot_range(
        start_date=date(2026, 3, 24),
        end_date=date(2026, 3, 24),
    )
    assert empty_snapshot_range.empty


def test_fetch_active_events_filters_only_active_rows_available_by_cutoff(geo_db) -> None:
    config, postgres_store, geo_store = geo_db
    _insert_assets(postgres_store.dsn(config.database.database_name), ["XLK"])

    cutoff = datetime(2026, 3, 22, 20, 5, tzinfo=UTC)
    base_event = NormalizedGeoEvent(
        event_id="evt-old",
        source_system="worldmonitor",
        source_endpoint="/api/example",
        source_record_id="record-old",
        payload_sha256="c" * 64,
        dedupe_group_id="group-old",
        event_type="UNREST",
        event_subtype="protest",
        country_iso3="FRA",
        region_code="EU",
        latitude=None,
        longitude=None,
        occurred_at=cutoff - timedelta(days=30),
        published_at=cutoff - timedelta(days=30),
        first_seen_at=cutoff - timedelta(days=30),
        available_at=cutoff - timedelta(days=30),
        effective_start_at=cutoff - timedelta(days=30),
        severity_norm=0.30,
        confidence_norm=0.90,
        normalization_confidence=0.80,
        fatalities_estimate=None,
        infrastructure_class=None,
        actor_1=None,
        actor_2=None,
    )
    future_event = replace(
        base_event,
        event_id="evt-future",
        source_record_id="record-future",
        payload_sha256="d" * 64,
        available_at=cutoff + timedelta(minutes=1),
        effective_start_at=cutoff + timedelta(minutes=1),
    )
    resolved_event = replace(
        base_event,
        event_id="evt-resolved",
        source_record_id="record-resolved",
        payload_sha256="e" * 64,
        status="RESOLVED",
    )

    geo_store.upsert_geo_events([base_event, future_event, resolved_event])

    active_events = geo_store.fetch_active_events(available_at_lte=cutoff)

    assert active_events["event_id"].tolist() == ["evt-old"]


def test_geo_store_health_fetches_latest_and_as_of_rows(geo_db) -> None:
    config, postgres_store, geo_store = geo_db

    base_time = datetime(2026, 3, 22, 20, 0, tzinfo=UTC)
    assert geo_store.insert_data_health(
        [
            GeoDataHealth(
                source_system="worldmonitor",
                checked_at=base_time,
                source_endpoint="/api/unrest",
                status="HEALTHY",
                freshness_minutes=5,
                max_allowed_staleness_minutes=60,
                records_seen_24h=10,
                expected_records_24h=12,
                coverage_score=0.80,
                last_success_at=base_time,
            ),
            GeoDataHealth(
                source_system="worldmonitor",
                checked_at=base_time + timedelta(minutes=10),
                source_endpoint="/api/unrest",
                status="PARTIAL",
                freshness_minutes=15,
                max_allowed_staleness_minutes=60,
                records_seen_24h=8,
                expected_records_24h=12,
                coverage_score=0.67,
                last_success_at=base_time,
                error_message="upstream lag",
            ),
            GeoDataHealth(
                source_system="worldmonitor",
                checked_at=base_time + timedelta(minutes=5),
                source_endpoint="/api/sanctions",
                status="HEALTHY",
                freshness_minutes=30,
                max_allowed_staleness_minutes=1440,
                records_seen_24h=2,
                expected_records_24h=2,
                coverage_score=1.0,
                last_success_at=base_time + timedelta(minutes=5),
            ),
        ]
    ) == 3

    latest_health = geo_store.fetch_latest_health()
    assert len(latest_health) == 2
    unrest_latest = latest_health.loc[latest_health["source_endpoint"] == "/api/unrest"].iloc[0]
    assert unrest_latest["status"] == "PARTIAL"
    assert unrest_latest["coverage_score"] == pytest.approx(0.67)

    health_as_of = geo_store.fetch_health_as_of(as_of=base_time + timedelta(minutes=6))
    assert len(health_as_of) == 2
    unrest_as_of = health_as_of.loc[health_as_of["source_endpoint"] == "/api/unrest"].iloc[0]
    assert unrest_as_of["status"] == "HEALTHY"
