from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

_DISABLE_DB = os.getenv("DISABLE_DB") == "true"
_DB_IMPORT_ERROR: Exception | None = None

if not _DISABLE_DB:
    try:
        import psycopg
        from pgserver import initdb, pg_ctl
        from psycopg import sql
    except Exception as exc:
        psycopg = None
        initdb = None
        pg_ctl = None
        sql = None
        _DB_IMPORT_ERROR = exc
else:
    psycopg = None
    initdb = None
    pg_ctl = None
    sql = None

from .config_loader import DatabaseConfig, PathsConfig

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS assets (
        ticker TEXT PRIMARY KEY,
        asset_type TEXT NOT NULL,
        source_name TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_runs (
        run_id TEXT PRIMARY KEY,
        source_name TEXT NOT NULL,
        started_at TIMESTAMPTZ NOT NULL,
        completed_at TIMESTAMPTZ,
        status TEXT NOT NULL,
        start_date DATE NOT NULL,
        end_date DATE,
        raw_rows INTEGER NOT NULL DEFAULT 0,
        valid_rows INTEGER NOT NULL DEFAULT 0,
        issue_rows INTEGER NOT NULL DEFAULT 0,
        message TEXT,
        config_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_prices (
        ticker TEXT NOT NULL REFERENCES assets(ticker),
        trade_date DATE NOT NULL,
        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        adj_close DOUBLE PRECISION,
        volume BIGINT,
        dividends DOUBLE PRECISION,
        stock_splits DOUBLE PRECISION,
        capital_gains DOUBLE PRECISION,
        daily_return DOUBLE PRECISION,
        is_valid BOOLEAN NOT NULL,
        validation_notes TEXT NOT NULL,
        cross_source_status TEXT NOT NULL,
        corporate_action_flag BOOLEAN NOT NULL,
        return_magnitude_flag BOOLEAN NOT NULL,
        zero_volume_flag BOOLEAN NOT NULL,
        continuity_flag BOOLEAN NOT NULL,
        split_detection_flag BOOLEAN NOT NULL,
        adjustment_gap_flag BOOLEAN NOT NULL,
        cross_source_validation_flag BOOLEAN NOT NULL,
        missing_business_days INTEGER NOT NULL,
        run_id TEXT NOT NULL REFERENCES ingestion_runs(run_id),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (ticker, trade_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_daily_prices_trade_date ON daily_prices (trade_date)",
    "CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker ON daily_prices (ticker)",
    """
    CREATE TABLE IF NOT EXISTS corporate_actions (
        ticker TEXT NOT NULL REFERENCES assets(ticker),
        action_date DATE NOT NULL,
        dividends DOUBLE PRECISION NOT NULL DEFAULT 0,
        stock_splits DOUBLE PRECISION NOT NULL DEFAULT 0,
        capital_gains DOUBLE PRECISION NOT NULL DEFAULT 0,
        run_id TEXT NOT NULL REFERENCES ingestion_runs(run_id),
        PRIMARY KEY (ticker, action_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS validation_events (
        run_id TEXT NOT NULL REFERENCES ingestion_runs(run_id),
        ticker TEXT NOT NULL REFERENCES assets(ticker),
        trade_date DATE NOT NULL,
        event_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        notes TEXT NOT NULL,
        PRIMARY KEY (run_id, ticker, trade_date, event_type)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_validation_events_trade_date ON validation_events (trade_date)",
    """
    CREATE TABLE IF NOT EXISTS data_quality_reports (
        run_id TEXT NOT NULL REFERENCES ingestion_runs(run_id),
        ticker TEXT NOT NULL REFERENCES assets(ticker),
        total_rows INTEGER NOT NULL,
        valid_rows INTEGER NOT NULL,
        invalid_rows INTEGER NOT NULL,
        return_magnitude_issues INTEGER NOT NULL,
        zero_volume_issues INTEGER NOT NULL,
        continuity_issues INTEGER NOT NULL,
        split_detection_rows INTEGER NOT NULL,
        adjustment_gap_rows INTEGER NOT NULL,
        corporate_action_events INTEGER NOT NULL,
        first_date DATE NOT NULL,
        last_date DATE NOT NULL,
        cross_source_status TEXT NOT NULL,
        PRIMARY KEY (run_id, ticker)
    )
    """,
]

PHASE2_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS phase2_runs (
        run_id TEXT PRIMARY KEY,
        started_at TIMESTAMPTZ NOT NULL,
        completed_at TIMESTAMPTZ,
        status TEXT NOT NULL,
        lookback_window INTEGER NOT NULL,
        diffusion_alpha DOUBLE PRECISION NOT NULL,
        diffusion_steps INTEGER NOT NULL,
        signal_threshold DOUBLE PRECISION NOT NULL,
        max_position_size DOUBLE PRECISION NOT NULL,
        total_signals INTEGER NOT NULL,
        total_trades INTEGER NOT NULL,
        sharpe_ratio DOUBLE PRECISION,
        max_drawdown DOUBLE PRECISION,
        win_rate DOUBLE PRECISION,
        profit_factor DOUBLE PRECISION,
        avg_holding_days DOUBLE PRECISION,
        annual_turnover DOUBLE PRECISION,
        profitable_month_fraction DOUBLE PRECISION,
        out_of_sample_months INTEGER NOT NULL,
        gate_passed BOOLEAN NOT NULL,
        config_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS phase2_daily_signals (
        run_id TEXT NOT NULL REFERENCES phase2_runs(run_id),
        signal_date DATE NOT NULL,
        ticker TEXT NOT NULL REFERENCES assets(ticker),
        current_return DOUBLE PRECISION NOT NULL,
        expected_return DOUBLE PRECISION NOT NULL,
        residual DOUBLE PRECISION NOT NULL,
        residual_zscore DOUBLE PRECISION,
        signal_direction INTEGER NOT NULL,
        target_position DOUBLE PRECISION NOT NULL,
        forward_return DOUBLE PRECISION,
        sigma DOUBLE PRECISION NOT NULL,
        edge_density DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (run_id, signal_date, ticker)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_phase2_daily_signals_date ON phase2_daily_signals (signal_date)",
    """
    CREATE TABLE IF NOT EXISTS phase2_trade_log (
        trade_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES phase2_runs(run_id),
        ticker TEXT NOT NULL REFERENCES assets(ticker),
        entry_date DATE NOT NULL,
        exit_date DATE NOT NULL,
        position_direction INTEGER NOT NULL,
        entry_zscore DOUBLE PRECISION,
        exit_zscore DOUBLE PRECISION,
        holding_days INTEGER NOT NULL,
        entry_weight DOUBLE PRECISION NOT NULL,
        gross_return DOUBLE PRECISION NOT NULL,
        net_return DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_phase2_trade_log_run ON phase2_trade_log (run_id)",
    """
    CREATE TABLE IF NOT EXISTS phase2_monthly_results (
        run_id TEXT NOT NULL REFERENCES phase2_runs(run_id),
        test_month DATE NOT NULL,
        training_end_date DATE NOT NULL,
        monthly_return DOUBLE PRECISION NOT NULL,
        profitable BOOLEAN NOT NULL,
        PRIMARY KEY (run_id, test_month)
    )
    """,
]

GEO_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS geo_event (
        event_id TEXT PRIMARY KEY,
        source_system TEXT NOT NULL,
        source_endpoint TEXT NOT NULL,
        source_record_id TEXT NOT NULL,
        payload_sha256 CHAR(64) NOT NULL,
        dedupe_group_id TEXT NOT NULL,
        event_type TEXT NOT NULL CHECK (event_type IN ('SANCTION', 'CONFLICT_ESCALATION', 'INFRA_DISRUPTION', 'UNREST')),
        event_subtype TEXT NOT NULL,
        country_iso3 CHAR(3),
        region_code TEXT,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        occurred_at TIMESTAMPTZ,
        published_at TIMESTAMPTZ,
        first_seen_at TIMESTAMPTZ NOT NULL,
        available_at TIMESTAMPTZ NOT NULL,
        effective_start_at TIMESTAMPTZ NOT NULL,
        severity_norm DOUBLE PRECISION NOT NULL CHECK (severity_norm >= 0.0 AND severity_norm <= 1.0),
        confidence_norm DOUBLE PRECISION NOT NULL CHECK (confidence_norm >= 0.0 AND confidence_norm <= 1.0),
        normalization_confidence DOUBLE PRECISION NOT NULL CHECK (normalization_confidence >= 0.0 AND normalization_confidence <= 1.0),
        fatalities_estimate INTEGER CHECK (fatalities_estimate IS NULL OR fatalities_estimate >= 0),
        infrastructure_class TEXT,
        actor_1 TEXT,
        actor_2 TEXT,
        tags_json JSONB NOT NULL DEFAULT '[]'::jsonb,
        raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'RESOLVED', 'RETRACTED', 'DUPLICATE')),
        normalization_version TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (source_system, source_record_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_geo_event_available_at ON geo_event (available_at)",
    "CREATE INDEX IF NOT EXISTS idx_geo_event_occurred_at ON geo_event (occurred_at)",
    "CREATE INDEX IF NOT EXISTS idx_geo_event_published_at ON geo_event (published_at)",
    "CREATE INDEX IF NOT EXISTS idx_geo_event_country_iso3 ON geo_event (country_iso3)",
    "CREATE INDEX IF NOT EXISTS idx_geo_event_region_available ON geo_event (region_code, available_at)",
    "CREATE INDEX IF NOT EXISTS idx_geo_event_type_available ON geo_event (event_type, available_at)",
    "CREATE INDEX IF NOT EXISTS idx_geo_event_dedupe_group ON geo_event (dedupe_group_id)",
    "CREATE INDEX IF NOT EXISTS idx_geo_event_status ON geo_event (status)",
    "CREATE INDEX IF NOT EXISTS idx_geo_event_tags_gin ON geo_event USING GIN (tags_json)",
    """
    CREATE TABLE IF NOT EXISTS geo_event_asset_impact (
        event_id TEXT NOT NULL REFERENCES geo_event(event_id) ON DELETE CASCADE,
        asset TEXT NOT NULL REFERENCES assets(ticker),
        mapping_version TEXT NOT NULL,
        impact_direction SMALLINT NOT NULL CHECK (impact_direction IN (-1, 0, 1)),
        directness_score DOUBLE PRECISION NOT NULL CHECK (directness_score >= 0.0 AND directness_score <= 1.0),
        region_exposure DOUBLE PRECISION NOT NULL CHECK (region_exposure >= 0.0 AND region_exposure <= 1.0),
        sector_exposure DOUBLE PRECISION NOT NULL CHECK (sector_exposure >= 0.0 AND sector_exposure <= 1.0),
        infra_exposure DOUBLE PRECISION NOT NULL CHECK (infra_exposure >= 0.0 AND infra_exposure <= 1.0),
        mapping_confidence DOUBLE PRECISION NOT NULL CHECK (mapping_confidence >= 0.0 AND mapping_confidence <= 1.0),
        impact_score DOUBLE PRECISION NOT NULL,
        impact_components_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (event_id, asset)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_geo_event_asset_impact_event_id ON geo_event_asset_impact (event_id)",
    "CREATE INDEX IF NOT EXISTS idx_geo_event_asset_impact_asset ON geo_event_asset_impact (asset)",
    "CREATE INDEX IF NOT EXISTS idx_geo_event_asset_impact_direction ON geo_event_asset_impact (asset, impact_direction)",
    "CREATE INDEX IF NOT EXISTS idx_geo_event_asset_impact_confidence ON geo_event_asset_impact (asset, mapping_confidence)",
    """
    CREATE TABLE IF NOT EXISTS geo_feature_snapshot (
        trade_date DATE NOT NULL,
        asset TEXT NOT NULL REFERENCES assets(ticker),
        snapshot_cutoff_at TIMESTAMPTZ NOT NULL,
        geo_net_raw DOUBLE PRECISION NOT NULL,
        geo_net_score DOUBLE PRECISION NOT NULL CHECK (geo_net_score >= -1.0 AND geo_net_score <= 1.0),
        geo_structural_score DOUBLE PRECISION NOT NULL CHECK (geo_structural_score >= -1.0 AND geo_structural_score <= 1.0),
        geo_harm_score DOUBLE PRECISION NOT NULL CHECK (geo_harm_score >= 0.0 AND geo_harm_score <= 1.0),
        geo_break_risk DOUBLE PRECISION NOT NULL CHECK (geo_break_risk >= 0.0 AND geo_break_risk <= 1.0),
        geo_velocity_3d DOUBLE PRECISION NOT NULL,
        geo_cluster_72h DOUBLE PRECISION NOT NULL CHECK (geo_cluster_72h >= 0.0),
        region_stress DOUBLE PRECISION NOT NULL,
        sector_disruption DOUBLE PRECISION NOT NULL,
        infra_disruption DOUBLE PRECISION NOT NULL,
        sanctions_score DOUBLE PRECISION NOT NULL,
        avg_mapping_confidence DOUBLE PRECISION NOT NULL CHECK (avg_mapping_confidence >= 0.0 AND avg_mapping_confidence <= 1.0),
        coverage_score DOUBLE PRECISION NOT NULL CHECK (coverage_score >= 0.0 AND coverage_score <= 1.0),
        data_freshness_minutes INTEGER NOT NULL CHECK (data_freshness_minutes >= 0),
        hard_override BOOLEAN NOT NULL,
        contributing_event_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (trade_date, asset)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_geo_feature_snapshot_asset_date ON geo_feature_snapshot (asset, trade_date)",
    "CREATE INDEX IF NOT EXISTS idx_geo_feature_snapshot_date ON geo_feature_snapshot (trade_date)",
    "CREATE INDEX IF NOT EXISTS idx_geo_feature_snapshot_override ON geo_feature_snapshot (trade_date, hard_override)",
    """
    CREATE TABLE IF NOT EXISTS geo_data_health (
        source_system TEXT NOT NULL,
        source_endpoint TEXT NOT NULL,
        checked_at TIMESTAMPTZ NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('HEALTHY', 'PARTIAL', 'STALE', 'FAILED')),
        freshness_minutes INTEGER NOT NULL CHECK (freshness_minutes >= 0),
        max_allowed_staleness_minutes INTEGER NOT NULL CHECK (max_allowed_staleness_minutes > 0),
        records_seen_24h INTEGER NOT NULL CHECK (records_seen_24h >= 0),
        expected_records_24h INTEGER NOT NULL CHECK (expected_records_24h >= 0),
        coverage_score DOUBLE PRECISION NOT NULL CHECK (coverage_score >= 0.0 AND coverage_score <= 1.0),
        last_success_at TIMESTAMPTZ,
        error_message TEXT,
        details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        PRIMARY KEY (source_system, source_endpoint, checked_at)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_geo_data_health_status_checked ON geo_data_health (status, checked_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_geo_data_health_last_success ON geo_data_health (last_success_at DESC)",
]


@dataclass(frozen=True)
class DatabaseInitResult:
    dsn: str
    timescaledb_enabled: bool


@dataclass(frozen=True)
class DatabaseVerificationResult:
    daily_prices_row_count: int
    distinct_tickers: int
    earliest_trade_date: str | None
    latest_trade_date: str | None
    indexes_present: bool
    quality_report_rows: int


@dataclass(frozen=True)
class Phase2RunSummary:
    run_id: str
    sharpe_ratio: float | None
    max_drawdown: float | None
    win_rate: float | None
    profit_factor: float | None
    avg_holding_days: float | None
    annual_turnover: float | None
    profitable_month_fraction: float | None
    out_of_sample_months: int
    gate_passed: bool


@dataclass(frozen=True)
class Phase2RunArtifacts:
    run_id: str
    config_snapshot: dict[str, object]
    daily_signals: pd.DataFrame


class NoOpDatabase:
    def __init__(self, config: DatabaseConfig, paths: PathsConfig, logger) -> None:
        self.config = config
        self.paths = paths
        self.logger = logger
        self.disabled = True

    def initialize(self) -> DatabaseInitResult:
        return DatabaseInitResult(dsn=self.dsn(self.config.database_name), timescaledb_enabled=False)

    def stop(self) -> None:
        return None

    def dsn(self, database_name: str | None = None) -> str:
        target_db = database_name or self.config.database_name
        return f"disabled://{target_db}"

    def persist_pipeline_run(
        self,
        *,
        run_id: str,
        source_name: str,
        start_date: str,
        end_date: str | None,
        config_snapshot: dict[str, object],
        raw_prices: pd.DataFrame,
        validated_prices: pd.DataFrame,
        quality_report: pd.DataFrame,
        issue_report: pd.DataFrame,
        started_at: datetime,
        completed_at: datetime,
    ) -> None:
        return None

    def verify_phase1_gate(self) -> DatabaseVerificationResult:
        validated_path = self.paths.processed_dir / "sector_etf_prices_validated.csv"
        quality_path = self.paths.processed_dir / "sector_etf_quality_report.csv"
        validated_prices = pd.read_csv(validated_path, parse_dates=["date"]) if validated_path.exists() else pd.DataFrame()
        quality_report = pd.read_csv(quality_path) if quality_path.exists() else pd.DataFrame()

        earliest_trade_date = None
        latest_trade_date = None
        if not validated_prices.empty and "date" in validated_prices.columns:
            earliest = pd.to_datetime(validated_prices["date"]).min()
            latest = pd.to_datetime(validated_prices["date"]).max()
            earliest_trade_date = earliest.date().isoformat() if pd.notna(earliest) else None
            latest_trade_date = latest.date().isoformat() if pd.notna(latest) else None

        distinct_tickers = 0
        if "ticker" in validated_prices.columns:
            distinct_tickers = int(validated_prices["ticker"].nunique())

        return DatabaseVerificationResult(
            daily_prices_row_count=int(len(validated_prices)),
            distinct_tickers=distinct_tickers,
            earliest_trade_date=earliest_trade_date,
            latest_trade_date=latest_trade_date,
            indexes_present=False,
            quality_report_rows=int(len(quality_report)),
        )

    def ensure_phase2_schema(self) -> None:
        return None

    def ensure_geo_schema(self) -> None:
        return None

    def fetch_validated_price_history(self, tickers: list[str]) -> pd.DataFrame:
        validated_path = self.paths.processed_dir / "sector_etf_prices_validated.csv"
        if not validated_path.exists():
            return pd.DataFrame(columns=["date", "ticker", "adj_close"])

        prices = pd.read_csv(validated_path, parse_dates=["date"])
        required_columns = {"date", "ticker", "adj_close"}
        if not required_columns.issubset(prices.columns):
            return pd.DataFrame(columns=["date", "ticker", "adj_close"])
        if "is_valid" in prices.columns:
            prices = prices.loc[prices["is_valid"]]
        prices = prices.loc[prices["ticker"].isin(tickers), ["date", "ticker", "adj_close"]].copy()
        return prices.sort_values(["date", "ticker"]).reset_index(drop=True)

    def persist_phase2_run(
        self,
        *,
        run_id: str,
        config_snapshot: dict[str, object],
        summary_metrics: dict[str, object],
        daily_signals: pd.DataFrame,
        trade_log: pd.DataFrame,
        monthly_results: pd.DataFrame,
        started_at: datetime,
        completed_at: datetime,
    ) -> None:
        return None

    def fetch_latest_phase2_run_summary(self) -> Phase2RunSummary | None:
        summary_path = self.paths.processed_dir / "phase2_summary.json"
        if not summary_path.exists():
            return None

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        run_id = payload.get("run_id")
        if run_id is None:
            return None

        return Phase2RunSummary(
            run_id=str(run_id),
            sharpe_ratio=_optional_float(payload.get("sharpe_ratio")),
            max_drawdown=_optional_float(payload.get("max_drawdown")),
            win_rate=_optional_float(payload.get("win_rate")),
            profit_factor=_optional_float(payload.get("profit_factor")),
            avg_holding_days=_optional_float(payload.get("avg_holding_days")),
            annual_turnover=_optional_float(payload.get("annual_turnover")),
            profitable_month_fraction=_optional_float(payload.get("profitable_month_fraction")),
            out_of_sample_months=_optional_int(payload.get("out_of_sample_months")) or 0,
            gate_passed=bool(payload.get("gate_passed", False)),
        )

    def fetch_phase2_run_artifacts(self, run_id: str) -> Phase2RunArtifacts:
        summary_path = self.paths.processed_dir / "phase2_summary.json"
        signals_path = self.paths.processed_dir / "phase2_daily_signals.csv"
        payload: dict[str, object] = {}
        stored_run_id = run_id

        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_run_id = payload.get("run_id")
            if summary_run_id is not None:
                stored_run_id = str(summary_run_id)
                if stored_run_id != run_id:
                    raise ValueError(f"Phase 2 run '{run_id}' was not found.")

        if not signals_path.exists():
            raise ValueError(f"Phase 2 run '{run_id}' has no stored daily signals.")

        daily_signals = pd.read_csv(signals_path, parse_dates=["date"])
        if daily_signals.empty:
            raise ValueError(f"Phase 2 run '{run_id}' has no stored daily signals.")

        config_snapshot = payload.get("config_snapshot", {})
        if not isinstance(config_snapshot, dict):
            config_snapshot = {}

        return Phase2RunArtifacts(
            run_id=stored_run_id,
            config_snapshot=config_snapshot,
            daily_signals=daily_signals,
        )


class _PostgresStoreImpl:
    def __init__(self, config: DatabaseConfig, paths: PathsConfig, logger) -> None:
        self.config = config
        self.paths = paths
        self.logger = logger

    def initialize(self) -> DatabaseInitResult:
        self.paths.postgres_dir.mkdir(parents=True, exist_ok=True)
        self.paths.postgres_log_file.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_cluster()
        self._start_server()
        self._ensure_database_exists()
        timescaledb_enabled = self._ensure_schema()
        return DatabaseInitResult(
            dsn=self.dsn(self.config.database_name),
            timescaledb_enabled=timescaledb_enabled,
        )

    def stop(self) -> None:
        try:
            if self._is_running():
                pg_ctl(["-w", "stop"], pgdata=self.paths.postgres_dir, timeout=30)
        except subprocess.CalledProcessError:
            self.logger.warning("PostgreSQL stop command failed")

    def dsn(self, database_name: str | None = None) -> str:
        target_db = database_name or self.config.database_name
        return (
            f"postgresql://{self.config.user}@{self.config.host}:{self.config.port}/{target_db}"
        )

    def persist_pipeline_run(
        self,
        *,
        run_id: str,
        source_name: str,
        start_date: str,
        end_date: str | None,
        config_snapshot: dict[str, object],
        raw_prices: pd.DataFrame,
        validated_prices: pd.DataFrame,
        quality_report: pd.DataFrame,
        issue_report: pd.DataFrame,
        started_at: datetime,
        completed_at: datetime,
    ) -> None:
        with psycopg.connect(self.dsn(self.config.database_name)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ingestion_runs (
                        run_id, source_name, started_at, completed_at, status,
                        start_date, end_date, raw_rows, valid_rows, issue_rows,
                        message, config_snapshot
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s::jsonb
                    )
                    ON CONFLICT (run_id) DO UPDATE SET
                        completed_at = EXCLUDED.completed_at,
                        status = EXCLUDED.status,
                        raw_rows = EXCLUDED.raw_rows,
                        valid_rows = EXCLUDED.valid_rows,
                        issue_rows = EXCLUDED.issue_rows,
                        message = EXCLUDED.message,
                        config_snapshot = EXCLUDED.config_snapshot
                    """,
                    (
                        run_id,
                        source_name,
                        started_at,
                        completed_at,
                        "completed",
                        start_date,
                        end_date,
                        len(raw_prices),
                        int(validated_prices["is_valid"].sum()),
                        len(issue_report),
                        "Phase 1 pipeline completed successfully",
                        json.dumps(config_snapshot),
                    ),
                )
                self._upsert_assets(cur, sorted(validated_prices["ticker"].unique().tolist()), source_name)
                self._upsert_daily_prices(cur, validated_prices, run_id)
                self._upsert_corporate_actions(cur, validated_prices, run_id)
                self._upsert_validation_events(cur, validated_prices, run_id)
                self._upsert_quality_report(cur, quality_report, run_id)
            conn.commit()

    def verify_phase1_gate(self) -> DatabaseVerificationResult:
        with psycopg.connect(self.dsn(self.config.database_name)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(trade_date), MAX(trade_date)
                    FROM daily_prices
                    """
                )
                row_count, ticker_count, earliest_date, latest_date = cur.fetchone()

                cur.execute("SELECT COUNT(*) FROM data_quality_reports")
                quality_report_rows = cur.fetchone()[0]

                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM pg_indexes
                    WHERE tablename = 'daily_prices'
                      AND indexname IN ('daily_prices_pkey', 'idx_daily_prices_trade_date', 'idx_daily_prices_ticker')
                    """
                )
                indexes_present = cur.fetchone()[0] == 3

        return DatabaseVerificationResult(
            daily_prices_row_count=row_count,
            distinct_tickers=ticker_count,
            earliest_trade_date=earliest_date.isoformat() if earliest_date else None,
            latest_trade_date=latest_date.isoformat() if latest_date else None,
            indexes_present=indexes_present,
            quality_report_rows=quality_report_rows,
        )

    def ensure_phase2_schema(self) -> None:
        with psycopg.connect(self.dsn(self.config.database_name)) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                for statement in PHASE2_SCHEMA_STATEMENTS:
                    cur.execute(statement)

    def ensure_geo_schema(self) -> None:
        with psycopg.connect(self.dsn(self.config.database_name)) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                for statement in GEO_SCHEMA_STATEMENTS:
                    cur.execute(statement)

    def fetch_validated_price_history(self, tickers: list[str]) -> pd.DataFrame:
        query = """
            SELECT trade_date AS date, ticker, adj_close
            FROM daily_prices
            WHERE is_valid = TRUE
              AND ticker = ANY(%s)
            ORDER BY trade_date, ticker
        """
        with psycopg.connect(self.dsn(self.config.database_name)) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (tickers,))
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
        prices = pd.DataFrame(rows, columns=columns)
        prices["date"] = pd.to_datetime(prices["date"])
        return prices

    def persist_phase2_run(
        self,
        *,
        run_id: str,
        config_snapshot: dict[str, object],
        summary_metrics: dict[str, object],
        daily_signals: pd.DataFrame,
        trade_log: pd.DataFrame,
        monthly_results: pd.DataFrame,
        started_at: datetime,
        completed_at: datetime,
    ) -> None:
        with psycopg.connect(self.dsn(self.config.database_name)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO phase2_runs (
                        run_id, started_at, completed_at, status,
                        lookback_window, diffusion_alpha, diffusion_steps,
                        signal_threshold, max_position_size, total_signals, total_trades,
                        sharpe_ratio, max_drawdown, win_rate, profit_factor,
                        avg_holding_days, annual_turnover, profitable_month_fraction,
                        out_of_sample_months, gate_passed, config_snapshot
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s::jsonb
                    )
                    ON CONFLICT (run_id) DO UPDATE SET
                        completed_at = EXCLUDED.completed_at,
                        status = EXCLUDED.status,
                        total_signals = EXCLUDED.total_signals,
                        total_trades = EXCLUDED.total_trades,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        max_drawdown = EXCLUDED.max_drawdown,
                        win_rate = EXCLUDED.win_rate,
                        profit_factor = EXCLUDED.profit_factor,
                        avg_holding_days = EXCLUDED.avg_holding_days,
                        annual_turnover = EXCLUDED.annual_turnover,
                        profitable_month_fraction = EXCLUDED.profitable_month_fraction,
                        out_of_sample_months = EXCLUDED.out_of_sample_months,
                        gate_passed = EXCLUDED.gate_passed,
                        config_snapshot = EXCLUDED.config_snapshot
                    """,
                    (
                        run_id,
                        started_at,
                        completed_at,
                        "completed",
                        int(summary_metrics["lookback_window"]),
                        float(summary_metrics["diffusion_alpha"]),
                        int(summary_metrics["diffusion_steps"]),
                        float(summary_metrics["signal_threshold"]),
                        float(summary_metrics["max_position_size"]),
                        int(summary_metrics["total_signals"]),
                        int(summary_metrics["total_trades"]),
                        _optional_float(summary_metrics.get("sharpe_ratio")),
                        _optional_float(summary_metrics.get("max_drawdown")),
                        _optional_float(summary_metrics.get("win_rate")),
                        _optional_float(summary_metrics.get("profit_factor")),
                        _optional_float(summary_metrics.get("avg_holding_days")),
                        _optional_float(summary_metrics.get("annual_turnover")),
                        _optional_float(summary_metrics.get("profitable_month_fraction")),
                        int(summary_metrics["out_of_sample_months"]),
                        bool(summary_metrics["gate_passed"]),
                        json.dumps(config_snapshot),
                    ),
                )
                self._upsert_phase2_daily_signals(cur, run_id, daily_signals)
                self._upsert_phase2_trade_log(cur, trade_log)
                self._upsert_phase2_monthly_results(cur, run_id, monthly_results)
            conn.commit()

    def fetch_latest_phase2_run_summary(self) -> Phase2RunSummary | None:
        query = """
            SELECT run_id, sharpe_ratio, max_drawdown, win_rate, profit_factor,
                   avg_holding_days, annual_turnover, profitable_month_fraction,
                   out_of_sample_months, gate_passed
            FROM phase2_runs
            ORDER BY completed_at DESC NULLS LAST, started_at DESC
            LIMIT 1
        """
        with psycopg.connect(self.dsn(self.config.database_name)) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                row = cur.fetchone()

        if row is None:
            return None

        return Phase2RunSummary(
            run_id=row[0],
            sharpe_ratio=_optional_float(row[1]),
            max_drawdown=_optional_float(row[2]),
            win_rate=_optional_float(row[3]),
            profit_factor=_optional_float(row[4]),
            avg_holding_days=_optional_float(row[5]),
            annual_turnover=_optional_float(row[6]),
            profitable_month_fraction=_optional_float(row[7]),
            out_of_sample_months=int(row[8]),
            gate_passed=bool(row[9]),
        )

    def fetch_phase2_run_artifacts(self, run_id: str) -> Phase2RunArtifacts:
        config_query = """
            SELECT config_snapshot
            FROM phase2_runs
            WHERE run_id = %s
        """
        signals_query = """
            SELECT
                signal_date AS date,
                ticker,
                current_return,
                expected_return,
                residual,
                residual_zscore AS zscore,
                signal_direction,
                target_position,
                forward_return,
                sigma,
                edge_density
            FROM phase2_daily_signals
            WHERE run_id = %s
            ORDER BY signal_date, ticker
        """
        with psycopg.connect(self.dsn(self.config.database_name)) as conn:
            with conn.cursor() as cur:
                cur.execute(config_query, (run_id,))
                config_row = cur.fetchone()
                if config_row is None:
                    raise ValueError(f"Phase 2 run '{run_id}' was not found.")

                cur.execute(signals_query, (run_id,))
                signal_rows = cur.fetchall()
                signal_columns = [desc[0] for desc in cur.description]

        daily_signals = pd.DataFrame(signal_rows, columns=signal_columns)
        if daily_signals.empty:
            raise ValueError(f"Phase 2 run '{run_id}' has no stored daily signals.")
        daily_signals["date"] = pd.to_datetime(daily_signals["date"])

        return Phase2RunArtifacts(
            run_id=run_id,
            config_snapshot=dict(config_row[0]),
            daily_signals=daily_signals,
        )

    def _initialize_cluster(self) -> None:
        if (self.paths.postgres_dir / "PG_VERSION").exists():
            return

        initdb(
            ["--auth=trust", "--auth-local=trust", "--encoding=utf8", "-U", self.config.user],
            pgdata=self.paths.postgres_dir,
        )
        self.logger.info(
            "Initialized PostgreSQL cluster",
            extra={"context": {"pgdata": str(self.paths.postgres_dir)}},
        )

    def _start_server(self) -> None:
        if self._is_running():
            return

        pg_ctl(
            [
                "-w",
                "-o",
                f"-h {self.config.host}",
                "-o",
                f"-p {self.config.port}",
                "-l",
                str(self.paths.postgres_log_file),
                "start",
            ],
            pgdata=self.paths.postgres_dir,
            timeout=30,
        )
        self.logger.info(
            "Started PostgreSQL server",
            extra={
                "context": {
                    "host": self.config.host,
                    "port": self.config.port,
                    "database": self.config.database_name,
                }
            },
        )

    def _is_running(self) -> bool:
        try:
            pg_ctl(["status"], pgdata=self.paths.postgres_dir, timeout=5)
            return True
        except subprocess.CalledProcessError:
            return False

    def _ensure_database_exists(self) -> None:
        with psycopg.connect(self.dsn(self.config.admin_database), autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (self.config.database_name,),
                )
                if cur.fetchone() is None:
                    cur.execute(
                        sql.SQL("CREATE DATABASE {}").format(
                            sql.Identifier(self.config.database_name)
                        )
                    )

    def _ensure_schema(self) -> bool:
        timescaledb_enabled = False
        with psycopg.connect(self.dsn(self.config.database_name)) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
                    timescaledb_enabled = True
                except Exception as exc:
                    if self.config.require_timescaledb:
                        raise
                    self.logger.warning(
                        "TimescaleDB extension unavailable; continuing with standard PostgreSQL",
                        extra={"context": {"error": str(exc)}},
                    )
                    conn.rollback()

                for statement in SCHEMA_STATEMENTS:
                    cur.execute(statement)

        return timescaledb_enabled

    def _upsert_assets(self, cursor: psycopg.Cursor[Any], tickers: list[str], source_name: str) -> None:
        cursor.executemany(
            """
            INSERT INTO assets (ticker, asset_type, source_name)
            VALUES (%s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
                asset_type = EXCLUDED.asset_type,
                source_name = EXCLUDED.source_name
            """,
            [(ticker, "sector_etf", source_name) for ticker in tickers],
        )

    def _upsert_daily_prices(
        self,
        cursor: psycopg.Cursor[Any],
        validated_prices: pd.DataFrame,
        run_id: str,
    ) -> None:
        rows = []
        for record in validated_prices.itertuples(index=False):
            rows.append(
                (
                    record.ticker,
                    record.date.date(),
                    _optional_float(record.open),
                    _optional_float(record.high),
                    _optional_float(record.low),
                    _optional_float(record.close),
                    _optional_float(record.adj_close),
                    _optional_int(record.volume),
                    _optional_float(record.dividends),
                    _optional_float(record.stock_splits),
                    _optional_float(record.capital_gains),
                    _optional_float(record.daily_return),
                    bool(record.is_valid),
                    str(record.validation_notes),
                    str(record.cross_source_status),
                    bool(record.corporate_action_flag),
                    bool(record.return_magnitude_flag),
                    bool(record.zero_volume_flag),
                    bool(record.continuity_flag),
                    bool(record.split_detection_flag),
                    bool(record.adjustment_gap_flag),
                    bool(record.cross_source_validation_flag),
                    int(record.missing_business_days),
                    run_id,
                )
            )

        cursor.executemany(
            """
            INSERT INTO daily_prices (
                ticker, trade_date, open, high, low, close, adj_close, volume,
                dividends, stock_splits, capital_gains, daily_return, is_valid,
                validation_notes, cross_source_status, corporate_action_flag,
                return_magnitude_flag, zero_volume_flag, continuity_flag,
                split_detection_flag, adjustment_gap_flag, cross_source_validation_flag,
                missing_business_days, run_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s
            )
            ON CONFLICT (ticker, trade_date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                adj_close = EXCLUDED.adj_close,
                volume = EXCLUDED.volume,
                dividends = EXCLUDED.dividends,
                stock_splits = EXCLUDED.stock_splits,
                capital_gains = EXCLUDED.capital_gains,
                daily_return = EXCLUDED.daily_return,
                is_valid = EXCLUDED.is_valid,
                validation_notes = EXCLUDED.validation_notes,
                cross_source_status = EXCLUDED.cross_source_status,
                corporate_action_flag = EXCLUDED.corporate_action_flag,
                return_magnitude_flag = EXCLUDED.return_magnitude_flag,
                zero_volume_flag = EXCLUDED.zero_volume_flag,
                continuity_flag = EXCLUDED.continuity_flag,
                split_detection_flag = EXCLUDED.split_detection_flag,
                adjustment_gap_flag = EXCLUDED.adjustment_gap_flag,
                cross_source_validation_flag = EXCLUDED.cross_source_validation_flag,
                missing_business_days = EXCLUDED.missing_business_days,
                run_id = EXCLUDED.run_id,
                updated_at = NOW()
            """,
            rows,
        )

    def _upsert_corporate_actions(
        self,
        cursor: psycopg.Cursor[Any],
        validated_prices: pd.DataFrame,
        run_id: str,
    ) -> None:
        corporate_actions = validated_prices.loc[validated_prices["corporate_action_flag"]].copy()
        if corporate_actions.empty:
            return

        cursor.executemany(
            """
            INSERT INTO corporate_actions (
                ticker, action_date, dividends, stock_splits, capital_gains, run_id
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, action_date) DO UPDATE SET
                dividends = EXCLUDED.dividends,
                stock_splits = EXCLUDED.stock_splits,
                capital_gains = EXCLUDED.capital_gains,
                run_id = EXCLUDED.run_id
            """,
            [
                (
                    record.ticker,
                    record.date.date(),
                    _optional_float(record.dividends) or 0.0,
                    _optional_float(record.stock_splits) or 0.0,
                    _optional_float(record.capital_gains) or 0.0,
                    run_id,
                )
                for record in corporate_actions.itertuples(index=False)
            ],
        )

    def _upsert_validation_events(
        self,
        cursor: psycopg.Cursor[Any],
        validated_prices: pd.DataFrame,
        run_id: str,
    ) -> None:
        events: list[tuple[Any, ...]] = []
        event_specs = [
            ("return_magnitude_flag", "return_magnitude", "error"),
            ("zero_volume_flag", "zero_volume", "error"),
            ("continuity_flag", "continuity_gap", "error"),
            ("corporate_action_flag", "corporate_action", "info"),
        ]
        for record in validated_prices.itertuples(index=False):
            for attr_name, event_type, severity in event_specs:
                if bool(getattr(record, attr_name)):
                    events.append(
                        (
                            run_id,
                            record.ticker,
                            record.date.date(),
                            event_type,
                            severity,
                            record.validation_notes,
                        )
                    )

        if not events:
            return

        cursor.executemany(
            """
            INSERT INTO validation_events (
                run_id, ticker, trade_date, event_type, severity, notes
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, ticker, trade_date, event_type) DO UPDATE SET
                severity = EXCLUDED.severity,
                notes = EXCLUDED.notes
            """,
            events,
        )

    def _upsert_quality_report(
        self,
        cursor: psycopg.Cursor[Any],
        quality_report: pd.DataFrame,
        run_id: str,
    ) -> None:
        cursor.executemany(
            """
            INSERT INTO data_quality_reports (
                run_id, ticker, total_rows, valid_rows, invalid_rows,
                return_magnitude_issues, zero_volume_issues, continuity_issues,
                split_detection_rows, adjustment_gap_rows, corporate_action_events,
                first_date, last_date, cross_source_status
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (run_id, ticker) DO UPDATE SET
                total_rows = EXCLUDED.total_rows,
                valid_rows = EXCLUDED.valid_rows,
                invalid_rows = EXCLUDED.invalid_rows,
                return_magnitude_issues = EXCLUDED.return_magnitude_issues,
                zero_volume_issues = EXCLUDED.zero_volume_issues,
                continuity_issues = EXCLUDED.continuity_issues,
                split_detection_rows = EXCLUDED.split_detection_rows,
                adjustment_gap_rows = EXCLUDED.adjustment_gap_rows,
                corporate_action_events = EXCLUDED.corporate_action_events,
                first_date = EXCLUDED.first_date,
                last_date = EXCLUDED.last_date,
                cross_source_status = EXCLUDED.cross_source_status
            """,
            [
                (
                    run_id,
                    record.ticker,
                    int(record.total_rows),
                    int(record.valid_rows),
                    int(record.invalid_rows),
                    int(record.return_magnitude_issues),
                    int(record.zero_volume_issues),
                    int(record.continuity_issues),
                    int(record.split_detection_rows),
                    int(record.adjustment_gap_rows),
                    int(record.corporate_action_events),
                    pd.to_datetime(record.first_date).date(),
                    pd.to_datetime(record.last_date).date(),
                    str(record.cross_source_status),
                )
                for record in quality_report.itertuples(index=False)
            ],
        )

    def _upsert_phase2_daily_signals(
        self,
        cursor: psycopg.Cursor[Any],
        run_id: str,
        daily_signals: pd.DataFrame,
    ) -> None:
        if daily_signals.empty:
            return

        cursor.executemany(
            """
            INSERT INTO phase2_daily_signals (
                run_id, signal_date, ticker, current_return, expected_return, residual,
                residual_zscore, signal_direction, target_position, forward_return, sigma, edge_density
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (run_id, signal_date, ticker) DO UPDATE SET
                current_return = EXCLUDED.current_return,
                expected_return = EXCLUDED.expected_return,
                residual = EXCLUDED.residual,
                residual_zscore = EXCLUDED.residual_zscore,
                signal_direction = EXCLUDED.signal_direction,
                target_position = EXCLUDED.target_position,
                forward_return = EXCLUDED.forward_return,
                sigma = EXCLUDED.sigma,
                edge_density = EXCLUDED.edge_density
            """,
            [
                (
                    run_id,
                    record.date.date(),
                    record.ticker,
                    float(record.current_return),
                    float(record.expected_return),
                    float(record.residual),
                    _optional_float(record.zscore),
                    int(record.signal_direction),
                    float(record.target_position),
                    _optional_float(record.forward_return),
                    float(record.sigma),
                    float(record.edge_density),
                )
                for record in daily_signals.itertuples(index=False)
            ],
        )

    def _upsert_phase2_trade_log(
        self,
        cursor: psycopg.Cursor[Any],
        trade_log: pd.DataFrame,
    ) -> None:
        if trade_log.empty:
            return

        cursor.executemany(
            """
            INSERT INTO phase2_trade_log (
                trade_id, run_id, ticker, entry_date, exit_date, position_direction,
                entry_zscore, exit_zscore, holding_days, entry_weight, gross_return, net_return
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (trade_id) DO UPDATE SET
                exit_date = EXCLUDED.exit_date,
                position_direction = EXCLUDED.position_direction,
                entry_zscore = EXCLUDED.entry_zscore,
                exit_zscore = EXCLUDED.exit_zscore,
                holding_days = EXCLUDED.holding_days,
                entry_weight = EXCLUDED.entry_weight,
                gross_return = EXCLUDED.gross_return,
                net_return = EXCLUDED.net_return
            """,
            [
                (
                    str(record.trade_id),
                    str(record.run_id),
                    record.ticker,
                    pd.to_datetime(record.entry_date).date(),
                    pd.to_datetime(record.exit_date).date(),
                    int(record.position_direction),
                    _optional_float(record.entry_zscore),
                    _optional_float(record.exit_zscore),
                    int(record.holding_days),
                    float(record.entry_weight),
                    float(record.gross_return),
                    float(record.net_return),
                )
                for record in trade_log.itertuples(index=False)
            ],
        )

    def _upsert_phase2_monthly_results(
        self,
        cursor: psycopg.Cursor[Any],
        run_id: str,
        monthly_results: pd.DataFrame,
    ) -> None:
        if monthly_results.empty:
            return

        cursor.executemany(
            """
            INSERT INTO phase2_monthly_results (
                run_id, test_month, training_end_date, monthly_return, profitable
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (run_id, test_month) DO UPDATE SET
                training_end_date = EXCLUDED.training_end_date,
                monthly_return = EXCLUDED.monthly_return,
                profitable = EXCLUDED.profitable
            """,
            [
                (
                    run_id,
                    pd.to_datetime(record.test_month).date(),
                    pd.to_datetime(record.training_end_date).date(),
                    float(record.monthly_return),
                    bool(record.profitable),
                )
                for record in monthly_results.itertuples(index=False)
            ],
        )


def PostgresStore(config: DatabaseConfig, paths: PathsConfig, logger) -> _PostgresStoreImpl | NoOpDatabase:
    if _DISABLE_DB or _DB_IMPORT_ERROR is not None:
        return NoOpDatabase(config, paths, logger)
    return _PostgresStoreImpl(config, paths, logger)


def _optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)
