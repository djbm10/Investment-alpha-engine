from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config_loader import PathsConfig, PipelineConfig
from .ingestion import download_universe_data
from .logging_utils import setup_logger
from .validation import build_issue_report, build_quality_report, validate_prices


@dataclass(frozen=True)
class PriceArtifactSpec:
    dataset: str
    tickers: list[str]
    raw_path: Path
    validated_path: Path
    quality_path: Path
    issue_path: Path
    required_columns: tuple[str, ...]
    phase: str


def ensure_output_directories(paths: PathsConfig) -> None:
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    if os.getenv("DISABLE_DB") != "true":
        paths.postgres_dir.mkdir(parents=True, exist_ok=True)


def save_pipeline_outputs(
    raw_prices: pd.DataFrame,
    validated_prices: pd.DataFrame,
    quality_report: pd.DataFrame,
    issue_report: pd.DataFrame,
    paths: PathsConfig,
) -> dict[str, Path]:
    ensure_output_directories(paths)

    raw_path = paths.raw_dir / "sector_etf_prices_raw.csv"
    clean_path = paths.processed_dir / "sector_etf_prices_validated.csv"
    quality_path = paths.processed_dir / "sector_etf_quality_report.csv"
    issues_path = paths.processed_dir / "sector_etf_validation_issues.csv"

    raw_prices.to_csv(raw_path, index=False)
    validated_prices.to_csv(clean_path, index=False)
    quality_report.to_csv(quality_path, index=False)
    issue_report.to_csv(issues_path, index=False)

    return {
        "raw_prices": raw_path,
        "validated_prices": clean_path,
        "quality_report": quality_path,
        "issue_report": issues_path,
    }


def load_validated_price_data(
    config: PipelineConfig,
    *,
    dataset: str,
    logger=None,
) -> pd.DataFrame:
    spec = _price_artifact_spec(config, dataset)
    ensure_output_directories(config.paths)

    artifact_ready, reason = _validated_artifact_ready(spec)
    if not artifact_ready:
        active_logger = logger or setup_logger(
            config.paths.pipeline_log_file,
            task="data-bootstrap",
            dataset=spec.dataset,
            phase=spec.phase,
        )
        active_logger.warning(
            "Validated price artifact unavailable; regenerating upstream data.",
            extra={
                "context": {
                    "dataset": spec.dataset,
                    "validated_path": str(spec.validated_path),
                    "reason": reason,
                    "tickers": spec.tickers,
                }
            },
        )
        return _refresh_validated_price_data(config, spec, active_logger)

    return pd.read_csv(spec.validated_path, parse_dates=["date"])


def _price_artifact_spec(config: PipelineConfig, dataset: str) -> PriceArtifactSpec:
    if dataset == "sector":
        return PriceArtifactSpec(
            dataset="sector",
            tickers=list(config.tickers),
            raw_path=config.paths.raw_dir / "sector_etf_prices_raw.csv",
            validated_path=config.paths.processed_dir / "sector_etf_prices_validated.csv",
            quality_path=config.paths.processed_dir / "sector_etf_quality_report.csv",
            issue_path=config.paths.processed_dir / "sector_etf_validation_issues.csv",
            required_columns=("date", "ticker", "adj_close", "is_valid"),
            phase="phase1",
        )
    if dataset == "trend":
        return PriceArtifactSpec(
            dataset="trend",
            tickers=list(config.phase5.trend_tickers),
            raw_path=config.paths.raw_dir / "trend_universe_prices_raw.csv",
            validated_path=config.paths.processed_dir / "trend_universe_prices_validated.csv",
            quality_path=config.paths.processed_dir / "trend_universe_quality_report.csv",
            issue_path=config.paths.processed_dir / "trend_universe_validation_issues.csv",
            required_columns=("date", "ticker", "adj_close", "volume", "is_valid"),
            phase="phase5",
        )
    raise ValueError(f"Unsupported validated price dataset '{dataset}'.")


def _validated_artifact_ready(spec: PriceArtifactSpec) -> tuple[bool, str]:
    if not spec.validated_path.exists():
        return False, "file_missing"

    try:
        header = pd.read_csv(spec.validated_path, nrows=5)
    except Exception as exc:  # pragma: no cover - defensive guard
        return False, f"unreadable:{exc}"

    missing_columns = sorted(set(spec.required_columns) - set(header.columns))
    if missing_columns:
        return False, f"missing_columns:{','.join(missing_columns)}"

    available_tickers = set(
        pd.read_csv(spec.validated_path, usecols=["ticker"])["ticker"].dropna().astype(str)
    )
    missing_tickers = sorted(set(spec.tickers) - available_tickers)
    if missing_tickers:
        return False, f"missing_tickers:{','.join(missing_tickers)}"

    return True, "ready"


def _refresh_validated_price_data(
    config: PipelineConfig,
    spec: PriceArtifactSpec,
    logger,
) -> pd.DataFrame:
    raw_prices = download_universe_data(
        tickers=spec.tickers,
        start_date=config.start_date,
        end_date=config.end_date,
        cache_dir=config.paths.cache_dir,
        logger=logger,
    )
    validated_prices = validate_prices(raw_prices, config.validation)
    quality_report = build_quality_report(validated_prices)
    issue_report = build_issue_report(validated_prices)

    spec.raw_path.parent.mkdir(parents=True, exist_ok=True)
    spec.validated_path.parent.mkdir(parents=True, exist_ok=True)
    raw_prices.to_csv(spec.raw_path, index=False)
    validated_prices.to_csv(spec.validated_path, index=False)
    quality_report.to_csv(spec.quality_path, index=False)
    issue_report.to_csv(spec.issue_path, index=False)
    return validated_prices
