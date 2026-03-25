from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config_loader import PathsConfig


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
