from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    processed_dir: Path
    log_file: Path
    cache_dir: Path


@dataclass(frozen=True)
class ValidationConfig:
    max_abs_daily_return: float
    max_missing_business_days: int


@dataclass(frozen=True)
class PipelineConfig:
    tickers: list[str]
    start_date: str
    end_date: str | None
    paths: PathsConfig
    validation: ValidationConfig


def load_config(config_path: str | Path) -> PipelineConfig:
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    paths = PathsConfig(
        raw_dir=Path(data["paths"]["raw_dir"]),
        processed_dir=Path(data["paths"]["processed_dir"]),
        log_file=Path(data["paths"]["log_file"]),
        cache_dir=Path(data["paths"]["cache_dir"]),
    )
    validation = ValidationConfig(
        max_abs_daily_return=float(data["validation"]["max_abs_daily_return"]),
        max_missing_business_days=int(data["validation"]["max_missing_business_days"]),
    )

    return PipelineConfig(
        tickers=list(data["tickers"]),
        start_date=data["start_date"],
        end_date=data.get("end_date"),
        paths=paths,
        validation=validation,
    )
