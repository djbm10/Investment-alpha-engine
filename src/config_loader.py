from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    log_dir: Path
    pipeline_log_file: Path
    postgres_log_file: Path
    cache_dir: Path
    postgres_dir: Path


@dataclass(frozen=True)
class ValidationConfig:
    max_abs_daily_return: float
    max_missing_business_days: int
    adj_close_close_tolerance: float


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    user: str
    admin_database: str
    database_name: str
    require_timescaledb: bool


@dataclass(frozen=True)
class ScheduleConfig:
    timezone: str
    hour: int
    minute: int


@dataclass(frozen=True)
class Phase2Config:
    lookback_window: int
    diffusion_alpha: float
    diffusion_steps: int
    sigma_scale: float
    min_weight: float
    zscore_lookback: int
    signal_threshold: float
    tier2_fraction: float
    tier2_size_fraction: float
    full_size_zscore: float
    max_position_size: float
    risk_budget_utilization: float
    max_drawdown_limit: float
    enforce_dollar_neutral: bool
    max_holding_days: int
    stop_loss: float
    min_training_months: int
    annualization_days: int
    commission_bps: float
    bid_ask_bps: float
    market_impact_bps: float
    slippage_bps: float
    tier2_enabled: bool = False
    corr_floor: float = 0.30
    density_floor: float = 0.40
    node_corr_floor: float = 0.20


@dataclass(frozen=True)
class Phase2SweepConfig:
    risk_budget_utilizations: list[float]
    corr_floors: list[float]
    density_floors: list[float]


@dataclass(frozen=True)
class PipelineConfig:
    price_source: str
    tickers: list[str]
    start_date: str
    end_date: str | None
    paths: PathsConfig
    database: DatabaseConfig
    validation: ValidationConfig
    schedule: ScheduleConfig
    phase2: Phase2Config
    phase2_sweep: Phase2SweepConfig


def load_config(config_path: str | Path) -> PipelineConfig:
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    project_root = _resolve_project_root(path)

    paths = PathsConfig(
        project_root=project_root,
        raw_dir=_resolve_path(project_root, data["paths"]["raw_dir"]),
        processed_dir=_resolve_path(project_root, data["paths"]["processed_dir"]),
        log_dir=_resolve_path(project_root, data["paths"]["log_dir"]),
        pipeline_log_file=_resolve_path(project_root, data["paths"]["pipeline_log_file"]),
        postgres_log_file=_resolve_path(project_root, data["paths"]["postgres_log_file"]),
        cache_dir=_resolve_path(project_root, data["paths"]["cache_dir"]),
        postgres_dir=_resolve_path(project_root, data["paths"]["postgres_dir"]),
    )
    database = DatabaseConfig(
        host=str(data["database"]["host"]),
        port=int(data["database"]["port"]),
        user=str(data["database"]["user"]),
        admin_database=str(data["database"]["admin_database"]),
        database_name=str(data["database"]["database_name"]),
        require_timescaledb=bool(data["database"]["require_timescaledb"]),
    )
    validation = ValidationConfig(
        max_abs_daily_return=float(data["validation"]["max_abs_daily_return"]),
        max_missing_business_days=int(data["validation"]["max_missing_business_days"]),
        adj_close_close_tolerance=float(data["validation"]["adj_close_close_tolerance"]),
    )
    schedule = ScheduleConfig(
        timezone=str(data["schedule"]["timezone"]),
        hour=int(data["schedule"]["hour"]),
        minute=int(data["schedule"]["minute"]),
    )
    phase2 = Phase2Config(
        lookback_window=int(data["phase2"]["lookback_window"]),
        diffusion_alpha=float(data["phase2"]["diffusion_alpha"]),
        diffusion_steps=int(data["phase2"]["diffusion_steps"]),
        sigma_scale=float(data["phase2"].get("sigma_scale", 1.0)),
        min_weight=float(data["phase2"]["min_weight"]),
        zscore_lookback=int(data["phase2"]["zscore_lookback"]),
        signal_threshold=float(data["phase2"]["signal_threshold"]),
        tier2_fraction=float(data["phase2"].get("tier2_fraction", 0.65)),
        tier2_size_fraction=float(data["phase2"].get("tier2_size_fraction", 0.5)),
        full_size_zscore=float(data["phase2"]["full_size_zscore"]),
        max_position_size=float(data["phase2"]["max_position_size"]),
        risk_budget_utilization=float(data["phase2"].get("risk_budget_utilization", 0.5)),
        max_drawdown_limit=float(data["phase2"].get("max_drawdown_limit", 0.20)),
        enforce_dollar_neutral=bool(data["phase2"].get("enforce_dollar_neutral", False)),
        max_holding_days=int(data["phase2"]["max_holding_days"]),
        stop_loss=float(data["phase2"]["stop_loss"]),
        min_training_months=int(data["phase2"]["min_training_months"]),
        annualization_days=int(data["phase2"]["annualization_days"]),
        commission_bps=float(data["phase2"]["commission_bps"]),
        bid_ask_bps=float(data["phase2"]["bid_ask_bps"]),
        market_impact_bps=float(data["phase2"]["market_impact_bps"]),
        slippage_bps=float(data["phase2"]["slippage_bps"]),
        tier2_enabled=bool(data["phase2"].get("tier2_enabled", False)),
        corr_floor=float(data["phase2"].get("corr_floor", 0.30)),
        density_floor=float(data["phase2"].get("density_floor", 0.40)),
        node_corr_floor=float(data["phase2"].get("node_corr_floor", 0.20)),
    )
    phase2_sweep = _load_phase2_sweep_config(data.get("phase2_sweep", {}), phase2)

    return PipelineConfig(
        price_source=str(data["price_source"]),
        tickers=list(data["tickers"]),
        start_date=data["start_date"],
        end_date=data.get("end_date"),
        paths=paths,
        database=database,
        validation=validation,
        schedule=schedule,
        phase2=phase2,
        phase2_sweep=phase2_sweep,
    )


def config_to_dict(config: PipelineConfig) -> dict[str, object]:
    payload = asdict(config)
    return _serialize_paths(payload)


def _resolve_project_root(config_path: Path) -> Path:
    if config_path.parent.name == "config":
        return config_path.parent.parent.resolve()
    return config_path.parent.resolve()


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _serialize_paths(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_paths(item) for item in value]
    return value


def _load_phase2_sweep_config(data: dict[str, object], phase2: Phase2Config) -> Phase2SweepConfig:
    return Phase2SweepConfig(
        risk_budget_utilizations=[
            float(value)
            for value in data.get("risk_budget_utilizations", [phase2.risk_budget_utilization])
        ],
        corr_floors=[float(value) for value in data.get("corr_floors", [phase2.corr_floor])],
        density_floors=[float(value) for value in data.get("density_floors", [phase2.density_floor])],
    )
