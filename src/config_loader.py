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
class Phase3Config:
    rolling_window: int
    wasserstein_lookback: int
    transition_threshold_sigma: float
    new_regime_threshold_sigma: float
    confirmation_window: int
    confirmation_required_transition: int
    confirmation_required_new_regime: int
    transition_position_scale: float
    transition_threshold_mult: float
    new_regime_position_scale: float
    new_regime_threshold_mult: float
    transition_lookback_cap: int
    new_regime_freeze_days: int
    emergency_recalib_days: int
    emergency_lookback: int


@dataclass(frozen=True)
class Phase4Config:
    tcn_enabled: bool
    hidden_channels: int
    n_blocks: int
    dropout: float
    n_ensemble: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    batch_size: int
    sequence_length: int
    validation_fraction: float
    reversion_confirm_threshold: float


@dataclass(frozen=True)
class Phase5Config:
    trend_tickers: list[str]
    trend_short_ma: int
    trend_long_ma: int
    trend_vol_lookback: int
    trend_cost_bps: float
    cash_return_annual: float
    utility_lambda_uncertainty: float
    utility_lambda_cost: float
    softmax_temperature: float
    min_allocation: float
    max_allocation: float
    rebalance_frequency_days: int
    performance_lookback: int
    daily_loss_limit: float


@dataclass(frozen=True)
class BayesianLearningConfig:
    evaluation_window: int
    update_smoothing: float
    grid_resolution: int
    sharpe_scaling: float


@dataclass(frozen=True)
class KillSwitchConfig:
    reduction_threshold: float
    quarantine_threshold: float
    reactivation_threshold: float
    reactivation_days: int
    reduction_lookback_days: int
    quarantine_lookback_days: int
    reactivation_lookback_days: int


@dataclass(frozen=True)
class LearningConfig:
    trade_journal_path: str
    bayesian: BayesianLearningConfig
    kill_switch: KillSwitchConfig


@dataclass(frozen=True)
class Phase7RiskLimitsConfig:
    max_daily_loss_pct: float
    max_weekly_loss_pct: float
    max_monthly_loss_pct: float
    max_single_position_pct: float
    max_gross_exposure_pct: float
    max_net_exposure_pct: float
    max_order_pct: float
    max_spy_correlation_20d: float


@dataclass(frozen=True)
class Phase7Config:
    mode: str
    credentials_path: Path
    alpaca_base_url: str
    fill_timeout_seconds: int
    mock_slippage_bps_min: float
    mock_slippage_bps_max: float
    risk_limits: Phase7RiskLimitsConfig


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
    phase3: Phase3Config
    phase4: Phase4Config
    phase5: Phase5Config
    learning: LearningConfig
    phase7: Phase7Config


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
    phase3 = _load_phase3_config(data.get("phase3", {}), phase2)
    phase4 = _load_phase4_config(data.get("phase4", {}))
    phase5 = _load_phase5_config(data.get("phase5", {}))
    learning = _load_learning_config(data.get("learning", {}))
    phase7 = _load_phase7_config(project_root, data.get("phase7", {}))

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
        phase3=phase3,
        phase4=phase4,
        phase5=phase5,
        learning=learning,
        phase7=phase7,
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


def _load_phase3_config(data: dict[str, object], phase2: Phase2Config) -> Phase3Config:
    return Phase3Config(
        rolling_window=int(data.get("rolling_window", phase2.lookback_window)),
        wasserstein_lookback=int(data.get("wasserstein_lookback", 20)),
        transition_threshold_sigma=float(data.get("transition_threshold_sigma", 1.5)),
        new_regime_threshold_sigma=float(data.get("new_regime_threshold_sigma", 2.5)),
        confirmation_window=int(data.get("confirmation_window", 5)),
        confirmation_required_transition=int(data.get("confirmation_required_transition", 3)),
        confirmation_required_new_regime=int(data.get("confirmation_required_new_regime", 4)),
        transition_position_scale=float(data.get("transition_position_scale", 0.5)),
        transition_threshold_mult=float(data.get("transition_threshold_mult", 1.25)),
        new_regime_position_scale=float(data.get("new_regime_position_scale", 0.5)),
        new_regime_threshold_mult=float(data.get("new_regime_threshold_mult", 1.25)),
        transition_lookback_cap=int(data.get("transition_lookback_cap", 30)),
        new_regime_freeze_days=int(data.get("new_regime_freeze_days", 0)),
        emergency_recalib_days=int(data.get("emergency_recalib_days", 5)),
        emergency_lookback=int(data.get("emergency_lookback", 20)),
    )


def _load_phase4_config(data: dict[str, object]) -> Phase4Config:
    return Phase4Config(
        tcn_enabled=bool(data.get("tcn_enabled", True)),
        hidden_channels=int(data.get("hidden_channels", 32)),
        n_blocks=int(data.get("n_blocks", 3)),
        dropout=float(data.get("dropout", 0.2)),
        n_ensemble=int(data.get("n_ensemble", 3)),
        learning_rate=float(data.get("learning_rate", 0.001)),
        weight_decay=float(data.get("weight_decay", 0.0001)),
        max_epochs=int(data.get("max_epochs", 100)),
        patience=int(data.get("patience", 10)),
        batch_size=int(data.get("batch_size", 32)),
        sequence_length=int(data.get("sequence_length", 20)),
        validation_fraction=float(data.get("validation_fraction", 0.15)),
        reversion_confirm_threshold=float(data.get("reversion_confirm_threshold", 0.5)),
    )


def _load_phase5_config(data: dict[str, object]) -> Phase5Config:
    default_trend_tickers = ["SPY", "TLT", "GLD", "UUP"]
    return Phase5Config(
        trend_tickers=[str(value) for value in data.get("trend_tickers", default_trend_tickers)],
        trend_short_ma=int(data.get("trend_short_ma", 50)),
        trend_long_ma=int(data.get("trend_long_ma", 200)),
        trend_vol_lookback=int(data.get("trend_vol_lookback", 20)),
        trend_cost_bps=float(data.get("trend_cost_bps", 2.0)),
        cash_return_annual=float(data.get("cash_return_annual", 0.0)),
        utility_lambda_uncertainty=float(data.get("utility_lambda_uncertainty", 1.0)),
        utility_lambda_cost=float(data.get("utility_lambda_cost", 2.0)),
        softmax_temperature=float(data.get("softmax_temperature", 1.0)),
        min_allocation=float(data.get("min_allocation", 0.15)),
        max_allocation=float(data.get("max_allocation", 0.85)),
        rebalance_frequency_days=int(data.get("rebalance_frequency_days", 5)),
        performance_lookback=int(data.get("performance_lookback", 20)),
        daily_loss_limit=float(data.get("daily_loss_limit", 0.02)),
    )


def _load_learning_config(data: dict[str, object]) -> LearningConfig:
    bayesian_data = data.get("bayesian", {})
    kill_switch_data = data.get("kill_switch", {})
    return LearningConfig(
        trade_journal_path=str(data.get("trade_journal_path", "data/trade_journal.db")),
        bayesian=BayesianLearningConfig(
            evaluation_window=int(bayesian_data.get("evaluation_window", 60)),
            update_smoothing=float(bayesian_data.get("update_smoothing", 0.7)),
            grid_resolution=int(bayesian_data.get("grid_resolution", 5)),
            sharpe_scaling=float(bayesian_data.get("sharpe_scaling", 2.0)),
        ),
        kill_switch=KillSwitchConfig(
            reduction_threshold=float(kill_switch_data.get("reduction_threshold", -0.5)),
            quarantine_threshold=float(kill_switch_data.get("quarantine_threshold", -0.5)),
            reactivation_threshold=float(kill_switch_data.get("reactivation_threshold", 0.0)),
            reactivation_days=int(kill_switch_data.get("reactivation_days", 40)),
            reduction_lookback_days=int(kill_switch_data.get("reduction_lookback_days", 60)),
            quarantine_lookback_days=int(kill_switch_data.get("quarantine_lookback_days", 120)),
            reactivation_lookback_days=int(kill_switch_data.get("reactivation_lookback_days", 40)),
        ),
    )


def _load_phase7_config(project_root: Path, data: dict[str, object]) -> Phase7Config:
    default_limits = {
        "max_daily_loss_pct": 0.02,
        "max_weekly_loss_pct": 0.05,
        "max_monthly_loss_pct": 0.10,
        "max_single_position_pct": 0.20,
        "max_gross_exposure_pct": 2.00,
        "max_net_exposure_pct": 0.50,
        "max_order_pct": 0.05,
        "max_spy_correlation_20d": 0.30,
    }
    risk_limit_data = data.get("risk_limits", {})
    risk_limits = Phase7RiskLimitsConfig(
        max_daily_loss_pct=_conservative_cap(
            float(risk_limit_data.get("max_daily_loss_pct", default_limits["max_daily_loss_pct"])),
            default_limits["max_daily_loss_pct"],
        ),
        max_weekly_loss_pct=_conservative_cap(
            float(risk_limit_data.get("max_weekly_loss_pct", default_limits["max_weekly_loss_pct"])),
            default_limits["max_weekly_loss_pct"],
        ),
        max_monthly_loss_pct=_conservative_cap(
            float(risk_limit_data.get("max_monthly_loss_pct", default_limits["max_monthly_loss_pct"])),
            default_limits["max_monthly_loss_pct"],
        ),
        max_single_position_pct=_conservative_cap(
            float(risk_limit_data.get("max_single_position_pct", default_limits["max_single_position_pct"])),
            default_limits["max_single_position_pct"],
        ),
        max_gross_exposure_pct=_conservative_cap(
            float(risk_limit_data.get("max_gross_exposure_pct", default_limits["max_gross_exposure_pct"])),
            default_limits["max_gross_exposure_pct"],
        ),
        max_net_exposure_pct=_conservative_cap(
            float(risk_limit_data.get("max_net_exposure_pct", default_limits["max_net_exposure_pct"])),
            default_limits["max_net_exposure_pct"],
        ),
        max_order_pct=_conservative_cap(
            float(risk_limit_data.get("max_order_pct", default_limits["max_order_pct"])),
            default_limits["max_order_pct"],
        ),
        max_spy_correlation_20d=_conservative_cap(
            float(risk_limit_data.get("max_spy_correlation_20d", default_limits["max_spy_correlation_20d"])),
            default_limits["max_spy_correlation_20d"],
        ),
    )
    return Phase7Config(
        mode=str(data.get("mode", "mock")),
        credentials_path=_resolve_path(project_root, str(data.get("credentials_path", "config/credentials.yaml"))),
        alpaca_base_url=str(data.get("alpaca_base_url", "https://paper-api.alpaca.markets")),
        fill_timeout_seconds=int(data.get("fill_timeout_seconds", 60)),
        mock_slippage_bps_min=float(data.get("mock_slippage_bps_min", 0.0)),
        mock_slippage_bps_max=float(data.get("mock_slippage_bps_max", 3.0)),
        risk_limits=risk_limits,
    )


def _conservative_cap(value: float, hard_limit: float) -> float:
    return float(min(value, hard_limit))
