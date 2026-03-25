from pathlib import Path

import pytest
import yaml

from src.config_loader import load_config


def test_load_config_reads_yaml_values() -> None:
    config = load_config(Path("config/phase1.yaml"))

    assert config.price_source == "yfinance"
    assert len(config.tickers) == 8
    assert config.database.database_name == "investment_alpha_engine"
    assert config.paths.raw_dir.name == "raw"
    assert config.schedule.hour == 16
    assert config.phase2.lookback_window == 60
    assert config.phase2.sigma_scale == 1.0
    assert config.phase2.zscore_lookback == 75
    assert config.phase2.signal_threshold == 2.8
    assert config.phase2.min_weight == 0.1
    assert config.phase2.tier2_enabled is False
    assert config.phase2.tier2_fraction == 0.75
    assert config.phase2.tier2_size_fraction == 0.5
    assert config.phase2.risk_budget_utilization == 0.3
    assert config.phase2.max_drawdown_limit == 0.20
    assert config.phase2.enforce_dollar_neutral is False
    assert config.phase2.corr_floor == 0.30
    assert config.phase2.density_floor == 0.30
    assert config.phase2.node_corr_floor == 0.20
    assert config.phase2_sweep.risk_budget_utilizations == [0.3, 0.5, 0.7]
    assert config.phase2_sweep.corr_floors == [0.25, 0.30, 0.35]
    assert config.phase2_sweep.density_floors == [0.30, 0.40, 0.50]
    assert config.phase3.rolling_window == 60
    assert config.phase3.wasserstein_lookback == 20
    assert config.phase3.transition_threshold_sigma == 1.5
    assert config.phase3.new_regime_threshold_sigma == 2.5


def test_load_phase3_config_reads_phase3_section() -> None:
    config = load_config(Path("config/phase3.yaml"))

    assert config.phase2.max_holding_days == 9
    assert config.phase3.transition_position_scale == 0.75
    assert config.phase3.transition_threshold_mult == 1.10
    assert config.phase3.new_regime_position_scale == 0.50
    assert config.phase3.new_regime_threshold_mult == 1.25
    assert config.phase3.confirmation_window == 5
    assert config.phase3.confirmation_required_transition == 3
    assert config.phase3.confirmation_required_new_regime == 4
    assert config.phase3.emergency_recalib_days == 5


def test_load_phase4_config_reads_phase4_section() -> None:
    config = load_config(Path("config/phase4.yaml"))

    assert config.phase2.max_holding_days == 9
    assert config.phase4.tcn_enabled is False
    assert config.phase4.hidden_channels == 32
    assert config.phase4.n_blocks == 3
    assert config.phase4.sequence_length == 20
    assert config.phase4.validation_fraction == 0.15


def test_load_phase5_config_reads_phase5_section() -> None:
    config = load_config(Path("config/phase5.yaml"))

    assert config.phase5.trend_tickers == ["SPY", "TLT", "GLD", "UUP"]
    assert config.phase5.trend_short_ma == 50
    assert config.phase5.trend_long_ma == 200
    assert config.phase5.min_allocation == 0.15
    assert config.phase5.max_allocation == 0.85
    assert config.learning.trade_journal_path == "data/trade_journal.db"


def test_load_phase6_config_reads_learning_section() -> None:
    config = load_config(Path("config/phase6.yaml"))

    assert config.learning.trade_journal_path == "data/trade_journal.db"
    assert config.learning.bayesian.evaluation_window == 60
    assert config.learning.bayesian.update_smoothing == 0.7
    assert config.learning.bayesian.grid_resolution == 5
    assert config.learning.kill_switch.reduction_threshold == -0.5
    assert config.learning.kill_switch.quarantine_lookback_days == 120


def test_load_phase7_config_reads_risk_limits() -> None:
    config = load_config(Path("config/phase7.yaml"))

    assert config.phase7.mode == "mock"
    assert config.phase7.credentials_path.name == "credentials.yaml"
    assert config.phase7.fill_timeout_seconds == 60
    assert config.phase7.risk_limits.max_daily_loss_pct == 0.02
    assert config.phase7.risk_limits.max_gross_exposure_pct == 2.0
    assert config.phase7.risk_limits.max_spy_correlation_20d == 0.30


def test_load_phase8_config_reads_deployment_controls() -> None:
    config = load_config(Path("config/phase8.yaml"))

    assert config.deployment.mode == "paper"
    assert config.deployment.min_capital == 5000
    assert config.deployment.live_confirmation_path.name == "live_confirmed.txt"
    assert config.deployment.scaling_schedule.weeks_1_4 == 0.25
    assert config.deployment.scaling_schedule.weeks_25_plus == 1.0


def test_load_phase1_config_defaults_geo_when_missing() -> None:
    config = load_config(Path("config/phase1.yaml"))

    assert config.geo.enabled is False
    assert config.geo.optional_overlay is True
    assert config.geo.cutoff_time_et == "16:10"
    assert config.geo.state_path.is_absolute()
    assert config.geo.exposure_files.region.is_absolute()
    assert config.geo.exposure_files.region.name == "asset_region_exposure.csv"
    assert config.geo.half_life_days.sanctions == 20


def test_load_phase7_config_reads_geo_section_and_resolves_absolute_paths() -> None:
    config = load_config(Path("config/phase7.yaml"))

    assert config.geo.enabled is False
    assert config.geo.mapping_version == "geo_map_v1"
    assert config.geo.cutoff_time_et == "16:10"
    assert config.geo.state_path.is_absolute()
    assert config.geo.state_path.name == "geo_freeze_state.json"
    assert config.geo.exposure_files.region.is_absolute()
    assert config.geo.exposure_files.sector.is_absolute()
    assert config.geo.exposure_files.infra.is_absolute()
    assert config.geo.exposure_files.betas.is_absolute()


@pytest.mark.parametrize("invalid_cutoff", ["4:10", "16-10", "4:10 PM"])
def test_load_config_rejects_invalid_geo_cutoff_format(tmp_path: Path, invalid_cutoff: str) -> None:
    payload = yaml.safe_load(Path("config/phase1.yaml").read_text(encoding="utf-8"))
    payload["geo"] = {
        "enabled": False,
        "cutoff_time_et": invalid_cutoff,
    }
    config_path = tmp_path / "phase1_invalid_geo.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="cutoff_time_et"):
        load_config(config_path)


def test_load_config_accepts_valid_geo_cutoff_format(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("config/phase1.yaml").read_text(encoding="utf-8"))
    payload["geo"] = {
        "enabled": False,
        "cutoff_time_et": "04:10",
    }
    config_path = tmp_path / "phase1_valid_geo.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_config(config_path)

    assert config.geo.cutoff_time_et == "04:10"
