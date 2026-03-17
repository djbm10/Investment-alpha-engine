from pathlib import Path

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
