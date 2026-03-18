from src.capital_scaler import CapitalScaler
from src.config_loader import load_config


def test_capital_scaler_transitions_through_schedule_buckets() -> None:
    config = load_config("config/phase8.yaml")
    scaler = CapitalScaler(config, "2026-01-05")

    assert scaler.get_scale_factor("2026-01-05") == 0.25
    assert scaler.get_scale_factor("2026-01-28") == 0.25
    assert scaler.get_scale_factor("2026-02-02") == 0.50
    assert scaler.get_scale_factor("2026-04-06") == 0.75
    assert scaler.get_scale_factor("2026-06-22") == 1.00


def test_capital_scaler_applies_scale_factor_to_target_positions() -> None:
    config = load_config("config/phase8.yaml")
    scaler = CapitalScaler(config, "2026-01-05")

    scaled = scaler.apply_scaling(
        {
            "XLK": {"quantity": 20.0, "price": 100.0, "market_value": 2_000.0},
            "SPY": {"quantity": 10.0, "price": 500.0, "market_value": 5_000.0},
        },
        portfolio_value=100_000.0,
        current_date="2026-01-06",
    )

    assert scaled["XLK"]["quantity"] == 5.0
    assert scaled["XLK"]["market_value"] == 500.0
    assert scaled["SPY"]["quantity"] == 2.5
    assert scaled["SPY"]["market_value"] == 1_250.0
