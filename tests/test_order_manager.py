from src.broker_mock import MockBrokerClient
from src.config_loader import load_config
from src.order_manager import OrderManager


def _make_manager() -> OrderManager:
    config = load_config("config/phase7.yaml")
    broker = MockBrokerClient(starting_equity=100_000.0, seed=7)
    return OrderManager(config, broker)


def test_generate_orders_creates_buy_and_sell_deltas() -> None:
    manager = _make_manager()
    orders = manager.generate_orders(
        target_positions={
            "XLK": {"quantity": 15, "price": 100.0},
            "XLE": {"quantity": -5, "price": 80.0},
        },
        current_positions={
            "XLK": {"quantity": 10, "price": 100.0},
            "XLE": {"quantity": 0, "price": 80.0},
        },
    )

    assert len(orders) == 2
    assert orders[0].asset == "XLE"
    assert orders[0].side == "sell"
    assert orders[0].quantity == 5.0
    assert orders[1].asset == "XLK"
    assert orders[1].side == "buy"
    assert orders[1].quantity == 5.0


def test_submit_orders_and_check_fills_capture_slippage() -> None:
    manager = _make_manager()
    orders = manager.generate_orders(
        target_positions={"XLK": {"quantity": 10, "price": 100.0}},
        current_positions={},
    )

    submissions = manager.submit_orders(orders)
    fills = manager.check_fills([submissions[0]["order_id"]])

    assert len(submissions) == 1
    assert submissions[0]["status"] == "submitted"
    assert len(fills) == 1
    assert fills[0]["status"] == "filled"
    assert fills[0]["fill_quantity"] == 10.0
    assert 0.0 <= fills[0]["slippage_bps"] <= 3.0


def test_reconcile_flags_broker_position_mismatches() -> None:
    manager = _make_manager()
    discrepancies = manager.reconcile(
        expected_positions={"XLK": {"quantity": 10}},
        broker_positions={"XLK": {"quantity": 9.5}},
    )

    assert len(discrepancies) == 1
    assert discrepancies[0]["asset"] == "XLK"
    assert discrepancies[0]["difference"] == -0.5
