from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

from src.broker_alpaca import AlpacaBrokerClient


def _write_credentials(path: Path) -> None:
    payload = {
        "alpaca": {
            "api_key": "test-key",
            "api_secret": "test-secret",
            "base_url": "https://paper-api.alpaca.markets",
        }
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _build_config(credentials_path: Path):
    phase7 = SimpleNamespace(
        credentials_path=credentials_path,
        alpaca_base_url="https://paper-api.alpaca.markets",
        mode="paper",
    )
    return SimpleNamespace(phase7=phase7)


@patch("src.broker_alpaca.tradeapi.REST")
def test_alpaca_client_initializes_with_credentials(mock_rest: MagicMock, tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    _write_credentials(credentials_path)

    AlpacaBrokerClient(_build_config(credentials_path))

    mock_rest.assert_called_once_with(
        key_id="test-key",
        secret_key="test-secret",
        base_url="https://paper-api.alpaca.markets",
    )


@patch("src.broker_alpaca.tradeapi.REST")
def test_submit_order_uses_expected_alpaca_arguments(mock_rest: MagicMock, tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    _write_credentials(credentials_path)
    order = SimpleNamespace(id="order-123")
    mock_client = MagicMock()
    mock_client.submit_order.return_value = order
    mock_rest.return_value = mock_client

    client = AlpacaBrokerClient(_build_config(credentials_path))
    order_id = client.submit_order("SPY", "buy", 5, order_type="market")

    assert order_id == "order-123"
    mock_client.submit_order.assert_called_once_with(
        symbol="SPY",
        qty=5,
        side="buy",
        type="market",
        time_in_force="day",
    )


@patch("src.broker_alpaca.tradeapi.REST")
def test_get_positions_parses_alpaca_position_payloads(mock_rest: MagicMock, tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    _write_credentials(credentials_path)
    mock_client = MagicMock()
    mock_client.list_positions.return_value = [
        SimpleNamespace(symbol="GLD", qty="10", market_value="4320.48", current_price="432.048"),
    ]
    mock_rest.return_value = mock_client

    client = AlpacaBrokerClient(_build_config(credentials_path))
    positions = client.get_positions()

    assert positions == {
        "GLD": {
            "quantity": 10.0,
            "price": 432.048,
            "market_value": 4320.48,
        }
    }


@patch("src.broker_alpaca.tradeapi.REST")
def test_close_all_positions_uses_alpaca_bulk_close(mock_rest: MagicMock, tmp_path: Path) -> None:
    credentials_path = tmp_path / "credentials.yaml"
    _write_credentials(credentials_path)
    mock_client = MagicMock()
    mock_client.close_all_positions.return_value = [
        SimpleNamespace(id="close-1", status="accepted", symbol="SPY"),
    ]
    mock_rest.return_value = mock_client

    client = AlpacaBrokerClient(_build_config(credentials_path))
    closed = client.close_all_positions()

    mock_client.close_all_positions.assert_called_once_with(cancel_orders=True)
    assert closed == [{"order_id": "close-1", "status": "accepted", "symbol": "SPY"}]
