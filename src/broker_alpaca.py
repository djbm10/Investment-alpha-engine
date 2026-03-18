from __future__ import annotations

from pathlib import Path
from typing import Any

import alpaca_trade_api as tradeapi
import yaml


class AlpacaBrokerClient:
    def __init__(self, config: Any) -> None:
        phase7 = config.phase7
        credentials = _load_credentials(Path(phase7.credentials_path))
        alpaca_config = credentials.get("alpaca", {})
        api_key = str(alpaca_config.get("api_key", "")).strip()
        api_secret = str(alpaca_config.get("api_secret", "")).strip()
        base_url = str(alpaca_config.get("base_url", phase7.alpaca_base_url)).strip()
        if not api_key or not api_secret:
            raise ValueError("Alpaca credentials are missing api_key or api_secret.")

        self.mode = phase7.mode
        self.base_url = base_url
        self.client = tradeapi.REST(key_id=api_key, secret_key=api_secret, base_url=base_url)

    def submit_order(
        self,
        asset: str,
        side: str,
        qty: float,
        order_type: str = "market",
        expected_price: float | None = None,
    ) -> str:
        del expected_price
        order = self.client.submit_order(
            symbol=asset,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force="day",
        )
        return str(order.id)

    def get_order_status(self, order_id: str) -> dict[str, float | str]:
        order = self.client.get_order(order_id)
        return {
            "status": str(order.status),
            "fill_price": float(getattr(order, "filled_avg_price", 0.0) or 0.0),
            "fill_qty": float(getattr(order, "filled_qty", 0.0) or 0.0),
        }

    def get_positions(self) -> dict[str, dict[str, float]]:
        positions: dict[str, dict[str, float]] = {}
        for position in self.client.list_positions():
            symbol = str(position.symbol)
            qty = float(position.qty)
            market_value = float(position.market_value)
            current_price = float(getattr(position, "current_price", 0.0) or getattr(position, "avg_entry_price", 0.0) or 0.0)
            positions[symbol] = {
                "quantity": qty,
                "price": current_price,
                "market_value": market_value,
            }
        return positions

    def get_portfolio_value(self) -> float:
        account = self.get_account()
        return float(account.equity)

    def get_account(self) -> Any:
        return self.client.get_account()

    def close_all_positions(self) -> list[dict[str, Any]]:
        responses = self.client.close_all_positions(cancel_orders=True)
        closed: list[dict[str, Any]] = []
        for response in responses or []:
            closed.append(
                {
                    "order_id": str(getattr(response, "id", "")),
                    "status": str(getattr(response, "status", "submitted")),
                    "symbol": str(getattr(response, "symbol", "")),
                }
            )
        return closed


def _load_credentials(credentials_path: Path) -> dict[str, Any]:
    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Alpaca credentials file was not found at '{credentials_path}'. "
            "Copy config/credentials.yaml.template to config/credentials.yaml and fill in your paper-trading keys."
        )
    return yaml.safe_load(credentials_path.read_text(encoding="utf-8")) or {}
