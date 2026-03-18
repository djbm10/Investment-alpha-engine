from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .config_loader import PipelineConfig


@dataclass(frozen=True)
class Order:
    asset: str
    side: str
    quantity: float
    order_type: str
    expected_price: float
    notional: float


class OrderManager:
    def __init__(self, config: PipelineConfig, broker_client: Any) -> None:
        self.config = config
        self.broker_client = broker_client
        self.mode = config.phase7.mode
        self._submitted_orders: dict[str, dict[str, Any]] = {}

    def generate_orders(
        self,
        target_positions: dict[str, Any],
        current_positions: dict[str, Any],
    ) -> list[Order]:
        orders: list[Order] = []
        assets = sorted(set(target_positions) | set(current_positions))
        for asset in assets:
            target_quantity = self._position_quantity(target_positions.get(asset, {}))
            current_quantity = self._position_quantity(current_positions.get(asset, {}))
            delta_quantity = target_quantity - current_quantity
            if abs(delta_quantity) <= 1e-9:
                continue

            reference_payload = target_positions.get(asset) or current_positions.get(asset) or {}
            expected_price = self._position_price(reference_payload)
            side = "buy" if delta_quantity > 0 else "sell"
            quantity = abs(delta_quantity)
            orders.append(
                Order(
                    asset=str(asset),
                    side=side,
                    quantity=float(quantity),
                    order_type="market",
                    expected_price=float(expected_price),
                    notional=float(quantity * expected_price),
                )
            )
        return orders

    def submit_orders(self, orders: list[Order]) -> list[dict[str, Any]]:
        submissions: list[dict[str, Any]] = []
        for order in orders:
            order_id = self.broker_client.submit_order(
                asset=order.asset,
                side=order.side,
                qty=order.quantity,
                order_type=order.order_type,
                expected_price=order.expected_price,
            )
            submitted_at = datetime.now(timezone.utc).isoformat()
            record = {
                "order_id": order_id,
                "status": "submitted",
                "submitted_at": submitted_at,
                **asdict(order),
            }
            self._submitted_orders[order_id] = record
            submissions.append(record)
        return submissions

    def check_fills(self, order_ids: list[str], timeout_seconds: int = 60) -> list[dict[str, Any]]:
        del timeout_seconds
        fills: list[dict[str, Any]] = []
        for order_id in order_ids:
            if order_id not in self._submitted_orders:
                raise KeyError(f"Order '{order_id}' was not submitted through this manager.")
            order_record = self._submitted_orders[order_id]
            status = self.broker_client.get_order_status(order_id)
            expected_price = float(order_record["expected_price"])
            fill_price = float(status["fill_price"])
            slippage_bps = 0.0 if expected_price == 0.0 else ((fill_price / expected_price) - 1.0) * 10_000.0
            if str(order_record["side"]).lower() == "sell":
                slippage_bps *= -1.0
            fills.append(
                {
                    "order_id": order_id,
                    "status": str(status["status"]),
                    "fill_price": fill_price,
                    "fill_quantity": float(status["fill_qty"]),
                    "fill_time": datetime.now(timezone.utc).isoformat(),
                    "expected_price": expected_price,
                    "slippage_bps": float(slippage_bps),
                    "asset": str(order_record["asset"]),
                    "side": str(order_record["side"]),
                }
            )
        return fills

    def reconcile(
        self,
        expected_positions: dict[str, Any],
        broker_positions: dict[str, Any],
    ) -> list[dict[str, Any]]:
        discrepancies: list[dict[str, Any]] = []
        assets = sorted(set(expected_positions) | set(broker_positions))
        for asset in assets:
            expected_quantity = self._position_quantity(expected_positions.get(asset, {}))
            broker_quantity = self._position_quantity(broker_positions.get(asset, {}))
            difference = broker_quantity - expected_quantity
            if abs(difference) > 1e-9:
                discrepancies.append(
                    {
                        "asset": str(asset),
                        "expected_quantity": float(expected_quantity),
                        "broker_quantity": float(broker_quantity),
                        "difference": float(difference),
                    }
                )
        return discrepancies

    def _position_quantity(self, payload: Any) -> float:
        if isinstance(payload, dict):
            if "quantity" in payload:
                return float(payload.get("quantity", 0.0))
            market_value = float(payload.get("market_value", 0.0))
            price = self._position_price(payload)
            return 0.0 if price == 0.0 else float(market_value / price)
        if payload is None:
            return 0.0
        return float(payload)

    def _position_price(self, payload: Any) -> float:
        if not isinstance(payload, dict):
            return 0.0
        for key in ("price", "expected_price", "last_price", "avg_entry_price"):
            if key in payload and payload[key] is not None:
                return float(payload[key])
        return 0.0
