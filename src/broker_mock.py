from __future__ import annotations

import random
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True)
class MockOrderStatus:
    order_id: str
    status: str
    asset: str
    side: str
    fill_price: float
    fill_qty: float


class MockBrokerClient:
    def __init__(
        self,
        *,
        starting_equity: float = 100_000.0,
        min_slippage_bps: float = 0.0,
        max_slippage_bps: float = 3.0,
        seed: int = 42,
    ) -> None:
        self._cash = float(starting_equity)
        self._min_slippage_bps = float(min_slippage_bps)
        self._max_slippage_bps = float(max_slippage_bps)
        self._rng = random.Random(seed)
        self._positions: dict[str, float] = {}
        self._market_prices: dict[str, float] = {}
        self._orders: dict[str, dict[str, float | str]] = {}

    def submit_order(
        self,
        asset: str,
        side: str,
        qty: float,
        order_type: str = "market",
        expected_price: float | None = None,
    ) -> str:
        if qty <= 0:
            raise ValueError("Order quantity must be positive.")
        side_normalized = side.lower()
        if side_normalized not in {"buy", "sell"}:
            raise ValueError(f"Unsupported order side '{side}'.")

        base_price = float(expected_price if expected_price is not None else self._market_prices.get(asset, 100.0))
        slippage_bps = self._rng.uniform(self._min_slippage_bps, self._max_slippage_bps)
        slippage_multiplier = 1.0 + (slippage_bps / 10_000.0) if side_normalized == "buy" else 1.0 - (slippage_bps / 10_000.0)
        fill_price = base_price * slippage_multiplier

        signed_qty = qty if side_normalized == "buy" else -qty
        self._positions[asset] = self._positions.get(asset, 0.0) + signed_qty
        self._market_prices[asset] = base_price
        self._cash -= signed_qty * fill_price

        order_id = str(uuid4())
        self._orders[order_id] = {
            "status": "filled",
            "asset": asset,
            "side": side_normalized,
            "fill_price": fill_price,
            "fill_qty": float(qty),
            "expected_price": base_price,
            "order_type": order_type,
        }
        return order_id

    def get_order_status(self, order_id: str) -> dict[str, float | str]:
        if order_id not in self._orders:
            raise KeyError(f"Unknown order id '{order_id}'.")
        return dict(self._orders[order_id])

    def get_positions(self) -> dict[str, dict[str, float]]:
        positions: dict[str, dict[str, float]] = {}
        for asset, quantity in self._positions.items():
            market_price = float(self._market_prices.get(asset, 0.0))
            positions[asset] = {
                "quantity": float(quantity),
                "price": market_price,
                "market_value": float(quantity * market_price),
            }
        return positions

    def get_portfolio_value(self) -> float:
        market_value = sum(
            float(quantity) * float(self._market_prices.get(asset, 0.0))
            for asset, quantity in self._positions.items()
        )
        return float(self._cash + market_value)

    def set_market_prices(self, price_map: dict[str, float]) -> None:
        for asset, price in price_map.items():
            self._market_prices[str(asset)] = float(price)
