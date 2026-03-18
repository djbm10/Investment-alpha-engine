from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config_loader import Phase7Config, Phase7RiskLimitsConfig, PipelineConfig


@dataclass(frozen=True)
class CorrelationCheckResult:
    correlation: float
    warning: bool
    reduce_positions: bool
    reduction_scale: float
    consecutive_high_days: int


class RiskManager:
    CORRELATION_REDUCTION_THRESHOLD = 0.40
    CORRELATION_REDUCTION_DAYS = 5
    CORRELATION_REDUCTION_SCALE = 0.75

    def __init__(self, config: PipelineConfig | Phase7Config | Phase7RiskLimitsConfig) -> None:
        if isinstance(config, PipelineConfig):
            self.config = config.phase7
        elif isinstance(config, Phase7Config):
            self.config = config
        else:
            self.config = Phase7Config(
                mode="mock",
                credentials_path=Path("config/credentials.yaml"),
                alpaca_base_url="https://paper-api.alpaca.markets",
                fill_timeout_seconds=60,
                mock_slippage_bps_min=0.0,
                mock_slippage_bps_max=3.0,
                risk_limits=config,
            )
        self.risk_limits = self.config.risk_limits
        self._expected_positions: dict[str, float] | None = None
        self._expected_portfolio_value: float | None = None
        self._high_correlation_streak = 0

    def check_pre_trade(
        self,
        proposed_orders: list[dict[str, Any]],
        current_positions: dict[str, Any],
        portfolio_value: float,
        daily_pnl: float,
        weekly_pnl: float,
        monthly_pnl: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
        circuit_action, circuit_reason = self.check_circuit_breakers(
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            monthly_pnl=monthly_pnl,
            portfolio_value=portfolio_value,
        )
        if circuit_action != "CONTINUE":
            rejected = [order.copy() for order in proposed_orders]
            rejection_reasons = {
                self._order_ref(order): [circuit_reason]
                for order in proposed_orders
            }
            self._expected_positions = self._normalize_positions(current_positions)
            self._expected_portfolio_value = float(portfolio_value)
            return [], rejected, rejection_reasons

        working_positions = self._normalize_positions(current_positions)
        approved_orders: list[dict[str, Any]] = []
        rejected_orders: list[dict[str, Any]] = []
        rejection_reasons: dict[str, list[str]] = {}

        for order in proposed_orders:
            reasons = self._evaluate_order_limits(
                order=order,
                positions=working_positions,
                portfolio_value=portfolio_value,
            )
            if reasons:
                rejected_orders.append(order.copy())
                rejection_reasons[self._order_ref(order)] = reasons
                continue

            approved_orders.append(order.copy())
            asset = str(order["asset"])
            working_positions[asset] = working_positions.get(asset, 0.0) + self._signed_order_notional(order)

        self._expected_positions = dict(working_positions)
        self._expected_portfolio_value = float(portfolio_value)
        return approved_orders, rejected_orders, rejection_reasons

    def check_post_trade(
        self,
        current_positions: dict[str, Any],
        portfolio_value: float,
    ) -> list[dict[str, Any]]:
        if self._expected_positions is None:
            return []
        actual_positions = self._normalize_positions(current_positions)
        discrepancies: list[dict[str, Any]] = []
        all_assets = sorted(set(self._expected_positions) | set(actual_positions))
        for asset in all_assets:
            expected_value = float(self._expected_positions.get(asset, 0.0))
            actual_value = float(actual_positions.get(asset, 0.0))
            difference = actual_value - expected_value
            if abs(difference) > 1e-6:
                discrepancies.append(
                    {
                        "asset": asset,
                        "expected_market_value": expected_value,
                        "actual_market_value": actual_value,
                        "difference": difference,
                    }
                )
        if self._expected_portfolio_value is not None and portfolio_value <= 0:
            discrepancies.append(
                {
                    "asset": "__portfolio__",
                    "expected_market_value": float(self._expected_portfolio_value),
                    "actual_market_value": float(portfolio_value),
                    "difference": float(portfolio_value - self._expected_portfolio_value),
                }
            )
        return discrepancies

    def check_circuit_breakers(
        self,
        daily_pnl: float,
        weekly_pnl: float,
        monthly_pnl: float,
        portfolio_value: float,
    ) -> tuple[str, str]:
        if portfolio_value <= 0:
            return "HALT_INDEFINITE", "Portfolio value is non-positive."

        limits = self.risk_limits
        daily_loss_pct = max(0.0, -float(daily_pnl) / portfolio_value)
        weekly_loss_pct = max(0.0, -float(weekly_pnl) / portfolio_value)
        monthly_loss_pct = max(0.0, -float(monthly_pnl) / portfolio_value)

        if monthly_loss_pct > limits.max_monthly_loss_pct:
            return "HALT_INDEFINITE", "Monthly loss limit breached."
        if weekly_loss_pct > limits.max_weekly_loss_pct:
            return "HALT_WEEK", "Weekly loss limit breached."
        if daily_loss_pct > limits.max_daily_loss_pct:
            return "HALT_TODAY", "Daily loss limit breached."
        return "CONTINUE", "Within circuit breaker limits."

    def check_spy_correlation(
        self,
        portfolio_returns: pd.Series,
        spy_returns: pd.Series,
        current_date: str | pd.Timestamp,
    ) -> CorrelationCheckResult:
        current = pd.Timestamp(current_date)
        aligned = pd.concat(
            [
                portfolio_returns.sort_index().loc[:current],
                spy_returns.sort_index().loc[:current],
            ],
            axis=1,
            join="inner",
        ).tail(20)
        if len(aligned) < 20:
            self._high_correlation_streak = 0
            return CorrelationCheckResult(
                correlation=0.0,
                warning=False,
                reduce_positions=False,
                reduction_scale=1.0,
                consecutive_high_days=0,
            )

        correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        correlation_value = 0.0 if pd.isna(correlation) else float(correlation)
        warning = correlation_value > self.risk_limits.max_spy_correlation_20d
        if correlation_value > self.CORRELATION_REDUCTION_THRESHOLD:
            self._high_correlation_streak += 1
        else:
            self._high_correlation_streak = 0

        reduce_positions = self._high_correlation_streak >= self.CORRELATION_REDUCTION_DAYS
        return CorrelationCheckResult(
            correlation=correlation_value,
            warning=warning,
            reduce_positions=reduce_positions,
            reduction_scale=self.CORRELATION_REDUCTION_SCALE if reduce_positions else 1.0,
            consecutive_high_days=self._high_correlation_streak,
        )

    def _evaluate_order_limits(
        self,
        *,
        order: dict[str, Any],
        positions: dict[str, float],
        portfolio_value: float,
    ) -> list[str]:
        if portfolio_value <= 0:
            return ["Portfolio value is non-positive."]

        reasons: list[str] = []
        order_notional = abs(self._order_notional(order))
        order_pct = order_notional / portfolio_value
        if order_pct > self.risk_limits.max_order_pct:
            reasons.append(
                f"Order exceeds max_order_pct ({order_pct:.2%} > {self.risk_limits.max_order_pct:.2%})."
            )

        candidate_positions = dict(positions)
        asset = str(order["asset"])
        candidate_positions[asset] = candidate_positions.get(asset, 0.0) + self._signed_order_notional(order)

        position_pct = abs(candidate_positions[asset]) / portfolio_value
        if position_pct > self.risk_limits.max_single_position_pct:
            reasons.append(
                "Resulting position exceeds max_single_position_pct "
                f"({position_pct:.2%} > {self.risk_limits.max_single_position_pct:.2%})."
            )

        gross_exposure_pct = sum(abs(value) for value in candidate_positions.values()) / portfolio_value
        if gross_exposure_pct > self.risk_limits.max_gross_exposure_pct:
            reasons.append(
                "Resulting gross exposure exceeds max_gross_exposure_pct "
                f"({gross_exposure_pct:.2%} > {self.risk_limits.max_gross_exposure_pct:.2%})."
            )

        net_exposure_pct = abs(sum(candidate_positions.values())) / portfolio_value
        if net_exposure_pct > self.risk_limits.max_net_exposure_pct:
            reasons.append(
                "Resulting net exposure exceeds max_net_exposure_pct "
                f"({net_exposure_pct:.2%} > {self.risk_limits.max_net_exposure_pct:.2%})."
            )

        return reasons

    def _normalize_positions(self, current_positions: dict[str, Any]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for asset, payload in current_positions.items():
            if isinstance(payload, dict):
                market_value = payload.get("market_value")
                quantity = payload.get("quantity", 0.0)
                price = payload.get("price", payload.get("last_price", payload.get("avg_entry_price", 0.0)))
                signed_value = market_value
                if signed_value is None:
                    signed_value = float(quantity) * float(price)
                normalized[str(asset)] = float(signed_value)
            else:
                normalized[str(asset)] = float(payload)
        return normalized

    def _signed_order_notional(self, order: dict[str, Any]) -> float:
        notional = self._order_notional(order)
        side = str(order.get("side", "")).lower()
        if side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported order side '{side}'.")
        return notional if side == "buy" else -notional

    def _order_notional(self, order: dict[str, Any]) -> float:
        if "notional" in order and order["notional"] is not None:
            return float(order["notional"])
        quantity = float(order.get("quantity", 0.0))
        price = float(order.get("price", order.get("expected_price", 0.0)))
        return abs(quantity * price)

    def _order_ref(self, order: dict[str, Any]) -> str:
        return str(order.get("order_id", f"{order.get('asset')}:{order.get('side')}:{order.get('quantity')}"))
