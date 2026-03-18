import pandas as pd

from src.config_loader import load_config
from src.risk_manager import RiskManager


def _make_manager() -> RiskManager:
    return RiskManager(load_config("config/phase7.yaml"))


def test_pre_trade_allows_order_exactly_at_max_order_limit() -> None:
    manager = _make_manager()
    approved, rejected, reasons = manager.check_pre_trade(
        proposed_orders=[{"asset": "XLK", "side": "buy", "quantity": 50, "price": 100.0}],
        current_positions={},
        portfolio_value=100_000.0,
        daily_pnl=0.0,
        weekly_pnl=0.0,
        monthly_pnl=0.0,
    )

    assert len(approved) == 1
    assert rejected == []
    assert reasons == {}


def test_pre_trade_rejects_orders_with_clear_reasons_for_each_limit() -> None:
    manager = _make_manager()
    approved, rejected, reasons = manager.check_pre_trade(
        proposed_orders=[
            {"asset": "XLK", "side": "buy", "quantity": 60, "price": 100.0},
            {"asset": "XLV", "side": "buy", "notional": 3_000.0},
            {"asset": "SPY", "side": "buy", "notional": 10_000.0},
            {"asset": "TLT", "side": "sell", "notional": 10_000.0},
        ],
        current_positions={
            "XLV": {"market_value": 18_000.0},
            "SPY": {"market_value": 195_000.0},
            "TLT": {"market_value": 45_000.0},
        },
        portfolio_value=100_000.0,
        daily_pnl=0.0,
        weekly_pnl=0.0,
        monthly_pnl=0.0,
    )

    assert approved == []
    assert len(rejected) == 4
    assert "Order exceeds max_order_pct" in reasons["XLK:buy:60"][0]
    assert any("max_single_position_pct" in reason for reason in reasons["XLV:buy:None"])
    assert any("max_gross_exposure_pct" in reason for reason in reasons["SPY:buy:None"])
    assert any("max_net_exposure_pct" in reason for reason in reasons["TLT:sell:None"])


def test_circuit_breakers_allow_exact_limit_and_halt_on_breach() -> None:
    manager = _make_manager()

    action, _ = manager.check_circuit_breakers(
        daily_pnl=-2_000.0,
        weekly_pnl=-5_000.0,
        monthly_pnl=-10_000.0,
        portfolio_value=100_000.0,
    )
    assert action == "CONTINUE"

    assert manager.check_circuit_breakers(-2_001.0, 0.0, 0.0, 100_000.0)[0] == "HALT_TODAY"
    assert manager.check_circuit_breakers(0.0, -5_001.0, 0.0, 100_000.0)[0] == "HALT_WEEK"
    assert manager.check_circuit_breakers(0.0, 0.0, -10_001.0, 100_000.0)[0] == "HALT_INDEFINITE"


def test_pre_trade_rejects_all_orders_when_circuit_breaker_tripped() -> None:
    manager = _make_manager()
    approved, rejected, reasons = manager.check_pre_trade(
        proposed_orders=[{"asset": "XLK", "side": "buy", "notional": 1_000.0}],
        current_positions={},
        portfolio_value=100_000.0,
        daily_pnl=-2_500.0,
        weekly_pnl=0.0,
        monthly_pnl=0.0,
    )

    assert approved == []
    assert len(rejected) == 1
    assert reasons["XLK:buy:None"] == ["Daily loss limit breached."]


def test_post_trade_flags_position_discrepancies() -> None:
    manager = _make_manager()
    manager.check_pre_trade(
        proposed_orders=[{"asset": "XLK", "side": "buy", "notional": 5_000.0}],
        current_positions={"XLK": {"market_value": 10_000.0}},
        portfolio_value=100_000.0,
        daily_pnl=0.0,
        weekly_pnl=0.0,
        monthly_pnl=0.0,
    )

    discrepancies = manager.check_post_trade(
        current_positions={"XLK": {"market_value": 14_500.0}},
        portfolio_value=100_000.0,
    )

    assert len(discrepancies) == 1
    assert discrepancies[0]["asset"] == "XLK"
    assert discrepancies[0]["difference"] == -500.0


def test_spy_correlation_warning_and_reduction_trigger() -> None:
    manager = _make_manager()
    dates = pd.bdate_range("2024-01-01", periods=30)
    portfolio_returns = pd.Series([0.001 * (idx + 1) for idx in range(30)], index=dates)
    spy_returns = portfolio_returns * 1.02

    last_result = None
    for date in dates[-5:]:
        last_result = manager.check_spy_correlation(portfolio_returns, spy_returns, date)

    assert last_result is not None
    assert last_result.warning is True
    assert last_result.reduce_positions is True
    assert last_result.reduction_scale == 0.75
    assert last_result.consecutive_high_days == 5
