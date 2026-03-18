from pathlib import Path

from src.trade_journal import TradeJournal


def _sample_trade(trade_id: str, strategy: str, exit_date: str, net_pnl: float) -> dict[str, object]:
    return {
        "trade_id": trade_id,
        "strategy": strategy,
        "asset": "XLK",
        "direction": "long",
        "entry_date": "2024-01-02",
        "exit_date": exit_date,
        "holding_days": 3,
        "entry_price": 100.0,
        "exit_price": 101.5,
        "entry_zscore": -2.7,
        "entry_node_corr": 0.32,
        "entry_regime": "TRADEABLE",
        "exit_reason": "reversion",
        "gross_pnl": net_pnl + 0.001,
        "transaction_cost": 0.001,
        "net_pnl": net_pnl,
        "predicted_residual": None,
        "actual_residual": -0.001,
        "prediction_error": None,
        "portfolio_exposure_at_entry": 0.75,
        "concurrent_positions": 3,
    }


def test_trade_journal_round_trip_and_filters(tmp_path: Path) -> None:
    journal = TradeJournal(tmp_path / "trade_journal.db")
    journal.log_trade(_sample_trade("t1", "A", "2024-01-05", 0.012))
    journal.log_trade(_sample_trade("t2", "B", "2024-02-10", -0.02))

    trades = journal.get_trades()
    assert len(trades) == 2
    assert set(trades["trade_id"]) == {"t1", "t2"}
    assert float(trades.loc[trades["trade_id"] == "t1", "entry_price"].iloc[0]) == 100.0

    strategy_b = journal.get_trades(strategy="B")
    assert len(strategy_b) == 1
    assert strategy_b.iloc[0]["trade_id"] == "t2"

    february = journal.get_trades(start_date="2024-02-01", end_date="2024-02-29")
    assert len(february) == 1
    assert february.iloc[0]["trade_id"] == "t2"

    losing = journal.get_losing_trades()
    assert len(losing) == 1
    assert losing.iloc[0]["trade_id"] == "t2"

    journal.close()
