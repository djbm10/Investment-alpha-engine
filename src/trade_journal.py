from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


class TradeJournal:
    def __init__(self, db_path: str | Path = "data/trade_journal.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def close(self) -> None:
        self.connection.close()

    def log_trade(self, trade_data: dict[str, Any]) -> None:
        payload = _normalized_trade_payload(trade_data)
        columns = ", ".join(payload)
        placeholders = ", ".join(f":{column}" for column in payload)
        query = f"INSERT OR REPLACE INTO trades ({columns}) VALUES ({placeholders})"
        self.connection.execute(query, payload)
        self.connection.commit()

    def get_trades(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        strategy: str | None = None,
    ) -> pd.DataFrame:
        where_clauses: list[str] = []
        params: dict[str, Any] = {}
        if start_date is not None:
            where_clauses.append("exit_date >= :start_date")
            params["start_date"] = _normalize_date(start_date)
        if end_date is not None:
            where_clauses.append("exit_date <= :end_date")
            params["end_date"] = _normalize_date(end_date)
        if strategy is not None:
            where_clauses.append("strategy = :strategy")
            params["strategy"] = strategy

        query = "SELECT * FROM trades"
        if where_clauses:
            query = f"{query} WHERE {' AND '.join(where_clauses)}"
        query = f"{query} ORDER BY exit_date, asset, trade_id"
        return pd.read_sql_query(
            query,
            self.connection,
            params=params,
            parse_dates=["entry_date", "exit_date"],
        )

    def get_losing_trades(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        trades = self.get_trades(start_date=start_date, end_date=end_date)
        if trades.empty:
            return trades
        return trades.loc[trades["net_pnl"] < 0].reset_index(drop=True)

    def _initialize_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                strategy TEXT NOT NULL,
                asset TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                exit_date TEXT NOT NULL,
                holding_days INTEGER NOT NULL,
                entry_price REAL,
                exit_price REAL,
                entry_zscore REAL,
                entry_node_corr REAL,
                entry_regime TEXT,
                exit_reason TEXT,
                gross_pnl REAL NOT NULL,
                transaction_cost REAL NOT NULL,
                net_pnl REAL NOT NULL,
                predicted_residual REAL,
                actual_residual REAL,
                prediction_error REAL,
                portfolio_exposure_at_entry REAL,
                concurrent_positions INTEGER
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_exit_date ON trades (exit_date)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy)"
        )
        self.connection.commit()


def _normalized_trade_payload(trade_data: dict[str, Any]) -> dict[str, Any]:
    columns = {
        "trade_id": None,
        "strategy": None,
        "asset": None,
        "direction": None,
        "entry_date": None,
        "exit_date": None,
        "holding_days": None,
        "entry_price": None,
        "exit_price": None,
        "entry_zscore": None,
        "entry_node_corr": None,
        "entry_regime": None,
        "exit_reason": None,
        "gross_pnl": None,
        "transaction_cost": None,
        "net_pnl": None,
        "predicted_residual": None,
        "actual_residual": None,
        "prediction_error": None,
        "portfolio_exposure_at_entry": None,
        "concurrent_positions": None,
    }
    payload = {**columns, **trade_data}
    payload["entry_date"] = _normalize_date(payload["entry_date"])
    payload["exit_date"] = _normalize_date(payload["exit_date"])
    payload["holding_days"] = int(payload["holding_days"])
    payload["concurrent_positions"] = (
        None if payload["concurrent_positions"] is None else int(payload["concurrent_positions"])
    )
    for field in (
        "entry_price",
        "exit_price",
        "entry_zscore",
        "entry_node_corr",
        "gross_pnl",
        "transaction_cost",
        "net_pnl",
        "predicted_residual",
        "actual_residual",
        "prediction_error",
        "portfolio_exposure_at_entry",
    ):
        payload[field] = _optional_float(payload[field])
    return payload


def _normalize_date(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
