from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .config_loader import PipelineConfig, load_config
from .trend_strategy import load_or_fetch_trend_price_history


class PerformanceTracker:
    def __init__(self, db_path: str | Path = "data/performance.db", config: PipelineConfig | None = None) -> None:
        self.config = config
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        self.connection.close()

    def record_daily(
        self,
        date: str,
        portfolio_value: float,
        daily_pnl: float,
        positions: dict[str, Any],
        allocation_weights: dict[str, float],
        regime_state: str,
        risk_headroom: dict[str, float],
        *,
        capital_scale_factor: float = 1.0,
        health_status: str = "UNKNOWN",
        tracking_error_pct: float = 0.0,
        spy_return: float = 0.0,
    ) -> None:
        weights = allocation_weights or {}
        strategy_a_weight = float(weights.get("strategy_a", 0.0))
        strategy_b_weight = float(weights.get("strategy_b", 0.0))
        strategy_a_contribution = float(daily_pnl * strategy_a_weight)
        strategy_b_contribution = float(daily_pnl * strategy_b_weight)
        daily_return = self._daily_return_estimate(date, portfolio_value, daily_pnl)

        self.connection.execute(
            """
            INSERT OR REPLACE INTO daily_performance (
                date,
                portfolio_value,
                daily_pnl,
                daily_return,
                positions_json,
                allocation_weights_json,
                regime_state,
                risk_headroom_json,
                strategy_a_weight,
                strategy_b_weight,
                strategy_a_contribution,
                strategy_b_contribution,
                capital_scale_factor,
                health_status,
                tracking_error_pct,
                spy_return
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(date),
                float(portfolio_value),
                float(daily_pnl),
                float(daily_return),
                json.dumps(positions, default=str),
                json.dumps(allocation_weights, default=str),
                str(regime_state),
                json.dumps(risk_headroom, default=str),
                strategy_a_weight,
                strategy_b_weight,
                strategy_a_contribution,
                strategy_b_contribution,
                float(capital_scale_factor),
                str(health_status),
                float(tracking_error_pct),
                float(spy_return),
            ),
        )
        self.connection.commit()

    def compute_statistics(self, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
        frame = self._load_frame(start_date=start_date, end_date=end_date)
        if frame.empty:
            return {
                "row_count": 0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "current_drawdown": 0.0,
                "cumulative_return": 0.0,
                "monthly_returns": {},
                "strategy_attribution": {"strategy_a": 0.0, "strategy_b": 0.0},
                "rolling_60d_sharpe": 0.0,
                "spy_comparison": {"beta": 0.0, "alpha": 0.0, "correlation": 0.0},
            }

        returns = frame["daily_return"].fillna(0.0)
        sharpe_ratio = _annualized_sharpe(returns)
        equity = frame["portfolio_value"].astype(float)
        running_max = equity.cummax()
        drawdown = (equity / running_max) - 1.0
        max_drawdown = float(drawdown.min())
        current_drawdown = float(drawdown.iloc[-1])
        cumulative_return = float((equity.iloc[-1] / equity.iloc[0]) - 1.0) if len(equity) > 1 else 0.0
        monthly_returns = (
            returns.add(1.0)
            .resample("ME")
            .prod()
            .sub(1.0)
        )
        rolling_sharpe = returns.rolling(60, min_periods=20).apply(_rolling_sharpe, raw=False)
        spy_comparison = _spy_statistics(returns, frame["spy_return"].fillna(0.0))
        return {
            "row_count": int(len(frame)),
            "start_date": frame.index.min().date().isoformat(),
            "end_date": frame.index.max().date().isoformat(),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "current_drawdown": current_drawdown,
            "cumulative_return": cumulative_return,
            "monthly_returns": {index.strftime("%Y-%m"): float(value) for index, value in monthly_returns.items()},
            "strategy_attribution": {
                "strategy_a": float(frame["strategy_a_contribution"].sum()),
                "strategy_b": float(frame["strategy_b_contribution"].sum()),
            },
            "rolling_60d_sharpe": 0.0 if rolling_sharpe.empty or pd.isna(rolling_sharpe.iloc[-1]) else float(rolling_sharpe.iloc[-1]),
            "spy_comparison": spy_comparison,
            "avg_allocation": {
                "strategy_a": float(frame["strategy_a_weight"].mean()),
                "strategy_b": float(frame["strategy_b_weight"].mean()),
            },
        }

    def generate_report(
        self,
        output_path: str | Path = "diagnostics/performance_report.md",
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Path:
        stats = self.compute_statistics(start_date=start_date, end_date=end_date)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        monthly_preview = stats["monthly_returns"]
        lines = [
            "# Performance Report",
            "",
            f"- Rows: `{stats['row_count']}`",
            f"- Start date: `{stats.get('start_date', 'n/a')}`",
            f"- End date: `{stats.get('end_date', 'n/a')}`",
            f"- Sharpe ratio: `{stats['sharpe_ratio']:.4f}`",
            f"- Max drawdown: `{stats['max_drawdown']:.2%}`",
            f"- Current drawdown: `{stats['current_drawdown']:.2%}`",
            f"- Cumulative return: `{stats['cumulative_return']:.2%}`",
            f"- Rolling 60-day Sharpe: `{stats['rolling_60d_sharpe']:.4f}`",
            f"- Strategy A attribution: `{stats['strategy_attribution']['strategy_a']:.2f}`",
            f"- Strategy B attribution: `{stats['strategy_attribution']['strategy_b']:.2f}`",
            f"- Avg allocation A/B: `{stats['avg_allocation']['strategy_a']:.2%}` / `{stats['avg_allocation']['strategy_b']:.2%}`",
            f"- SPY beta: `{stats['spy_comparison']['beta']:.4f}`",
            f"- SPY alpha: `{stats['spy_comparison']['alpha']:.4f}`",
            f"- SPY correlation: `{stats['spy_comparison']['correlation']:.4f}`",
            "",
            "## Monthly Returns",
        ]
        if not monthly_preview:
            lines.append("- No monthly returns available.")
        else:
            for month, value in monthly_preview.items():
                lines.append(f"- {month}: `{value:.2%}`")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _ensure_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_performance (
                date TEXT PRIMARY KEY,
                portfolio_value REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                daily_return REAL NOT NULL,
                positions_json TEXT NOT NULL,
                allocation_weights_json TEXT NOT NULL,
                regime_state TEXT NOT NULL,
                risk_headroom_json TEXT NOT NULL,
                strategy_a_weight REAL NOT NULL,
                strategy_b_weight REAL NOT NULL,
                strategy_a_contribution REAL NOT NULL,
                strategy_b_contribution REAL NOT NULL,
                capital_scale_factor REAL NOT NULL,
                health_status TEXT NOT NULL,
                tracking_error_pct REAL NOT NULL,
                spy_return REAL NOT NULL
            )
            """
        )
        self.connection.commit()

    def _load_frame(self, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM daily_performance WHERE 1=1"
        params: list[Any] = []
        if start_date is not None:
            query += " AND date >= ?"
            params.append(str(start_date))
        if end_date is not None:
            query += " AND date <= ?"
            params.append(str(end_date))
        query += " ORDER BY date ASC"
        frame = pd.read_sql_query(query, self.connection, params=params, parse_dates=["date"])
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        return frame

    def _daily_return_estimate(self, date: str, portfolio_value: float, daily_pnl: float) -> float:
        cursor = self.connection.execute(
            "SELECT portfolio_value FROM daily_performance WHERE date < ? ORDER BY date DESC LIMIT 1",
            (str(date),),
        )
        row = cursor.fetchone()
        if row is not None and float(row["portfolio_value"]) != 0.0:
            previous_value = float(row["portfolio_value"])
        else:
            previous_value = float(portfolio_value - daily_pnl)
        previous_value = previous_value if abs(previous_value) > 1e-9 else 1e-9
        return float(daily_pnl / previous_value)


def generate_performance_report(
    config_path: str | Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path)
    tracker = PerformanceTracker(config.paths.project_root / "data" / "performance.db", config=config)
    try:
        _sync_spy_returns(tracker, config)
        stats = tracker.compute_statistics(start_date=start_date, end_date=end_date)
        output_path = tracker.generate_report(
            config.paths.project_root / "diagnostics" / "performance_report.md",
            start_date=start_date,
            end_date=end_date,
        )
        return stats, output_path
    finally:
        tracker.close()


def build_performance_summary(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    tracker = PerformanceTracker(config.paths.project_root / "data" / "performance.db", config=config)
    try:
        _sync_spy_returns(tracker, config)
        return tracker.compute_statistics()
    finally:
        tracker.close()


def _sync_spy_returns(tracker: PerformanceTracker, config: PipelineConfig) -> None:
    frame = tracker._load_frame()
    if frame.empty:
        return
    if frame["spy_return"].abs().sum() > 0:
        return
    trend_prices = load_or_fetch_trend_price_history(config)
    price_matrix = (
        trend_prices.pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
        .reindex(columns=config.phase5.trend_tickers)
    )
    spy_returns = price_matrix["SPY"].pct_change().fillna(0.0)
    for timestamp, value in spy_returns.reindex(frame.index).fillna(0.0).items():
        tracker.connection.execute(
            "UPDATE daily_performance SET spy_return = ? WHERE date = ?",
            (float(value), timestamp.date().isoformat()),
        )
    tracker.connection.commit()


def _annualized_sharpe(returns: pd.Series) -> float:
    cleaned = returns.dropna()
    if cleaned.empty:
        return 0.0
    std = float(cleaned.std(ddof=0))
    if std <= 1e-12:
        return 0.0
    return float(cleaned.mean() / std * (252 ** 0.5))


def _rolling_sharpe(window: pd.Series) -> float:
    return _annualized_sharpe(window)


def _spy_statistics(returns: pd.Series, spy_returns: pd.Series) -> dict[str, float]:
    aligned = pd.concat([returns.rename("portfolio"), spy_returns.rename("spy")], axis=1).dropna()
    if aligned.empty or len(aligned) < 2:
        return {"beta": 0.0, "alpha": 0.0, "correlation": 0.0}
    variance = float(aligned["spy"].var(ddof=0))
    if variance <= 1e-12:
        beta = 0.0
    else:
        beta = float(aligned["portfolio"].cov(aligned["spy"]) / variance)
    alpha = float((aligned["portfolio"].mean() - (beta * aligned["spy"].mean())) * 252.0)
    correlation = aligned["portfolio"].corr(aligned["spy"])
    return {
        "beta": beta,
        "alpha": alpha,
        "correlation": 0.0 if pd.isna(correlation) else float(correlation),
    }
