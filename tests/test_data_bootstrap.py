from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest import BacktestResult, RiskBudgetScalingResult
from src.config_loader import load_config
from src.phase2 import run_phase2_pipeline
from src.storage import load_validated_price_data
from src.trend_strategy import load_or_fetch_trend_price_history
from src.validation import validate_prices


def _write_bootstrap_config(
    tmp_path: Path,
    *,
    sector_tickers: list[str] | None = None,
    trend_tickers: list[str] | None = None,
) -> Path:
    config_path = tmp_path / "config" / "phase8.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_load(Path("config/phase8.yaml").read_text(encoding="utf-8"))
    payload["tickers"] = sector_tickers or ["XLK", "XLE"]
    payload["phase5"]["trend_tickers"] = trend_tickers or ["SPY", "TLT"]
    payload["start_date"] = "2024-01-02"
    payload["end_date"] = "2025-03-31"
    payload["paths"]["raw_dir"] = str(tmp_path / "data" / "raw")
    payload["paths"]["processed_dir"] = str(tmp_path / "data" / "processed")
    payload["paths"]["log_dir"] = str(tmp_path / "logs")
    payload["paths"]["pipeline_log_file"] = str(tmp_path / "logs" / "phase1_pipeline.jsonl")
    payload["paths"]["postgres_log_file"] = str(tmp_path / "logs" / "postgres.log")
    payload["paths"]["cache_dir"] = str(tmp_path / "data" / ".cache" / "yfinance")
    payload["paths"]["postgres_dir"] = str(tmp_path / "data" / "postgres" / "phase1")
    payload["deployment"]["paper_state_path"] = str(tmp_path / "data" / "processed" / "phase7_state.json")
    payload["deployment"]["phase7_gate_artifact"] = str(tmp_path / "config" / "phase7_cleared.yaml")
    payload["deployment"]["live_confirmation_path"] = str(tmp_path / "config" / "live_confirmed.txt")
    payload["phase7"]["credentials_path"] = str(tmp_path / "config" / "credentials.yaml")
    payload["learning"]["trade_journal_path"] = str(tmp_path / "data" / "trade_journal.db")
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _build_raw_price_frame(tickers: list[str], *, periods: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    rows: list[dict[str, object]] = []
    for idx, ticker in enumerate(tickers):
        base_series = 100 + (idx * 7.5) + np.linspace(0, 18, len(dates))
        wobble = np.sin(np.arange(len(dates)) / (7 + idx)) * (1.5 + idx * 0.1)
        prices = base_series + wobble
        for date, price in zip(dates, prices):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price * 0.995,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000 + (idx * 1_000),
                    "dividends": 0.0,
                    "stock_splits": 0.0,
                    "capital_gains": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_load_validated_price_data_bootstraps_missing_sector_artifact(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_bootstrap_config(tmp_path)
    config = load_config(config_path)
    download_calls: list[list[str]] = []

    def fake_download_universe_data(*, tickers, start_date, end_date, cache_dir, logger):
        download_calls.append(list(tickers))
        return _build_raw_price_frame(list(tickers))

    monkeypatch.setattr("src.storage.download_universe_data", fake_download_universe_data)

    validated = load_validated_price_data(config, dataset="sector")

    assert download_calls == [config.tickers]
    assert set(validated["ticker"].unique()) == set(config.tickers)
    assert (tmp_path / "data" / "raw").exists()
    assert (tmp_path / "data" / "processed").exists()
    assert (tmp_path / "logs").exists()
    assert config.paths.processed_dir.joinpath("sector_etf_prices_validated.csv").exists()


def test_load_or_fetch_trend_price_history_bootstraps_missing_trend_artifact(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_bootstrap_config(tmp_path)
    config = load_config(config_path)
    download_calls: list[list[str]] = []

    def fake_download_universe_data(*, tickers, start_date, end_date, cache_dir, logger):
        download_calls.append(list(tickers))
        return _build_raw_price_frame(list(tickers))

    monkeypatch.setattr("src.storage.download_universe_data", fake_download_universe_data)

    trend_prices = load_or_fetch_trend_price_history(config)

    assert download_calls == [config.phase5.trend_tickers]
    assert set(trend_prices["ticker"].unique()) == set(config.phase5.trend_tickers)
    assert trend_prices["volume"].gt(0).all()
    assert config.paths.processed_dir.joinpath("trend_universe_prices_validated.csv").exists()


def test_run_phase2_pipeline_bootstraps_when_store_has_no_prices(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_bootstrap_config(tmp_path)
    config = load_config(config_path)
    raw_prices = _build_raw_price_frame(config.tickers)
    validated_prices = validate_prices(raw_prices, config.validation)
    fallback_calls = {"count": 0}

    class FakeStore:
        def __init__(self) -> None:
            self.persisted = False

        def initialize(self) -> None:
            return None

        def ensure_phase2_schema(self) -> None:
            return None

        def fetch_validated_price_history(self, tickers: list[str]) -> pd.DataFrame:
            return pd.DataFrame(columns=["date", "ticker", "adj_close"])

        def persist_phase2_run(self, **kwargs) -> None:
            self.persisted = True

        def stop(self) -> None:
            return None

    fake_store = FakeStore()

    def fake_load_validated_price_data(active_config, *, dataset, logger=None):
        assert dataset == "sector"
        fallback_calls["count"] += 1
        return validated_prices

    signal_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "ticker": config.tickers,
            "target_position": [0.25, -0.25],
        }
    )
    daily_results = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "gross_portfolio_return": [0.01],
            "net_portfolio_return": [0.009],
            "portfolio_return": [0.009],
            "gross_exposure": [0.5],
            "turnover": [0.1],
            "transaction_cost": [0.001],
        }
    )
    trade_log = pd.DataFrame(
        {
            "trade_id": ["trade-1"],
            "ticker": [config.tickers[0]],
            "entry_date": pd.to_datetime(["2025-01-02"]),
            "exit_date": pd.to_datetime(["2025-01-03"]),
            "position_direction": [1],
            "holding_days": [1],
            "entry_weight": [0.25],
            "gross_return": [0.01],
            "net_return": [0.009],
        }
    )
    monthly_results = pd.DataFrame(
        {
            "test_month": pd.to_datetime(["2025-01-01"]),
            "training_end_date": pd.to_datetime(["2024-12-31"]),
            "monthly_return": [0.009],
            "profitable": [True],
        }
    )

    monkeypatch.setattr("src.phase2.PostgresStore", lambda *args, **kwargs: fake_store)
    monkeypatch.setattr("src.phase2.load_validated_price_data", fake_load_validated_price_data)
    monkeypatch.setattr("src.phase2.compute_graph_signals", lambda price_history, tickers, phase2_config: signal_frame)
    monkeypatch.setattr(
        "src.phase2.scale_signals_to_risk_budget",
        lambda signals, phase2_config: RiskBudgetScalingResult(
            scaled_signals=signals,
            baseline_max_drawdown=0.12,
            target_max_drawdown=0.08,
            scale_factor=0.75,
        ),
    )
    monkeypatch.setattr(
        "src.phase2.run_walk_forward_backtest",
        lambda scaled_signals, phase2_config, run_id: BacktestResult(
            daily_results=daily_results,
            trade_log=trade_log,
            monthly_results=monthly_results,
            summary_metrics={
                "gate_passed": True,
                "sharpe_ratio": 1.1,
                "max_drawdown": 0.08,
                "profitable_month_fraction": 1.0,
            },
        ),
    )

    result = run_phase2_pipeline(config_path)

    assert fallback_calls["count"] == 1
    assert fake_store.persisted is True
    assert result.summary_metrics["position_scale_factor"] == 0.75
    assert result.output_paths["signals"].exists()
