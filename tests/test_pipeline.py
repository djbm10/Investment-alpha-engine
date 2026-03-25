from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.pipeline import DailyPipeline


def _build_pipeline(available_dates: list[str], processed_dates: list[str]) -> DailyPipeline:
    pipeline = DailyPipeline.__new__(DailyPipeline)
    pipeline.available_dates = [pd.Timestamp(value) for value in available_dates]
    pipeline.state = {
        "daily_records": [{"date": value} for value in processed_dates],
        "allocations": {"strategy_a": 0.5, "strategy_b": 0.5},
    }
    return pipeline


def test_resolve_trading_date_returns_latest_available_when_state_is_behind() -> None:
    pipeline = _build_pipeline(
        available_dates=["2026-03-14", "2026-03-16", "2026-03-17"],
        processed_dates=["2026-03-16"],
    )

    resolved = pipeline._resolve_trading_date(None)

    assert resolved == pd.Timestamp("2026-03-17")


def test_resolve_trading_date_raises_when_no_new_date_is_available() -> None:
    pipeline = _build_pipeline(
        available_dates=["2026-03-14", "2026-03-16"],
        processed_dates=["2026-03-16"],
    )

    with pytest.raises(ValueError, match="No new trading date is available for Phase 7"):
        pipeline._resolve_trading_date(None)


def test_resolve_trading_date_rejects_out_of_order_requested_dates() -> None:
    pipeline = _build_pipeline(
        available_dates=["2026-03-14", "2026-03-16", "2026-03-17"],
        processed_dates=["2026-03-16"],
    )

    with pytest.raises(ValueError, match="cannot be processed out of order"):
        pipeline._resolve_trading_date("2026-03-15")


def test_run_daily_refreshes_paper_inputs_before_resolving_date(tmp_path) -> None:
    halt_flag = tmp_path / "emergency_halt.flag"
    halt_flag.write_text("manual halt", encoding="utf-8")

    pipeline = _build_pipeline(
        available_dates=["2026-03-16"],
        processed_dates=["2026-03-16"],
    )
    pipeline.mode = "paper"
    pipeline.config = SimpleNamespace(
        end_date=None,
        paths=SimpleNamespace(log_dir=tmp_path),
    )
    pipeline.broker_client = SimpleNamespace(get_positions=lambda: {})
    pipeline.logger = MagicMock()
    pipeline._build_daily_summary = lambda **kwargs: {}
    pipeline._current_capital_scale_factor = lambda trading_date: 1.0
    pipeline._finalize_result = lambda result: result

    refresh_calls = {"count": 0}

    def _refresh() -> None:
        refresh_calls["count"] += 1
        pipeline.available_dates = [
            pd.Timestamp("2026-03-16"),
            pd.Timestamp("2026-03-17"),
        ]

    pipeline._refresh_strategy_inputs = _refresh

    result = pipeline.run_daily()

    assert refresh_calls["count"] == 1
    assert result.date == pd.Timestamp("2026-03-17")
    assert result.aborted is True
