import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.config_loader import load_config
from src.geo.evaluation import build_geo_overlay_evaluation_report, run_geo_overlay_evaluation
from src.trend_strategy import TrendStrategyBacktest


def _write_phase8_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config" / "phase8.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_load(Path("config/phase8.yaml").read_text(encoding="utf-8"))
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
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _make_backtest(
    *,
    dates: pd.DatetimeIndex,
    daily_returns: list[float],
    turnovers: list[float],
    trade_log: pd.DataFrame,
    daily_signals: pd.DataFrame | None,
    sharpe_ratio: float,
    max_drawdown: float,
    annual_turnover: float,
    contradiction_loss_rate: float,
) -> TrendStrategyBacktest:
    signal_frame = daily_signals.copy() if daily_signals is not None else _build_geo_daily_signals(dates)
    daily_results = pd.DataFrame(
        {
            "date": dates,
            "gross_portfolio_return": daily_returns,
            "net_portfolio_return": daily_returns,
            "portfolio_return": daily_returns,
            "gross_exposure": [0.10] * len(dates),
            "turnover": turnovers,
            "transaction_cost": [0.0] * len(dates),
        }
    )
    daily_positions = pd.DataFrame(
        [{"date": date, "ticker": "XLK", "target_position": 0.10} for date in dates]
    )
    monthly_results = pd.DataFrame(
        {
            "test_month": [pd.Timestamp(dates[-1]).to_period("M").to_timestamp()],
            "training_end_date": [pd.Timestamp(dates[-1]) - pd.Timedelta(days=1)],
            "monthly_return": [0.01],
            "profitable": [True],
            "active_month": [True],
        }
    )
    return TrendStrategyBacktest(
        daily_positions=daily_positions,
        daily_results=daily_results,
        trade_log=trade_log,
        monthly_results=monthly_results,
        summary_metrics={
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "annual_turnover": annual_turnover,
            "contradiction_loss_rate": contradiction_loss_rate,
            "total_trades": int(len(trade_log)),
        },
        daily_signals=signal_frame,
    )


def _build_trade_log(
    *,
    dates: pd.DatetimeIndex,
    contradictory_return: float,
    include_removed_trade: bool,
) -> pd.DataFrame:
    rows = [
        {
            "trade_id": "phase5-strategy-a:XLK:2024-01-02:2024-01-03:1",
            "ticker": "XLK",
            "entry_date": dates[1],
            "exit_date": dates[2],
            "position_direction": 1,
            "net_return": 0.02,
            "entry_contradiction": 0.0,
        },
        {
            "trade_id": "phase5-strategy-a:XLE:2024-01-12:2024-01-15:-1",
            "ticker": "XLE",
            "entry_date": dates[9],
            "exit_date": dates[10],
            "position_direction": -1,
            "net_return": contradictory_return,
            "entry_contradiction": 0.90,
        },
        {
            "trade_id": "phase5-strategy-a:XLU:2024-01-03:2024-01-04:1",
            "ticker": "XLU",
            "entry_date": dates[2],
            "exit_date": dates[3],
            "position_direction": 1,
            "net_return": 0.004,
            "entry_contradiction": 0.0,
        },
        {
            "trade_id": "phase5-strategy-a:XLB:2024-01-08:2024-01-09:-1",
            "ticker": "XLB",
            "entry_date": dates[5],
            "exit_date": dates[6],
            "position_direction": -1,
            "net_return": 0.003,
            "entry_contradiction": 0.0,
        },
    ]
    if include_removed_trade:
        rows.append(
            {
                "trade_id": "phase5-strategy-a:XLF:2024-01-05:2024-01-08:1",
                "ticker": "XLF",
                "entry_date": dates[4],
                "exit_date": dates[5],
                "position_direction": 1,
                "net_return": -0.01,
                "entry_contradiction": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _build_geo_daily_signals(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in dates:
        high_stress = pd.Timestamp(date) == pd.Timestamp(dates[9])
        rows.extend(
            [
                {
                    "date": date,
                    "ticker": "XLK",
                    "adj_close": 100.0 + float((date - dates[0]).days),
                    "current_return": 0.001,
                    "forward_return": 0.002,
                    "allow_new_entries": True,
                    "node_tradeable": True,
                    "geo_net_score": 0.90 if high_stress else 0.05,
                    "geo_structural_score": 0.85 if high_stress else 0.04,
                    "coverage_score": 0.80,
                    "avg_mapping_confidence": 0.90,
                    "data_freshness_minutes": 30,
                    "position_scale": 1.0,
                    "final_signal_direction": 1,
                    "final_target_position": 0.10,
                    "geo_entry_blocked": False,
                },
                {
                    "date": date,
                    "ticker": "XLE",
                    "adj_close": 90.0 + float((date - dates[0]).days),
                    "current_return": -0.001,
                    "forward_return": -0.002,
                    "allow_new_entries": True,
                    "node_tradeable": True,
                    "geo_net_score": -0.80 if high_stress else 0.02,
                    "geo_structural_score": -0.75 if high_stress else -0.01,
                    "coverage_score": 0.80,
                    "avg_mapping_confidence": 0.90,
                    "data_freshness_minutes": 30,
                    "position_scale": 0.75 if high_stress else 1.0,
                    "final_signal_direction": -1,
                    "final_target_position": -0.075 if high_stress else -0.10,
                    "geo_entry_blocked": False,
                },
                {
                    "date": date,
                    "ticker": "XLU",
                    "adj_close": 70.0 + float((date - dates[0]).days),
                    "current_return": 0.0005,
                    "forward_return": 0.001,
                    "allow_new_entries": True,
                    "node_tradeable": True,
                    "geo_net_score": 0.01,
                    "geo_structural_score": 0.01,
                    "coverage_score": 0.80,
                    "avg_mapping_confidence": 0.90,
                    "data_freshness_minutes": 30,
                    "position_scale": 1.0,
                    "final_signal_direction": 1,
                    "final_target_position": 0.10,
                    "geo_entry_blocked": False,
                },
                {
                    "date": date,
                    "ticker": "XLB",
                    "adj_close": 65.0 + float((date - dates[0]).days),
                    "current_return": -0.0005,
                    "forward_return": 0.0003,
                    "allow_new_entries": True,
                    "node_tradeable": True,
                    "geo_net_score": -0.03 if pd.Timestamp(date) == pd.Timestamp(dates[5]) else 0.01,
                    "geo_structural_score": -0.02 if pd.Timestamp(date) == pd.Timestamp(dates[5]) else 0.01,
                    "coverage_score": 0.80,
                    "avg_mapping_confidence": 0.90,
                    "data_freshness_minutes": 30,
                    "position_scale": 0.60 if pd.Timestamp(date) == pd.Timestamp(dates[5]) else 1.0,
                    "final_signal_direction": -1,
                    "final_target_position": -0.06 if pd.Timestamp(date) == pd.Timestamp(dates[5]) else -0.10,
                    "geo_entry_blocked": False,
                },
                {
                    "date": date,
                    "ticker": "XLF",
                    "adj_close": 80.0 + float((date - dates[0]).days),
                    "current_return": 0.0002,
                    "forward_return": -0.0008,
                    "allow_new_entries": True,
                    "node_tradeable": True,
                    "geo_net_score": 0.20 if pd.Timestamp(date) == pd.Timestamp(dates[4]) else 0.01,
                    "geo_structural_score": 0.90 if pd.Timestamp(date) == pd.Timestamp(dates[4]) else 0.01,
                    "coverage_score": 0.80,
                    "avg_mapping_confidence": 0.90,
                    "data_freshness_minutes": 30,
                    "position_scale": 0.0 if pd.Timestamp(date) == pd.Timestamp(dates[4]) else 1.0,
                    "final_signal_direction": 0 if pd.Timestamp(date) == pd.Timestamp(dates[4]) else 1,
                    "final_target_position": 0.0 if pd.Timestamp(date) == pd.Timestamp(dates[4]) else 0.10,
                    "geo_entry_blocked": bool(pd.Timestamp(date) == pd.Timestamp(dates[4])),
                },
            ]
        )
    return pd.DataFrame(rows)


def _build_geo_snapshot(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in dates:
        high_stress = pd.Timestamp(date) == pd.Timestamp(dates[9])
        rows.extend(
            [
                {
                    "trade_date": date.date(),
                    "asset": "XLK",
                    "geo_net_score": 0.90 if high_stress else 0.05,
                    "geo_structural_score": 0.85 if high_stress else 0.04,
                    "coverage_score": 0.80,
                    "avg_mapping_confidence": 0.90,
                    "data_freshness_minutes": 30,
                },
                {
                    "trade_date": date.date(),
                    "asset": "XLE",
                    "geo_net_score": -0.80 if high_stress else 0.02,
                    "geo_structural_score": -0.75 if high_stress else -0.01,
                    "coverage_score": 0.80,
                    "avg_mapping_confidence": 0.90,
                    "data_freshness_minutes": 30,
                },
                {
                    "trade_date": date.date(),
                    "asset": "XLU",
                    "geo_net_score": 0.01,
                    "geo_structural_score": 0.01,
                    "coverage_score": 0.80,
                    "avg_mapping_confidence": 0.90,
                    "data_freshness_minutes": 30,
                },
                {
                    "trade_date": date.date(),
                    "asset": "XLB",
                    "geo_net_score": -0.03 if pd.Timestamp(date) == pd.Timestamp(dates[5]) else 0.01,
                    "geo_structural_score": -0.02 if pd.Timestamp(date) == pd.Timestamp(dates[5]) else 0.01,
                    "coverage_score": 0.80,
                    "avg_mapping_confidence": 0.90,
                    "data_freshness_minutes": 30,
                },
                {
                    "trade_date": date.date(),
                    "asset": "XLF",
                    "geo_net_score": 0.20 if pd.Timestamp(date) == pd.Timestamp(dates[4]) else 0.01,
                    "geo_structural_score": 0.90 if pd.Timestamp(date) == pd.Timestamp(dates[4]) else 0.01,
                    "coverage_score": 0.80,
                    "avg_mapping_confidence": 0.90,
                    "data_freshness_minutes": 30,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_build_geo_overlay_evaluation_report_computes_required_metrics_and_slices() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.001, 0.002, 0.0, 0.001, -0.001, 0.0, 0.001, 0.0, 0.001, -0.02, 0.001, 0.0],
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.03, include_removed_trade=True),
        daily_signals=None,
        sharpe_ratio=1.00,
        max_drawdown=0.10,
        annual_turnover=12.0,
        contradiction_loss_rate=1.0,
    )
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.001, 0.002, 0.0, 0.001, 0.0, 0.0, 0.001, 0.0, 0.001, 0.002, 0.001, 0.0],
        turnovers=[0.095] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=0.01, include_removed_trade=False),
        daily_signals=_build_geo_daily_signals(dates),
        sharpe_ratio=1.02,
        max_drawdown=0.08,
        annual_turnover=11.0,
        contradiction_loss_rate=0.0,
    )

    report = build_geo_overlay_evaluation_report(
        baseline_result=baseline,
        geo_enabled_result=geo_enabled,
        annualization_days=252,
    )

    assert report["baseline"]["sharpe_ratio"] == 1.0
    assert report["baseline"]["avg_net_pnl_per_trade"] == pytest.approx(-0.0026)
    assert report["geo_enabled"]["max_drawdown"] == 0.08
    assert report["geo_enabled"]["net_sharpe_after_costs"] is not None
    assert report["geo_stress"]["days_with_geo_signal"] == 12
    assert report["geo_stress"]["median_coverage_on_geo_days"] == 0.8
    assert report["geo_stress"]["top_decile_avg_coverage"] == 0.8
    assert report["slices"]["top_decile_geo_stress"]["date_count"] == 2
    assert report["slices"]["top_decile_geo_stress"]["baseline"]["contradiction_loss_rate"] == 1.0
    assert report["slices"]["top_decile_geo_stress"]["geo_enabled"]["contradiction_loss_rate"] == 0.0
    assert report["slices"]["high_geo_blocks"]["block_count"] == 0
    assert report["trade_diagnostics"]["baseline"]["contradictory_trade_count"] == 1
    assert report["trade_diagnostics"]["geo_enabled"]["avg_pnl_contradictory"] == 0.01
    assert report["filtering_impact"]["removed_trade_count"] == 1
    assert report["filtering_impact"]["removed_trade_fraction"] == 0.2
    assert report["filtering_impact"]["avg_removed_trade_pnl"] == -0.01
    assert report["filtering_impact"]["avg_all_trade_pnl"] == pytest.approx(-0.0026)
    assert report["bootstrap"]["random_seed"] == 1729
    assert len(report["bootstrap"]["sample_indices"]) == 250
    assert report["bootstrap"]["delta_sharpe"]["p25"] is not None
    assert report["geo_enabled"]["net_cvar_5"] is not None
    assert report["curves"]["baseline"]["equity"][0] == pytest.approx(1.001)
    assert report["curves"]["geo_enabled"]["drawdown"][-1] <= 0.0
    assert report["geo_stress_series"][0]["high_geo_block"] is False
    assert report["per_asset_delta_pnl"]["top_positive"][0]["ticker"] == "XLE"
    assert report["acceptance"]["passed"] is True
    assert report["acceptance"]["criteria"]["overall_net_sharpe_after_costs_not_materially_worse"] is True
    assert report["acceptance"]["criteria"]["bootstrap_delta_sharpe_median_non_negative"] is True
    assert report["acceptance"]["criteria"]["bootstrap_delta_sharpe_p25_within_tolerance"] is True
    assert report["input_parity"]["passed"] is True
    assert report["input_parity"]["market_data_columns_checked"] == ["adj_close", "current_return", "forward_return"]
    assert report["input_parity"]["tradability_columns_checked"] == ["allow_new_entries", "node_tradeable"]
    assert report["slices"]["global"]["date_count"] == 12
    assert report["slices"]["global"]["median_coverage"] == 0.8
    assert report["slices"]["top_decile_geo_stress"]["baseline"]["trade_count"] == 2
    assert report["slices"]["top_decile_geo_stress"]["median_coverage"] == 0.8
    assert report["overlay_action_diagnostics"]["blocked"]["trade_count"] == 1
    assert report["overlay_action_diagnostics"]["blocked"]["trade_fraction"] == pytest.approx(0.2)
    assert report["overlay_action_diagnostics"]["blocked"]["avg_baseline_pnl"] == -0.01
    assert report["overlay_action_diagnostics"]["scaled"]["trade_count"] == 2
    assert report["overlay_action_diagnostics"]["scaled"]["trade_fraction"] == pytest.approx(0.4)
    assert report["overlay_action_diagnostics"]["scaled"]["avg_geo_enabled_pnl"] == pytest.approx(0.0065)
    assert report["overlay_action_diagnostics"]["scaled"]["delta_avg_pnl"] == pytest.approx(0.02)
    assert report["overlay_action_diagnostics"]["untouched"]["trade_count"] == 2
    assert report["overlay_action_diagnostics"]["untouched"]["trade_fraction"] == pytest.approx(0.4)
    assert report["data_sanity"]["extreme_return_days"] == []
    assert report["data_sanity"]["extreme_return_treated_identically"] is True


def test_build_geo_overlay_evaluation_report_fails_without_geo_stress_signal() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.0] * len(dates),
        turnovers=[0.0] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.01, include_removed_trade=False),
        daily_signals=None,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        annual_turnover=0.0,
        contradiction_loss_rate=1.0,
    )
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.0] * len(dates),
        turnovers=[0.0] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.01, include_removed_trade=False),
        daily_signals=_build_geo_daily_signals(dates).assign(
            geo_net_score=0.0,
            geo_structural_score=0.0,
            coverage_score=0.0,
            avg_mapping_confidence=0.0,
            data_freshness_minutes=9999,
            final_signal_direction=0,
            final_target_position=0.0,
            geo_entry_blocked=False,
        ),
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        annual_turnover=0.0,
        contradiction_loss_rate=1.0,
    )

    report = build_geo_overlay_evaluation_report(
        baseline_result=baseline,
        geo_enabled_result=geo_enabled,
        annualization_days=252,
    )

    assert report["geo_stress"]["available"] is False
    assert report["acceptance"]["criteria"]["geo_stress_days_sufficient"] is False
    assert report["acceptance"]["passed"] is False


def test_run_geo_overlay_evaluation_writes_deterministic_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = _write_phase8_config(tmp_path)
    config = load_config(config_path)
    dates = pd.bdate_range("2024-01-01", periods=12)
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.001, 0.002, 0.0, 0.001, -0.001, 0.0, 0.001, 0.0, 0.001, -0.02, 0.001, 0.0],
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.03, include_removed_trade=True),
        daily_signals=None,
        sharpe_ratio=1.00,
        max_drawdown=0.10,
        annual_turnover=12.0,
        contradiction_loss_rate=1.0,
    )
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.001, 0.002, 0.0, 0.001, 0.0, 0.0, 0.001, 0.0, 0.001, 0.002, 0.001, 0.0],
        turnovers=[0.095] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=0.01, include_removed_trade=False),
        daily_signals=_build_geo_daily_signals(dates),
        sharpe_ratio=1.02,
        max_drawdown=0.08,
        annual_turnover=11.0,
        contradiction_loss_rate=0.0,
    )

    def fake_load_phase2_baseline_backtest(active_config, trade_journal=None, geo_store=None, geo_snapshot=None):
        if active_config.geo.enabled:
            return geo_enabled
        return baseline

    monkeypatch.setattr("src.geo.evaluation.load_phase2_baseline_backtest", fake_load_phase2_baseline_backtest)

    result_one = run_geo_overlay_evaluation(config_path)
    payload_one = json.loads(result_one.output_path.read_text(encoding="utf-8"))
    result_two = run_geo_overlay_evaluation(config_path, geo_snapshot=_build_geo_snapshot(dates))
    payload_two = json.loads(result_two.output_path.read_text(encoding="utf-8"))

    assert result_one.output_path == config.paths.processed_dir / "geo_overlay_evaluation.json"
    assert payload_one["report_version"] == payload_two["report_version"]
    assert payload_one["report_schema_hash"] == payload_two["report_schema_hash"]
    assert payload_one["metadata"]["random_seed"] == 1729
    assert "config_hash" in payload_one["metadata"]
    assert "cost_model_hash" in payload_one["metadata"]
    assert "code_version" in payload_one["metadata"]
    assert payload_one["metadata"]["data_range"]["start_date"] == "2024-01-01"
    assert payload_one["metadata"]["data_range"]["end_date"] == "2024-01-16"
    assert "universe_hash" in payload_one["metadata"]
    assert "sample_indices" in payload_one["bootstrap"]
    assert payload_two["alignment_check"]["performed"] is True
    assert payload_two["alignment_check"]["passed"] is True
    assert payload_two["snapshot_timing_check"]["performed"] is True
    assert payload_two["snapshot_timing_check"]["enforceable"] is False
    assert payload_two["metadata"]["random_seed"] == payload_one["metadata"]["random_seed"]
    assert payload_one["input_parity"]["passed"] is True


def test_build_geo_overlay_evaluation_report_fails_fast_on_missing_overlay_columns() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.0] * len(dates),
        turnovers=[0.0] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.01, include_removed_trade=False),
        daily_signals=None,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        annual_turnover=0.0,
        contradiction_loss_rate=1.0,
    )
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.0] * len(dates),
        turnovers=[0.0] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.01, include_removed_trade=False),
        daily_signals=_build_geo_daily_signals(dates).loc[:, ["date", "ticker", "geo_net_score"]].copy(),
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        annual_turnover=0.0,
        contradiction_loss_rate=1.0,
    )

    with pytest.raises(ValueError, match="overlay signal contract"):
        build_geo_overlay_evaluation_report(
            baseline_result=baseline,
            geo_enabled_result=geo_enabled,
            annualization_days=252,
        )


def test_build_geo_overlay_evaluation_report_rejects_non_entry_contradiction_fields() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    bad_trade_log = _build_trade_log(dates=dates, contradictory_return=-0.03, include_removed_trade=False)
    bad_trade_log["intratrade_contradiction"] = 0.0
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.0] * len(dates),
        turnovers=[0.0] * len(dates),
        trade_log=bad_trade_log,
        daily_signals=None,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        annual_turnover=0.0,
        contradiction_loss_rate=1.0,
    )
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.0] * len(dates),
        turnovers=[0.0] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=0.01, include_removed_trade=False),
        daily_signals=_build_geo_daily_signals(dates),
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        annual_turnover=0.0,
        contradiction_loss_rate=0.0,
    )

    with pytest.raises(ValueError, match="entry-time contradiction"):
        build_geo_overlay_evaluation_report(
            baseline_result=baseline,
            geo_enabled_result=geo_enabled,
            annualization_days=252,
        )


def test_build_geo_overlay_evaluation_report_flags_overfiltering_when_removed_trades_are_positive() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    baseline_trade_log = pd.DataFrame(
        [
            {
                "trade_id": f"phase5-strategy-a:XLK:2024-01-{day:02d}:2024-01-{day + 1:02d}:1",
                "ticker": "XLK",
                "entry_date": date,
                "exit_date": date + pd.Timedelta(days=1),
                "position_direction": 1,
                "net_return": 0.02,
                "entry_contradiction": 0.0,
            }
            for day, date in enumerate(dates[:10], start=1)
        ]
    )
    geo_trade_log = baseline_trade_log.head(7).copy()
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=baseline_trade_log,
        daily_signals=None,
        sharpe_ratio=1.0,
        max_drawdown=0.10,
        annual_turnover=10.0,
        contradiction_loss_rate=0.0,
    )
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.08] * len(dates),
        trade_log=geo_trade_log,
        daily_signals=_build_geo_daily_signals(dates),
        sharpe_ratio=1.0,
        max_drawdown=0.09,
        annual_turnover=8.0,
        contradiction_loss_rate=0.0,
    )

    report = build_geo_overlay_evaluation_report(
        baseline_result=baseline,
        geo_enabled_result=geo_enabled,
        annualization_days=252,
    )

    assert report["filtering_impact"]["removed_trade_fraction"] == 0.3
    assert report["acceptance"]["criteria"]["trade_removal_not_excessive_or_value_additive"] is False
    assert report["acceptance"]["passed"] is False


def test_build_geo_overlay_evaluation_report_flags_stale_geo_window() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    stale_signals = _build_geo_daily_signals(dates)
    stale_signals.loc[stale_signals["date"].isin(dates[6:10]), "data_freshness_minutes"] = 600
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.03, include_removed_trade=False),
        daily_signals=None,
        sharpe_ratio=1.0,
        max_drawdown=0.10,
        annual_turnover=10.0,
        contradiction_loss_rate=1.0,
    )
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=0.01, include_removed_trade=False),
        daily_signals=stale_signals,
        sharpe_ratio=1.0,
        max_drawdown=0.09,
        annual_turnover=10.0,
        contradiction_loss_rate=0.0,
    )

    report = build_geo_overlay_evaluation_report(
        baseline_result=baseline,
        geo_enabled_result=geo_enabled,
        annualization_days=252,
    )

    assert report["geo_stress"]["stale_window_detected"] is True
    assert report["acceptance"]["criteria"]["geo_windows_not_stale"] is False
    assert report["acceptance"]["passed"] is False


def test_build_geo_overlay_evaluation_report_null_geo_signal_fails_presence_gate() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.03, include_removed_trade=False),
        daily_signals=None,
        sharpe_ratio=1.0,
        max_drawdown=0.10,
        annual_turnover=10.0,
        contradiction_loss_rate=1.0,
    )
    null_signals = _build_geo_daily_signals(dates)
    null_signals["geo_net_score"] = 0.0
    null_signals["geo_structural_score"] = 0.0
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.03, include_removed_trade=False),
        daily_signals=null_signals,
        sharpe_ratio=1.0,
        max_drawdown=0.10,
        annual_turnover=10.0,
        contradiction_loss_rate=1.0,
    )

    report = build_geo_overlay_evaluation_report(
        baseline_result=baseline,
        geo_enabled_result=geo_enabled,
        annualization_days=252,
    )

    assert report["comparison"]["sharpe_ratio"]["delta"] == 0.0
    assert report["geo_stress"]["available"] is False
    assert report["acceptance"]["passed"] is False


def test_build_geo_overlay_evaluation_report_adversarial_geo_signal_fails_gate() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.001, 0.002, 0.0, 0.001, 0.0, 0.0, 0.001, 0.0, 0.001, 0.001, 0.001, 0.0],
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=0.01, include_removed_trade=False),
        daily_signals=None,
        sharpe_ratio=1.0,
        max_drawdown=0.08,
        annual_turnover=10.0,
        contradiction_loss_rate=0.0,
    )
    adversarial_signals = _build_geo_daily_signals(dates)
    adversarial_trade_log = _build_trade_log(dates=dates, contradictory_return=-0.05, include_removed_trade=False)
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.001, 0.002, 0.0, 0.001, 0.0, 0.0, 0.001, 0.0, 0.001, -0.01, 0.001, 0.0],
        turnovers=[0.10] * len(dates),
        trade_log=adversarial_trade_log,
        daily_signals=adversarial_signals,
        sharpe_ratio=0.9,
        max_drawdown=0.10,
        annual_turnover=10.0,
        contradiction_loss_rate=1.0,
    )

    report = build_geo_overlay_evaluation_report(
        baseline_result=baseline,
        geo_enabled_result=geo_enabled,
        annualization_days=252,
    )

    assert report["slices"]["top_decile_geo_stress"]["geo_enabled"]["contradiction_loss_rate"] == 1.0
    assert report["acceptance"]["criteria"]["top_decile_contradiction_loss_reduced"] is False
    assert report["acceptance"]["passed"] is False


def test_run_geo_overlay_evaluation_fails_alignment_on_duplicate_snapshot_keys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = _write_phase8_config(tmp_path)
    dates = pd.bdate_range("2024-01-01", periods=12)
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.03, include_removed_trade=False),
        daily_signals=None,
        sharpe_ratio=1.0,
        max_drawdown=0.10,
        annual_turnover=10.0,
        contradiction_loss_rate=1.0,
    )
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=0.01, include_removed_trade=False),
        daily_signals=_build_geo_daily_signals(dates),
        sharpe_ratio=1.0,
        max_drawdown=0.09,
        annual_turnover=10.0,
        contradiction_loss_rate=0.0,
    )

    def fake_load_phase2_baseline_backtest(active_config, trade_journal=None, geo_store=None, geo_snapshot=None):
        if active_config.geo.enabled:
            return geo_enabled
        return baseline

    monkeypatch.setattr("src.geo.evaluation.load_phase2_baseline_backtest", fake_load_phase2_baseline_backtest)
    duplicate_snapshot = pd.concat([_build_geo_snapshot(dates), _build_geo_snapshot(dates).head(1)], ignore_index=True)

    with pytest.raises(ValueError, match="snapshot keys"):
        run_geo_overlay_evaluation(config_path, geo_snapshot=duplicate_snapshot)


def test_build_geo_overlay_evaluation_report_fails_on_universe_mismatch() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.03, include_removed_trade=False),
        daily_signals=_build_geo_daily_signals(dates).loc[lambda frame: frame["ticker"] != "XLE"].reset_index(drop=True),
        sharpe_ratio=1.0,
        max_drawdown=0.10,
        annual_turnover=10.0,
        contradiction_loss_rate=1.0,
    )
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=0.01, include_removed_trade=False),
        daily_signals=_build_geo_daily_signals(dates),
        sharpe_ratio=1.0,
        max_drawdown=0.09,
        annual_turnover=10.0,
        contradiction_loss_rate=0.0,
    )

    with pytest.raises(ValueError, match="Universe parity"):
        build_geo_overlay_evaluation_report(
            baseline_result=baseline,
            geo_enabled_result=geo_enabled,
            annualization_days=252,
        )


def test_build_geo_overlay_evaluation_report_fails_on_calendar_mismatch() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    shorter_dates = dates[:-1]
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.03, include_removed_trade=False),
        daily_signals=_build_geo_daily_signals(dates),
        sharpe_ratio=1.0,
        max_drawdown=0.10,
        annual_turnover=10.0,
        contradiction_loss_rate=1.0,
    )
    geo_enabled = _make_backtest(
        dates=shorter_dates,
        daily_returns=[0.001] * len(shorter_dates),
        turnovers=[0.10] * len(shorter_dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=0.01, include_removed_trade=False),
        daily_signals=_build_geo_daily_signals(dates),
        sharpe_ratio=1.0,
        max_drawdown=0.09,
        annual_turnover=10.0,
        contradiction_loss_rate=0.0,
    )

    with pytest.raises(ValueError, match="Calendar parity"):
        build_geo_overlay_evaluation_report(
            baseline_result=baseline,
            geo_enabled_result=geo_enabled,
            annualization_days=252,
        )


def test_build_geo_overlay_evaluation_report_fails_on_market_data_parity_mismatch() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    baseline_signals = _build_geo_daily_signals(dates)
    geo_signals = _build_geo_daily_signals(dates)
    geo_signals.loc[
        (geo_signals["date"] == dates[3]) & (geo_signals["ticker"] == "XLK"),
        "forward_return",
    ] = 0.5
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.03, include_removed_trade=False),
        daily_signals=baseline_signals,
        sharpe_ratio=1.0,
        max_drawdown=0.10,
        annual_turnover=10.0,
        contradiction_loss_rate=1.0,
    )
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=0.01, include_removed_trade=False),
        daily_signals=geo_signals,
        sharpe_ratio=1.0,
        max_drawdown=0.09,
        annual_turnover=10.0,
        contradiction_loss_rate=0.0,
    )

    with pytest.raises(ValueError, match="Market data parity"):
        build_geo_overlay_evaluation_report(
            baseline_result=baseline,
            geo_enabled_result=geo_enabled,
            annualization_days=252,
        )


def test_build_geo_overlay_evaluation_report_accepts_market_data_within_numeric_tolerance() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    baseline_signals = _build_geo_daily_signals(dates)
    geo_signals = _build_geo_daily_signals(dates)
    geo_signals.loc[
        (geo_signals["date"] == dates[3]) & (geo_signals["ticker"] == "XLK"),
        "forward_return",
    ] += 1e-15
    geo_signals["ticker"] = geo_signals["ticker"].map(lambda value: f" {value.lower()} ")
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.03, include_removed_trade=False),
        daily_signals=baseline_signals,
        sharpe_ratio=1.0,
        max_drawdown=0.10,
        annual_turnover=10.0,
        contradiction_loss_rate=1.0,
    )
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=0.01, include_removed_trade=False),
        daily_signals=geo_signals,
        sharpe_ratio=1.0,
        max_drawdown=0.09,
        annual_turnover=10.0,
        contradiction_loss_rate=0.0,
    )

    report = build_geo_overlay_evaluation_report(
        baseline_result=baseline,
        geo_enabled_result=geo_enabled,
        annualization_days=252,
    )

    assert report["input_parity"]["passed"] is True


def test_build_geo_overlay_evaluation_report_rejects_duplicate_identifiers_after_normalization() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    bad_signals = _build_geo_daily_signals(dates)
    bad_signals.loc[
        (bad_signals["date"] == dates[0]) & (bad_signals["ticker"] == "XLE"),
        "ticker",
    ] = " xlk "
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.03, include_removed_trade=False),
        daily_signals=_build_geo_daily_signals(dates),
        sharpe_ratio=1.0,
        max_drawdown=0.10,
        annual_turnover=10.0,
        contradiction_loss_rate=1.0,
    )
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=0.01, include_removed_trade=False),
        daily_signals=bad_signals,
        sharpe_ratio=1.0,
        max_drawdown=0.09,
        annual_turnover=10.0,
        contradiction_loss_rate=0.0,
    )

    with pytest.raises(ValueError, match="normalized"):
        build_geo_overlay_evaluation_report(
            baseline_result=baseline,
            geo_enabled_result=geo_enabled,
            annualization_days=252,
        )


def test_build_geo_overlay_evaluation_report_treats_low_coverage_asset_days_as_neutral() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    low_coverage_signals = _build_geo_daily_signals(dates)
    low_coverage_mask = (low_coverage_signals["date"] == dates[9]) & (low_coverage_signals["ticker"] != "XLK")
    low_coverage_signals.loc[low_coverage_mask, "coverage_score"] = 0.10
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.03, include_removed_trade=False),
        daily_signals=_build_geo_daily_signals(dates),
        sharpe_ratio=1.0,
        max_drawdown=0.10,
        annual_turnover=10.0,
        contradiction_loss_rate=1.0,
    )
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=0.01, include_removed_trade=False),
        daily_signals=low_coverage_signals,
        sharpe_ratio=1.0,
        max_drawdown=0.09,
        annual_turnover=10.0,
        contradiction_loss_rate=0.0,
    )

    report = build_geo_overlay_evaluation_report(
        baseline_result=baseline,
        geo_enabled_result=geo_enabled,
        annualization_days=252,
    )

    high_stress_row = next(item for item in report["geo_stress_series"] if item["date"] == dates[9].date().isoformat())
    assert report["geo_stress"]["per_asset_coverage_floor"] == 0.5
    assert high_stress_row["geo_stress"] == 0.0


def test_build_geo_overlay_evaluation_report_flags_extreme_returns_in_data_sanity() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    extreme_signals = _build_geo_daily_signals(dates)
    extreme_mask = (extreme_signals["date"] == dates[6]) & (extreme_signals["ticker"] == "XLK")
    extreme_signals.loc[extreme_mask, "forward_return"] = 0.60
    baseline = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=-0.03, include_removed_trade=False),
        daily_signals=extreme_signals,
        sharpe_ratio=1.0,
        max_drawdown=0.10,
        annual_turnover=10.0,
        contradiction_loss_rate=1.0,
    )
    geo_enabled = _make_backtest(
        dates=dates,
        daily_returns=[0.001] * len(dates),
        turnovers=[0.10] * len(dates),
        trade_log=_build_trade_log(dates=dates, contradictory_return=0.01, include_removed_trade=False),
        daily_signals=extreme_signals.copy(),
        sharpe_ratio=1.0,
        max_drawdown=0.09,
        annual_turnover=10.0,
        contradiction_loss_rate=0.0,
    )

    report = build_geo_overlay_evaluation_report(
        baseline_result=baseline,
        geo_enabled_result=geo_enabled,
        annualization_days=252,
    )

    assert report["data_sanity"]["extreme_return_treated_identically"] is True
    assert report["data_sanity"]["extreme_return_days"] == [dates[6].date().isoformat()]
