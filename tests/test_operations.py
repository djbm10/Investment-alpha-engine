import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import yaml

from src.config_loader import load_config
from src.operations import emergency_halt, verify_phase7_gate
from src.performance_tracker import PerformanceTracker


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


def test_verify_phase7_gate_passes_with_complete_synthetic_history(tmp_path: Path) -> None:
    config_path = _write_phase8_config(tmp_path)
    config = load_config(config_path)
    processed_dir = config.paths.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    business_dates = pd.bdate_range("2025-12-17", "2026-03-16")
    pd.DataFrame({"date": business_dates}).to_csv(processed_dir / "sector_etf_prices_validated.csv", index=False)
    pd.DataFrame({"date": business_dates}).to_csv(processed_dir / "trend_universe_prices_validated.csv", index=False)

    expected_frame = pd.DataFrame({"date": business_dates})
    expected_frame["week"] = expected_frame["date"].dt.strftime("%G-%V")
    expected_frame["month"] = expected_frame["date"].dt.strftime("%Y-%m")
    weekly_due = set(expected_frame.groupby("week")["date"].max().dt.date.astype(str).tolist())
    monthly_due = set(expected_frame.groupby("month")["date"].max().dt.date.astype(str).tolist())

    records = []
    for date in business_dates:
        iso_date = date.date().isoformat()
        learning_actions: list[str] = []
        if iso_date in weekly_due:
            learning_actions.append(f"weekly_mistake_analysis:/tmp/{iso_date}.md")
        if iso_date in monthly_due:
            learning_actions.append(f"monthly_bayesian_update:/tmp/{iso_date}.json")
        records.append(
            {
                "date": iso_date,
                "portfolio_value": 100_000.0,
                "day_pnl": 0.0,
                "week_pnl": 0.0,
                "month_pnl": 0.0,
                "allocation_strategy_a": 0.6,
                "allocation_strategy_b": 0.4,
                "circuit_action": "CONTINUE",
                "manual_review": False,
                "learning_actions": learning_actions,
            }
        )
    config.deployment.paper_state_path.parent.mkdir(parents=True, exist_ok=True)
    config.deployment.paper_state_path.write_text(json.dumps({"daily_records": records}, indent=2), encoding="utf-8")

    tracker = PerformanceTracker(config.paths.project_root / "data" / "performance.db", config=config)
    try:
        for idx, date in enumerate(business_dates):
            tracker.record_daily(
                date=date.date().isoformat(),
                portfolio_value=100_000.0 + idx,
                daily_pnl=1.0,
                positions={},
                allocation_weights={"strategy_a": 0.6, "strategy_b": 0.4},
                regime_state="TRADEABLE",
                risk_headroom={"daily_loss_pct_remaining": 0.02, "weekly_loss_pct_remaining": 0.05, "monthly_loss_pct_remaining": 0.1},
                health_status="HEALTHY",
                tracking_error_pct=0.001,
                spy_return=0.0005,
            )
    finally:
        tracker.close()

    result = verify_phase7_gate(config_path)

    assert result.gate_passed is True
    assert result.critical_day_count == 0
    assert result.learning_loops_on_schedule is True
    assert (tmp_path / "config" / "phase7_cleared.yaml").exists()


@patch("src.operations.DeploymentManager")
def test_emergency_halt_creates_flag_and_closes_positions(mock_manager_cls: MagicMock, tmp_path: Path) -> None:
    config_path = _write_phase8_config(tmp_path)
    client = MagicMock()
    client.close_all_positions.return_value = [{"order_id": "1", "symbol": "SPY"}]
    mock_manager_cls.return_value.get_broker_client.return_value = client

    result = emergency_halt(config_path, reason="test_halt")

    assert Path(result["halt_flag_path"]).exists()
    client.close_all_positions.assert_called_once()
    assert result["closed_positions"] == [{"order_id": "1", "symbol": "SPY"}]
