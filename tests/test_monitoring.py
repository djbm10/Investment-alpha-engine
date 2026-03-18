from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from src.config_loader import load_config
from src.monitoring import Monitor


def _make_monitor(tmp_path: Path) -> Monitor:
    config = load_config("config/phase8.yaml")
    paths = replace(
        config.paths,
        log_dir=tmp_path / "logs",
        pipeline_log_file=tmp_path / "logs" / "pipeline.log",
        postgres_log_file=tmp_path / "logs" / "postgres.log",
    )
    return Monitor(replace(config, paths=paths))


def _result(**overrides: object) -> SimpleNamespace:
    payload = {
        "date": "2026-03-18",
        "approved_orders": [],
        "fills": [],
        "discrepancies": [],
        "alerts": [],
        "manual_review": False,
        "aborted": False,
        "learning_actions": [],
        "daily_summary": {
            "date": "2026-03-18",
            "portfolio_value": 100_000.0,
            "day_pnl": 0.0,
            "week_pnl": 0.0,
            "month_pnl": 0.0,
            "ytd_pnl": 0.0,
            "risk_limit_headroom": {
                "daily_loss_pct_remaining": 0.02,
                "weekly_loss_pct_remaining": 0.05,
                "monthly_loss_pct_remaining": 0.10,
            },
            "allocation_weights": {"strategy_a": 0.6, "strategy_b": 0.4},
            "capital_scale_factor": 1.0,
            "regime_state": "TRADEABLE",
            "positions": {},
            "tracking_error_pct": 0.0,
        },
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_send_alert_writes_alert_logs(tmp_path: Path) -> None:
    monitor = _make_monitor(tmp_path)

    monitor.send_alert("CRITICAL", "Test critical alert", {"detail": "value"})

    alerts_log = (tmp_path / "logs" / "alerts.log").read_text(encoding="utf-8")
    critical_log = (tmp_path / "logs" / "critical_alerts.log").read_text(encoding="utf-8")
    assert "Test critical alert" in alerts_log
    assert "Test critical alert" in critical_log


def test_daily_health_check_classifies_degraded_conditions(tmp_path: Path) -> None:
    monitor = _make_monitor(tmp_path)
    result = _result(
        manual_review=True,
        alerts=["Weekly mistake analysis failed: boom"],
        daily_summary={
            **_result().daily_summary,
            "tracking_error_pct": 0.012,
            "risk_limit_headroom": {
                "daily_loss_pct_remaining": 0.009,
                "weekly_loss_pct_remaining": 0.05,
                "monthly_loss_pct_remaining": 0.10,
            },
        },
    )

    health = monitor.daily_health_check(result)

    assert health.status == "DEGRADED"
    assert "manual_review_required" in health.issues
    assert "weekly_learning_missed" in health.issues


def test_generate_daily_email_writes_report_file(tmp_path: Path) -> None:
    monitor = _make_monitor(tmp_path)
    result = _result()
    health = monitor.daily_health_check(result)

    report_path = monitor.generate_daily_email(result, health)

    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Health: HEALTHY" in report_text
    assert "Allocation weights" in report_text
