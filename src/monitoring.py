from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HealthCheckResult:
    status: str
    issues: list[str]
    tracking_error_pct: float


class Monitor:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.alert_methods = ["console", "file"]
        self.alert_log_path = config.paths.log_dir / "alerts.log"
        self.critical_alert_log_path = config.paths.log_dir / "critical_alerts.log"
        self.daily_report_dir = config.paths.log_dir / "daily_reports"
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_report_dir.mkdir(parents=True, exist_ok=True)

    def send_alert(self, severity: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        normalized = severity.upper()
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": normalized,
            "message": message,
            "details": details or {},
        }
        line = json.dumps(payload, default=str)
        self._append_line(self.alert_log_path, line)
        print(f"[{normalized}] {message}")
        if normalized == "CRITICAL":
            self._append_line(self.critical_alert_log_path, line)
        return payload

    def daily_health_check(self, pipeline_result: Any) -> HealthCheckResult:
        summary = dict(getattr(pipeline_result, "daily_summary", {}) or {})
        issues: list[str] = []
        status = "HEALTHY"
        tracking_error_pct = float(summary.get("tracking_error_pct", 0.0))

        if getattr(pipeline_result, "aborted", False):
            status = "CRITICAL"
            issues.append("pipeline_aborted")
        if getattr(pipeline_result, "manual_review", False):
            status = self._escalate(status, "DEGRADED")
            issues.append("manual_review_required")

        approved_orders = getattr(pipeline_result, "approved_orders", [])
        fills = getattr(pipeline_result, "fills", [])
        if len(fills) < len(approved_orders):
            status = "CRITICAL"
            issues.append("unfilled_orders")

        if getattr(pipeline_result, "discrepancies", []):
            status = "CRITICAL"
            issues.append("position_discrepancies")

        if tracking_error_pct > 0.02:
            status = "CRITICAL"
            issues.append("tracking_error_above_2pct")
        elif tracking_error_pct > 0.01:
            status = self._escalate(status, "DEGRADED")
            issues.append("tracking_error_above_1pct")

        risk_headroom = summary.get("risk_limit_headroom", {})
        thresholds = {
            "daily_loss_pct_remaining": float(self.config.phase7.risk_limits.max_daily_loss_pct),
            "weekly_loss_pct_remaining": float(self.config.phase7.risk_limits.max_weekly_loss_pct),
            "monthly_loss_pct_remaining": float(self.config.phase7.risk_limits.max_monthly_loss_pct),
        }
        for key, threshold in thresholds.items():
            remaining = float(risk_headroom.get(key, threshold))
            if remaining < 0.0:
                status = "CRITICAL"
                issues.append(f"{key}_breached")
            elif remaining <= threshold * 0.5:
                status = self._escalate(status, "DEGRADED")
                issues.append(f"{key}_within_50pct")

        alerts = [str(alert) for alert in getattr(pipeline_result, "alerts", [])]
        if any("Weekly mistake analysis failed" in alert for alert in alerts):
            status = self._escalate(status, "DEGRADED")
            issues.append("weekly_learning_missed")
        if any("Monthly Bayesian update failed" in alert for alert in alerts):
            status = self._escalate(status, "DEGRADED")
            issues.append("monthly_learning_missed")

        return HealthCheckResult(status=status, issues=issues, tracking_error_pct=tracking_error_pct)

    def generate_daily_email(self, pipeline_result: Any, health_result: HealthCheckResult) -> Path:
        summary = dict(getattr(pipeline_result, "daily_summary", {}) or {})
        date_label = str(summary.get("date", "unknown"))
        output_path = self.daily_report_dir / f"report_{date_label}.txt"
        lines = [
            f"Date: {date_label}",
            f"Health: {health_result.status}",
            f"Portfolio value: {summary.get('portfolio_value', 0.0)}",
            f"Day P&L: {summary.get('day_pnl', 0.0)}",
            f"Week P&L: {summary.get('week_pnl', 0.0)}",
            f"Month P&L: {summary.get('month_pnl', 0.0)}",
            f"YTD P&L: {summary.get('ytd_pnl', 0.0)}",
            f"Risk headroom: {summary.get('risk_limit_headroom', {})}",
            f"Regime state: {summary.get('regime_state', 'UNKNOWN')}",
            f"Allocation weights: {summary.get('allocation_weights', {})}",
            f"Capital scale factor: {summary.get('capital_scale_factor', 1.0)}",
            f"Tracking error pct: {health_result.tracking_error_pct:.6f}",
            f"Alerts: {getattr(pipeline_result, 'alerts', [])}",
            f"Learning actions: {getattr(pipeline_result, 'learning_actions', [])}",
            f"Health issues: {health_result.issues}",
            f"Positions: {summary.get('positions', {})}",
        ]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    def _append_line(self, path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _escalate(self, current: str, candidate: str) -> str:
        order = {"HEALTHY": 0, "DEGRADED": 1, "CRITICAL": 2}
        return candidate if order[candidate] > order[current] else current
