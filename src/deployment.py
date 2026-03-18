from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .broker_alpaca import AlpacaBrokerClient
from .broker_mock import MockBrokerClient
from .config_loader import PipelineConfig, load_config


class DeploymentManager:
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.mode = config.deployment.mode

    def get_broker_client(self) -> Any:
        if self.mode == "backtest":
            return MockBrokerClient(
                starting_equity=100_000.0,
                min_slippage_bps=self.config.phase7.mock_slippage_bps_min,
                max_slippage_bps=self.config.phase7.mock_slippage_bps_max,
            )
        if self.mode == "paper":
            return AlpacaBrokerClient(self._with_alpaca_url(self.PAPER_URL))
        if self.mode == "live":
            self._require_live_confirmation()
            return AlpacaBrokerClient(self._with_alpaca_url(self.LIVE_URL))
        raise ValueError(f"Unsupported deployment mode '{self.mode}'.")

    def validate_live_readiness(self) -> tuple[bool, list[str]]:
        issues: list[str] = []
        paper_state_path = self.config.deployment.paper_state_path
        if not paper_state_path.exists():
            issues.append(f"Paper trading state file not found at {paper_state_path}.")
        else:
            state = json.loads(paper_state_path.read_text(encoding="utf-8"))
            if len(state.get("daily_records", [])) < 90:
                issues.append("Paper trading history has fewer than 90 daily records.")

        if not self.config.deployment.phase7_gate_artifact.exists():
            issues.append(
                f"Phase 7 gate artifact not found at {self.config.deployment.phase7_gate_artifact}."
            )

        confirmation_issues = self._confirmation_issues()
        issues.extend(confirmation_issues)

        if not confirmation_issues:
            try:
                client = AlpacaBrokerClient(self._with_alpaca_url(self.LIVE_URL))
                equity = float(client.get_account().equity)
                if equity < self.config.deployment.min_capital:
                    issues.append(
                        f"Alpaca equity {equity:.2f} is below required minimum capital {self.config.deployment.min_capital:.2f}."
                    )
            except Exception as exc:
                issues.append(f"Unable to query Alpaca live account readiness: {exc}")

        return len(issues) == 0, issues

    def deploy_live(self, config_path: str | Path) -> tuple[bool, list[str]]:
        ready, issues = self.validate_live_readiness()
        if not ready:
            return False, issues

        config_file = Path(config_path)
        updated = config_file.read_text(encoding="utf-8").replace("mode: paper", "mode: live", 1)
        config_file.write_text(updated, encoding="utf-8")
        return True, []

    def _with_alpaca_url(self, base_url: str) -> PipelineConfig:
        return replace(
            self.config,
            phase7=replace(self.config.phase7, alpaca_base_url=base_url, mode=self.mode),
        )

    def _require_live_confirmation(self) -> None:
        issues = self._confirmation_issues()
        if issues:
            raise ValueError("; ".join(issues))

    def _confirmation_issues(self) -> list[str]:
        path = self.config.deployment.live_confirmation_path
        if not path.exists():
            return [f"Live confirmation file not found at {path}."]

        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        today = datetime.now().date().isoformat()
        if not lines or lines[0] != today:
            return [f"Live confirmation file must start with today's date ({today})."]
        if len(lines) < 2 or not lines[1].lower().startswith("signature:"):
            return ["Live confirmation file must include a second line beginning with 'signature:'."]
        return []


def check_live_readiness(config_path: str | Path) -> tuple[bool, list[str]]:
    config = load_config(config_path)
    manager = DeploymentManager(config)
    return manager.validate_live_readiness()


def deploy_live_mode(config_path: str | Path) -> tuple[bool, list[str]]:
    config = load_config(config_path)
    manager = DeploymentManager(config)
    return manager.deploy_live(config_path)
