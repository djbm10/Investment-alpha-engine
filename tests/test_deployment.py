import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from src.config_loader import load_config
from src.deployment import DeploymentManager


def test_deployment_manager_returns_mock_broker_for_backtest() -> None:
    config = load_config("config/phase8.yaml")
    deployment = DeploymentManager(config=replace(config, deployment=replace(config.deployment, mode="backtest")))

    client = deployment.get_broker_client()

    assert client.__class__.__name__ == "MockBrokerClient"


@patch("src.deployment.AlpacaBrokerClient")
def test_paper_mode_uses_paper_broker_url(mock_alpaca: MagicMock) -> None:
    config = load_config("config/phase8.yaml")
    deployment = DeploymentManager(config=replace(config, deployment=replace(config.deployment, mode="paper")))

    deployment.get_broker_client()

    patched_config = mock_alpaca.call_args.args[0]
    assert patched_config.phase7.alpaca_base_url == "https://paper-api.alpaca.markets"


def test_live_readiness_reports_missing_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "phase8.yaml"
    config_path.write_text(Path("config/phase8.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    state_path = data_dir / "phase7_state.json"
    state_path.write_text(json.dumps({"daily_records": [{"date": "2026-01-01"}] * 30}), encoding="utf-8")
    confirmation_path = tmp_path / "config" / "live_confirmed.txt"
    confirmation_path.parent.mkdir(parents=True)
    confirmation_path.write_text("2026-03-18\nsignature: test\n", encoding="utf-8")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["deployment"]["paper_state_path"] = str(state_path)
    payload["deployment"]["live_confirmation_path"] = str(confirmation_path)
    payload["deployment"]["phase7_gate_artifact"] = str(tmp_path / "config" / "phase7_cleared.yaml")
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_config(config_path)
    deployment = DeploymentManager(config)
    ready, issues = deployment.validate_live_readiness()

    assert ready is False
    assert any("fewer than 90" in issue for issue in issues)
    assert any("Phase 7 gate artifact" in issue for issue in issues)
