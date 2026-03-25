from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.config_loader import load_config
from src.geo.evaluation import GeoOverlayEvaluationResult
from src.geo.variant_experiments import _load_snapshot_frame, run_geo_snapshot_variant_experiments


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


def test_run_geo_snapshot_variant_experiments_writes_variant_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = _write_phase8_config(tmp_path)
    build_calls: list[str] = []

    def fake_write_minimal_geo_feature_snapshot(
        *,
        output_path: str | Path,
        tickers: list[str],
        start_date,
        end_date,
        reference_price_history=None,
        profile: str = "base",
    ) -> Path:
        build_calls.append(profile)
        frame = pd.DataFrame(
            {
                "trade_date": ["2024-01-02"],
                "asset": [tickers[0]],
                "snapshot_cutoff_at": ["2024-01-02T16:10:00-05:00"],
                "geo_net_raw": [0.1],
                "geo_net_score": [0.1 if profile == "A" else 0.2 if profile == "B" else 0.05],
                "geo_structural_score": [0.05 if profile == "A" else 0.1 if profile == "B" else 0.03],
                "geo_harm_score": [0.0],
                "geo_break_risk": [0.0],
                "geo_velocity_3d": [0.0],
                "geo_cluster_72h": [1.0],
                "region_stress": [0.2],
                "sector_disruption": [0.1],
                "infra_disruption": [0.0],
                "sanctions_score": [0.0],
                "avg_mapping_confidence": [0.7],
                "coverage_score": [0.75],
                "data_freshness_minutes": [60],
                "hard_override": [False],
                "contributing_event_ids": [json.dumps([f"{profile}-event"])],
                "profile": [profile],
            }
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        return output

    def fake_run_geo_overlay_evaluation(config_path_arg, *, geo_snapshot=None):
        assert geo_snapshot is not None
        profile = str(geo_snapshot["profile"].iloc[0])
        report = {
            "acceptance": {"passed": profile == "B"},
            "comparison": {
                "sharpe_ratio": {"delta": 0.01 if profile == "B" else -0.02},
                "max_drawdown": {"delta": -0.01 if profile == "B" else 0.02},
                "annual_turnover": {"delta": 0.0},
                "contradiction_loss_rate": {"delta": -0.10 if profile == "B" else 0.05},
            },
            "slices": {
                "high_geo_blocks": {
                    "baseline": {"contradiction_loss_rate": 0.20, "max_drawdown": 0.06},
                    "geo_enabled": {
                        "contradiction_loss_rate": 0.05 if profile == "B" else 0.25,
                        "max_drawdown": 0.05 if profile == "B" else 0.07,
                    },
                },
                "normal_periods": {
                    "baseline": {"sharpe_ratio": 0.60},
                    "geo_enabled": {"sharpe_ratio": 0.61 if profile == "B" else 0.55},
                },
            },
            "filtering_impact": {
                "removed_trade_count": 1 if profile == "B" else 2,
                "removed_trade_fraction": 0.05 if profile == "B" else 0.15,
                "avg_net_pnl_removed": -0.01 if profile == "B" else 0.01,
            },
            "bootstrap": {
                "delta_sharpe": {"median": 0.01 if profile == "B" else -0.03, "p25": -0.01 if profile == "B" else -0.08},
            },
        }
        config = load_config(config_path_arg)
        default_output = config.paths.processed_dir / "geo_overlay_evaluation.json"
        default_output.parent.mkdir(parents=True, exist_ok=True)
        default_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return GeoOverlayEvaluationResult(report=report, output_path=default_output)

    monkeypatch.setattr(
        "src.geo.variant_experiments.write_minimal_geo_feature_snapshot",
        fake_write_minimal_geo_feature_snapshot,
    )
    monkeypatch.setattr(
        "src.geo.variant_experiments.run_geo_overlay_evaluation",
        fake_run_geo_overlay_evaluation,
    )

    result = run_geo_snapshot_variant_experiments(
        config_path,
        profiles=("A", "B", "C"),
        output_dir=tmp_path / "artifacts",
        start_date="2024-01-02",
        end_date="2024-01-05",
    )

    assert build_calls == ["A", "B", "C"]
    assert result.output_path == tmp_path / "artifacts" / "geo_overlay_variant_comparison.json"
    assert result.output_path.exists()

    report = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert report["profiles"] == ["A", "B", "C"]
    assert report["best_profile"] == "B"
    assert report["results"]["B"]["acceptance_passed"] is True
    assert report["results"]["A"]["acceptance_passed"] is False
    assert report["results"]["B"]["snapshot_path"].endswith("geo_feature_snapshot_minimal_b.csv")
    assert report["results"]["C"]["report_path"].endswith("geo_overlay_evaluation_c.json")


def test_load_snapshot_frame_parses_mixed_dst_offsets_without_failure(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "mixed_offsets.csv"
    pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-06-03"],
            "asset": ["XLK", "XLK"],
            "snapshot_cutoff_at": [
                "2024-01-02T16:10:00-05:00",
                "2024-06-03T16:10:00-04:00",
            ],
        }
    ).to_csv(snapshot_path, index=False)

    frame = _load_snapshot_frame(snapshot_path)

    assert frame["snapshot_cutoff_at"].notna().all()
    assert str(frame["snapshot_cutoff_at"].dt.tz) == "UTC"
