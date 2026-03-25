from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.config_loader import load_config
from src.geo.regime_experiment import (
    GeoRegimePolicy,
    GeoRegimePolicyRun,
    _apply_policy_to_signals,
    _build_conclusions,
    run_geo_regime_stage1_experiment,
)
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


def _build_fake_baseline_result() -> TrendStrategyBacktest:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    daily_results = pd.DataFrame(
        {
            "date": dates,
            "gross_portfolio_return": [0.01, -0.02, 0.01],
            "net_portfolio_return": [0.009, -0.021, 0.009],
            "portfolio_return": [0.009, -0.021, 0.009],
            "gross_exposure": [0.2, 0.2, 0.0],
            "turnover": [0.2, 0.0, 0.2],
            "transaction_cost": [0.001, 0.0, 0.001],
        }
    )
    trade_log = pd.DataFrame(
        {
            "ticker": ["XLE"],
            "entry_date": [dates[0]],
            "exit_date": [dates[2]],
            "position_direction": [1],
            "entry_zscore": [-3.0],
            "exit_zscore": [0.1],
            "holding_days": [2],
            "entry_weight": [0.2],
            "gross_return": [-0.01],
            "net_return": [-0.011],
            "exit_reason": ["reversion"],
        }
    )
    daily_signals = pd.DataFrame(
        {
            "date": [dates[0], dates[1]],
            "ticker": ["XLE", "XLE"],
            "zscore": [3.0, 3.5],
            "signal_direction": [-1, -1],
            "target_position": [-0.2, -0.2],
            "regime_threshold_multiplier": [1.0, 1.0],
            "regime_position_scale": [1.0, 1.0],
            "allow_new_entries": [True, True],
            "forward_return": [-0.10, 0.02],
        }
    )
    return TrendStrategyBacktest(
        daily_positions=daily_signals.loc[:, ["date", "ticker", "target_position"]].copy(),
        daily_results=daily_results,
        trade_log=trade_log,
        monthly_results=pd.DataFrame(),
        summary_metrics={},
        daily_signals=daily_signals,
    )


def _write_matching_baseline_artifacts(processed_dir: Path, baseline_result: TrendStrategyBacktest) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    baseline_result.daily_results.to_csv(processed_dir / "phase5_strategy_a_daily.csv", index=False)
    baseline_result.trade_log.loc[
        :,
        [
            "ticker",
            "entry_date",
            "exit_date",
            "position_direction",
            "entry_zscore",
            "exit_zscore",
            "holding_days",
            "entry_weight",
            "gross_return",
            "net_return",
        ],
    ].assign(
        trade_id="fake-trade",
        run_id="fake-run",
    ).loc[
        :,
        [
            "trade_id",
            "run_id",
            "ticker",
            "entry_date",
            "exit_date",
            "position_direction",
            "entry_zscore",
            "exit_zscore",
            "holding_days",
            "entry_weight",
            "gross_return",
            "net_return",
        ],
    ].to_csv(processed_dir / "phase2_trade_log.csv", index=False)


def _write_snapshot(processed_dir: Path) -> Path:
    snapshot_path = processed_dir / "geo_feature_snapshot_minimal.csv"
    pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04"],
            "asset": ["XLE", "XLK", "XLE", "XLK", "XLE", "XLK"],
            "geo_net_score": [0.02, 0.02, 0.02, 0.02, 0.0, 0.0],
            "geo_structural_score": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "coverage_score": [0.57, 0.57, 0.57, 0.57, 0.57, 0.57],
        }
    ).to_csv(snapshot_path, index=False)
    return snapshot_path


def test_apply_policy_to_signals_changes_threshold_and_sizing_by_regime(tmp_path: Path) -> None:
    config_path = _write_phase8_config(tmp_path)
    config = load_config(config_path)
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    daily_signals = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["XLE", "XLE"],
            "zscore": [3.0, 3.5],
            "signal_direction": [-1, -1],
            "target_position": [-0.2, -0.2],
            "regime_threshold_multiplier": [1.0, 1.0],
            "regime_position_scale": [1.0, 1.0],
            "allow_new_entries": [True, True],
        }
    )
    regime_frame = pd.DataFrame(
        {
            "date": dates,
            "geo_regime": ["GEO_STRESS", "STRUCTURAL_GEO"],
        }
    )

    threshold_variant = _apply_policy_to_signals(
        daily_signals,
        regime_frame=regime_frame,
        policy=GeoRegimePolicy(
            name="threshold_only",
            stress_threshold_multiplier=1.10,
            structural_threshold_multiplier=1.25,
        ),
        phase2_config=config.phase2,
    )
    sizing_variant = _apply_policy_to_signals(
        daily_signals,
        regime_frame=regime_frame,
        policy=GeoRegimePolicy(
            name="sizing_only",
            stress_position_scale=0.75,
            structural_position_scale=0.50,
        ),
        phase2_config=config.phase2,
    )

    assert int(threshold_variant.loc[threshold_variant["date"] == dates[0], "signal_direction"].iloc[0]) == 0
    assert float(sizing_variant.loc[sizing_variant["date"] == dates[0], "target_position"].iloc[0]) == pytest.approx(-0.15)
    assert float(sizing_variant.loc[sizing_variant["date"] == dates[1], "target_position"].iloc[0]) == pytest.approx(-0.10)


def test_run_geo_regime_stage1_experiment_hard_fails_on_baseline_parity_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = _write_phase8_config(tmp_path)
    config = load_config(config_path)
    baseline_result = _build_fake_baseline_result()
    _write_matching_baseline_artifacts(config.paths.processed_dir, baseline_result)
    _write_snapshot(config.paths.processed_dir)

    daily_artifact = config.paths.processed_dir / "phase5_strategy_a_daily.csv"
    broken = pd.read_csv(daily_artifact)
    broken.loc[0, "net_portfolio_return"] = 99.0
    broken.to_csv(daily_artifact, index=False)

    monkeypatch.setattr("src.geo.regime_experiment.load_phase2_baseline_backtest", lambda active_config: baseline_result)

    with pytest.raises(ValueError, match="Baseline parity hard-failed"):
        run_geo_regime_stage1_experiment(config_path)


def test_build_conclusions_splits_research_signal_from_production_gate() -> None:
    baseline_by_regime = pd.DataFrame(
        [
            {
                "regime": "NORMAL",
                "date_count": 100,
                "trade_count": 20,
                "net_sharpe_after_costs": 0.60,
                "avg_net_pnl_per_trade": 0.010,
                "win_rate": 0.60,
                "abs_zscore_decay_per_day": 1.00,
            },
            {
                "regime": "GEO_STRESS",
                "date_count": 15,
                "trade_count": 8,
                "net_sharpe_after_costs": 0.20,
                "avg_net_pnl_per_trade": 0.004,
                "win_rate": 0.50,
                "abs_zscore_decay_per_day": 0.70,
            },
            {
                "regime": "STRUCTURAL_GEO",
                "date_count": 40,
                "trade_count": 10,
                "net_sharpe_after_costs": 0.10,
                "avg_net_pnl_per_trade": 0.005,
                "win_rate": 0.45,
                "abs_zscore_decay_per_day": 0.60,
            },
        ]
    )
    best_variant = GeoRegimePolicyRun(
        variant="threshold_only",
        label_source="REAL",
        daily_results=pd.DataFrame(),
        trade_log=pd.DataFrame(),
        daily_signals=pd.DataFrame(),
        metrics={
            "delta_net_sharpe_after_costs": 0.03,
            "delta_max_drawdown": 0.0,
            "delta_net_cvar_5": 0.0,
            "quiet_delta_net_sharpe_after_costs": -0.01,
            "removed_trade_fraction": 0.10,
            "avg_removed_trade_pnl": -0.001,
        },
    )
    placebo_run = GeoRegimePolicyRun(
        variant="threshold_only",
        label_source="shuffled_counts_preserved",
        daily_results=pd.DataFrame(),
        trade_log=pd.DataFrame(),
        daily_signals=pd.DataFrame(),
        metrics={"delta_net_sharpe_after_costs": 0.01},
    )
    bootstrap = {
        "delta_net_sharpe_after_costs": {"median": 0.02, "p25": -0.01},
        "delta_max_drawdown": {"median": 0.0},
        "delta_net_cvar_5": {"median": 0.0},
    }

    conclusions = _build_conclusions(
        baseline_by_regime=baseline_by_regime,
        best_real_variant=best_variant,
        placebo_runs=[placebo_run],
        bootstrap=bootstrap,
    )

    assert conclusions["research_signal_present"]["passed"] is True
    assert conclusions["production_gate_pass"]["passed"] is False


def test_run_geo_regime_stage1_experiment_writes_stage1_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = _write_phase8_config(tmp_path)
    config = load_config(config_path)
    baseline_result = _build_fake_baseline_result()
    _write_matching_baseline_artifacts(config.paths.processed_dir, baseline_result)
    _write_snapshot(config.paths.processed_dir)

    def fake_load_phase2_baseline_backtest(active_config):
        return baseline_result

    def fake_build_baseline_by_regime_frame(*, baseline_result, regime_frame, annualization_days):
        return pd.DataFrame(
            [
                {
                    "regime": "NORMAL",
                    "date_count": 100,
                    "active_signal_day_count": 20,
                    "trade_count": 20,
                    "net_sharpe_after_costs": 0.60,
                    "gross_sharpe": 0.70,
                    "max_drawdown": 0.05,
                    "net_cvar_5": -0.02,
                    "avg_net_pnl_per_trade": 0.010,
                    "avg_holding_days": 1.5,
                    "annual_turnover": 1000.0,
                    "win_rate": 0.60,
                    "abs_zscore_decay_per_day": 1.00,
                    "reversion_exit_rate": 0.80,
                    "worst_25_day_share": 0.50,
                    "worst_25_day_ratio": 1.00,
                },
                {
                    "regime": "GEO_STRESS",
                    "date_count": 20,
                    "active_signal_day_count": 5,
                    "trade_count": 8,
                    "net_sharpe_after_costs": 0.20,
                    "gross_sharpe": 0.30,
                    "max_drawdown": 0.03,
                    "net_cvar_5": -0.03,
                    "avg_net_pnl_per_trade": 0.003,
                    "avg_holding_days": 2.0,
                    "annual_turnover": 1200.0,
                    "win_rate": 0.50,
                    "abs_zscore_decay_per_day": 0.70,
                    "reversion_exit_rate": 0.60,
                    "worst_25_day_share": 0.20,
                    "worst_25_day_ratio": 1.00,
                },
                {
                    "regime": "STRUCTURAL_GEO",
                    "date_count": 30,
                    "active_signal_day_count": 6,
                    "trade_count": 10,
                    "net_sharpe_after_costs": 0.10,
                    "gross_sharpe": 0.20,
                    "max_drawdown": 0.06,
                    "net_cvar_5": -0.04,
                    "avg_net_pnl_per_trade": 0.004,
                    "avg_holding_days": 2.0,
                    "annual_turnover": 900.0,
                    "win_rate": 0.45,
                    "abs_zscore_decay_per_day": 0.60,
                    "reversion_exit_rate": 0.50,
                    "worst_25_day_share": 0.30,
                    "worst_25_day_ratio": 1.50,
                },
            ]
        )

    def fake_real_variants(*, baseline_result, regime_frame, phase2_config, annualization_days, baseline_lookup):
        threshold_run = GeoRegimePolicyRun(
            variant="threshold_only",
            label_source="REAL",
            daily_results=baseline_result.daily_results,
            trade_log=baseline_result.trade_log,
            daily_signals=baseline_result.daily_signals,
            metrics={
                "variant": "threshold_only",
                "label_source": "REAL",
                "net_sharpe_after_costs": 0.63,
                "delta_net_sharpe_after_costs": 0.03,
                "gross_sharpe": 0.72,
                "delta_gross_sharpe": 0.02,
                "max_drawdown": 0.05,
                "delta_max_drawdown": 0.0,
                "net_cvar_5": -0.02,
                "delta_net_cvar_5": 0.0,
                "trade_count": 1,
                "trade_count_delta_pct": 0.0,
                "avg_net_pnl_per_trade": 0.01,
                "avg_holding_days": 2.0,
                "annual_turnover": 900.0,
                "win_rate": 1.0,
                "quiet_net_sharpe_after_costs": 0.59,
                "quiet_delta_net_sharpe_after_costs": -0.01,
                "stress_net_sharpe_after_costs": 0.30,
                "structural_net_sharpe_after_costs": 0.20,
                "removed_trade_fraction": 0.10,
                "avg_removed_trade_pnl": -0.001,
            },
        )
        sizing_run = GeoRegimePolicyRun(
            variant="sizing_only",
            label_source="REAL",
            daily_results=baseline_result.daily_results,
            trade_log=baseline_result.trade_log,
            daily_signals=baseline_result.daily_signals,
            metrics={
                "variant": "sizing_only",
                "label_source": "REAL",
                "net_sharpe_after_costs": 0.61,
                "delta_net_sharpe_after_costs": 0.01,
                "gross_sharpe": 0.71,
                "delta_gross_sharpe": 0.01,
                "max_drawdown": 0.05,
                "delta_max_drawdown": 0.0,
                "net_cvar_5": -0.02,
                "delta_net_cvar_5": 0.0,
                "trade_count": 1,
                "trade_count_delta_pct": 0.0,
                "avg_net_pnl_per_trade": 0.01,
                "avg_holding_days": 2.0,
                "annual_turnover": 950.0,
                "win_rate": 1.0,
                "quiet_net_sharpe_after_costs": 0.58,
                "quiet_delta_net_sharpe_after_costs": -0.02,
                "stress_net_sharpe_after_costs": 0.25,
                "structural_net_sharpe_after_costs": 0.18,
                "removed_trade_fraction": 0.0,
                "avg_removed_trade_pnl": 0.0,
            },
        )
        return [threshold_run, sizing_run]

    def fake_placebo_variants(*, baseline_result, regime_frame, selected_variant, phase2_config, annualization_days, baseline_lookup):
        return [
            GeoRegimePolicyRun(
                variant="threshold_only",
                label_source="shuffled_counts_preserved",
                daily_results=baseline_result.daily_results,
                trade_log=baseline_result.trade_log,
                daily_signals=baseline_result.daily_signals,
                metrics={
                    "variant": "threshold_only",
                    "label_source": "shuffled_counts_preserved",
                    "delta_net_sharpe_after_costs": 0.01,
                },
            ),
            GeoRegimePolicyRun(
                variant="threshold_only",
                label_source="lag_broken_63d",
                daily_results=baseline_result.daily_results,
                trade_log=baseline_result.trade_log,
                daily_signals=baseline_result.daily_signals,
                metrics={
                    "variant": "threshold_only",
                    "label_source": "lag_broken_63d",
                    "delta_net_sharpe_after_costs": 0.00,
                },
            ),
        ]

    def fake_bootstrap(**kwargs):
        return {
            "delta_net_sharpe_after_costs": {"median": 0.02, "p25": -0.01},
            "delta_max_drawdown": {"median": 0.0},
            "delta_net_cvar_5": {"median": 0.0},
        }

    monkeypatch.setattr("src.geo.regime_experiment.load_phase2_baseline_backtest", fake_load_phase2_baseline_backtest)
    monkeypatch.setattr("src.geo.regime_experiment._build_baseline_by_regime_frame", fake_build_baseline_by_regime_frame)
    monkeypatch.setattr("src.geo.regime_experiment._evaluate_real_policy_variants", fake_real_variants)
    monkeypatch.setattr("src.geo.regime_experiment._evaluate_placebo_variants", fake_placebo_variants)
    monkeypatch.setattr("src.geo.regime_experiment._build_bootstrap_report", fake_bootstrap)

    result = run_geo_regime_stage1_experiment(config_path)

    assert result.output_path.exists()
    report = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert report["report_version"] == "geo_regime_stage1_v1"
    assert report["best_real_variant"]["variant"] == "threshold_only"
    assert report["conclusions"]["research_signal_present"]["passed"] is True
    assert report["conclusions"]["production_gate_pass"]["passed"] is False
    assert Path(report["artifacts"]["labels"]).exists()
    assert Path(report["artifacts"]["baseline_by_regime"]).exists()
    assert Path(report["artifacts"]["policy_results"]).exists()
    assert Path(report["artifacts"]["placebo_results"]).exists()
