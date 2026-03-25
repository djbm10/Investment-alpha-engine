from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.backtest import (
    apply_phase2_geo_overlay,
    apply_phase3_regime_overlay,
    apply_phase4_tcn_filter,
    run_walk_forward_backtest,
    scale_signals_to_risk_budget,
)
from src.config_loader import GeoConfig, GeoExposureFilesConfig, GeoHalfLivesConfig, Phase2Config, Phase3Config


def test_walk_forward_backtest_returns_summary_and_logs() -> None:
    dates = pd.bdate_range("2022-01-03", periods=320)
    rows = []
    for idx, date in enumerate(dates):
        paired_signal = idx % 40 == 0
        for ticker in ["XLK", "XLF"]:
            if paired_signal and ticker == "XLK":
                zscore = -2.0
            elif paired_signal and ticker == "XLF":
                zscore = 2.0
            else:
                zscore = 0.0
            signal_direction = 1 if zscore <= -1.5 else -1 if zscore >= 1.5 else 0
            target_position = signal_direction * 0.1
            forward_return = 0.002 if signal_direction != 0 else 0.0
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "current_return": 0.001,
                    "expected_return": 0.0,
                    "residual": 0.001,
                    "sigma": 1.0,
                    "edge_density": 0.5,
                    "forward_return": forward_return,
                    "zscore": zscore,
                    "signal_direction": signal_direction,
                    "target_position": target_position,
                }
            )

    daily_signals = pd.DataFrame(rows)
    config = Phase2Config(
        lookback_window=60,
        diffusion_alpha=0.05,
        diffusion_steps=3,
        sigma_scale=1.0,
        min_weight=0.1,
        zscore_lookback=60,
        signal_threshold=1.5,
        tier2_fraction=0.65,
        tier2_size_fraction=0.5,
        full_size_zscore=3.0,
        max_position_size=0.2,
        risk_budget_utilization=0.5,
        max_drawdown_limit=0.20,
        enforce_dollar_neutral=False,
        max_holding_days=10,
        stop_loss=0.05,
        min_training_months=6,
        annualization_days=252,
        commission_bps=0.0,
        bid_ask_bps=2.0,
        market_impact_bps=2.0,
        slippage_bps=1.0,
    )

    result = run_walk_forward_backtest(daily_signals, config, run_id="phase2-test")

    assert not result.daily_results.empty
    assert result.summary_metrics["total_signals"] > 0
    assert result.summary_metrics["total_trades"] >= 0
    assert "sharpe_ratio" in result.summary_metrics
    assert "annualized_return" in result.summary_metrics
    assert result.summary_metrics["corr_floor"] == 0.30
    assert result.summary_metrics["density_floor"] == 0.40


def test_walk_forward_backtest_keeps_single_sided_positions_when_not_neutral() -> None:
    dates = pd.bdate_range("2022-01-03", periods=320)
    rows = []
    for idx, date in enumerate(dates):
        zscore = -2.0 if idx % 20 == 0 else 0.0
        signal_direction = 1 if zscore <= -1.5 else 0
        target_position = signal_direction * 0.1
        rows.append(
            {
                "date": date,
                "ticker": "XLK",
                "current_return": 0.001,
                "expected_return": 0.0,
                "residual": 0.001,
                "sigma": 1.0,
                "edge_density": 0.5,
                "forward_return": 0.002 if signal_direction != 0 else 0.0,
                "zscore": zscore,
                "signal_direction": signal_direction,
                "target_position": target_position,
            }
        )

    daily_signals = pd.DataFrame(rows)
    config = Phase2Config(
        lookback_window=60,
        diffusion_alpha=0.05,
        diffusion_steps=3,
        sigma_scale=1.0,
        min_weight=0.1,
        zscore_lookback=60,
        signal_threshold=1.5,
        tier2_fraction=0.65,
        tier2_size_fraction=0.5,
        full_size_zscore=3.0,
        max_position_size=0.2,
        risk_budget_utilization=0.5,
        max_drawdown_limit=0.20,
        enforce_dollar_neutral=False,
        max_holding_days=10,
        stop_loss=0.05,
        min_training_months=6,
        annualization_days=252,
        commission_bps=0.0,
        bid_ask_bps=2.0,
        market_impact_bps=2.0,
        slippage_bps=1.0,
    )

    result = run_walk_forward_backtest(daily_signals, config, run_id="phase2-single-sided")

    assert result.summary_metrics["total_signals"] > 0
    assert result.summary_metrics["total_trades"] > 0


def test_scale_signals_to_risk_budget_increases_target_positions_when_drawdown_is_small() -> None:
    dates = pd.bdate_range("2022-01-03", periods=320)
    rows = []
    for idx, date in enumerate(dates):
        paired_signal = idx % 25 == 0
        for ticker in ["XLK", "XLF"]:
            if paired_signal and ticker == "XLK":
                zscore = -2.0
            elif paired_signal and ticker == "XLF":
                zscore = 2.0
            else:
                zscore = 0.0
            signal_direction = 1 if zscore <= -1.5 else -1 if zscore >= 1.5 else 0
            target_position = signal_direction * 0.05
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "current_return": 0.001,
                    "expected_return": 0.0,
                    "residual": 0.001,
                    "sigma": 1.0,
                    "edge_density": 0.5,
                    "forward_return": 0.0015 if signal_direction != 0 else 0.0,
                    "zscore": zscore,
                    "signal_direction": signal_direction,
                    "target_position": target_position,
                }
            )

    daily_signals = pd.DataFrame(rows)
    config = Phase2Config(
        lookback_window=60,
        diffusion_alpha=0.05,
        diffusion_steps=3,
        sigma_scale=1.0,
        min_weight=0.1,
        zscore_lookback=60,
        signal_threshold=1.5,
        tier2_fraction=0.65,
        tier2_size_fraction=0.5,
        full_size_zscore=3.0,
        max_position_size=0.2,
        risk_budget_utilization=0.5,
        max_drawdown_limit=0.20,
        enforce_dollar_neutral=False,
        max_holding_days=10,
        stop_loss=0.05,
        min_training_months=6,
        annualization_days=252,
        commission_bps=0.0,
        bid_ask_bps=2.0,
        market_impact_bps=2.0,
        slippage_bps=1.0,
    )

    scaling_result = scale_signals_to_risk_budget(daily_signals, config)

    assert scaling_result.scale_factor > 1.0
    assert scaling_result.target_max_drawdown == 0.10
    assert scaling_result.scaled_signals["target_position"].abs().max() > daily_signals["target_position"].abs().max()


def test_walk_forward_backtest_keeps_existing_position_during_no_trade_regime() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    daily_signals = pd.DataFrame(
        [
            {
                "date": dates[0],
                "ticker": "XLK",
                "current_return": 0.001,
                "expected_return": 0.0,
                "residual": 0.001,
                "sigma": 1.0,
                "edge_density": 0.5,
                "forward_return": 0.01,
                "zscore": -2.5,
                "signal_direction": 1,
                "target_position": 0.1,
            },
            {
                "date": dates[1],
                "ticker": "XLK",
                "current_return": 0.001,
                "expected_return": 0.0,
                "residual": 0.001,
                "sigma": 1.0,
                "edge_density": 0.1,
                "forward_return": 0.01,
                "zscore": -0.5,
                "signal_direction": 0,
                "target_position": 0.0,
            },
            {
                "date": dates[2],
                "ticker": "XLK",
                "current_return": 0.001,
                "expected_return": 0.0,
                "residual": 0.001,
                "sigma": 1.0,
                "edge_density": 0.1,
                "forward_return": 0.0,
                "zscore": 0.1,
                "signal_direction": 0,
                "target_position": 0.0,
            },
        ]
    )
    config = Phase2Config(
        lookback_window=60,
        diffusion_alpha=0.05,
        diffusion_steps=3,
        sigma_scale=1.0,
        min_weight=0.1,
        zscore_lookback=60,
        signal_threshold=1.5,
        tier2_fraction=0.65,
        tier2_size_fraction=0.5,
        full_size_zscore=3.0,
        max_position_size=0.2,
        risk_budget_utilization=0.5,
        max_drawdown_limit=0.20,
        enforce_dollar_neutral=False,
        max_holding_days=10,
        stop_loss=0.05,
        min_training_months=0,
        annualization_days=252,
        commission_bps=0.0,
        bid_ask_bps=2.0,
        market_impact_bps=2.0,
        slippage_bps=1.0,
        tier2_enabled=False,
        corr_floor=0.30,
        density_floor=0.40,
    )

    result = run_walk_forward_backtest(daily_signals, config, run_id="phase2-no-trade")

    assert len(result.trade_log) == 1
    assert result.trade_log.iloc[0]["holding_days"] == 2


def test_profitable_month_fraction_excludes_flat_inactive_months() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"])
    daily_signals = pd.DataFrame(
        [
            {
                "date": dates[0],
                "ticker": "XLK",
                "current_return": 0.001,
                "expected_return": 0.0,
                "residual": 0.001,
                "sigma": 1.0,
                "edge_density": 0.5,
                "forward_return": 0.01,
                "zscore": -2.5,
                "signal_direction": 1,
                "target_position": 0.1,
            },
            {
                "date": dates[1],
                "ticker": "XLK",
                "current_return": 0.001,
                "expected_return": 0.0,
                "residual": 0.001,
                "sigma": 1.0,
                "edge_density": 0.5,
                "forward_return": 0.0,
                "zscore": 0.1,
                "signal_direction": 0,
                "target_position": 0.0,
            },
            {
                "date": dates[2],
                "ticker": "XLK",
                "current_return": 0.0,
                "expected_return": 0.0,
                "residual": 0.0,
                "sigma": 1.0,
                "edge_density": 0.1,
                "forward_return": 0.0,
                "zscore": 0.0,
                "signal_direction": 0,
                "target_position": 0.0,
            },
            {
                "date": dates[3],
                "ticker": "XLK",
                "current_return": 0.0,
                "expected_return": 0.0,
                "residual": 0.0,
                "sigma": 1.0,
                "edge_density": 0.1,
                "forward_return": 0.0,
                "zscore": 0.0,
                "signal_direction": 0,
                "target_position": 0.0,
            },
        ]
    )
    config = Phase2Config(
        lookback_window=60,
        diffusion_alpha=0.05,
        diffusion_steps=3,
        sigma_scale=1.0,
        min_weight=0.1,
        zscore_lookback=60,
        signal_threshold=1.5,
        tier2_fraction=0.65,
        tier2_size_fraction=0.5,
        full_size_zscore=3.0,
        max_position_size=0.2,
        risk_budget_utilization=0.5,
        max_drawdown_limit=0.20,
        enforce_dollar_neutral=False,
        max_holding_days=10,
        stop_loss=0.05,
        min_training_months=0,
        annualization_days=252,
        commission_bps=0.0,
        bid_ask_bps=2.0,
        market_impact_bps=2.0,
        slippage_bps=1.0,
        tier2_enabled=False,
        corr_floor=0.30,
        density_floor=0.40,
    )

    result = run_walk_forward_backtest(daily_signals, config, run_id="phase2-flat-months")

    february = result.monthly_results.loc[result.monthly_results["test_month"] == pd.Timestamp("2024-02-01")].iloc[0]
    assert bool(february["active_month"]) is False
    assert result.summary_metrics["profitable_month_fraction"] == 1.0
    assert result.summary_metrics["active_out_of_sample_months"] == 1


def test_apply_phase3_regime_overlay_reduces_and_freezes_entries() -> None:
    daily_signals = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "ticker": "XLK",
                "zscore": -3.0,
                "regime_threshold_multiplier": 1.0,
                "regime_position_scale": 1.0,
                "allow_new_entries": True,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "ticker": "XLK",
                "zscore": -3.0,
                "regime_threshold_multiplier": 1.0,
                "regime_position_scale": 1.0,
                "allow_new_entries": True,
            },
        ]
    )
    phase2_config = Phase2Config(
        lookback_window=60,
        diffusion_alpha=0.05,
        diffusion_steps=3,
        sigma_scale=1.0,
        min_weight=0.1,
        zscore_lookback=60,
        signal_threshold=2.0,
        tier2_fraction=0.65,
        tier2_size_fraction=0.5,
        full_size_zscore=3.0,
        max_position_size=0.2,
        risk_budget_utilization=0.3,
        max_drawdown_limit=0.20,
        enforce_dollar_neutral=False,
        max_holding_days=9,
        stop_loss=0.05,
        min_training_months=12,
        annualization_days=252,
        commission_bps=0.0,
        bid_ask_bps=2.0,
        market_impact_bps=2.0,
        slippage_bps=1.0,
        tier2_enabled=False,
        corr_floor=0.30,
        density_floor=0.30,
        node_corr_floor=0.20,
    )
    phase3_config = Phase3Config(
        rolling_window=60,
        wasserstein_lookback=20,
        transition_threshold_sigma=1.5,
        new_regime_threshold_sigma=2.5,
        confirmation_window=5,
        confirmation_required_transition=3,
        confirmation_required_new_regime=4,
        transition_position_scale=0.75,
        transition_threshold_mult=1.10,
        new_regime_position_scale=0.5,
        new_regime_threshold_mult=1.25,
        transition_lookback_cap=30,
        new_regime_freeze_days=0,
        emergency_recalib_days=5,
        emergency_lookback=20,
    )

    overlaid = apply_phase3_regime_overlay(
        daily_signals,
        phase2_config,
        phase3_config,
        {
            pd.Timestamp("2024-01-02"): "TRANSITIONING",
            pd.Timestamp("2024-01-03"): "NEW_REGIME",
        },
    )

    transition_row = overlaid.loc[overlaid["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    new_regime_row = overlaid.loc[overlaid["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    assert abs(transition_row["target_position"] - 0.15) < 1e-12
    assert abs(new_regime_row["target_position"] - 0.1) < 1e-12


def test_apply_phase4_tcn_filter_vetoes_persistent_same_sign_predictions() -> None:
    daily_signals = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "ticker": "XLK",
                "signal_direction": 1,
                "target_position": 0.2,
                "residual": -0.04,
                "zscore": -3.0,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "ticker": "XLV",
                "signal_direction": -1,
                "target_position": -0.2,
                "residual": 0.05,
                "zscore": 2.5,
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "signal_date": pd.Timestamp("2024-01-02"),
                "ticker": "XLK",
                "predicted_residual_mean": -0.03,
                "predicted_residual_std": 0.02,
                "actual_next_residual": -0.01,
            },
            {
                "signal_date": pd.Timestamp("2024-01-02"),
                "ticker": "XLV",
                "predicted_residual_mean": -0.01,
                "predicted_residual_std": 0.03,
                "actual_next_residual": 0.01,
            },
        ]
    )

    filtered = apply_phase4_tcn_filter(daily_signals, predictions, reversion_confirm_threshold=0.5)

    vetoed = filtered.loc[filtered["ticker"] == "XLK"].iloc[0]
    retained = filtered.loc[filtered["ticker"] == "XLV"].iloc[0]
    assert bool(vetoed["tcn_veto"]) is True
    assert vetoed["target_position"] == 0.0
    assert bool(retained["tcn_veto"]) is False
    assert retained["target_position"] != 0.0


def test_apply_phase2_geo_overlay_preserves_baseline_when_geo_disabled() -> None:
    daily_signals = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "ticker": "XLK",
                "zscore": -2.5,
                "residual": -0.03,
                "signal_direction": 1,
                "target_position": 0.1,
                "forward_return": 0.01,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "ticker": "XLE",
                "zscore": 2.2,
                "residual": 0.02,
                "signal_direction": -1,
                "target_position": -0.1,
                "forward_return": -0.01,
            },
        ]
    )
    geo_snapshot = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp("2024-01-02").date(),
                "asset": "XLK",
                "geo_net_score": -1.0,
                "geo_structural_score": -1.0,
                "avg_mapping_confidence": 1.0,
                "coverage_score": 1.0,
                "data_freshness_minutes": 0,
            }
        ]
    )
    config = Phase2Config(
        lookback_window=60,
        diffusion_alpha=0.05,
        diffusion_steps=3,
        sigma_scale=1.0,
        min_weight=0.1,
        zscore_lookback=60,
        signal_threshold=1.5,
        tier2_fraction=0.65,
        tier2_size_fraction=0.5,
        full_size_zscore=3.0,
        max_position_size=0.2,
        risk_budget_utilization=0.5,
        max_drawdown_limit=0.20,
        enforce_dollar_neutral=False,
        max_holding_days=10,
        stop_loss=0.05,
        min_training_months=0,
        annualization_days=252,
        commission_bps=0.0,
        bid_ask_bps=2.0,
        market_impact_bps=2.0,
        slippage_bps=1.0,
    )
    geo_config = GeoConfig(
        enabled=False,
        optional_overlay=True,
        state_path=Path("data/processed/geo_freeze_state.json").resolve(),
        normalization_version="geo_norm_v1",
        mapping_version="geo_map_v1",
        cutoff_time_et="16:10",
        gamma=0.75,
        lambda_g=1.5,
        hard_override_threshold=0.8,
        min_mapping_confidence=0.7,
        min_coverage_score=0.7,
        half_life_days=GeoHalfLivesConfig(20, 10, 3, 2),
        exposure_files=GeoExposureFilesConfig(
            region=Path("config/geo/asset_region_exposure.csv").resolve(),
            sector=Path("config/geo/asset_sector_exposure.csv").resolve(),
            infra=Path("config/geo/asset_infra_exposure.csv").resolve(),
            betas=Path("config/geo/event_betas.csv").resolve(),
        ),
    )

    overlaid = apply_phase2_geo_overlay(daily_signals, config, geo_config, geo_snapshot=geo_snapshot)
    baseline_result = run_walk_forward_backtest(daily_signals, config, run_id="baseline-disabled-raw")
    overlaid_result = run_walk_forward_backtest(overlaid, config, run_id="baseline-disabled-geo")

    pd.testing.assert_series_equal(overlaid["target_position"], daily_signals["target_position"], check_names=False)
    pd.testing.assert_series_equal(overlaid["signal_direction"], daily_signals["signal_direction"], check_names=False)
    pd.testing.assert_frame_equal(
        baseline_result.daily_results.reset_index(drop=True),
        overlaid_result.daily_results.reset_index(drop=True),
        check_dtype=False,
    )


def test_apply_phase2_geo_overlay_uses_trade_date_snapshot_and_reduces_contradiction_losses() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4)
    daily_signals = pd.DataFrame(
        [
            {
                "date": dates[0],
                "ticker": "XLK",
                "zscore": -2.8,
                "residual": -0.03,
                "signal_direction": 1,
                "target_position": 0.1,
                "forward_return": -0.04,
            },
            {
                "date": dates[1],
                "ticker": "XLK",
                "zscore": -2.8,
                "residual": -0.03,
                "signal_direction": 1,
                "target_position": 0.1,
                "forward_return": -0.03,
            },
            {
                "date": dates[2],
                "ticker": "XLK",
                "zscore": -2.8,
                "residual": -0.03,
                "signal_direction": 1,
                "target_position": 0.1,
                "forward_return": 0.02,
            },
            {
                "date": dates[3],
                "ticker": "XLK",
                "zscore": 0.1,
                "residual": 0.0,
                "signal_direction": 0,
                "target_position": 0.0,
                "forward_return": 0.0,
            },
        ]
    )
    geo_snapshot = pd.DataFrame(
        [
            {
                "trade_date": dates[0].date(),
                "asset": "XLK",
                "geo_net_score": -1.0,
                "geo_structural_score": -1.0,
                "avg_mapping_confidence": 0.95,
                "coverage_score": 0.95,
                "data_freshness_minutes": 30,
            },
            {
                "trade_date": dates[1].date(),
                "asset": "XLK",
                "geo_net_score": -0.1,
                "geo_structural_score": -0.1,
                "avg_mapping_confidence": 0.95,
                "coverage_score": 0.95,
                "data_freshness_minutes": 30,
            },
            {
                "trade_date": dates[2].date(),
                "asset": "XLK",
                "geo_net_score": 0.0,
                "geo_structural_score": 0.0,
                "avg_mapping_confidence": 0.95,
                "coverage_score": 0.95,
                "data_freshness_minutes": 30,
            },
        ]
    )
    config = Phase2Config(
        lookback_window=60,
        diffusion_alpha=0.05,
        diffusion_steps=3,
        sigma_scale=1.0,
        min_weight=0.1,
        zscore_lookback=60,
        signal_threshold=1.5,
        tier2_fraction=0.65,
        tier2_size_fraction=0.5,
        full_size_zscore=3.0,
        max_position_size=0.2,
        risk_budget_utilization=0.5,
        max_drawdown_limit=0.20,
        enforce_dollar_neutral=False,
        max_holding_days=10,
        stop_loss=0.50,
        min_training_months=0,
        annualization_days=252,
        commission_bps=0.0,
        bid_ask_bps=0.0,
        market_impact_bps=0.0,
        slippage_bps=0.0,
    )
    geo_config = GeoConfig(
        enabled=True,
        optional_overlay=True,
        state_path=Path("data/processed/geo_freeze_state.json").resolve(),
        normalization_version="geo_norm_v1",
        mapping_version="geo_map_v1",
        cutoff_time_et="16:10",
        gamma=0.75,
        lambda_g=1.5,
        hard_override_threshold=0.8,
        min_mapping_confidence=0.7,
        min_coverage_score=0.7,
        half_life_days=GeoHalfLivesConfig(20, 10, 3, 2),
        exposure_files=GeoExposureFilesConfig(
            region=Path("config/geo/asset_region_exposure.csv").resolve(),
            sector=Path("config/geo/asset_sector_exposure.csv").resolve(),
            infra=Path("config/geo/asset_infra_exposure.csv").resolve(),
            betas=Path("config/geo/event_betas.csv").resolve(),
        ),
    )

    disabled_geo_config = replace(geo_config, enabled=False)
    baseline_signals = apply_phase2_geo_overlay(
        daily_signals,
        config,
        disabled_geo_config,
        geo_snapshot=geo_snapshot,
    )
    baseline_result = run_walk_forward_backtest(baseline_signals, config, run_id="baseline-geo")
    overlaid_signals = apply_phase2_geo_overlay(daily_signals, config, geo_config, geo_snapshot=geo_snapshot)
    overlaid_result = run_walk_forward_backtest(overlaid_signals, config, run_id="enabled-geo")

    assert overlaid_signals.loc[overlaid_signals["date"] == dates[0], "target_position"].iloc[0] == 0.0
    assert overlaid_signals.loc[overlaid_signals["date"] == dates[1], "target_position"].iloc[0] > 0.0
    assert not overlaid_result.daily_results.equals(baseline_result.daily_results)
    assert overlaid_result.summary_metrics["max_drawdown"] <= baseline_result.summary_metrics["max_drawdown"]
    assert (
        overlaid_result.summary_metrics["contradiction_loss_rate"]
        <= baseline_result.summary_metrics["contradiction_loss_rate"]
    )
