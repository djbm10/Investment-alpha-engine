import math

import numpy as np
import pandas as pd

from src.config_loader import Phase2Config
from src.features import ExpandingFeatureScaler, FeatureBuilder


def _phase2_config() -> Phase2Config:
    return Phase2Config(
        lookback_window=40,
        diffusion_alpha=0.05,
        diffusion_steps=3,
        sigma_scale=1.0,
        min_weight=0.1,
        zscore_lookback=20,
        signal_threshold=2.0,
        tier2_fraction=0.75,
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


def _price_history() -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=220)
    tickers = ["XLK", "XLE", "XLV"]
    rows = []
    for ticker_idx, ticker in enumerate(tickers):
        for day_idx, date in enumerate(dates):
            trend = 100.0 + day_idx * (0.12 + 0.02 * ticker_idx)
            seasonal = math.sin(day_idx / (4.0 + ticker_idx)) * (1.0 + ticker_idx * 0.2)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adj_close": trend + seasonal,
                    "volume": 1_000_000 + ticker_idx * 50_000 + day_idx * 1_500,
                }
            )
    return pd.DataFrame(rows)


def test_feature_builder_returns_expected_matrix_shape() -> None:
    builder = FeatureBuilder(_phase2_config())
    state = builder.prepare_graph_engine_state(_price_history(), ["XLK", "XLE", "XLV"])
    target_date = sorted(pd.to_datetime(state.scaled_feature_frame["date"]).unique())[30]

    features = builder.build_features(target_date, state)

    assert features["feature_matrix"].shape == (3, len(state.feature_names))
    assert features["raw_matrix"].shape == (3, len(state.feature_names))
    assert list(features["asset_index"]) == ["XLK", "XLE", "XLV"]


def test_feature_builder_has_no_nans_after_initial_warmup() -> None:
    builder = FeatureBuilder(_phase2_config())
    state = builder.prepare_graph_engine_state(_price_history(), ["XLK", "XLE", "XLV"])
    feature_history = builder.build_feature_history(state)

    warm_dates = sorted(feature_history)[20:]
    assert warm_dates
    for date in warm_dates:
        assert not np.isnan(feature_history[date]).any()


def test_expanding_feature_scaler_uses_only_past_and_current_values() -> None:
    scaler = ExpandingFeatureScaler(["signal"])
    frame = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "ticker": "XLK", "signal": 1.0},
            {"date": pd.Timestamp("2024-01-03"), "ticker": "XLK", "signal": 2.0},
            {"date": pd.Timestamp("2024-01-04"), "ticker": "XLK", "signal": 100.0},
        ]
    )

    transformed = scaler.transform(frame)

    second_day = pd.Timestamp("2024-01-03")
    snapshot = scaler.snapshots[second_day]["signal"]
    assert snapshot["count"] == 2.0
    assert math.isclose(snapshot["mean"], 1.5)
    assert math.isclose(snapshot["std"], 0.5)
    assert math.isclose(float(transformed.loc[transformed["date"] == second_day, "signal"].iloc[0]), 1.0)
