from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config_loader import Phase2Config
from .correlation_filter import (
    average_pairwise_correlation,
    classify_universe_regime,
    graph_density,
    node_average_correlations,
    node_tradeable_mask,
    regime_controls,
)


@dataclass(frozen=True)
class GraphMatrixSnapshot:
    correlation_matrix: np.ndarray
    distance_matrix: np.ndarray


def build_price_matrix(price_history: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    price_matrix = (
        price_history.pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
        .reindex(columns=tickers)
    )
    return price_matrix.dropna(how="any")


def compute_daily_graph_matrices(
    price_history: pd.DataFrame,
    tickers: list[str],
    lookback_window: int,
) -> dict[pd.Timestamp, GraphMatrixSnapshot]:
    price_matrix = build_price_matrix(price_history, tickers)
    log_returns = np.log(price_matrix / price_matrix.shift(1)).dropna(how="any")

    if len(log_returns) < lookback_window:
        raise ValueError("Not enough return history to build graph matrices.")

    snapshots: dict[pd.Timestamp, GraphMatrixSnapshot] = {}
    for end_idx in range(lookback_window - 1, len(log_returns)):
        window = log_returns.iloc[end_idx - lookback_window + 1 : end_idx + 1]
        current_date = pd.Timestamp(log_returns.index[end_idx])
        correlation = window.corr().to_numpy(dtype=float)
        correlation = np.nan_to_num(correlation, nan=0.0)
        np.fill_diagonal(correlation, 1.0)
        distance = _correlation_to_distance(correlation)
        snapshots[current_date] = GraphMatrixSnapshot(
            correlation_matrix=correlation,
            distance_matrix=distance,
        )

    return snapshots


def compute_graph_signals(
    price_history: pd.DataFrame,
    tickers: list[str],
    config: Phase2Config,
    graph_matrices: dict[pd.Timestamp, GraphMatrixSnapshot] | None = None,
) -> pd.DataFrame:
    price_matrix = build_price_matrix(price_history, tickers)
    log_returns = np.log(price_matrix / price_matrix.shift(1)).dropna(how="any")
    forward_returns = price_matrix.pct_change().shift(-1)

    if len(log_returns) < config.lookback_window:
        raise ValueError("Not enough return history to build the graph engine.")

    rows: list[dict[str, object]] = []
    active_graph_matrices = (
        graph_matrices
        if graph_matrices is not None
        else compute_daily_graph_matrices(price_history, tickers, config.lookback_window)
    )
    for end_idx in range(config.lookback_window - 1, len(log_returns)):
        window = log_returns.iloc[end_idx - config.lookback_window + 1 : end_idx + 1]
        current_date = log_returns.index[end_idx]
        current_return = window.iloc[-1].to_numpy(dtype=float)

        snapshot = active_graph_matrices[pd.Timestamp(current_date)]
        correlation = snapshot.correlation_matrix
        avg_pairwise_corr = average_pairwise_correlation(correlation)
        node_avg_corr = node_average_correlations(correlation)
        node_tradeable = node_tradeable_mask(correlation, config)

        distance = snapshot.distance_matrix
        sigma = _scaled_sigma(_estimate_sigma(distance), config.sigma_scale)
        weights = _distance_to_weights(distance, sigma, config.min_weight)
        density = graph_density(weights)
        regime_state = classify_universe_regime(avg_pairwise_corr, density, config)
        regime_threshold_multiplier, regime_position_scale, universe_allow_entries = regime_controls(
            regime_state
        )
        laplacian = _normalized_laplacian(weights)
        expected_return = _apply_diffusion_filter(
            current_return,
            laplacian,
            alpha=config.diffusion_alpha,
            steps=config.diffusion_steps,
        )
        residual = current_return - expected_return
        forward_row = forward_returns.loc[current_date]

        for ticker_idx, ticker in enumerate(log_returns.columns):
            rows.append(
                {
                    "date": current_date,
                    "ticker": ticker,
                    "current_return": float(current_return[ticker_idx]),
                    "expected_return": float(expected_return[ticker_idx]),
                    "residual": float(residual[ticker_idx]),
                    "sigma": float(sigma),
                    "avg_pairwise_corr": float(avg_pairwise_corr),
                    "node_avg_corr": float(node_avg_corr[ticker_idx]),
                    "node_tradeable": bool(node_tradeable[ticker_idx]),
                    "graph_density": float(density),
                    "edge_density": float(density),
                    "graph_regime": regime_state,
                    "regime_threshold_multiplier": float(regime_threshold_multiplier),
                    "regime_position_scale": float(regime_position_scale),
                    "allow_new_entries": bool(universe_allow_entries and node_tradeable[ticker_idx]),
                    "forward_return": _optional_float(forward_row[ticker]),
                }
            )

    signals = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    signals["zscore"] = signals.groupby("ticker", group_keys=False)["residual"].transform(
        lambda series: _rolling_zscore(series, config.zscore_lookback)
    )
    return apply_signal_rules(signals, config)


def apply_signal_rules(signals: pd.DataFrame, config: Phase2Config) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()

    applied = signals.copy()
    abs_zscores = applied["zscore"].abs().fillna(0.0)
    threshold_multiplier = applied.get("regime_threshold_multiplier", pd.Series(1.0, index=applied.index))
    position_scale = applied.get("regime_position_scale", pd.Series(1.0, index=applied.index))
    allow_new_entries = applied.get("allow_new_entries", pd.Series(True, index=applied.index)).astype(bool)
    strong_threshold = config.signal_threshold * threshold_multiplier
    if config.tier2_enabled:
        tier2_threshold = strong_threshold * config.tier2_fraction
        applied["signal_direction"] = np.select(
            [
                allow_new_entries & (applied["zscore"] <= -tier2_threshold),
                allow_new_entries & (applied["zscore"] >= tier2_threshold),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        applied["signal_tier"] = np.select(
            [abs_zscores >= strong_threshold, abs_zscores >= tier2_threshold],
            [2, 1],
            default=0,
        ).astype(int)
        size_fraction = np.select(
            [applied["signal_tier"] == 2, applied["signal_tier"] == 1],
            [1.0, config.tier2_size_fraction],
            default=0.0,
        )
    else:
        applied["signal_direction"] = np.select(
            [
                allow_new_entries & (applied["zscore"] <= -strong_threshold),
                allow_new_entries & (applied["zscore"] >= strong_threshold),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        applied["signal_tier"] = np.where(abs_zscores >= strong_threshold, 2, 0).astype(int)
        size_fraction = np.where(applied["signal_tier"] == 2, 1.0, 0.0)
    applied["target_position"] = (
        applied["signal_direction"].astype(float)
        * config.max_position_size
        * size_fraction
        * position_scale
    )
    return applied


def _correlation_to_distance(correlation: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(2.0 * (1.0 - correlation), a_min=0.0, a_max=None))


def _estimate_sigma(distance: np.ndarray) -> float:
    upper_triangle = distance[np.triu_indices_from(distance, k=1)]
    positive_distances = upper_triangle[upper_triangle > 0]
    if positive_distances.size == 0:
        return 1.0
    return float(np.median(positive_distances))


def _scaled_sigma(base_sigma: float, sigma_scale: float) -> float:
    scaled = base_sigma * sigma_scale
    if scaled <= 0:
        return 1.0
    return float(scaled)


def _distance_to_weights(distance: np.ndarray, sigma: float, min_weight: float) -> np.ndarray:
    sigma = sigma if sigma > 0 else 1.0
    weights = np.exp(-(distance**2) / (2.0 * sigma**2))
    np.fill_diagonal(weights, 0.0)
    weights[weights < min_weight] = 0.0
    return weights


def _normalized_laplacian(weights: np.ndarray) -> np.ndarray:
    degree = weights.sum(axis=1)
    inverse_sqrt_degree = np.zeros_like(degree)
    nonzero = degree > 0
    inverse_sqrt_degree[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    degree_matrix = np.diag(inverse_sqrt_degree)
    identity = np.eye(weights.shape[0])
    return identity - degree_matrix @ weights @ degree_matrix


def _apply_diffusion_filter(signal: np.ndarray, laplacian: np.ndarray, alpha: float, steps: int) -> np.ndarray:
    filter_matrix = np.eye(laplacian.shape[0]) - alpha * laplacian
    smoothed = signal.copy()
    for _ in range(steps):
        smoothed = filter_matrix @ smoothed
    return smoothed
def _rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    rolling_mean = series.shift(1).rolling(lookback).mean()
    rolling_std = series.shift(1).rolling(lookback).std(ddof=0)
    zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore


def _optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
