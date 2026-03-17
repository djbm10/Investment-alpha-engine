from __future__ import annotations

import numpy as np
import pandas as pd

from .config_loader import Phase2Config


def build_price_matrix(price_history: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    price_matrix = (
        price_history.pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
        .reindex(columns=tickers)
    )
    return price_matrix.dropna(how="any")


def compute_graph_signals(
    price_history: pd.DataFrame,
    tickers: list[str],
    config: Phase2Config,
) -> pd.DataFrame:
    price_matrix = build_price_matrix(price_history, tickers)
    log_returns = np.log(price_matrix / price_matrix.shift(1)).dropna(how="any")
    forward_returns = price_matrix.pct_change().shift(-1)

    if len(log_returns) < config.lookback_window:
        raise ValueError("Not enough return history to build the graph engine.")

    rows: list[dict[str, object]] = []
    for end_idx in range(config.lookback_window - 1, len(log_returns)):
        window = log_returns.iloc[end_idx - config.lookback_window + 1 : end_idx + 1]
        current_date = log_returns.index[end_idx]
        current_return = window.iloc[-1].to_numpy(dtype=float)

        correlation = window.corr().to_numpy(dtype=float)
        correlation = np.nan_to_num(correlation, nan=0.0)
        np.fill_diagonal(correlation, 1.0)

        distance = _correlation_to_distance(correlation)
        sigma = _scaled_sigma(_estimate_sigma(distance), config.sigma_scale)
        weights = _distance_to_weights(distance, sigma, config.min_weight)
        laplacian = _normalized_laplacian(weights)
        expected_return = _apply_diffusion_filter(
            current_return,
            laplacian,
            alpha=config.diffusion_alpha,
            steps=config.diffusion_steps,
        )
        residual = current_return - expected_return
        edge_density = _edge_density(weights)
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
                    "edge_density": float(edge_density),
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
    zscores = applied["zscore"]
    directions = np.select(
        [zscores <= -config.signal_threshold, zscores >= config.signal_threshold],
        [1, -1],
        default=0,
    )
    scales = (zscores.abs() / config.full_size_zscore).clip(upper=1.0).fillna(0.0)
    applied["signal_direction"] = directions.astype(int)
    applied["target_position"] = (
        applied["signal_direction"].astype(float) * config.max_position_size * scales
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


def _edge_density(weights: np.ndarray) -> float:
    possible_edges = weights.shape[0] * (weights.shape[0] - 1)
    if possible_edges == 0:
        return 0.0
    nonzero_edges = np.count_nonzero(weights)
    return float(nonzero_edges / possible_edges)


def _rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    rolling_mean = series.shift(1).rolling(lookback).mean()
    rolling_std = series.shift(1).rolling(lookback).std(ddof=0)
    zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore


def _optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
