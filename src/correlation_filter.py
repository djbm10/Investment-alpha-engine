from __future__ import annotations

import numpy as np

from .config_loader import Phase2Config

TRADEABLE_REGIME = "TRADEABLE"
REDUCED_REGIME = "REDUCED"
NO_TRADE_REGIME = "NO_TRADE"
REDUCED_POSITION_SCALE = 0.5
REDUCED_THRESHOLD_MULTIPLIER = 1.25


def average_pairwise_correlation(correlation: np.ndarray) -> float:
    upper_triangle = correlation[np.triu_indices_from(correlation, k=1)]
    if upper_triangle.size == 0:
        return 0.0
    return float(np.mean(upper_triangle))


def graph_density(weights: np.ndarray) -> float:
    possible_edges = weights.shape[0] * (weights.shape[0] - 1) / 2
    if possible_edges == 0:
        return 0.0
    nonzero_edges = np.count_nonzero(np.triu(weights > 0, k=1))
    return float(nonzero_edges / possible_edges)


def classify_universe_regime(avg_pairwise_corr: float, density: float, config: Phase2Config) -> str:
    if avg_pairwise_corr >= config.corr_floor and density >= config.density_floor:
        return TRADEABLE_REGIME
    if avg_pairwise_corr >= config.corr_floor * 0.75 or density >= config.density_floor * 0.75:
        return REDUCED_REGIME
    return NO_TRADE_REGIME


def regime_controls(regime_state: str) -> tuple[float, float, bool]:
    if regime_state == TRADEABLE_REGIME:
        return 1.0, 1.0, True
    if regime_state == REDUCED_REGIME:
        return REDUCED_THRESHOLD_MULTIPLIER, REDUCED_POSITION_SCALE, True
    return 1.0, 0.0, False


def node_average_correlations(correlation: np.ndarray) -> np.ndarray:
    if correlation.shape[0] <= 1:
        return np.zeros(correlation.shape[0], dtype=float)

    row_sums = correlation.sum(axis=1) - np.diag(correlation)
    return row_sums / (correlation.shape[0] - 1)


def node_tradeable_mask(
    correlation: np.ndarray,
    config: Phase2Config,
    regime_state: str = TRADEABLE_REGIME,
) -> np.ndarray:
    node_corr = node_average_correlations(correlation)
    floor = config.node_corr_floor
    if regime_state == REDUCED_REGIME:
        floor = floor * config.reduced_node_corr_multiplier
    return node_corr >= floor
