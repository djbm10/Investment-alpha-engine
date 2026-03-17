from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from .config_loader import Phase2Config, PipelineConfig
from .graph_engine import build_price_matrix, compute_daily_graph_matrices, compute_graph_signals

FEATURE_NAMES = [
    "e_i",
    "e_i_ma5",
    "e_i_std5",
    "z_i",
    "z_i_ma5",
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "vol_20d",
    "vol_ratio",
    "vol_rel",
    "vol_trend",
    "node_degree",
    "neighbor_mean_ret",
    "node_corr",
    "z_rank",
    "dispersion",
    "pct_same_sign",
]


@dataclass(frozen=True)
class GraphEngineState:
    tickers: list[str]
    feature_names: list[str]
    raw_feature_frame: pd.DataFrame
    scaled_feature_frame: pd.DataFrame
    scaler_snapshots: dict[pd.Timestamp, dict[str, dict[str, float]]]


@dataclass
class RunningFeatureStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0


class ExpandingFeatureScaler:
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = list(feature_names)
        self.stats = {feature_name: RunningFeatureStats() for feature_name in self.feature_names}
        self.snapshots: dict[pd.Timestamp, dict[str, dict[str, float]]] = {}

    def transform(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        if feature_frame.empty:
            return feature_frame.copy()

        scaled_frames: list[pd.DataFrame] = []
        for date, date_frame in feature_frame.groupby("date", sort=True):
            transformed = date_frame.copy()
            snapshot: dict[str, dict[str, float]] = {}
            for feature_name in self.feature_names:
                values = date_frame[feature_name].astype(float).to_numpy()
                stats = self.stats[feature_name]
                _update_running_stats(stats, values)
                std = _stats_std(stats)
                transformed[feature_name] = 0.0 if std == 0.0 else (values - stats.mean) / std
                snapshot[feature_name] = {
                    "count": float(stats.count),
                    "mean": float(stats.mean),
                    "std": float(std),
                }
            self.snapshots[pd.Timestamp(date)] = snapshot
            scaled_frames.append(transformed)

        return pd.concat(scaled_frames, ignore_index=True)


class FeatureBuilder:
    def __init__(self, config: PipelineConfig | Phase2Config) -> None:
        self.phase2_config = config.phase2 if isinstance(config, PipelineConfig) else config
        self.feature_names = list(FEATURE_NAMES)

    def prepare_graph_engine_state(
        self,
        price_history: pd.DataFrame,
        tickers: list[str],
    ) -> GraphEngineState:
        normalized = _normalize_price_history(price_history, tickers)
        signals = compute_graph_signals(normalized.loc[:, ["date", "ticker", "adj_close"]], tickers, self.phase2_config)
        raw_feature_frame = self._build_raw_feature_frame(normalized, signals, tickers)
        scaler = ExpandingFeatureScaler(self.feature_names)
        scaled_feature_frame = scaler.transform(raw_feature_frame)
        return GraphEngineState(
            tickers=list(tickers),
            feature_names=list(self.feature_names),
            raw_feature_frame=raw_feature_frame,
            scaled_feature_frame=scaled_feature_frame,
            scaler_snapshots=scaler.snapshots,
        )

    def build_features(
        self,
        date: pd.Timestamp | str,
        graph_engine_state: GraphEngineState,
    ) -> dict[str, np.ndarray]:
        timestamp = pd.Timestamp(date)
        scaled = _frame_for_date(graph_engine_state.scaled_feature_frame, timestamp, graph_engine_state.tickers, self.feature_names)
        raw = _frame_for_date(graph_engine_state.raw_feature_frame, timestamp, graph_engine_state.tickers, self.feature_names)
        if scaled.empty or raw.empty:
            raise ValueError(f"No features were available for {timestamp.date()}.")
        return {
            "feature_matrix": scaled.loc[:, self.feature_names].to_numpy(dtype=float),
            "raw_matrix": raw.loc[:, self.feature_names].to_numpy(dtype=float),
            "asset_index": np.asarray(graph_engine_state.tickers, dtype=object),
        }

    def build_feature_history(self, graph_engine_state: GraphEngineState) -> dict[pd.Timestamp, np.ndarray]:
        history: dict[pd.Timestamp, np.ndarray] = {}
        for date in sorted(pd.to_datetime(graph_engine_state.scaled_feature_frame["date"]).unique()):
            history[pd.Timestamp(date)] = self.build_features(date, graph_engine_state)["feature_matrix"]
        return history

    def build_residual_history(self, graph_engine_state: GraphEngineState) -> dict[pd.Timestamp, np.ndarray]:
        residual_history: dict[pd.Timestamp, np.ndarray] = {}
        for date in sorted(pd.to_datetime(graph_engine_state.raw_feature_frame["date"]).unique()):
            date_frame = _frame_for_date(
                graph_engine_state.raw_feature_frame,
                pd.Timestamp(date),
                graph_engine_state.tickers,
                ["e_i"],
            )
            residual_history[pd.Timestamp(date)] = date_frame["e_i"].to_numpy(dtype=float)
        return residual_history

    def _build_raw_feature_frame(
        self,
        price_history: pd.DataFrame,
        signals: pd.DataFrame,
        tickers: list[str],
    ) -> pd.DataFrame:
        feature_frame = (
            signals.loc[:, ["date", "ticker", "residual", "zscore", "current_return", "node_avg_corr"]]
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
            .copy()
        )
        feature_frame["e_i"] = feature_frame["residual"].astype(float)
        feature_frame["e_i_ma5"] = (
            feature_frame.groupby("ticker")["e_i"].transform(lambda series: series.rolling(5, min_periods=1).mean())
        )
        feature_frame["e_i_std5"] = (
            feature_frame.groupby("ticker")["e_i"].transform(lambda series: series.rolling(5, min_periods=1).std(ddof=0))
        )
        feature_frame["z_i"] = feature_frame["zscore"].astype(float).fillna(0.0)
        feature_frame["z_i_ma5"] = (
            feature_frame.groupby("ticker")["z_i"].transform(lambda series: series.rolling(5, min_periods=1).mean())
        )

        signal_dates = sorted(pd.to_datetime(feature_frame["date"]).unique())
        price_matrix = build_price_matrix(price_history.loc[:, ["date", "ticker", "adj_close"]], tickers)
        volume_matrix = (
            price_history.pivot(index="date", columns="ticker", values="volume")
            .sort_index()
            .reindex(columns=tickers)
            .fillna(0.0)
        )
        daily_returns = price_matrix.pct_change().fillna(0.0)
        derived_frames = {
            "ret_1d": daily_returns,
            "ret_5d": price_matrix.pct_change(5),
            "ret_10d": price_matrix.pct_change(10),
            "ret_20d": price_matrix.pct_change(20),
            "vol_20d": daily_returns.rolling(20, min_periods=1).std(ddof=0),
            "vol_ratio": (
                daily_returns.rolling(5, min_periods=1).std(ddof=0)
                / daily_returns.rolling(20, min_periods=1).std(ddof=0).replace(0.0, np.nan)
            ),
            "vol_rel": volume_matrix / volume_matrix.rolling(20, min_periods=1).mean().replace(0.0, np.nan),
            "vol_trend": (
                volume_matrix.rolling(5, min_periods=1).mean()
                / volume_matrix.rolling(20, min_periods=1).mean().replace(0.0, np.nan)
            ),
        }
        for feature_name, derived_frame in derived_frames.items():
            feature_frame = feature_frame.merge(
                _stack_feature_frame(derived_frame, signal_dates, tickers, feature_name),
                on=["date", "ticker"],
                how="left",
            )

        graph_matrices = compute_daily_graph_matrices(price_history.loc[:, ["date", "ticker", "adj_close"]], tickers, self.phase2_config.lookback_window)
        network_frame = self._build_network_features(feature_frame, graph_matrices, tickers)
        feature_frame = feature_frame.merge(network_frame, on=["date", "ticker"], how="left")

        numeric_columns = list(self.feature_names)
        feature_frame[numeric_columns] = (
            feature_frame[numeric_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
        return feature_frame.loc[:, ["date", "ticker", *numeric_columns]].sort_values(["date", "ticker"]).reset_index(drop=True)

    def _build_network_features(
        self,
        feature_frame: pd.DataFrame,
        graph_matrices,
        tickers: list[str],
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for date in sorted(pd.to_datetime(feature_frame["date"]).unique()):
            date_frame = (
                feature_frame.loc[feature_frame["date"] == date, ["ticker", "e_i", "z_i", "current_return", "node_avg_corr"]]
                .set_index("ticker")
                .reindex(tickers)
            )
            snapshot = graph_matrices[pd.Timestamp(date)]
            weights = _distance_to_weights(
                snapshot.distance_matrix,
                _scaled_sigma(_estimate_sigma(snapshot.distance_matrix), self.phase2_config.sigma_scale),
                self.phase2_config.min_weight,
            )
            node_degree = weights.sum(axis=1)
            current_returns = date_frame["current_return"].fillna(0.0).to_numpy(dtype=float)
            neighbor_mean_ret = np.divide(
                weights @ current_returns,
                node_degree,
                out=np.zeros(len(tickers), dtype=float),
                where=node_degree > 0,
            )
            residuals = date_frame["e_i"].fillna(0.0).to_numpy(dtype=float)
            zscores = date_frame["z_i"].fillna(0.0).to_numpy(dtype=float)
            z_rank = _normalized_rank(zscores)
            dispersion = float(np.std(residuals, ddof=0))
            pct_same_sign = _pct_same_sign(residuals)
            node_corr = date_frame["node_avg_corr"].fillna(0.0).to_numpy(dtype=float)

            for ticker_idx, ticker in enumerate(tickers):
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "ticker": ticker,
                        "node_degree": float(node_degree[ticker_idx]),
                        "neighbor_mean_ret": float(neighbor_mean_ret[ticker_idx]),
                        "node_corr": float(node_corr[ticker_idx]),
                        "z_rank": float(z_rank[ticker_idx]),
                        "dispersion": float(dispersion),
                        "pct_same_sign": float(pct_same_sign[ticker_idx]),
                    }
                )
        return pd.DataFrame(rows)


def _normalize_price_history(price_history: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    normalized = price_history.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    if "volume" not in normalized.columns:
        normalized["volume"] = 0.0
    required_columns = {"date", "ticker", "adj_close", "volume"}
    missing_columns = required_columns - set(normalized.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required price history columns: {missing}")
    normalized = normalized.loc[normalized["ticker"].isin(tickers), ["date", "ticker", "adj_close", "volume"]].copy()
    normalized["volume"] = normalized["volume"].fillna(0.0).astype(float)
    return normalized.sort_values(["date", "ticker"]).reset_index(drop=True)


def _frame_for_date(
    feature_frame: pd.DataFrame,
    date: pd.Timestamp,
    tickers: list[str],
    feature_names: list[str],
) -> pd.DataFrame:
    return (
        feature_frame.loc[feature_frame["date"] == pd.Timestamp(date), ["ticker", *feature_names]]
        .set_index("ticker")
        .reindex(tickers)
        .reset_index()
    )


def _stack_feature_frame(
    frame: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    tickers: list[str],
    feature_name: str,
) -> pd.DataFrame:
    aligned = frame.reindex(index=signal_dates, columns=tickers)
    stacked = aligned.stack().rename(feature_name).reset_index()
    stacked.columns = ["date", "ticker", feature_name]
    return stacked


def _update_running_stats(stats: RunningFeatureStats, values: np.ndarray) -> None:
    for raw_value in values:
        value = float(raw_value)
        stats.count += 1
        delta = value - stats.mean
        stats.mean += delta / stats.count
        delta2 = value - stats.mean
        stats.m2 += delta * delta2


def _stats_std(stats: RunningFeatureStats) -> float:
    if stats.count <= 1:
        return 0.0
    variance = stats.m2 / stats.count
    if variance <= 0.0:
        return 0.0
    return float(sqrt(variance))


def _estimate_sigma(distance: np.ndarray) -> float:
    upper_triangle = distance[np.triu_indices_from(distance, k=1)]
    positive_distances = upper_triangle[upper_triangle > 0]
    if positive_distances.size == 0:
        return 1.0
    return float(np.median(positive_distances))


def _scaled_sigma(base_sigma: float, sigma_scale: float) -> float:
    scaled = base_sigma * sigma_scale
    if scaled <= 0.0:
        return 1.0
    return float(scaled)


def _distance_to_weights(distance: np.ndarray, sigma: float, min_weight: float) -> np.ndarray:
    sigma = sigma if sigma > 0 else 1.0
    weights = np.exp(-(distance**2) / (2.0 * sigma**2))
    np.fill_diagonal(weights, 0.0)
    weights[weights < min_weight] = 0.0
    return weights


def _normalized_rank(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    return (ranks - 1.0) / (len(values) - 1.0)


def _pct_same_sign(residuals: np.ndarray) -> np.ndarray:
    if len(residuals) == 0:
        return np.empty(0, dtype=float)
    signs = np.sign(residuals)
    fractions = np.zeros(len(residuals), dtype=float)
    for idx, sign in enumerate(signs):
        fractions[idx] = float(np.mean(signs == sign))
    return fractions
