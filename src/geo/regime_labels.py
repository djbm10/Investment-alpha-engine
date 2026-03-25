from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NORMAL_REGIME = "NORMAL"
GEO_STRESS_REGIME = "GEO_STRESS"
STRUCTURAL_GEO_REGIME = "STRUCTURAL_GEO"
REGIME_ORDER = (NORMAL_REGIME, GEO_STRESS_REGIME, STRUCTURAL_GEO_REGIME)
PLACEBO_RANDOM_SEED = 1729
PLACEBO_LAG_DAYS = 63


@dataclass(frozen=True)
class GeoRegimeDefinition:
    coverage_floor: float = 0.57
    stress_abs_net_floor: float = 0.0
    structural_abs_score_floor: float = 0.0
    stress_persistence_days: int = 2
    structural_persistence_days: int = 3

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def load_geo_snapshot_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "trade_date" not in frame.columns:
        raise ValueError("geo snapshot history must contain a 'trade_date' column")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    if frame["trade_date"].isna().any():
        raise ValueError("geo snapshot history contained invalid trade_date values")
    return frame


def aggregate_daily_geo_features(snapshot_frame: pd.DataFrame) -> pd.DataFrame:
    _validate_snapshot_columns(snapshot_frame)
    frame = snapshot_frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["geo_net_score"] = pd.to_numeric(frame["geo_net_score"], errors="coerce").fillna(0.0)
    frame["geo_structural_score"] = pd.to_numeric(frame["geo_structural_score"], errors="coerce").fillna(0.0)
    frame["coverage_score"] = pd.to_numeric(frame["coverage_score"], errors="coerce").fillna(0.0)
    daily = (
        frame.groupby("trade_date", sort=True)
        .agg(
            mean_abs_geo_net_score=("geo_net_score", lambda series: float(series.abs().mean())),
            mean_abs_geo_structural_score=("geo_structural_score", lambda series: float(series.abs().mean())),
            median_coverage_score=("coverage_score", "median"),
            asset_count=("asset", "nunique"),
        )
        .reset_index()
        .rename(columns={"trade_date": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return daily.sort_values("date", kind="mergesort").reset_index(drop=True)


def build_geo_regime_labels(
    snapshot_frame: pd.DataFrame,
    *,
    min_asset_count: int,
    definition: GeoRegimeDefinition | None = None,
) -> pd.DataFrame:
    if min_asset_count <= 0:
        raise ValueError("min_asset_count must be positive")
    active_definition = definition or GeoRegimeDefinition()
    daily = aggregate_daily_geo_features(snapshot_frame)
    daily["regime_eligible"] = (
        (daily["median_coverage_score"] >= active_definition.coverage_floor)
        & (daily["asset_count"] >= int(min_asset_count))
    )
    daily["stress_raw"] = daily["regime_eligible"] & (
        daily["mean_abs_geo_net_score"] > float(active_definition.stress_abs_net_floor)
    )
    daily["structural_raw"] = daily["regime_eligible"] & (
        daily["mean_abs_geo_structural_score"] > float(active_definition.structural_abs_score_floor)
    )
    daily["stress_persistence_count"] = _consecutive_true_counts(daily["stress_raw"])
    daily["structural_persistence_count"] = _consecutive_true_counts(daily["structural_raw"])

    labels = pd.Series(NORMAL_REGIME, index=daily.index, dtype="object")
    labels.loc[
        daily["stress_persistence_count"] >= int(active_definition.stress_persistence_days)
    ] = GEO_STRESS_REGIME
    labels.loc[
        daily["structural_persistence_count"] >= int(active_definition.structural_persistence_days)
    ] = STRUCTURAL_GEO_REGIME
    labels.loc[~daily["regime_eligible"]] = NORMAL_REGIME
    daily["geo_regime"] = labels
    return daily.loc[
        :,
        [
            "date",
            "mean_abs_geo_net_score",
            "mean_abs_geo_structural_score",
            "median_coverage_score",
            "asset_count",
            "regime_eligible",
            "stress_raw",
            "structural_raw",
            "stress_persistence_count",
            "structural_persistence_count",
            "geo_regime",
        ],
    ].copy()


def build_placebo_regime_frames(
    regime_frame: pd.DataFrame,
    *,
    seed: int = PLACEBO_RANDOM_SEED,
    lag_days: int = PLACEBO_LAG_DAYS,
) -> dict[str, pd.DataFrame]:
    _validate_regime_frame(regime_frame)
    normalized = regime_frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    normalized["regime_eligible"] = normalized["regime_eligible"].astype(bool)
    normalized["geo_regime"] = normalized["geo_regime"].astype(str)

    eligible_mask = normalized["regime_eligible"]
    placebos = {
        "shuffled_counts_preserved": _with_replaced_labels(
            normalized,
            _shuffle_eligible_labels(normalized.loc[eligible_mask, "geo_regime"], seed=seed),
        ),
        "lag_broken_63d": _with_replaced_labels(
            normalized,
            normalized["geo_regime"].shift(int(lag_days), fill_value=NORMAL_REGIME),
        ),
        "constant_geo_stress": _with_constant_label(normalized, GEO_STRESS_REGIME),
        "constant_structural_geo": _with_constant_label(normalized, STRUCTURAL_GEO_REGIME),
    }
    return placebos


def _validate_snapshot_columns(snapshot_frame: pd.DataFrame) -> None:
    required = {"trade_date", "asset", "geo_net_score", "geo_structural_score", "coverage_score"}
    missing = sorted(required - set(snapshot_frame.columns))
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"geo snapshot history missing required columns: {missing_list}")


def _validate_regime_frame(regime_frame: pd.DataFrame) -> None:
    required = {"date", "geo_regime", "regime_eligible"}
    missing = sorted(required - set(regime_frame.columns))
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"regime frame missing required columns: {missing_list}")


def _consecutive_true_counts(mask: pd.Series) -> pd.Series:
    counts: list[int] = []
    running = 0
    for value in mask.fillna(False).astype(bool).tolist():
        running = running + 1 if value else 0
        counts.append(running)
    return pd.Series(counts, index=mask.index, dtype=int)


def _shuffle_eligible_labels(labels: pd.Series, *, seed: int) -> pd.Series:
    values = labels.astype(str).to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    return pd.Series(values, index=labels.index, dtype="object")


def _with_replaced_labels(regime_frame: pd.DataFrame, replacement_labels: pd.Series) -> pd.DataFrame:
    placed = regime_frame.copy()
    placed["geo_regime"] = placed["geo_regime"].astype(str)
    replacement = replacement_labels.reindex(placed.index)
    eligible_mask = placed["regime_eligible"].astype(bool)
    placed.loc[eligible_mask, "geo_regime"] = replacement.loc[eligible_mask].fillna(NORMAL_REGIME).astype(str)
    placed.loc[~eligible_mask, "geo_regime"] = NORMAL_REGIME
    return placed


def _with_constant_label(regime_frame: pd.DataFrame, label: str) -> pd.DataFrame:
    placed = regime_frame.copy()
    eligible_mask = placed["regime_eligible"].astype(bool)
    placed["geo_regime"] = np.where(eligible_mask, label, NORMAL_REGIME)
    return placed


__all__ = [
    "GEO_STRESS_REGIME",
    "GeoRegimeDefinition",
    "NORMAL_REGIME",
    "PLACEBO_LAG_DAYS",
    "PLACEBO_RANDOM_SEED",
    "REGIME_ORDER",
    "STRUCTURAL_GEO_REGIME",
    "aggregate_daily_geo_features",
    "build_geo_regime_labels",
    "build_placebo_regime_frames",
    "load_geo_snapshot_frame",
]
