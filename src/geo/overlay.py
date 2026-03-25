from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_SIGNAL_COLUMNS = frozenset({"ticker", "signal_direction", "target_position"})
REQUIRED_BASE_SIGNAL_COLUMNS = frozenset({"zscore", "residual"})
OVERLAY_OUTPUT_COLUMNS = (
    "base_entry_threshold",
    "base_signal_value",
    "geo_net_score",
    "geo_structural_score",
    "avg_mapping_confidence",
    "coverage_score",
    "data_freshness_minutes",
    "geo_reliability_weight",
    "effective_geo_net_score",
    "effective_geo_structural_score",
    "contradiction",
    "geo_break_risk",
    "adjusted_entry_threshold",
    "geo_entry_allowed",
    "geo_entry_blocked",
    "position_scale",
    "adjusted_target_position",
    "final_signal_direction",
    "final_target_position",
    "hard_override",
)


def compute_geo_net_score(geo_snapshot: pd.DataFrame | None) -> pd.Series:
    if geo_snapshot is None or geo_snapshot.empty:
        return pd.Series(dtype=float)

    if "geo_net_score" in geo_snapshot.columns:
        score = pd.to_numeric(geo_snapshot["geo_net_score"], errors="coerce")
    elif "geo_net_raw" in geo_snapshot.columns:
        raw = pd.to_numeric(geo_snapshot["geo_net_raw"], errors="coerce").fillna(0.0)
        score = pd.Series(np.tanh(raw.to_numpy(dtype=float)), index=geo_snapshot.index, dtype=float)
    else:
        score = pd.Series(0.0, index=geo_snapshot.index, dtype=float)

    return score.fillna(0.0).clip(-1.0, 1.0)


def compute_contradiction(signal_direction: pd.Series, geo_net_score: pd.Series) -> pd.Series:
    direction = pd.to_numeric(signal_direction, errors="coerce").fillna(0.0)
    geo_score = pd.to_numeric(geo_net_score, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    return (-direction * geo_score).clip(lower=0.0, upper=1.0)


def compute_geo_break_risk(geo_structural_score: pd.Series) -> pd.Series:
    structural_score = pd.to_numeric(geo_structural_score, errors="coerce").fillna(0.0)
    return structural_score.abs().clip(lower=0.0, upper=1.0)


def compute_adjusted_entry_threshold(
    base_entry_threshold: pd.Series,
    contradiction: pd.Series,
    geo_break_risk: pd.Series,
    *,
    gamma: float,
) -> pd.Series:
    base = pd.to_numeric(base_entry_threshold, errors="coerce").fillna(0.0)
    contradiction_term = pd.to_numeric(contradiction, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    break_term = pd.to_numeric(geo_break_risk, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return base * (1.0 + float(gamma) * contradiction_term + 0.5 * float(gamma) * break_term)


def compute_position_scale(contradiction: pd.Series, geo_break_risk: pd.Series) -> pd.Series:
    contradiction_term = pd.to_numeric(contradiction, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    break_term = pd.to_numeric(geo_break_risk, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return (1.0 - 0.50 * contradiction_term - 0.25 * break_term).clip(lower=0.0, upper=1.0)


def compute_adjusted_position_size(base_position: pd.Series, position_scale: pd.Series) -> pd.Series:
    position = pd.to_numeric(base_position, errors="coerce").fillna(0.0)
    scale = pd.to_numeric(position_scale, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return position * scale


def compute_hard_override(
    signal_direction: pd.Series,
    geo_structural_score: pd.Series,
    avg_mapping_confidence: pd.Series,
    coverage_score: pd.Series,
    data_freshness_minutes: pd.Series,
    *,
    hard_override_threshold: float,
    min_mapping_confidence: float,
    min_coverage_score: float,
    freshness_cutoff_minutes: int,
) -> pd.Series:
    direction = pd.to_numeric(signal_direction, errors="coerce").fillna(0.0)
    structural_score = pd.to_numeric(geo_structural_score, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    contradiction = (-direction * structural_score).clip(lower=0.0)
    mapping_confidence = pd.to_numeric(avg_mapping_confidence, errors="coerce").fillna(0.0)
    coverage = pd.to_numeric(coverage_score, errors="coerce").fillna(0.0)
    freshness = pd.to_numeric(data_freshness_minutes, errors="coerce").fillna(float("inf"))
    return (
        (contradiction >= float(hard_override_threshold))
        & (mapping_confidence >= float(min_mapping_confidence))
        & (coverage >= float(min_coverage_score))
        & (freshness <= int(freshness_cutoff_minutes))
    )


def compute_geo_reliability_weight(
    avg_mapping_confidence: pd.Series,
    coverage_score: pd.Series,
    *,
    min_mapping_confidence: float,
    min_coverage_score: float,
) -> pd.Series:
    mapping_confidence = pd.to_numeric(avg_mapping_confidence, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    coverage = pd.to_numeric(coverage_score, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    mapping_floor = max(float(min_mapping_confidence), 1e-9)
    coverage_floor = max(float(min_coverage_score), 1e-9)
    mapping_weight = (mapping_confidence / mapping_floor).clip(lower=0.0, upper=1.0)
    coverage_weight = (coverage / coverage_floor).clip(lower=0.0, upper=1.0)
    return pd.Series(
        np.minimum(mapping_weight.to_numpy(dtype=float), coverage_weight.to_numpy(dtype=float)),
        index=mapping_weight.index,
        dtype=float,
    )


def apply_geo_overlay(
    signals: pd.DataFrame,
    geo_snapshot: pd.DataFrame | None,
    *,
    base_signal_threshold: float,
    enabled: bool = True,
    gamma: float = 0.75,
    hard_override_threshold: float = 0.80,
    min_mapping_confidence: float = 0.70,
    min_coverage_score: float = 0.70,
    freshness_cutoff_minutes: int = 120,
) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()

    _validate_signal_contract(signals)
    applied = signals.copy()
    base_threshold = float(base_signal_threshold) * pd.to_numeric(
        applied.get("regime_threshold_multiplier", pd.Series(1.0, index=applied.index)),
        errors="coerce",
    ).fillna(1.0)
    applied["base_entry_threshold"] = base_threshold
    applied["base_signal_value"] = _extract_base_signal_value(applied)

    if not enabled or geo_snapshot is None or geo_snapshot.empty:
        return _apply_neutral_overlay(applied)

    merged_snapshot = _prepare_geo_snapshot(geo_snapshot)
    applied = _merge_geo_snapshot(applied, merged_snapshot)

    applied["geo_net_score"] = pd.to_numeric(applied["geo_net_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    applied["geo_structural_score"] = (
        pd.to_numeric(applied["geo_structural_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    )
    applied["avg_mapping_confidence"] = pd.to_numeric(
        applied["avg_mapping_confidence"], errors="coerce"
    ).fillna(0.0)
    applied["coverage_score"] = pd.to_numeric(applied["coverage_score"], errors="coerce").fillna(0.0)
    applied["data_freshness_minutes"] = pd.to_numeric(
        applied["data_freshness_minutes"], errors="coerce"
    ).fillna(float("inf"))
    applied["geo_reliability_weight"] = compute_geo_reliability_weight(
        applied["avg_mapping_confidence"],
        applied["coverage_score"],
        min_mapping_confidence=min_mapping_confidence,
        min_coverage_score=min_coverage_score,
    )
    applied["effective_geo_net_score"] = (
        applied["geo_net_score"] * applied["geo_reliability_weight"]
    ).clip(-1.0, 1.0)
    applied["effective_geo_structural_score"] = (
        applied["geo_structural_score"] * applied["geo_reliability_weight"]
    ).clip(-1.0, 1.0)

    applied["contradiction"] = compute_contradiction(
        applied["signal_direction"],
        applied["effective_geo_net_score"],
    )
    applied["geo_break_risk"] = compute_geo_break_risk(applied["effective_geo_structural_score"])
    applied["adjusted_entry_threshold"] = compute_adjusted_entry_threshold(
        applied["base_entry_threshold"],
        applied["contradiction"],
        applied["geo_break_risk"],
        gamma=gamma,
    )
    applied["geo_entry_allowed"] = (
        applied["base_signal_value"].abs() >= applied["adjusted_entry_threshold"]
    )
    applied["position_scale"] = compute_position_scale(applied["contradiction"], applied["geo_break_risk"])
    applied["adjusted_target_position"] = compute_adjusted_position_size(
        applied["target_position"],
        applied["position_scale"],
    )
    applied["hard_override"] = compute_hard_override(
        applied["signal_direction"],
        applied["geo_structural_score"],
        applied["avg_mapping_confidence"],
        applied["coverage_score"],
        applied["data_freshness_minutes"],
        hard_override_threshold=hard_override_threshold,
        min_mapping_confidence=min_mapping_confidence,
        min_coverage_score=min_coverage_score,
        freshness_cutoff_minutes=freshness_cutoff_minutes,
    )
    applied["geo_entry_blocked"] = (~applied["geo_entry_allowed"]) | applied["hard_override"]
    applied["final_signal_direction"] = applied["signal_direction"].where(~applied["geo_entry_blocked"], 0).astype(int)
    applied["final_target_position"] = (
        applied["adjusted_target_position"].where(~applied["geo_entry_blocked"], 0.0)
    )
    return applied


def _prepare_geo_snapshot(geo_snapshot: pd.DataFrame) -> pd.DataFrame:
    prepared = geo_snapshot.copy()
    if "asset" not in prepared.columns:
        raise ValueError("geo_snapshot must contain an 'asset' column")
    if "trade_date" in prepared.columns:
        prepared["trade_date"] = pd.to_datetime(prepared["trade_date"]).dt.date
        if prepared.duplicated(subset=["asset", "trade_date"]).any():
            raise ValueError("geo_snapshot must contain at most one row per asset and trade_date")
    elif prepared["asset"].duplicated().any():
        raise ValueError("geo_snapshot must contain at most one row per asset")

    prepared["geo_net_score"] = compute_geo_net_score(prepared)
    prepared["geo_structural_score"] = pd.to_numeric(
        prepared.get("geo_structural_score", pd.Series(0.0, index=prepared.index)),
        errors="coerce",
    ).fillna(0.0).clip(-1.0, 1.0)
    prepared["avg_mapping_confidence"] = pd.to_numeric(
        prepared.get("avg_mapping_confidence", pd.Series(0.0, index=prepared.index)),
        errors="coerce",
    ).fillna(0.0)
    prepared["coverage_score"] = pd.to_numeric(
        prepared.get("coverage_score", pd.Series(0.0, index=prepared.index)),
        errors="coerce",
    ).fillna(0.0)
    prepared["data_freshness_minutes"] = pd.to_numeric(
        prepared.get("data_freshness_minutes", pd.Series(float("inf"), index=prepared.index)),
        errors="coerce",
    ).fillna(float("inf"))
    columns = [
        "asset",
        "geo_net_score",
        "geo_structural_score",
        "avg_mapping_confidence",
        "coverage_score",
        "data_freshness_minutes",
    ]
    if "trade_date" in prepared.columns:
        columns.insert(1, "trade_date")
    return prepared[columns]


def _apply_neutral_overlay(signals: pd.DataFrame) -> pd.DataFrame:
    neutral = signals.copy()
    neutral["base_signal_value"] = _extract_base_signal_value(neutral)
    neutral["geo_net_score"] = 0.0
    neutral["geo_structural_score"] = 0.0
    neutral["avg_mapping_confidence"] = 0.0
    neutral["coverage_score"] = 0.0
    neutral["data_freshness_minutes"] = float("inf")
    neutral["geo_reliability_weight"] = 0.0
    neutral["effective_geo_net_score"] = 0.0
    neutral["effective_geo_structural_score"] = 0.0
    neutral["contradiction"] = 0.0
    neutral["geo_break_risk"] = 0.0
    neutral["adjusted_entry_threshold"] = neutral["base_entry_threshold"]
    neutral["geo_entry_allowed"] = True
    neutral["geo_entry_blocked"] = False
    neutral["position_scale"] = 1.0
    neutral["adjusted_target_position"] = pd.to_numeric(
        neutral["target_position"], errors="coerce"
    ).fillna(0.0)
    neutral["final_signal_direction"] = pd.to_numeric(
        neutral["signal_direction"], errors="coerce"
    ).fillna(0.0).astype(int)
    neutral["final_target_position"] = neutral["adjusted_target_position"]
    neutral["hard_override"] = False
    return neutral


def _validate_signal_contract(signals: pd.DataFrame) -> None:
    missing_required = REQUIRED_SIGNAL_COLUMNS.difference(signals.columns)
    if missing_required:
        missing_list = ", ".join(sorted(missing_required))
        raise ValueError(f"signals must contain required columns: {missing_list}")
    if not REQUIRED_BASE_SIGNAL_COLUMNS.intersection(signals.columns):
        expected = " or ".join(sorted(REQUIRED_BASE_SIGNAL_COLUMNS))
        raise ValueError(f"signals must contain at least one base signal column: {expected}")


def _extract_base_signal_value(signals: pd.DataFrame) -> pd.Series:
    if "zscore" in signals.columns:
        return pd.to_numeric(signals["zscore"], errors="coerce").fillna(0.0)
    return pd.to_numeric(signals["residual"], errors="coerce").fillna(0.0)


def _merge_geo_snapshot(signals: pd.DataFrame, geo_snapshot: pd.DataFrame) -> pd.DataFrame:
    if "date" in signals.columns and "trade_date" in geo_snapshot.columns:
        merged = signals.copy()
        merged["trade_date"] = pd.to_datetime(merged["date"]).dt.date
        merged = merged.merge(
            geo_snapshot,
            how="left",
            left_on=["ticker", "trade_date"],
            right_on=["asset", "trade_date"],
        )
        return merged.drop(columns=["asset", "trade_date"], errors="ignore")

    merged = signals.merge(geo_snapshot, how="left", left_on="ticker", right_on="asset")
    return merged.drop(columns=["asset"], errors="ignore")
