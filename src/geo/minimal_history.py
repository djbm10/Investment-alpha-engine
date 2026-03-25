from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ..config_loader import load_config

NY_TZ = ZoneInfo("America/New_York")
DEFAULT_START_DATE = date(2020, 1, 1)
DEFAULT_END_DATE = date(2025, 12, 31)
BASELINE_FRESHNESS_MINUTES = 180
BASELINE_COVERAGE = {
    "XLK": 0.58,
    "XLE": 0.64,
    "XLV": 0.57,
    "XLP": 0.56,
    "XLU": 0.56,
    "XLY": 0.58,
    "XLRE": 0.55,
    "XLB": 0.57,
}
BASELINE_MAPPING_CONFIDENCE = {
    "XLK": 0.62,
    "XLE": 0.66,
    "XLV": 0.60,
    "XLP": 0.59,
    "XLU": 0.59,
    "XLY": 0.61,
    "XLRE": 0.58,
    "XLB": 0.60,
}
MINIMAL_GEO_SNAPSHOT_COLUMNS = (
    "trade_date",
    "asset",
    "snapshot_cutoff_at",
    "geo_net_raw",
    "geo_net_score",
    "geo_structural_score",
    "geo_harm_score",
    "geo_break_risk",
    "geo_velocity_3d",
    "geo_cluster_72h",
    "region_stress",
    "sector_disruption",
    "infra_disruption",
    "sanctions_score",
    "avg_mapping_confidence",
    "coverage_score",
    "data_freshness_minutes",
    "hard_override",
    "contributing_event_ids",
)


@dataclass(frozen=True)
class SyntheticGeoEvent:
    event_id: str
    first_seen_date: date
    active_end_date: date
    half_life_days: int
    event_class: str
    coverage_score: float
    freshness_minutes: int
    region_stress: float
    sanctions_stress: float
    infra_stress: float
    impacts: dict[str, float]
    structural_impacts: dict[str, float]


@dataclass(frozen=True)
class SnapshotVariantProfile:
    name: str
    shock_impact_scale: float
    shock_structural_scale: float
    shock_region_scale: float
    shock_half_life_scale: float
    structural_impact_scale: float
    structural_structural_scale: float
    structural_region_scale: float
    structural_half_life_scale: float
    global_impact_scale: float = 1.0
    global_structural_scale: float = 1.0
    global_coverage_cap: float | None = None
    global_mapping_confidence_cap: float | None = None
    base_coverage_scale: float = 1.0
    base_mapping_confidence_scale: float = 1.0
    event_asset_impact_scales: dict[str, dict[str, float]] = field(default_factory=dict)
    event_asset_structural_scales: dict[str, dict[str, float]] = field(default_factory=dict)


SYNTHETIC_GEO_EVENTS = (
    SyntheticGeoEvent(
        event_id="2020-01-03-us-iran-escalation",
        first_seen_date=date(2020, 1, 3),
        active_end_date=date(2020, 1, 6),
        half_life_days=3,
        event_class="shock",
        coverage_score=0.68,
        freshness_minutes=120,
        region_stress=0.20,
        sanctions_stress=0.05,
        infra_stress=0.10,
        impacts={
            "XLE": 0.18,
            "XLB": 0.03,
            "XLK": -0.02,
            "XLY": -0.03,
            "XLRE": -0.02,
            "XLP": 0.01,
            "XLU": 0.02,
        },
        structural_impacts={
            "XLE": 0.03,
        },
    ),
    SyntheticGeoEvent(
        event_id="2022-02-24-russia-ukraine-invasion",
        first_seen_date=date(2022, 2, 24),
        active_end_date=date(2022, 2, 28),
        half_life_days=5,
        event_class="shock",
        coverage_score=0.78,
        freshness_minutes=45,
        region_stress=0.55,
        sanctions_stress=0.15,
        infra_stress=0.20,
        impacts={
            "XLE": 0.38,
            "XLB": 0.04,
            "XLK": -0.04,
            "XLY": -0.06,
            "XLRE": -0.04,
            "XLP": 0.03,
            "XLU": 0.04,
            "XLV": 0.02,
        },
        structural_impacts={
            "XLE": 0.06,
        },
    ),
    SyntheticGeoEvent(
        event_id="2020-05-15-huawei-semiconductor-rule",
        first_seen_date=date(2020, 5, 15),
        active_end_date=date(2020, 5, 29),
        half_life_days=18,
        event_class="structural",
        coverage_score=0.84,
        freshness_minutes=90,
        region_stress=0.36,
        sanctions_stress=0.82,
        infra_stress=0.02,
        impacts={
            "XLK": -0.50,
            "XLB": -0.06,
            "XLY": -0.02,
            "XLRE": -0.01,
        },
        structural_impacts={
            "XLK": -0.44,
            "XLB": -0.02,
        },
    ),
    SyntheticGeoEvent(
        event_id="2022-10-07-china-chip-export-controls",
        first_seen_date=date(2022, 10, 7),
        active_end_date=date(2022, 10, 21),
        half_life_days=20,
        event_class="structural",
        coverage_score=0.86,
        freshness_minutes=60,
        region_stress=0.48,
        sanctions_stress=0.90,
        infra_stress=0.05,
        impacts={
            "XLK": -0.58,
            "XLB": -0.05,
            "XLY": -0.03,
            "XLRE": -0.02,
        },
        structural_impacts={
            "XLK": -0.48,
            "XLB": -0.02,
        },
    ),
    SyntheticGeoEvent(
        event_id="2023-10-17-ai-chip-export-control-expansion",
        first_seen_date=date(2023, 10, 17),
        active_end_date=date(2023, 10, 31),
        half_life_days=18,
        event_class="structural",
        coverage_score=0.85,
        freshness_minutes=75,
        region_stress=0.42,
        sanctions_stress=0.88,
        infra_stress=0.03,
        impacts={
            "XLK": -0.46,
            "XLB": -0.05,
            "XLY": -0.02,
            "XLRE": -0.01,
        },
        structural_impacts={
            "XLK": -0.40,
            "XLB": -0.015,
        },
    ),
    SyntheticGeoEvent(
        event_id="2023-10-07-middle-east-conflict",
        first_seen_date=date(2023, 10, 9),
        active_end_date=date(2023, 10, 12),
        half_life_days=3,
        event_class="shock",
        coverage_score=0.72,
        freshness_minutes=75,
        region_stress=0.22,
        sanctions_stress=0.10,
        infra_stress=0.08,
        impacts={
            "XLE": 0.16,
            "XLY": -0.04,
            "XLK": -0.02,
            "XLRE": -0.02,
            "XLP": 0.02,
            "XLU": 0.02,
        },
        structural_impacts={},
    ),
    SyntheticGeoEvent(
        event_id="2023-11-19-red-sea-shipping-disruption",
        first_seen_date=date(2023, 11, 20),
        active_end_date=date(2023, 11, 24),
        half_life_days=4,
        event_class="shock",
        coverage_score=0.74,
        freshness_minutes=60,
        region_stress=0.26,
        sanctions_stress=0.05,
        infra_stress=0.55,
        impacts={
            "XLE": 0.05,
            "XLB": -0.08,
            "XLY": -0.03,
            "XLK": -0.01,
            "XLRE": -0.02,
        },
        structural_impacts={
            "XLB": -0.02,
        },
    ),
    SyntheticGeoEvent(
        event_id="2024-12-02-semiconductor-export-controls",
        first_seen_date=date(2024, 12, 2),
        active_end_date=date(2024, 12, 16),
        half_life_days=18,
        event_class="structural",
        coverage_score=0.84,
        freshness_minutes=60,
        region_stress=0.38,
        sanctions_stress=0.86,
        infra_stress=0.02,
        impacts={
            "XLK": -0.44,
            "XLB": -0.05,
            "XLY": -0.01,
            "XLRE": -0.01,
        },
        structural_impacts={
            "XLK": -0.38,
            "XLB": -0.015,
        },
    ),
    SyntheticGeoEvent(
        event_id="2024-04-13-iran-israel-direct-strikes",
        first_seen_date=date(2024, 4, 15),
        active_end_date=date(2024, 4, 16),
        half_life_days=3,
        event_class="shock",
        coverage_score=0.70,
        freshness_minutes=90,
        region_stress=0.18,
        sanctions_stress=0.05,
        infra_stress=0.06,
        impacts={
            "XLE": 0.12,
            "XLY": -0.02,
            "XLK": -0.01,
            "XLU": 0.02,
        },
        structural_impacts={},
    ),
)

SNAPSHOT_VARIANT_PROFILES = {
    "base": SnapshotVariantProfile(
        name="base",
        shock_impact_scale=1.0,
        shock_structural_scale=1.0,
        shock_region_scale=1.0,
        shock_half_life_scale=1.0,
        structural_impact_scale=1.0,
        structural_structural_scale=1.0,
        structural_region_scale=1.0,
        structural_half_life_scale=1.0,
    ),
    "A": SnapshotVariantProfile(
        name="A",
        shock_impact_scale=0.20,
        shock_structural_scale=0.0,
        shock_region_scale=0.35,
        shock_half_life_scale=1.0,
        structural_impact_scale=1.0,
        structural_structural_scale=1.0,
        structural_region_scale=1.0,
        structural_half_life_scale=1.0,
        global_coverage_cap=0.82,
        global_mapping_confidence_cap=0.82,
    ),
    "B": SnapshotVariantProfile(
        name="B",
        shock_impact_scale=0.20,
        shock_structural_scale=0.0,
        shock_region_scale=0.30,
        shock_half_life_scale=0.40,
        structural_impact_scale=1.0,
        structural_structural_scale=1.0,
        structural_region_scale=1.0,
        structural_half_life_scale=1.0,
        global_coverage_cap=0.80,
        global_mapping_confidence_cap=0.80,
    ),
    "C": SnapshotVariantProfile(
        name="C",
        shock_impact_scale=0.12,
        shock_structural_scale=0.0,
        shock_region_scale=0.25,
        shock_half_life_scale=0.35,
        structural_impact_scale=0.65,
        structural_structural_scale=0.70,
        structural_region_scale=0.75,
        structural_half_life_scale=0.85,
        global_impact_scale=0.75,
        global_structural_scale=0.75,
        global_coverage_cap=0.76,
        global_mapping_confidence_cap=0.76,
    ),
    "D": SnapshotVariantProfile(
        name="D",
        shock_impact_scale=0.10,
        shock_structural_scale=0.0,
        shock_region_scale=0.18,
        shock_half_life_scale=0.30,
        structural_impact_scale=0.55,
        structural_structural_scale=0.75,
        structural_region_scale=0.70,
        structural_half_life_scale=0.80,
        global_impact_scale=0.65,
        global_structural_scale=0.70,
        global_coverage_cap=0.74,
        global_mapping_confidence_cap=0.74,
        event_asset_impact_scales={
            "2020-01-03-us-iran-escalation": {"XLE": 1.0, "XLB": 0.25, "XLP": 0.10, "XLU": 0.10, "XLK": 0.0, "XLY": 0.0, "XLRE": 0.0},
            "2022-02-24-russia-ukraine-invasion": {"XLE": 1.0, "XLB": 0.40, "XLP": 0.15, "XLU": 0.15, "XLV": 0.10, "XLK": 0.0, "XLY": 0.0, "XLRE": 0.0},
            "2022-10-07-china-chip-export-controls": {"XLK": 1.0, "XLB": 0.30, "XLY": 0.0, "XLRE": 0.0},
            "2023-10-07-middle-east-conflict": {"XLE": 0.75, "XLP": 0.15, "XLU": 0.15, "XLY": 0.0, "XLK": 0.0, "XLRE": 0.0},
            "2023-11-19-red-sea-shipping-disruption": {"XLB": 0.45, "XLE": 0.20, "XLY": 0.0, "XLK": 0.0, "XLRE": 0.0},
            "2024-04-13-iran-israel-direct-strikes": {"XLE": 0.70, "XLU": 0.20, "XLY": 0.0, "XLK": 0.0},
        },
        event_asset_structural_scales={
            "2020-01-03-us-iran-escalation": {"XLE": 0.40},
            "2022-02-24-russia-ukraine-invasion": {"XLE": 0.50},
            "2022-10-07-china-chip-export-controls": {"XLK": 1.0, "XLB": 0.35},
            "2023-11-19-red-sea-shipping-disruption": {"XLB": 0.25},
        },
    ),
    "E": SnapshotVariantProfile(
        name="E",
        shock_impact_scale=0.06,
        shock_structural_scale=0.0,
        shock_region_scale=0.12,
        shock_half_life_scale=0.22,
        structural_impact_scale=0.42,
        structural_structural_scale=0.55,
        structural_region_scale=0.55,
        structural_half_life_scale=0.70,
        global_impact_scale=0.52,
        global_structural_scale=0.55,
        global_coverage_cap=0.70,
        global_mapping_confidence_cap=0.70,
        event_asset_impact_scales={
            "2020-01-03-us-iran-escalation": {"XLE": 0.80, "XLB": 0.10, "XLP": 0.05, "XLU": 0.05, "XLK": 0.0, "XLY": 0.0, "XLRE": 0.0},
            "2022-02-24-russia-ukraine-invasion": {"XLE": 0.75, "XLB": 0.20, "XLP": 0.08, "XLU": 0.08, "XLV": 0.05, "XLK": 0.0, "XLY": 0.0, "XLRE": 0.0},
            "2022-10-07-china-chip-export-controls": {"XLK": 0.85, "XLB": 0.18, "XLY": 0.0, "XLRE": 0.0},
            "2023-10-07-middle-east-conflict": {"XLE": 0.55, "XLP": 0.08, "XLU": 0.08, "XLY": 0.0, "XLK": 0.0, "XLRE": 0.0},
            "2023-11-19-red-sea-shipping-disruption": {"XLB": 0.30, "XLE": 0.10, "XLY": 0.0, "XLK": 0.0, "XLRE": 0.0},
            "2024-04-13-iran-israel-direct-strikes": {"XLE": 0.45, "XLU": 0.10, "XLY": 0.0, "XLK": 0.0},
        },
        event_asset_structural_scales={
            "2020-01-03-us-iran-escalation": {"XLE": 0.20},
            "2022-02-24-russia-ukraine-invasion": {"XLE": 0.28},
            "2022-10-07-china-chip-export-controls": {"XLK": 0.80, "XLB": 0.20},
            "2023-11-19-red-sea-shipping-disruption": {"XLB": 0.12},
        },
    ),
    "F": SnapshotVariantProfile(
        name="F",
        shock_impact_scale=0.0,
        shock_structural_scale=0.0,
        shock_region_scale=0.0,
        shock_half_life_scale=0.20,
        structural_impact_scale=1.0,
        structural_structural_scale=1.0,
        structural_region_scale=1.0,
        structural_half_life_scale=1.0,
        global_impact_scale=1.0,
        global_structural_scale=1.0,
        base_coverage_scale=0.72,
        base_mapping_confidence_scale=0.70,
        event_asset_impact_scales={
            "2020-05-15-huawei-semiconductor-rule": {"XLK": 1.0, "XLB": 0.45, "XLY": 0.0, "XLRE": 0.0},
            "2022-10-07-china-chip-export-controls": {"XLK": 1.0, "XLB": 0.45, "XLY": 0.0, "XLRE": 0.0},
            "2023-10-17-ai-chip-export-control-expansion": {"XLK": 1.0, "XLB": 0.45, "XLY": 0.0, "XLRE": 0.0},
            "2024-12-02-semiconductor-export-controls": {"XLK": 1.0, "XLB": 0.45, "XLY": 0.0, "XLRE": 0.0},
        },
        event_asset_structural_scales={
            "2020-05-15-huawei-semiconductor-rule": {"XLK": 1.0, "XLB": 0.0},
            "2022-10-07-china-chip-export-controls": {"XLK": 1.0, "XLB": 0.0},
            "2023-10-17-ai-chip-export-control-expansion": {"XLK": 1.0, "XLB": 0.0},
            "2024-12-02-semiconductor-export-controls": {"XLK": 1.0, "XLB": 0.0},
        },
    ),
}


def build_minimal_geo_feature_snapshot(
    *,
    tickers: list[str],
    start_date: str | date = DEFAULT_START_DATE,
    end_date: str | date = DEFAULT_END_DATE,
    reference_price_history: pd.DataFrame | None = None,
    profile: str = "base",
) -> pd.DataFrame:
    variant_profile = _resolve_variant_profile(profile)
    normalized_tickers = sorted({_normalize_ticker(ticker) for ticker in tickers})
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    trade_dates = _build_trade_dates(
        tickers=normalized_tickers,
        start_date=start,
        end_date=end,
        reference_price_history=reference_price_history,
    )
    rows: list[dict[str, object]] = []
    per_asset_rows: dict[str, list[dict[str, object]]] = {ticker: [] for ticker in normalized_tickers}

    for trade_day in trade_dates:
        snapshot_cutoff = datetime.combine(trade_day, time(16, 10), tzinfo=NY_TZ)
        for ticker in normalized_tickers:
            row = _build_asset_day_snapshot(
                trade_day=trade_day,
                ticker=ticker,
                snapshot_cutoff=snapshot_cutoff,
                profile=variant_profile,
            )
            rows.append(row)
            per_asset_rows[ticker].append(row)

    for ticker, ticker_rows in per_asset_rows.items():
        previous_raw: list[float] = []
        for row in ticker_rows:
            current_raw = float(row["geo_net_raw"])
            lagged_raw = previous_raw[-3] if len(previous_raw) >= 3 else 0.0
            row["geo_velocity_3d"] = round(current_raw - lagged_raw, 6)
            previous_raw.append(current_raw)

    frame = pd.DataFrame(rows, columns=MINIMAL_GEO_SNAPSHOT_COLUMNS)
    frame = frame.sort_values(["trade_date", "asset"], kind="mergesort").reset_index(drop=True)
    return frame


def write_minimal_geo_feature_snapshot(
    *,
    output_path: str | Path,
    tickers: list[str],
    start_date: str | date = DEFAULT_START_DATE,
    end_date: str | date = DEFAULT_END_DATE,
    reference_price_history: pd.DataFrame | None = None,
    profile: str = "base",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = build_minimal_geo_feature_snapshot(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        reference_price_history=reference_price_history,
        profile=profile,
    )
    frame.to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic minimal geo_feature_snapshot history.")
    parser.add_argument("--config", default="config/phase8.yaml", help="Pipeline config path.")
    parser.add_argument(
        "--output",
        default="data/processed/geo_feature_snapshot_minimal.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat(), help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat(), help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument(
        "--profile",
        default="base",
        choices=sorted(SNAPSHOT_VARIANT_PROFILES.keys()),
        help="Deterministic snapshot profile to generate.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    reference_price_history = _load_reference_price_history(config.paths.processed_dir / "sector_etf_prices_validated.csv")
    output_path = write_minimal_geo_feature_snapshot(
        output_path=args.output,
        tickers=config.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        reference_price_history=reference_price_history,
        profile=args.profile,
    )
    print(str(output_path))


def _build_trade_dates(
    *,
    tickers: list[str],
    start_date: date,
    end_date: date,
    reference_price_history: pd.DataFrame | None,
) -> list[date]:
    if reference_price_history is None or reference_price_history.empty:
        return [timestamp.date() for timestamp in pd.bdate_range(start_date, end_date)]

    reference = reference_price_history.copy()
    reference["date"] = pd.to_datetime(reference["date"]).dt.date
    reference["ticker"] = reference["ticker"].map(_normalize_ticker)
    reference = reference.loc[
        reference["ticker"].isin(tickers)
        & (reference["date"] >= start_date)
        & (reference["date"] <= end_date)
    ].copy()
    if reference.empty:
        return [timestamp.date() for timestamp in pd.bdate_range(start_date, end_date)]

    daily_counts = reference.groupby("date")["ticker"].nunique()
    qualifying_dates = daily_counts.loc[daily_counts == len(tickers)].index.tolist()
    if not qualifying_dates:
        return [timestamp.date() for timestamp in pd.bdate_range(start_date, end_date)]
    return sorted(qualifying_dates)


def _build_asset_day_snapshot(
    *,
    trade_day: date,
    ticker: str,
    snapshot_cutoff: datetime,
    profile: SnapshotVariantProfile,
) -> dict[str, object]:
    geo_net_raw = 0.0
    geo_structural_raw = 0.0
    region_stress = 0.0
    sector_disruption = 0.0
    infra_disruption = 0.0
    sanctions_score = 0.0
    coverage_score = BASELINE_COVERAGE.get(ticker, 0.55) * profile.base_coverage_scale
    avg_mapping_confidence = (
        BASELINE_MAPPING_CONFIDENCE.get(ticker, 0.60) * profile.base_mapping_confidence_scale
    )
    data_freshness_minutes = BASELINE_FRESHNESS_MINUTES
    cluster_count = 0
    contributing_event_ids: list[str] = []

    for event in SYNTHETIC_GEO_EVENTS:
        weight = _event_weight(event, trade_day, profile=profile)
        if weight <= 0.0:
            continue

        base_impact = _scaled_event_impact(
            event.impacts.get(ticker, 0.0),
            event=event,
            ticker=ticker,
            profile=profile,
        )
        base_structural = _scaled_event_structural_impact(
            event.structural_impacts.get(ticker, 0.0),
            event=event,
            ticker=ticker,
            profile=profile,
        )
        if abs(base_impact) < 1e-12 and abs(base_structural) < 1e-12:
            continue

        weighted_impact = base_impact * weight
        weighted_structural = base_structural * weight
        geo_net_raw += weighted_impact
        geo_structural_raw += weighted_structural
        region_stress += _scaled_region_stress(event=event, profile=profile) * weight
        sector_disruption += abs(base_impact) * weight
        infra_disruption += abs(base_impact) * event.infra_stress * weight
        sanctions_score += abs(base_impact) * event.sanctions_stress * weight

        if abs(weighted_impact) >= 0.01 or abs(weighted_structural) >= 0.01:
            coverage_score = max(coverage_score, _scaled_coverage(event=event, profile=profile))
            avg_mapping_confidence = max(avg_mapping_confidence, _scaled_mapping_confidence(abs(base_impact), profile=profile))
            data_freshness_minutes = min(data_freshness_minutes, event.freshness_minutes)
            contributing_event_ids.append(event.event_id)
            if 0 <= (trade_day - event.first_seen_date).days <= 3:
                cluster_count += 1

    geo_net_score = _clip(math.tanh(geo_net_raw), -1.0, 1.0)
    geo_structural_score = _clip(math.tanh(geo_structural_raw), -1.0, 1.0)
    geo_break_risk = _clip(
        0.50 * abs(geo_structural_score) + 0.25 * abs(geo_net_score - geo_structural_score),
        0.0,
        1.0,
    )
    hard_override = bool(
        abs(geo_structural_score) >= 0.75
        and coverage_score >= 0.75
        and data_freshness_minutes <= 120
    )

    return {
        "trade_date": trade_day.isoformat(),
        "asset": ticker,
        "snapshot_cutoff_at": snapshot_cutoff.isoformat(),
        "geo_net_raw": round(geo_net_raw, 6),
        "geo_net_score": round(geo_net_score, 6),
        "geo_structural_score": round(geo_structural_score, 6),
        "geo_harm_score": round(max(0.0, -geo_net_score), 6),
        "geo_break_risk": round(geo_break_risk, 6),
        "geo_velocity_3d": 0.0,
        "geo_cluster_72h": float(cluster_count),
        "region_stress": round(region_stress, 6),
        "sector_disruption": round(sector_disruption, 6),
        "infra_disruption": round(infra_disruption, 6),
        "sanctions_score": round(sanctions_score, 6),
        "avg_mapping_confidence": round(_clip(avg_mapping_confidence, 0.0, 1.0), 6),
        "coverage_score": round(_clip(coverage_score, 0.0, 1.0), 6),
        "data_freshness_minutes": int(data_freshness_minutes),
        "hard_override": hard_override,
        "contributing_event_ids": json.dumps(sorted(set(contributing_event_ids))),
    }


def _event_weight(
    event: SyntheticGeoEvent,
    trade_day: date,
    *,
    profile: SnapshotVariantProfile,
) -> float:
    if trade_day < event.first_seen_date:
        return 0.0
    if trade_day <= event.active_end_date:
        return 1.0
    decay_days = (trade_day - event.active_end_date).days
    variant_scale = (
        profile.shock_half_life_scale
        if event.event_class == "shock"
        else profile.structural_half_life_scale
    )
    adjusted_half_life = max(1.0, float(event.half_life_days) * variant_scale)
    weight = math.exp(-math.log(2.0) * decay_days / adjusted_half_life)
    return weight if weight >= 1e-4 else 0.0


def _resolve_variant_profile(profile: str) -> SnapshotVariantProfile:
    try:
        return SNAPSHOT_VARIANT_PROFILES[profile]
    except KeyError as exc:
        available = ", ".join(sorted(SNAPSHOT_VARIANT_PROFILES))
        raise ValueError(f"Unknown snapshot profile '{profile}'. Available: {available}") from exc


def _scaled_event_impact(
    base_impact: float,
    *,
    event: SyntheticGeoEvent,
    ticker: str,
    profile: SnapshotVariantProfile,
) -> float:
    event_scale = profile.shock_impact_scale if event.event_class == "shock" else profile.structural_impact_scale
    asset_scale = profile.event_asset_impact_scales.get(event.event_id, {}).get(ticker, 1.0)
    return base_impact * event_scale * profile.global_impact_scale * asset_scale


def _scaled_event_structural_impact(
    base_structural: float,
    *,
    event: SyntheticGeoEvent,
    ticker: str,
    profile: SnapshotVariantProfile,
) -> float:
    event_scale = (
        profile.shock_structural_scale
        if event.event_class == "shock"
        else profile.structural_structural_scale
    )
    asset_scale = profile.event_asset_structural_scales.get(event.event_id, {}).get(ticker, 1.0)
    return base_structural * event_scale * profile.global_structural_scale * asset_scale


def _scaled_region_stress(
    *,
    event: SyntheticGeoEvent,
    profile: SnapshotVariantProfile,
) -> float:
    event_scale = profile.shock_region_scale if event.event_class == "shock" else profile.structural_region_scale
    return event.region_stress * event_scale


def _scaled_coverage(
    *,
    event: SyntheticGeoEvent,
    profile: SnapshotVariantProfile,
) -> float:
    if profile.global_coverage_cap is None:
        return event.coverage_score
    return min(event.coverage_score, profile.global_coverage_cap)


def _scaled_mapping_confidence(
    impact_abs: float,
    *,
    profile: SnapshotVariantProfile,
) -> float:
    confidence = min(0.95, 0.70 + 0.20 * min(1.0, impact_abs))
    if profile.global_mapping_confidence_cap is None:
        return confidence
    return min(confidence, profile.global_mapping_confidence_cap)


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _normalize_ticker(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("ticker identifiers must be non-empty")
    return normalized


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _load_reference_price_history(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["date"])
    if "is_valid" in frame.columns:
        frame = frame.loc[frame["is_valid"]].copy()
    return frame.loc[:, [column for column in ("date", "ticker") if column in frame.columns]].copy()


if __name__ == "__main__":
    main()
