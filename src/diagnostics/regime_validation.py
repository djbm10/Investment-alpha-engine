from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config_loader import load_config
from ..graph_engine import compute_daily_graph_matrices
from ..tda_regime import RegimeObservation, RegimeState, TDARegimeDetector


@dataclass(frozen=True)
class CrisisWindow:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class CrisisValidationRecord:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    first_transition_date: pd.Timestamp | None
    first_new_regime_date: pd.Timestamp | None
    first_flag_date: pd.Timestamp | None
    detected_within_window: bool
    days_before_window_end: int | None


@dataclass(frozen=True)
class RegimeValidationResult:
    output_path: Path
    hit_rate: float
    false_positive_rate: float
    records: list[CrisisValidationRecord]
    observations: dict[pd.Timestamp, RegimeObservation]


KNOWN_CRISIS_WINDOWS = [
    CrisisWindow("COVID crash", pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23")),
    CrisisWindow("2022 rate shock", pd.Timestamp("2022-01-03"), pd.Timestamp("2022-06-16")),
    CrisisWindow("SVB crisis", pd.Timestamp("2023-03-08"), pd.Timestamp("2023-03-15")),
    CrisisWindow("Aug 2024 vol spike", pd.Timestamp("2024-08-01"), pd.Timestamp("2024-08-08")),
]


def validate_regime_detector(config_path: str | Path) -> RegimeValidationResult:
    config = load_config(config_path)
    price_history = _load_validated_price_history(config.paths.processed_dir, config.tickers)
    graph_matrices = compute_daily_graph_matrices(price_history, config.tickers, config.phase3.rolling_window)
    detector = TDARegimeDetector(config)
    observations = detector.compute_daily_regime(
        {date: snapshot.distance_matrix for date, snapshot in graph_matrices.items()}
    )
    records, hit_rate = evaluate_crisis_detection(observations, KNOWN_CRISIS_WINDOWS)
    false_positive_rate = compute_false_positive_rate(observations, KNOWN_CRISIS_WINDOWS)

    output_dir = config.paths.project_root / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "regime_detection_validation.md"
    output_path.write_text(
        build_regime_validation_markdown(records, hit_rate, false_positive_rate),
        encoding="utf-8",
    )

    print(
        f"Regime detector hit rate {hit_rate:.2%}; false positive rate {false_positive_rate:.2%}."
    )

    return RegimeValidationResult(
        output_path=output_path,
        hit_rate=hit_rate,
        false_positive_rate=false_positive_rate,
        records=records,
        observations=observations,
    )


def evaluate_crisis_detection(
    observations: dict[pd.Timestamp, RegimeObservation],
    crisis_windows: list[CrisisWindow],
) -> tuple[list[CrisisValidationRecord], float]:
    records: list[CrisisValidationRecord] = []
    for window in crisis_windows:
        crisis_days = _flagged_dates_between(observations, window.start, window.end)
        buffered_days = _flagged_dates_between(observations, window.start - pd.Timedelta(days=5), window.end)
        first_transition = _first_state_date(observations, window.start - pd.Timedelta(days=5), window.end, RegimeState.TRANSITIONING)
        first_new_regime = _first_state_date(observations, window.start - pd.Timedelta(days=5), window.end, RegimeState.NEW_REGIME)
        first_flag_date = min(buffered_days) if buffered_days else None
        records.append(
            CrisisValidationRecord(
                name=window.name,
                start=window.start,
                end=window.end,
                first_transition_date=first_transition,
                first_new_regime_date=first_new_regime,
                first_flag_date=first_flag_date,
                detected_within_window=bool(crisis_days),
                days_before_window_end=(window.end - first_flag_date).days if first_flag_date is not None else None,
            )
        )

    hit_rate = float(sum(record.detected_within_window for record in records) / len(records)) if records else 0.0
    return records, hit_rate


def compute_false_positive_rate(
    observations: dict[pd.Timestamp, RegimeObservation],
    crisis_windows: list[CrisisWindow],
) -> float:
    flagged_days = {
        date
        for date, observation in observations.items()
        if observation.state in {RegimeState.TRANSITIONING, RegimeState.NEW_REGIME}
    }
    if not flagged_days:
        return 0.0

    near_crisis_days: set[pd.Timestamp] = set()
    for window in crisis_windows:
        buffered_range = pd.date_range(window.start - pd.Timedelta(days=5), window.end + pd.Timedelta(days=5), freq="D")
        near_crisis_days.update(pd.Timestamp(day) for day in buffered_range)

    false_positive_days = {date for date in flagged_days if pd.Timestamp(date) not in near_crisis_days}
    return float(len(false_positive_days) / len(flagged_days))


def build_regime_validation_markdown(
    records: list[CrisisValidationRecord],
    hit_rate: float,
    false_positive_rate: float,
) -> str:
    lines = [
        "# Regime Detection Validation",
        "",
        f"- Hit rate: `{hit_rate:.2%}`",
        f"- False positive rate: `{false_positive_rate:.2%}`",
        "",
        "| Crisis | Window | First Transition | First New Regime | First Flag | Within Window | Days Before Window End |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    record.name,
                    f"{record.start.date()} to {record.end.date()}",
                    _format_date(record.first_transition_date),
                    _format_date(record.first_new_regime_date),
                    _format_date(record.first_flag_date),
                    "yes" if record.detected_within_window else "no",
                    str(record.days_before_window_end) if record.days_before_window_end is not None else "-",
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def _load_validated_price_history(processed_dir: Path, tickers: list[str]) -> pd.DataFrame:
    validated_path = processed_dir / "sector_etf_prices_validated.csv"
    if not validated_path.exists():
        raise ValueError(f"Validated price history was not found at '{validated_path}'.")

    price_history = pd.read_csv(validated_path, parse_dates=["date"])
    return price_history.loc[
        price_history["is_valid"] & price_history["ticker"].isin(tickers),
        ["date", "ticker", "adj_close"],
    ].copy()


def _flagged_dates_between(
    observations: dict[pd.Timestamp, RegimeObservation],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Timestamp]:
    return [
        pd.Timestamp(date)
        for date, observation in sorted(observations.items())
        if start <= pd.Timestamp(date) <= end
        and observation.state in {RegimeState.TRANSITIONING, RegimeState.NEW_REGIME}
    ]


def _first_state_date(
    observations: dict[pd.Timestamp, RegimeObservation],
    start: pd.Timestamp,
    end: pd.Timestamp,
    state: RegimeState,
) -> pd.Timestamp | None:
    for date, observation in sorted(observations.items()):
        timestamp = pd.Timestamp(date)
        if start <= timestamp <= end and observation.state == state:
            return timestamp
    return None


def _format_date(value: pd.Timestamp | None) -> str:
    return str(value.date()) if value is not None else "-"
