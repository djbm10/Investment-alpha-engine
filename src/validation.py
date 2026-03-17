from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config_loader import ValidationConfig


def validate_prices(prices: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    validated = prices.copy()
    validated["date"] = pd.to_datetime(validated["date"])

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
        "capital_gains",
    ]
    for column in numeric_columns:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")

    validated = validated.sort_values(["ticker", "date"]).reset_index(drop=True)

    validated["daily_return"] = (
        validated.groupby("ticker", group_keys=False)["adj_close"].pct_change().fillna(0.0)
    )
    validated["return_magnitude_flag"] = (
        validated["daily_return"].abs() > config.max_abs_daily_return
    )
    validated["zero_volume_flag"] = validated["volume"].fillna(0) <= 0
    validated["missing_business_days"] = _calculate_missing_business_days(validated)
    validated["continuity_flag"] = (
        validated["missing_business_days"] > config.max_missing_business_days
    )

    close_denominator = validated["close"].replace(0, pd.NA)
    adjustment_gap = (validated["adj_close"] - validated["close"]).abs() / close_denominator
    validated["adjustment_gap_flag"] = adjustment_gap.fillna(0).gt(
        config.adj_close_close_tolerance
    )
    validated["split_detection_flag"] = (
        validated["stock_splits"].fillna(0).ne(0) | validated["adjustment_gap_flag"]
    )
    validated["corporate_action_flag"] = (
        validated["dividends"].fillna(0).ne(0)
        | validated["stock_splits"].fillna(0).ne(0)
        | validated["capital_gains"].fillna(0).ne(0)
    )
    validated["cross_source_status"] = "skipped_single_source_dev"
    validated["cross_source_validation_flag"] = False

    blocking_flags = [
        "return_magnitude_flag",
        "zero_volume_flag",
        "continuity_flag",
    ]
    validated["is_valid"] = ~validated[blocking_flags].any(axis=1)
    validated["validation_notes"] = validated.apply(_build_validation_notes, axis=1)

    return validated


def build_quality_report(validated_prices: pd.DataFrame) -> pd.DataFrame:
    summary = (
        validated_prices.groupby("ticker")
        .agg(
            total_rows=("ticker", "size"),
            valid_rows=("is_valid", "sum"),
            invalid_rows=("is_valid", lambda values: (~values).sum()),
            return_magnitude_issues=("return_magnitude_flag", "sum"),
            zero_volume_issues=("zero_volume_flag", "sum"),
            continuity_issues=("continuity_flag", "sum"),
            split_detection_rows=("split_detection_flag", "sum"),
            adjustment_gap_rows=("adjustment_gap_flag", "sum"),
            corporate_action_events=("corporate_action_flag", "sum"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
    )
    summary["cross_source_status"] = "skipped_single_source_dev"
    return summary


def build_issue_report(validated_prices: pd.DataFrame) -> pd.DataFrame:
    issue_mask = (~validated_prices["is_valid"]) | validated_prices["corporate_action_flag"]
    return validated_prices.loc[issue_mask].copy()


def _calculate_missing_business_days(prices: pd.DataFrame) -> pd.Series:
    missing_days = pd.Series(0, index=prices.index, dtype="int64")

    for _, group in prices.groupby("ticker"):
        previous_dates = group["date"].shift()
        group_missing_days: list[int] = []
        for current_date, previous_date in zip(group["date"], previous_dates):
            if pd.isna(previous_date):
                group_missing_days.append(0)
                continue

            gap_size = len(pd.bdate_range(previous_date, current_date)) - 2
            group_missing_days.append(max(gap_size, 0))

        missing_days.loc[group.index] = group_missing_days

    return missing_days


def _build_validation_notes(row: pd.Series) -> str:
    notes: list[str] = []
    if bool(row["return_magnitude_flag"]):
        notes.append("return_magnitude")
    if bool(row["zero_volume_flag"]):
        notes.append("zero_volume")
    if bool(row["continuity_flag"]):
        notes.append("continuity_gap")
    if bool(row["corporate_action_flag"]):
        notes.append("corporate_action_detected")
    if not notes:
        return "ok"
    return ",".join(notes)
