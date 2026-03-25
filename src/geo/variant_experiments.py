from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..config_loader import load_config
from .evaluation import GeoOverlayEvaluationResult, run_geo_overlay_evaluation
from .minimal_history import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    _load_reference_price_history,
    write_minimal_geo_feature_snapshot,
)

DEFAULT_VARIANT_PROFILES: tuple[str, ...] = ("A", "B", "C")


@dataclass(frozen=True)
class GeoVariantExperimentSummary:
    report: dict[str, object]
    output_path: Path


def run_geo_snapshot_variant_experiments(
    config_path: str | Path,
    *,
    profiles: Sequence[str] = DEFAULT_VARIANT_PROFILES,
    output_dir: str | Path | None = None,
    start_date: str = DEFAULT_START_DATE.isoformat(),
    end_date: str = DEFAULT_END_DATE.isoformat(),
) -> GeoVariantExperimentSummary:
    config = load_config(config_path)
    output_root = Path(output_dir) if output_dir is not None else config.paths.processed_dir
    output_root.mkdir(parents=True, exist_ok=True)
    reference_price_history = _load_reference_price_history(
        config.paths.processed_dir / "sector_etf_prices_validated.csv"
    )

    ordered_profiles = tuple(str(profile) for profile in profiles)
    results: dict[str, dict[str, object]] = {}

    for profile in ordered_profiles:
        snapshot_path = output_root / f"geo_feature_snapshot_minimal_{profile.lower()}.csv"
        write_minimal_geo_feature_snapshot(
            output_path=snapshot_path,
            tickers=config.tickers,
            start_date=start_date,
            end_date=end_date,
            reference_price_history=reference_price_history,
            profile=profile,
        )
        geo_snapshot = _load_snapshot_frame(snapshot_path)
        evaluation_result = run_geo_overlay_evaluation(config_path, geo_snapshot=geo_snapshot)
        report_path = output_root / f"geo_overlay_evaluation_{profile.lower()}.json"
        report_path.write_text(
            json.dumps(evaluation_result.report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        results[profile] = _build_variant_result_entry(
            profile=profile,
            snapshot_path=snapshot_path,
            report_path=report_path,
            evaluation_result=evaluation_result,
        )

    comparison_report = {
        "profiles": list(ordered_profiles),
        "config_path": str(Path(config_path)),
        "data_range": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "results": results,
        "best_profile": _select_best_profile(ordered_profiles, results),
    }
    output_path = output_root / "geo_overlay_variant_comparison.json"
    output_path.write_text(json.dumps(comparison_report, indent=2, sort_keys=True), encoding="utf-8")
    return GeoVariantExperimentSummary(report=comparison_report, output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic A/B/C geo snapshot experiments.")
    parser.add_argument("--config", default="config/phase8.yaml", help="Pipeline config path.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(DEFAULT_VARIANT_PROFILES),
        help="Ordered snapshot profiles to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for variant CSVs and evaluation reports. Defaults to config.paths.processed_dir.",
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat(), help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat(), help="Inclusive end date (YYYY-MM-DD).")
    args = parser.parse_args()

    result = run_geo_snapshot_variant_experiments(
        args.config,
        profiles=args.profiles,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(str(result.output_path))


def _load_snapshot_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "trade_date" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    for column in ("snapshot_cutoff_at", "available_at"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    return frame


def _build_variant_result_entry(
    *,
    profile: str,
    snapshot_path: Path,
    report_path: Path,
    evaluation_result: GeoOverlayEvaluationResult,
) -> dict[str, object]:
    report = evaluation_result.report
    return {
        "profile": profile,
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "acceptance_passed": bool(report.get("acceptance", {}).get("passed", False)),
        "comparison": report.get("comparison", {}),
        "high_geo_blocks": report.get("slices", {}).get("high_geo_blocks", {}),
        "normal_periods": report.get("slices", {}).get("normal_periods", {}),
        "filtering_impact": report.get("filtering_impact", {}),
        "bootstrap": report.get("bootstrap", {}),
    }


def _select_best_profile(
    ordered_profiles: Sequence[str],
    results: dict[str, dict[str, object]],
) -> str | None:
    passing_profiles = [profile for profile in ordered_profiles if results.get(profile, {}).get("acceptance_passed")]
    if not passing_profiles:
        return None
    return max(passing_profiles, key=lambda profile: _profile_rank(results[profile]))


def _profile_rank(result: dict[str, object]) -> tuple[float, float, float]:
    comparison = result.get("comparison", {})
    sharpe_delta = float(comparison.get("sharpe_ratio", {}).get("delta", float("-inf")))
    drawdown_delta = float(comparison.get("max_drawdown", {}).get("delta", float("inf")))
    contradiction_delta = float(comparison.get("contradiction_loss_rate", {}).get("delta", float("inf")))
    return (sharpe_delta, -drawdown_delta, -contradiction_delta)


if __name__ == "__main__":
    main()
