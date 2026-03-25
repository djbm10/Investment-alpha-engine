from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config_loader import load_config
from ..storage import load_validated_price_data
from .regime_validation import KNOWN_CRISIS_WINDOWS, CrisisWindow


@dataclass(frozen=True)
class FalsePositiveDiagnosticResult:
    run_id: str
    output_path: Path
    summary_text: str
    breakdown: pd.DataFrame


def diagnose_regime_false_positives(config_path: str | Path) -> FalsePositiveDiagnosticResult:
    config = load_config(config_path)
    processed_dir = config.paths.processed_dir
    summary_path = processed_dir / "phase3_summary.json"
    regimes_path = processed_dir / "phase3_regime_states.csv"

    if not summary_path.exists():
        raise ValueError(f"Phase 3 summary was not found at '{summary_path}'.")
    if not regimes_path.exists():
        raise ValueError(f"Phase 3 regime states were not found at '{regimes_path}'.")

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    run_id = str(summary_payload.get("run_id", "unknown"))
    regime_history = pd.read_csv(regimes_path, parse_dates=["date"])
    price_history = load_validated_price_data(config, dataset="sector")
    price_history = price_history.loc[
        price_history["is_valid"] & price_history["ticker"].isin(config.tickers),
        ["date", "ticker", "adj_close"],
    ].copy()

    drawdown_start_dates = compute_drawdown_start_dates(price_history, threshold=0.01)
    breakdown = build_false_positive_breakdown(regime_history, KNOWN_CRISIS_WINDOWS, drawdown_start_dates)

    output_dir = config.paths.project_root / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "false_positive_analysis.md"
    output_path.write_text(
        build_false_positive_analysis_markdown(run_id, regime_history, breakdown, KNOWN_CRISIS_WINDOWS),
        encoding="utf-8",
    )

    summary_text = build_false_positive_summary_text(regime_history, breakdown, KNOWN_CRISIS_WINDOWS)
    print(summary_text)

    return FalsePositiveDiagnosticResult(
        run_id=run_id,
        output_path=output_path,
        summary_text=summary_text,
        breakdown=breakdown,
    )


def build_false_positive_breakdown(
    regime_history: pd.DataFrame,
    crisis_windows: list[CrisisWindow],
    drawdown_start_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    if regime_history.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "state",
                "total_distance",
                "classification",
                "days_to_drawdown_start",
                "cluster_size",
            ]
        )

    history = regime_history.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").reset_index(drop=True)
    history["flagged"] = history["state"].isin(["TRANSITIONING", "NEW_REGIME"])
    history["near_crisis"] = history["date"].apply(lambda value: _is_near_crisis(pd.Timestamp(value), crisis_windows))

    false_positive_rows = history.loc[history["flagged"] & ~history["near_crisis"]].copy()
    if false_positive_rows.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "state",
                "total_distance",
                "classification",
                "days_to_drawdown_start",
                "cluster_size",
            ]
        )

    flagged_indices = history.index[history["flagged"]].tolist()
    cluster_lookup = _build_cluster_lookup(flagged_indices)

    classifications: list[str] = []
    days_to_drawdown: list[int | None] = []
    cluster_sizes: list[int] = []
    for row_index, row in false_positive_rows.iterrows():
        next_drawdown = _days_to_next_drawdown_start(pd.Timestamp(row["date"]), drawdown_start_dates)
        cluster_size = cluster_lookup.get(int(row_index), 1)
        if next_drawdown is not None:
            classification = "PRE_DRAWDOWN"
        elif cluster_size == 1:
            classification = "ISOLATED"
        else:
            classification = "CLUSTER"

        classifications.append(classification)
        days_to_drawdown.append(next_drawdown)
        cluster_sizes.append(cluster_size)

    false_positive_rows["classification"] = classifications
    false_positive_rows["days_to_drawdown_start"] = days_to_drawdown
    false_positive_rows["cluster_size"] = cluster_sizes
    return false_positive_rows.loc[
        :,
        ["date", "state", "total_distance", "classification", "days_to_drawdown_start", "cluster_size"],
    ].reset_index(drop=True)


def build_false_positive_summary_text(
    regime_history: pd.DataFrame,
    breakdown: pd.DataFrame,
    crisis_windows: list[CrisisWindow],
) -> str:
    history = regime_history.copy()
    history["date"] = pd.to_datetime(history["date"])
    flagged = history.loc[history["state"].isin(["TRANSITIONING", "NEW_REGIME"])].copy()
    flagged["near_crisis"] = flagged["date"].apply(lambda value: _is_near_crisis(pd.Timestamp(value), crisis_windows))
    true_positive_count = int(flagged["near_crisis"].sum())
    false_positive_count = int((~flagged["near_crisis"]).sum())
    isolated_count = int((breakdown["classification"] == "ISOLATED").sum()) if not breakdown.empty else 0
    cluster_count = int((breakdown["classification"] == "CLUSTER").sum()) if not breakdown.empty else 0
    pre_drawdown_count = int((breakdown["classification"] == "PRE_DRAWDOWN").sum()) if not breakdown.empty else 0
    true_positive_distance = float(flagged.loc[flagged["near_crisis"], "total_distance"].dropna().mean()) if true_positive_count else 0.0
    false_positive_distance = float(flagged.loc[~flagged["near_crisis"], "total_distance"].dropna().mean()) if false_positive_count else 0.0
    return (
        f"{false_positive_count} of {len(flagged)} flagged regime days were outside the crisis buffers. "
        f"{isolated_count} were isolated spikes, {cluster_count} were clustered flags, and {pre_drawdown_count} preceded a >1% "
        f"equal-weight universe drawdown within 10 trading days. Average total distance was {true_positive_distance:.4f} "
        f"for true positives versus {false_positive_distance:.4f} for false positives."
    )


def build_false_positive_analysis_markdown(
    run_id: str,
    regime_history: pd.DataFrame,
    breakdown: pd.DataFrame,
    crisis_windows: list[CrisisWindow],
) -> str:
    history = regime_history.copy()
    history["date"] = pd.to_datetime(history["date"])
    flagged = history.loc[history["state"].isin(["TRANSITIONING", "NEW_REGIME"])].copy()
    flagged["near_crisis"] = flagged["date"].apply(lambda value: _is_near_crisis(pd.Timestamp(value), crisis_windows))
    true_positive_distance = float(flagged.loc[flagged["near_crisis"], "total_distance"].dropna().mean()) if not flagged.empty else 0.0
    false_positive_distance = float(flagged.loc[~flagged["near_crisis"], "total_distance"].dropna().mean()) if not flagged.empty else 0.0

    lines = [
        "# False Positive Analysis",
        "",
        f"- Run ID: `{run_id}`",
        f"- Flagged days: `{len(flagged)}`",
        f"- False positives outside crisis buffers: `{len(breakdown)}`",
        f"- Isolated spikes: `{int((breakdown['classification'] == 'ISOLATED').sum()) if not breakdown.empty else 0}`",
        f"- Clustered flags: `{int((breakdown['classification'] == 'CLUSTER').sum()) if not breakdown.empty else 0}`",
        f"- Pre-drawdown flags: `{int((breakdown['classification'] == 'PRE_DRAWDOWN').sum()) if not breakdown.empty else 0}`",
        f"- Average total distance, true positives: `{true_positive_distance:.4f}`",
        f"- Average total distance, false positives: `{false_positive_distance:.4f}`",
        "",
        "Classification uses the equal-weight universe drawdown start date for the current 8-asset Phase 2 universe.",
        "",
        "| Date | State | Total Distance | Classification | Days To Drawdown Start | Cluster Size |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in breakdown.itertuples(index=False):
        drawdown_value = "-" if pd.isna(row.days_to_drawdown_start) else int(row.days_to_drawdown_start)
        distance_value = "-" if pd.isna(row.total_distance) else f"{float(row.total_distance):.4f}"
        lines.append(
            f"| {row.date.date()} | {row.state} | {distance_value} | {row.classification} | {drawdown_value} | {int(row.cluster_size)} |"
        )

    return "\n".join(lines) + "\n"


def compute_drawdown_start_dates(price_history: pd.DataFrame, threshold: float) -> list[pd.Timestamp]:
    if price_history.empty:
        return []

    pivot = (
        price_history.pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
        .ffill()
        .dropna(how="any")
    )
    if pivot.empty:
        return []

    returns = pivot.pct_change().fillna(0.0)
    equal_weight_returns = returns.mean(axis=1)
    wealth = (1.0 + equal_weight_returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = (wealth / running_peak) - 1.0
    threshold_mask = drawdown <= -abs(threshold)
    start_mask = threshold_mask & ~threshold_mask.shift(1, fill_value=False)
    return [pd.Timestamp(date) for date in drawdown.index[start_mask]]


def _is_near_crisis(date: pd.Timestamp, crisis_windows: list[CrisisWindow]) -> bool:
    for window in crisis_windows:
        buffered_start = window.start - pd.Timedelta(days=5)
        buffered_end = window.end + pd.Timedelta(days=5)
        if buffered_start <= date <= buffered_end:
            return True
    return False


def _build_cluster_lookup(flagged_indices: list[int]) -> dict[int, int]:
    clusters: dict[int, int] = {}
    if not flagged_indices:
        return clusters

    current_cluster = [flagged_indices[0]]
    for index in flagged_indices[1:]:
        if index == current_cluster[-1] + 1:
            current_cluster.append(index)
            continue
        for cluster_index in current_cluster:
            clusters[cluster_index] = len(current_cluster)
        current_cluster = [index]

    for cluster_index in current_cluster:
        clusters[cluster_index] = len(current_cluster)
    return clusters


def _days_to_next_drawdown_start(date: pd.Timestamp, drawdown_start_dates: list[pd.Timestamp]) -> int | None:
    for start_date in drawdown_start_dates:
        delta = (pd.Timestamp(start_date) - date).days
        if 0 < delta <= 10:
            return delta
    return None
