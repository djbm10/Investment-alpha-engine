from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config_loader import load_config


@dataclass(frozen=True)
class AssetContributionDiagnosticResult:
    run_id: str
    output_path: Path
    summary_text: str
    breakdown: pd.DataFrame


def diagnose_asset_contribution(config_path: str | Path) -> AssetContributionDiagnosticResult:
    config = load_config(config_path)
    processed_dir = config.paths.processed_dir
    trade_log_path = processed_dir / "phase2_trade_log.csv"
    summary_path = processed_dir / "phase2_summary.json"

    if not trade_log_path.exists():
        raise ValueError(f"Phase 2 trade log was not found at '{trade_log_path}'.")
    if not summary_path.exists():
        raise ValueError(f"Phase 2 summary was not found at '{summary_path}'.")

    trade_log = pd.read_csv(trade_log_path, parse_dates=["entry_date", "exit_date"])
    if trade_log.empty:
        raise ValueError("Phase 2 trade log is empty; no asset contribution analysis is available.")

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    run_id = str(summary_payload.get("run_id", "unknown"))
    breakdown = build_asset_contribution_breakdown(trade_log)

    output_dir = config.paths.project_root / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "asset_contribution.csv"
    breakdown.to_csv(output_path, index=False)

    summary_text = build_asset_contribution_summary_text(breakdown)
    print(summary_text)

    return AssetContributionDiagnosticResult(
        run_id=run_id,
        output_path=output_path,
        summary_text=summary_text,
        breakdown=breakdown,
    )


def build_asset_contribution_breakdown(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "total_pnl_contribution",
                "asset_trade_sharpe",
                "trade_count",
                "win_rate",
                "avg_holding_days",
                "avg_entry_weight",
                "anchor_asset",
                "drag_asset",
                "inert_asset",
            ]
        )

    enriched = trade_log.copy()
    enriched["pnl_contribution"] = enriched["entry_weight"].astype(float) * enriched["net_return"].astype(float)
    per_asset = (
        enriched.groupby("ticker")
        .apply(
            lambda frame: pd.Series(
                {
                    "total_pnl_contribution": float(frame["pnl_contribution"].sum()),
                    "asset_trade_sharpe": _trade_sharpe(frame["net_return"]),
                    "trade_count": int(len(frame)),
                    "win_rate": float((frame["net_return"] > 0).mean()),
                    "avg_holding_days": float(frame["holding_days"].mean()),
                    "avg_entry_weight": float(frame["entry_weight"].mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    per_asset["anchor_asset"] = (
        per_asset["asset_trade_sharpe"].gt(0)
        & per_asset["trade_count"].ge(15)
        & per_asset["win_rate"].gt(0.55)
    )
    per_asset["drag_asset"] = (
        per_asset["total_pnl_contribution"].lt(0) | per_asset["asset_trade_sharpe"].lt(0)
    )
    per_asset["inert_asset"] = per_asset["trade_count"].lt(10)
    return per_asset.sort_values(
        by=["total_pnl_contribution", "asset_trade_sharpe", "trade_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_asset_contribution_summary_text(breakdown: pd.DataFrame) -> str:
    if breakdown.empty:
        return "No asset-level trade data was available for analysis."

    anchors = breakdown.loc[breakdown["anchor_asset"], "ticker"].tolist()
    drags = breakdown.loc[breakdown["drag_asset"], "ticker"].tolist()
    inert = breakdown.loc[breakdown["inert_asset"], "ticker"].tolist()
    best_asset = breakdown.iloc[0]
    worst_asset = breakdown.sort_values("total_pnl_contribution", ascending=True).iloc[0]

    summary_lines = [
        f"{len(anchors)} anchor assets, {len(drags)} drag assets, and {len(inert)} inert assets were identified.",
        f"Top contributor: {best_asset['ticker']} with contribution {best_asset['total_pnl_contribution']:.6f}, trade Sharpe {best_asset['asset_trade_sharpe']:.3f}, and {int(best_asset['trade_count'])} trades.",
        f"Worst contributor: {worst_asset['ticker']} with contribution {worst_asset['total_pnl_contribution']:.6f}, trade Sharpe {worst_asset['asset_trade_sharpe']:.3f}, and {int(worst_asset['trade_count'])} trades.",
    ]
    if anchors:
        summary_lines.append("Anchors: " + ", ".join(anchors) + ".")
    if drags:
        summary_lines.append("Drags: " + ", ".join(drags) + ".")
    if inert:
        summary_lines.append("Inert: " + ", ".join(inert) + ".")
    return " ".join(summary_lines)


def _trade_sharpe(returns: pd.Series) -> float:
    volatility = returns.std(ddof=0)
    if returns.empty or volatility == 0 or pd.isna(volatility):
        return 0.0
    return float((returns.mean() / volatility) * np.sqrt(len(returns)))
