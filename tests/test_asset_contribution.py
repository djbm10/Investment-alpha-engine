import pandas as pd

from src.diagnostics.asset_contribution import (
    build_asset_contribution_breakdown,
    build_asset_contribution_summary_text,
)


def test_build_asset_contribution_breakdown_classifies_anchor_drag_and_inert_assets() -> None:
    trade_log = pd.DataFrame(
        [
            {
                "ticker": "XLK",
                "entry_weight": 0.10,
                "net_return": 0.03,
                "holding_days": 2,
            }
            for _ in range(16)
        ]
        + [
            {
                "ticker": "XLK",
                "entry_weight": 0.10,
                "net_return": -0.01,
                "holding_days": 3,
            }
        ]
        + [
            {
                "ticker": "XLE",
                "entry_weight": 0.10,
                "net_return": -0.02,
                "holding_days": 4,
            }
            for _ in range(12)
        ]
        + [
            {
                "ticker": "XLU",
                "entry_weight": 0.08,
                "net_return": 0.01,
                "holding_days": 1,
            }
            for _ in range(3)
        ]
    )

    breakdown = build_asset_contribution_breakdown(trade_log)

    xlk = breakdown.loc[breakdown["ticker"] == "XLK"].iloc[0]
    xle = breakdown.loc[breakdown["ticker"] == "XLE"].iloc[0]
    xlu = breakdown.loc[breakdown["ticker"] == "XLU"].iloc[0]
    assert bool(xlk["anchor_asset"]) is True
    assert bool(xlk["drag_asset"]) is False
    assert bool(xle["drag_asset"]) is True
    assert bool(xlu["inert_asset"]) is True


def test_build_asset_contribution_summary_mentions_extremes() -> None:
    breakdown = pd.DataFrame(
        [
            {
                "ticker": "XLK",
                "total_pnl_contribution": 0.12,
                "asset_trade_sharpe": 1.5,
                "trade_count": 18,
                "win_rate": 0.61,
                "avg_holding_days": 2.0,
                "avg_entry_weight": 0.1,
                "anchor_asset": True,
                "drag_asset": False,
                "inert_asset": False,
            },
            {
                "ticker": "XLE",
                "total_pnl_contribution": -0.04,
                "asset_trade_sharpe": -0.5,
                "trade_count": 12,
                "win_rate": 0.4,
                "avg_holding_days": 3.0,
                "avg_entry_weight": 0.08,
                "anchor_asset": False,
                "drag_asset": True,
                "inert_asset": False,
            },
        ]
    )

    summary = build_asset_contribution_summary_text(breakdown)

    assert "Top contributor: XLK" in summary
    assert "Worst contributor: XLE" in summary
    assert "Anchors: XLK." in summary
    assert "Drags: XLE." in summary
