from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config_loader import PipelineConfig, load_config
from ..trade_journal import TradeJournal


CORRECTIVE_SIGNALS = {
    "COST_KILLED": "raise minimum z-score for this asset by 0.1",
    "FALSE_REVERSION": "increase momentum filter weight for next month",
    "CORRELATION_BREAKDOWN": "node_corr_floor may be too low",
    "HOLDING_TOO_LONG": "consider tightening max_hold_days by 1",
    "VOLATILITY_MISMATCH": "add vol-scaling to position sizing",
    "UNCATEGORIZED": "review trade manually for a novel loss pattern",
}


@dataclass(frozen=True)
class MistakeAnalysisResult:
    categorized_rate: float
    category_counts: dict[str, int]
    category_percentages: dict[str, float]
    corrective_signals: list[dict[str, object]]
    trade_records: pd.DataFrame
    output_path: Path | None = None


class MistakeAnalyzer:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._trade_details = self._load_trade_details()
        self._price_history = self._load_price_history()

    def analyze_period(
        self,
        trade_journal: TradeJournal,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> MistakeAnalysisResult:
        losing_trades = trade_journal.get_losing_trades(start_date=start_date, end_date=end_date).copy()
        if losing_trades.empty:
            return MistakeAnalysisResult(
                categorized_rate=1.0,
                category_counts={},
                category_percentages={},
                corrective_signals=[],
                trade_records=losing_trades,
            )

        losing_trades = losing_trades.merge(
            self._trade_details,
            on="trade_id",
            how="left",
            suffixes=("", "_detail"),
        )
        categories: list[str] = []
        corrective_signals: list[str] = []
        for _, trade in losing_trades.iterrows():
            category = self._classify_trade(trade)
            categories.append(category)
            corrective_signals.append(CORRECTIVE_SIGNALS[category])

        losing_trades["category"] = categories
        losing_trades["corrective_signal"] = corrective_signals

        category_counts = losing_trades["category"].value_counts().to_dict()
        total_losses = len(losing_trades)
        category_percentages = {
            category: float(count / total_losses)
            for category, count in sorted(category_counts.items())
        }
        categorized_rate = 1.0 - category_percentages.get("UNCATEGORIZED", 0.0)
        prioritized_signals = (
            losing_trades.loc[losing_trades["category"] != "UNCATEGORIZED", "corrective_signal"]
            .value_counts()
            .reset_index()
        )
        prioritized_signals.columns = ["signal", "count"]
        corrective_ranked = prioritized_signals.to_dict(orient="records")

        return MistakeAnalysisResult(
            categorized_rate=float(categorized_rate),
            category_counts={str(key): int(value) for key, value in category_counts.items()},
            category_percentages=category_percentages,
            corrective_signals=corrective_ranked,
            trade_records=losing_trades,
        )

    def generate_report(
        self,
        analysis_results: MistakeAnalysisResult,
        *,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> Path:
        start_label = pd.Timestamp(start_date).date().isoformat()
        end_label = pd.Timestamp(end_date).date().isoformat()
        output_path = self.config.paths.project_root / "diagnostics" / f"mistake_analysis_{end_label}.md"
        lines = [
            f"# Mistake Analysis: {start_label} to {end_label}",
            "",
            f"- Categorization rate: `{analysis_results.categorized_rate:.2%}`",
            f"- Total losing trades: `{len(analysis_results.trade_records)}`",
            "",
            "## Category Breakdown",
        ]
        for category, count in sorted(
            analysis_results.category_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            percentage = analysis_results.category_percentages.get(category, 0.0)
            lines.append(f"- `{category}`: `{count}` (`{percentage:.2%}`)")

        lines.append("")
        lines.append("## Corrective Signals")
        if analysis_results.corrective_signals:
            for item in analysis_results.corrective_signals:
                lines.append(f"- `{item['signal']}`: `{item['count']}`")
        else:
            lines.append("- None")

        if not analysis_results.trade_records.empty:
            lines.append("")
            lines.append("## Sample Losing Trades")
            preview = analysis_results.trade_records.loc[
                :,
                ["trade_id", "strategy", "asset", "category", "net_pnl", "exit_reason"],
            ].head(10)
            lines.append(preview.to_markdown(index=False))

        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    def _classify_trade(self, trade: pd.Series) -> str:
        if float(trade["gross_pnl"]) > 0 and float(trade["net_pnl"]) < 0:
            return "COST_KILLED"

        entry_zscore = trade.get("entry_zscore")
        exit_zscore = trade.get("exit_zscore")
        if (
            entry_zscore is not None
            and not pd.isna(entry_zscore)
            and abs(float(entry_zscore)) >= self.config.phase2.signal_threshold
            and exit_zscore is not None
            and not pd.isna(exit_zscore)
            and np.sign(float(entry_zscore)) == np.sign(float(exit_zscore))
            and abs(float(exit_zscore)) > abs(float(entry_zscore))
        ):
            return "FALSE_REVERSION"

        entry_node_corr = trade.get("entry_node_corr")
        if entry_node_corr is not None and not pd.isna(entry_node_corr) and float(entry_node_corr) < 0.20:
            return "CORRELATION_BREAKDOWN"

        if (
            int(trade["holding_days"]) == self.config.phase2.max_holding_days
            and float(trade["net_pnl"]) < 0
        ):
            return "HOLDING_TOO_LONG"

        if self._is_volatility_mismatch(trade):
            return "VOLATILITY_MISMATCH"

        return "UNCATEGORIZED"

    def _is_volatility_mismatch(self, trade: pd.Series) -> bool:
        asset = str(trade["asset"])
        if asset not in self._price_history:
            return False
        prices = self._price_history[asset]
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date = pd.Timestamp(trade["exit_date"])
        entry_history = prices.loc[prices.index < entry_date].tail(20)
        trade_history = prices.loc[(prices.index >= entry_date) & (prices.index <= exit_date)]
        if len(entry_history) < 20 or len(trade_history) < 2:
            return False
        entry_vol = float(entry_history.pct_change().dropna().std(ddof=0))
        realized_vol = float(trade_history.pct_change().dropna().std(ddof=0))
        if entry_vol <= 0:
            return False
        return realized_vol > (1.5 * entry_vol)

    def _load_trade_details(self) -> pd.DataFrame:
        trade_log_path = self.config.paths.processed_dir / "phase5_trade_log.csv"
        if not trade_log_path.exists():
            return pd.DataFrame(columns=["trade_id", "exit_zscore", "strategy_id"])
        details = pd.read_csv(trade_log_path, parse_dates=["entry_date", "exit_date"])
        keep_columns = [column for column in ["trade_id", "exit_zscore", "strategy_id"] if column in details.columns]
        return details.loc[:, keep_columns].copy()

    def _load_price_history(self) -> dict[str, pd.Series]:
        price_frames: list[pd.DataFrame] = []
        sector_path = self.config.paths.processed_dir / "sector_etf_prices_validated.csv"
        trend_path = self.config.paths.processed_dir / "trend_universe_prices_validated.csv"
        for path in (sector_path, trend_path):
            if not path.exists():
                continue
            frame = pd.read_csv(path, parse_dates=["date"])
            valid = frame.loc[frame["is_valid"], ["date", "ticker", "adj_close"]].copy()
            price_frames.append(valid)
        if not price_frames:
            return {}
        combined = pd.concat(price_frames, ignore_index=True)
        result: dict[str, pd.Series] = {}
        for ticker, group in combined.groupby("ticker"):
            result[str(ticker)] = group.sort_values("date").set_index("date")["adj_close"]
        return result


def run_mistake_analysis(
    config_path: str | Path,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> MistakeAnalysisResult:
    config = load_config(config_path)
    journal = TradeJournal(config.paths.project_root / config.learning.trade_journal_path)
    try:
        analyzer = MistakeAnalyzer(config)
        result = analyzer.analyze_period(journal, start_date, end_date)
        output_path = analyzer.generate_report(result, start_date=start_date, end_date=end_date)
    finally:
        journal.close()
    return MistakeAnalysisResult(
        categorized_rate=result.categorized_rate,
        category_counts=result.category_counts,
        category_percentages=result.category_percentages,
        corrective_signals=result.corrective_signals,
        trade_records=result.trade_records,
        output_path=output_path,
    )
