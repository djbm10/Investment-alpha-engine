from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

EXPECTED_COLUMNS = [
    "date",
    "ticker",
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

SOURCE_COLUMN_MAP = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
    "Capital Gains": "capital_gains",
}


def download_universe_data(
    tickers: list[str],
    start_date: str,
    end_date: str | None,
    cache_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        logger.info(
            "Downloading ticker history",
            extra={"context": {"ticker": ticker, "start_date": start_date, "end_date": end_date}},
        )
        frames.append(download_ticker_history(ticker, start_date, end_date))

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def download_ticker_history(ticker: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    history = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
    )
    if history.empty:
        raise RuntimeError(f"No data returned for ticker {ticker}. Check ticker symbol or network access.")

    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)

    history = history.reset_index().rename(columns=SOURCE_COLUMN_MAP)
    history["ticker"] = ticker

    for column in EXPECTED_COLUMNS:
        if column not in history.columns:
            history[column] = 0.0 if column not in {"date", "ticker"} else pd.NA

    return history[EXPECTED_COLUMNS]
