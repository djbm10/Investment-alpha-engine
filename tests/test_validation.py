import pandas as pd

from src.config_loader import ValidationConfig
from src.validation import build_issue_report, build_quality_report, validate_prices


def test_validate_prices_flags_expected_conditions() -> None:
    prices = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "ticker": "XLK",
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100,
                "adj_close": 100,
                "volume": 1000,
                "dividends": 0,
                "stock_splits": 0,
                "capital_gains": 0,
            },
            {
                "date": "2024-01-03",
                "ticker": "XLK",
                "open": 150,
                "high": 151,
                "low": 149,
                "close": 150,
                "adj_close": 150,
                "volume": 0,
                "dividends": 0.5,
                "stock_splits": 0,
                "capital_gains": 0,
            },
            {
                "date": "2024-01-10",
                "ticker": "XLK",
                "open": 151,
                "high": 152,
                "low": 150,
                "close": 151,
                "adj_close": 120,
                "volume": 1000,
                "dividends": 0,
                "stock_splits": 0,
                "capital_gains": 0,
            },
        ]
    )
    config = ValidationConfig(
        max_abs_daily_return=0.5,
        max_missing_business_days=3,
        adj_close_close_tolerance=0.001,
    )

    validated = validate_prices(prices, config)
    issue_report = build_issue_report(validated)
    quality_report = build_quality_report(validated)

    assert bool(validated.loc[1, "zero_volume_flag"]) is True
    assert bool(validated.loc[1, "corporate_action_flag"]) is True
    assert bool(validated.loc[2, "continuity_flag"]) is True
    assert bool(validated.loc[2, "split_detection_flag"]) is True
    assert issue_report.shape[0] == 2
    assert int(quality_report.loc[0, "invalid_rows"]) == 2
