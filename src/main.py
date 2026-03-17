from __future__ import annotations

import argparse
from pathlib import Path

from .config_loader import load_config
from .ingestion import download_universe_data
from .logging_utils import setup_logger
from .storage import ensure_output_directories, save_pipeline_outputs
from .validation import build_issue_report, build_quality_report, validate_prices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 1 data pipeline.")
    parser.add_argument(
        "--config",
        default="config/phase1.json",
        help="Path to the Phase 1 configuration file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_directories(config.paths)

    logger = setup_logger(config.paths.log_file)
    logger.info(
        "Starting Phase 1 pipeline",
        extra={"context": {"tickers": config.tickers, "config_path": str(Path(args.config))}},
    )

    raw_prices = download_universe_data(
        tickers=config.tickers,
        start_date=config.start_date,
        end_date=config.end_date,
        cache_dir=config.paths.cache_dir,
        logger=logger,
    )
    validated_prices = validate_prices(raw_prices, config.validation)
    quality_report = build_quality_report(validated_prices)
    issue_report = build_issue_report(validated_prices)
    output_paths = save_pipeline_outputs(
        raw_prices=raw_prices,
        validated_prices=validated_prices,
        quality_report=quality_report,
        issue_report=issue_report,
        paths=config.paths,
    )

    logger.info(
        "Completed Phase 1 pipeline",
        extra={
            "context": {
                "raw_rows": len(raw_prices),
                "clean_rows": int(validated_prices["is_valid"].sum()),
                "issue_rows": len(issue_report),
                "output_paths": {name: str(path) for name, path in output_paths.items()},
            }
        },
    )

    print("Phase 1 pipeline completed.")
    print(f"Raw rows: {len(raw_prices)}")
    print(f"Clean rows: {int(validated_prices['is_valid'].sum())}")
    print(f"Issue rows: {len(issue_report)}")
    for name, path in output_paths.items():
        print(f"{name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
