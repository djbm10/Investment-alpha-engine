# Investment Alpha Engine

Phase 1 implements the infrastructure and data pipeline from `docs/plan.md`.

## Stack

- Python 3.12
- Poetry with pinned dependencies in `pyproject.toml`
- `yfinance` for development historical data ingestion
- YAML configuration in `config/phase1.yaml`
- Structured JSON logging in `logs/`
- Local PostgreSQL via `pgserver` and `psycopg`
- APScheduler for the daily 4:15 PM ET schedule

## Commands

- `python3 -m src.main run-pipeline`
- `python3 -m src.main init-db`
- `python3 -m src.main verify-phase1`
- `python3 -m src.main show-schedule`
- `pytest`
- `poetry check`

## Outputs

- Raw download CSV: `data/raw/sector_etf_prices_raw.csv`
- Validated CSV: `data/processed/sector_etf_prices_validated.csv`
- Data quality report: `data/processed/sector_etf_quality_report.csv`
- Validation issues: `data/processed/sector_etf_validation_issues.csv`
- Structured pipeline log: `logs/phase1_pipeline.jsonl`
- Local PostgreSQL cluster: `data/postgres/phase1`
