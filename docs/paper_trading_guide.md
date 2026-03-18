# Paper Trading Guide

## Overview
Phase 7 code is complete. The gate is still pending because it requires 90 consecutive calendar days of Alpaca paper trading with daily execution, low tracking error, zero system-caused risk breaches, and the weekly/monthly learning loops running on schedule.

Official references used for the account setup details:
- Alpaca paper trading: https://docs.alpaca.markets/docs/trading/paper-trading/
- Alpaca authentication: https://docs.alpaca.markets/docs/authentication
- Alpaca Trading API getting started: https://docs.alpaca.markets/docs/getting-started-with-trading-api

## 1. Create an Alpaca Paper Trading Account
1. Go to Alpaca's paper trading page and create a paper-only account or sign in to an existing Alpaca account that has paper trading enabled.
2. Confirm that a paper account exists in the Alpaca dashboard.
3. If you create a new paper account, note that Alpaca issues paper credentials separately from live credentials.

## 2. Get API Keys
1. In the Alpaca dashboard, open the paper trading area.
2. Generate or view the paper API key and secret for the paper account you will use.
3. Confirm the paper trading base URL is `https://paper-api.alpaca.markets`.

## 3. Fill In `credentials.yaml`
1. Create the local credentials file from the committed template:

```bash
cp config/credentials.yaml.template config/credentials.yaml
```

2. Edit `config/credentials.yaml` and replace the placeholder values with the paper account key and secret.
3. Leave the base URL set to the paper endpoint unless you are explicitly switching to live trading later.
4. Do not commit `config/credentials.yaml`. The repo already ignores it.

## 4. Run the First Daily Pipeline Manually
1. Make sure the project environment is active and the Alpaca dependency is installed.
2. Run the paper pipeline manually after the market close:

```bash
python3 -m src.main run-daily --mode paper
```

3. Review the console output.
4. Confirm:
   - Orders were either submitted or explicitly rejected by the risk manager.
   - No reconciliation mismatch was reported.
   - The daily summary printed allocation weights, risk headroom, and any alerts.

## 5. Set Up Automated Daily Execution
Run the daily pipeline at 4:30 PM ET, after the market close and after the data inputs should be available.

Example crontab:

```cron
CRON_TZ=America/New_York
30 16 * * 1-5 cd /home/djmann/projects/Investment-alpha-engine && python3 -m src.main run-daily --mode paper >> logs/phase7_paper_trading.log 2>&1
```

Notes:
- `1-5` limits execution to weekdays.
- Keep the schedule 15 minutes after the close unless you explicitly validate a different delay.
- If you run from a different shell environment, use the full interpreter path for the environment that has both the project dependencies and `alpaca-trade-api` installed.

## 6. Monitor the System
Daily:
- Check the console or cron log for `Aborted`, `Manual review`, `Alerts`, and `circuit_action`.
- Check the Alpaca paper dashboard for positions, fills, and buying power.
- Run:

```bash
python3 -m src.main run-daily-summary --mode paper
```

Weekly:
- Confirm a new `diagnostics/mistake_analysis_*.md` report was generated near the end of the trading week.
- Review any repeated rejection reasons or reconciliation issues.

Monthly:
- Confirm `data/processed/bayesian_update_report.json` was refreshed after month-end.
- Review whether the suggested parameter moves are small and directionally sensible.

## 7. Failure Handling
Common failure classes and the expected action:

- `Pre-trade pipeline failure`
  Meaning: data load, signal generation, allocator logic, or target-position construction failed before orders were sent.
  Action: no trades should have been placed; inspect the traceback in the log, fix the issue, rerun manually the next day.

- `Execution failure`
  Meaning: risk check, order submission, or fill retrieval failed after the pipeline reached the broker path.
  Action: immediately review broker positions and rerun reconciliation before the next scheduled day.

- `Position reconciliation mismatch detected`
  Meaning: internal expected positions and broker-reported positions differ.
  Action: halt automation, inspect Alpaca positions manually, and do not resume until the discrepancy is understood.

- `Circuit breaker`
  Meaning: daily, weekly, or monthly loss limits were exceeded.
  Action: the system should already reduce or halt trading according to the configured limit; verify that behavior and do not override it casually.

- `Rejected order ... max_order_pct` or another hard risk limit
  Meaning: the order violated a hard risk constraint.
  Action: treat this as expected protection unless it indicates a sizing bug; investigate repeated occurrences.

## 8. Emergency Stop
If you need to stop the system immediately:
1. Disable the cron entry.
2. If the process is currently running, stop it.
3. Review current Alpaca paper positions and flatten manually in the dashboard if needed.
4. Optionally rename `config/credentials.yaml` to prevent accidental reconnects until you are ready to resume.

## 9. After 90 Days: Phase 7 Gate Review
There is not yet an automated `verify-phase7` command. Use the recorded paper-trading history to evaluate the gate manually:

1. Confirm 90 consecutive calendar days of scheduled paper operation from the cron log and daily state history.
2. Check that there were no missed trading days caused by system failures.
3. Compare realized paper equity against the expected risk-limited reference path and confirm portfolio tracking error stayed below 5% annualized.
4. Review all logged alerts and verify there were zero risk-limit breaches caused by software or execution bugs.
5. Confirm the weekly mistake analysis and monthly Bayesian update artifacts were created on schedule.

Minimum records to review:
- `logs/phase7_paper_trading.log`
- `data/processed/phase7_state.json`
- `data/trade_journal.db`
- Alpaca paper account history and order/fill records

## Environment Note
`alpaca-trade-api` is installed separately for the paper broker adapter because its pinned transitive dependencies conflict with the repo's `yfinance` stack. If you rebuild the environment, reinstall the base project dependencies first, then install `alpaca-trade-api`, and finally confirm that `pytest` still passes before restarting paper trading.
