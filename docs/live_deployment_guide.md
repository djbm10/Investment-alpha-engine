# Live Deployment Guide

## Overview
Phase 8 code is complete, but live trading is still gated behind:
- A cleared Phase 7 paper-trading verification.
- A same-day manual live confirmation file.
- Manual review that the Alpaca live account has sufficient capital.

The live path is designed so that deployment is a controlled configuration change, not a new engineering effort.

## 1. Confirm The Prerequisites
Before switching to live mode, confirm all of the following:
- `python3 -m src.main verify-phase7-gate` passes.
- `config/phase7_cleared.yaml` exists.
- Paper trading has completed without unresolved critical alerts.
- `config/credentials.yaml` contains the live Alpaca credentials you intend to use.
- The live account has at least the configured minimum capital.

Check readiness with:

```bash
python3 -m src.main check-live-readiness
```

## 2. Create `config/live_confirmed.txt`
This file is the manual tripwire that prevents accidental live deployment.

Create `config/live_confirmed.txt` with exactly two lines:

```text
YYYY-MM-DD
signature: YOUR_NAME_OR_INITIALS
```

Requirements:
- The first line must be today's date in `YYYY-MM-DD`.
- The second line must begin with `signature:`.
- Do not commit this file. It is ignored by git.

## 3. Switch To Live Mode
Once readiness checks pass:

```bash
python3 -m src.main deploy-live
```

This command:
- Re-runs the live readiness checks.
- Validates `live_confirmed.txt`.
- Confirms the account equity is above `deployment.min_capital`.
- Switches `deployment.mode` from `paper` to `live` in `config/phase8.yaml`.

## 4. Capital Scaling During The First 6 Months
Live capital is scaled in gradually through `deployment.scaling_schedule`:
- Weeks 1-4: 25%
- Weeks 5-12: 50%
- Weeks 13-24: 75%
- Weeks 25+: 100%

This scaling is applied automatically in live mode only. Paper trading continues to use full target sizing.

## 5. Start Live Execution
For the first live day, run manually:

```bash
python3 -m src.main run-daily --mode live
python3 -m src.main run-daily-summary --mode live
python3 -m src.main performance-summary
```

Once the manual run is clean, start the scheduler or cron job:

```bash
python3 -m src.main start-scheduler --mode live
```

## 6. What To Monitor In The First 30 Days
Every live day:
- `logs/daily_reports/report_YYYY-MM-DD.txt`
- `logs/alerts.log`
- `logs/critical_alerts.log`
- `data/performance.db`
- Alpaca account equity, buying power, positions, and fills

Watch specifically for:
- Tracking error drift versus paper expectations
- Any `CRITICAL` health day
- Reconciliation mismatches
- Unexpected position sizes
- Emergency halt flag creation

## 7. Emergency Procedures
To halt trading immediately:

```bash
python3 -m src.main emergency-halt --reason "manual_live_stop"
```

This does two things:
- Writes `logs/emergency_halt.flag`
- Calls the broker directly to close all positions

The scheduler checks for the halt flag and will skip future jobs while it exists.

## 8. Resume After An Emergency Halt
Before resuming:
- Confirm the account is flat.
- Review the triggering alert and root cause.
- Verify that no unresolved reconciliation issue remains.

Then:
1. Remove `logs/emergency_halt.flag`.
2. Re-run `python3 -m src.main check-live-readiness`.
3. Run one manual `python3 -m src.main run-daily --mode live`.
4. Restart the scheduler or cron entry.

## 9. Phase 8 Gate Review
The Phase 8 gate is evaluated later, after:
- Phase 7 paper trading is cleared.
- Live deployment runs for the first 30 days without system errors.
- Live tracking error versus paper remains below 3% annualized.
- All risk limits function correctly in live mode.
