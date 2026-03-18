# AGENTS.md

## Objective
Build the algorithmic trading system described in docs/plan.md.

## Rules
- Work step-by-step. Do NOT skip steps.
- Only implement ONE phase at a time.
- Do NOT jump ahead to advanced models.
- Always verify code runs before moving on.
- Keep code simple and modular.
- Use Python only.
- Continually update the `Current Focus` section whenever work progresses so it always shows the active task and next step.

## Current Focus
Phase 7: Code Complete, Gate Pending
- Active task: Phase 7 implementation is complete; the remaining work is the 90-day Alpaca paper-trading gate evaluation
- Next step: Create `config/credentials.yaml` from the template, add Alpaca paper keys, schedule the daily run, and begin paper trading

## Tasks
1. Set up data ingestion using yfinance
2. Pull historical price data for sector ETFs
3. Clean and validate data
4. Store data (start with CSV, upgrade later)
5. Create reusable functions in /src

## File Structure
- src/ -> all code
- data/ -> raw + processed data
- config/ -> parameters
- logs/ -> logs
- docs/ -> reference documents

## Execution Instructions
- After completing a task, immediately move to the next
- Do NOT stop unless there is an error
- If blocked, explain the issue and propose a fix
- Always update or create code files directly
- Keep `Current Focus` in `AGENTS.md` up to date during edits so progress and the next action are always clear

## Constraints
- No unnecessary libraries
- No overengineering
- No skipping validation steps
