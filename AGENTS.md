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
Phase 1: Empty-State Data Bootstrap Hardening
- Active task: make validated market-data artifacts auto-generate when missing so fresh environments can run without manual setup
- Current step: shared bootstrap fallbacks are implemented and verified for sector/trend validated price files, Phase 2 fallback loading, and Phase 7 date initialization
- Next step: rerun workflow-specific paper-trading or diagnostics commands and patch additional missing-file assumptions only if they surface

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
