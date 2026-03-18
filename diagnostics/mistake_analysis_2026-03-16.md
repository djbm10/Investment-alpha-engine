# Mistake Analysis: 2020-09-03 to 2026-03-16

- Categorization rate: `96.49%`
- Total losing trades: `57`

## Category Breakdown
- `REVERSAL_OVERSHOOT`: `41` (`71.93%`)
- `TREND_REVERSAL`: `9` (`15.79%`)
- `VOLATILITY_MISMATCH`: `3` (`5.26%`)
- `COST_KILLED`: `2` (`3.51%`)
- `UNCATEGORIZED`: `2` (`3.51%`)

## Corrective Signals
- `require a small positive P&L buffer before reversion exits`: `41`
- `de-risk the trend sleeve when trend breadth weakens`: `9`
- `add vol-scaling to position sizing`: `3`
- `raise minimum z-score for this asset by 0.1`: `2`

## Sample Losing Trades
| trade_id | strategy | asset | category | net_pnl | exit_reason |
| --- | --- | --- | --- | --- | --- |
| phase5-strategy-a:XLK:2020-09-03:2020-09-08:1 | A | XLK | VOLATILITY_MISMATCH | -0.059519008993288325 | stop_loss |
| phase5-strategy-a:XLV:2020-10-29:2020-10-30:1 | A | XLV | REVERSAL_OVERSHOOT | -0.0012949845767536985 | reversion |
| phase5-strategy-a:XLV:2020-11-04:2020-11-05:-1 | A | XLV | REVERSAL_OVERSHOOT | -0.003006150267102269 | reversion |
| phase5-strategy-a:XLE:2020-11-09:2020-11-11:-1 | A | XLE | REVERSAL_OVERSHOOT | -0.023442580053572804 | reversion |
| trend:TLT:2020-10-15:2020-11-13 | B | TLT | VOLATILITY_MISMATCH | -0.022090038171365834 | signal_flip |
| phase5-strategy-a:XLE:2020-11-23:2020-11-24:-1 | A | XLE | UNCATEGORIZED | -0.052382673705083094 | stop_loss |
| phase5-strategy-a:XLB:2021-01-06:2021-01-07:-1 | A | XLB | REVERSAL_OVERSHOOT | -0.008469356126659888 | reversion |
| trend:GLD:2020-10-15:2021-02-17 | B | GLD | TREND_REVERSAL | -0.07076662456257414 | signal_flip |
| phase5-strategy-a:XLY:2021-03-09:2021-03-10:-1 | A | XLY | REVERSAL_OVERSHOOT | -0.006093995675611973 | reversion |
| trend:GLD:2021-07-08:2021-08-06 | B | GLD | TREND_REVERSAL | -0.024177021357600626 | signal_flip |
