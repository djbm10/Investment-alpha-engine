# Phase 4 Runtime Profile

## Command

Profiled with a hard 10-minute cap:

```bash
timeout 600s python3 -u -m src.main run-phase4
```

## Captured Output

```text
[PROFILE] Graph engine setup: 12.8s
[PROFILE] Feature building: 127.1s
[PROFILE] Window 2022-07-01:2022-12-31 started
[PROFILE] Window 2022-07-01:2022-12-31 training: 176.9s
[PROFILE] Window 2022-07-01:2022-12-31 inference: 5.4s for 127 samples
[PROFILE] Window 2023-01-01:2023-06-30 started
[PROFILE] Window 2023-01-01:2023-06-30 training: 261.6s
[PROFILE] Window 2023-01-01:2023-06-30 inference: 1.5s for 124 samples
[PROFILE] Window 2023-07-01:2023-12-31 started
```

The command hit the `timeout` limit and exited with code `124`.

## Bottleneck

The dominant bottleneck is **Scenario A: walk-forward retraining**.

Why:
- Graph engine setup is only `12.8s`.
- Feature building is non-trivial at `127.1s`, so it is a meaningful secondary cost.
- Per-window inference is negligible: `5.4s` and `1.5s`.
- Per-window ensemble training is the dominant repeated cost:
  - Window 1 training: `176.9s`
  - Window 2 training: `261.6s`

Even before the third window completed, the run had already consumed the 10-minute budget. With eight walk-forward windows, retraining inside `run-phase4` is the primary reason the full backtest exceeds the target runtime.

## Conclusion

Priority order from this profile:
1. **A: Move walk-forward ensemble training out of `run-phase4`**
2. **C: Reduce feature-building overhead**
3. **B: Batched inference is optional optimization only**
4. **D: Graph engine recomputation is not the main issue**

The next fix should therefore be:
- pre-train and persist all walk-forward ensembles in `train-tcn`;
- make `run-phase4` load those saved ensembles and perform inference/backtest only.
