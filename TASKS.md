# Tasks

## Active

### RCA-1: Relax node_corr_floor in REDUCED regime
**Priority:** HIGH
**Status:** ✅ complete
**Files:** `src/correlation_filter.py`, `src/config_loader.py`, `config/phase7.yaml`, `tests/test_correlation_filter.py`

Steps:
- [ ] Add `reduced_node_corr_multiplier: float = 0.75` to `Phase2Config`
- [ ] Add param to `config/phase7.yaml`
- [ ] Update `node_tradeable_mask()` signature to accept `regime_state`
- [ ] Apply multiplier when `regime_state == REDUCED_REGIME`
- [ ] Update call-site in `graph_engine.py` to pass `regime_state`
- [ ] Add/update tests in `test_correlation_filter.py`

---

### RCA-2: Add node_avg_corr to top-signals diagnostic
**Priority:** MEDIUM
**Status:** ✅ complete
**Files:** `src/pipeline.py`

Steps:
- [ ] Extend `_z_pairs` tuples to include `node_avg_corr`
- [ ] Update `DecisionSummary.top_z_scores` and `closest_to_threshold` types to carry the extra field
- [ ] Update `_log_decision_summary` print block to show `node_avg_corr` per asset
- [ ] Update JSON persistence to include `node_avg_corr` in each tuple

---

### RCA-3: Log graph_density in decision summary
**Priority:** LOW / MONITOR
**Status:** ✅ complete (folded into RCA-2 implementation)
**Files:** `src/pipeline.py`

Steps:
- [x] Add `graph_density: float` field to `DecisionSummary`
- [x] Populate from `_first_snap.get("graph_density", 0.0)` in `run_daily`
- [x] Print in `_log_decision_summary`
- [x] Include in JSON persistence

---

## Completed

_(none yet)_
