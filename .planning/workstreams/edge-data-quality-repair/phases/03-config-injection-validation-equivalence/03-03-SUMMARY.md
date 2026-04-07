---
phase: 03-config-injection-validation-equivalence
plan: 03
subsystem: data_pre/bf_edge_v3/tests
tags: [equivalence-gate, testing, ref-06, bit-identical]
dependency_graph:
  requires: [03-01, 03-02]
  provides: [test_equivalence.py]
  affects: []
tech_stack:
  added: []
  patterns: [module-scoped-fixtures, np-array-equal]
key_files:
  created:
    - data_pre/bf_edge_v3/tests/test_equivalence.py
  modified: []
decisions:
  - "All comparisons use np.array_equal (bit-identical), zero tolerance-based comparisons"
  - "Module-scoped fixtures run pipeline once, reused across 9 tests"
  - "In-memory path test confirms build_support_dataset_v3 pattern matches per-stage results"
metrics:
  duration: ~3m
  completed: "2026-04-07"
  tasks_completed: 1
  tasks_total: 1
  files_created: 1
  files_modified: 0
---

# Phase 03 Plan 03: Equivalence Gate Summary

Implemented formal equivalence gate (REF-06) as a pytest suite proving the refactored pipeline with config system and validation hooks produces bit-identical output to pre-change reference data on scene 010101.

## Task Results

### Task 1: Equivalence gate pytest suite

**Commit:** 6749edc

Created `tests/test_equivalence.py` (308 lines) with 9 tests:

1. **test_stage1_boundary_centers_identical** — All NPZ fields in boundary_centers match reference
2. **test_stage2_local_clusters_identical** — All NPZ fields in local_clusters match reference
3. **test_stage3_supports_identical** — All 27 NPZ fields in supports match reference (covers to_runtime_dict() transition)
4. **test_stage4_edge_dist_identical** — edge_dist.npy bit-identical
5. **test_stage4_edge_dir_identical** — edge_dir.npy bit-identical
6. **test_stage4_edge_valid_identical** — edge_valid.npy bit-identical
7. **test_stage4_edge_support_id_identical** — edge_support_id.npy bit-identical
8. **test_validation_passes_on_refactored_output** — All 4 validation hooks pass without raising
9. **test_inmemory_path_matches_perstage** — Stages 1→2→3 in-memory path identical to per-stage

All tests pass. No `np.allclose`, `atol=`, or `rtol=` used anywhere — strict `np.array_equal` throughout.

### Task 2: Human verification checkpoint

Pending — full suite results:
- test_config.py: 10 passed
- test_validation.py: 12 passed
- test_equivalence.py: 9 passed
- **Total: 31 passed in ~70s**

## Deviations from Plan

None.

## Self-Check: PASSED

- [x] test_equivalence.py exists (308 lines, > 80 minimum)
- [x] 9 tests covering all 4 stages + validation + in-memory path
- [x] All tests use np.array_equal (grep confirms zero tolerance comparisons)
- [x] Module-scoped fixtures (pipeline runs once, not per-test)
- [x] Skip conditions for missing scene data / reference data
- [x] Diagnostic output on failure (field name, shape, first diffs, max |diff|)
- [x] Commit 6749edc found
