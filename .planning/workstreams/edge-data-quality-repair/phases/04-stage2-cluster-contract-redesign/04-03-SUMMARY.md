---
phase: 04-stage2-cluster-contract-redesign
plan: 03
subsystem: data_pre/bf_edge_v3 testing
tags: [contract-tests, density-rescue, equivalence-gate, non-regression]
dependency_graph:
  requires:
    - phase: 04-01
      provides: [rescue_noise_centers, refine_cluster_into_runs, validate_cluster_contract, Stage2Config redesign]
    - phase: 04-02
      provides: [trigger path eliminated, Stage3Config 7 fields, post_fitting.py]
  provides:
    - test_cluster_contract.py with 7 ALG-01 contract invariant tests
    - test_density_rescue.py with 5 ALG-02 rescue unit tests
    - Phase 4 reference_v2 baseline for Stages 2-4
    - Updated equivalence gate (Stage 1 exact-match, Phase 4 v2 for Stages 2-4)
    - Updated validation and config tests for Phase 4 schema
  affects: [Phase 5+ quality repair, future regression testing]
tech_stack:
  added: []
  patterns: [contract-invariant-testing, density-rescue-unit-testing, reference-v2-baseline]
key_files:
  created:
    - data_pre/bf_edge_v3/tests/test_cluster_contract.py
    - data_pre/bf_edge_v3/tests/test_density_rescue.py
  modified:
    - data_pre/bf_edge_v3/tests/test_equivalence.py
    - data_pre/bf_edge_v3/tests/test_validation.py
    - data_pre/bf_edge_v3/tests/test_config.py
    - data_pre/bf_edge_v3/tests/conftest.py
    - data_pre/bf_edge_v3/.gitignore
key-decisions:
  - "Phase 4 equivalence uses reference_v2/ directory (generated, not tracked) for Stages 2-4 while preserving Stage 1 against Part A reference/"
  - "Part A Stage 2-4 equivalence tests skipped with explicit reason (not deleted) to preserve audit trail"
  - "test_validation.py local_clusters tests now generate Phase 4 data on-the-fly instead of loading Part A reference (which contains cluster_trigger_flag)"
  - "test_config.py updated as Rule 3 auto-fix to match Phase 4 Stage2Config and Stage3Config schemas"
patterns-established:
  - "Contract invariant tests: direction consistency, spatial continuity, lateral spread verified per-cluster"
  - "Non-regression checks: assigned center count >= 80% of Part A baseline, cluster count increased"
  - "Dual reference baseline: reference/ (Part A) for Stage 1, reference_v2/ (Phase 4) for Stages 2-4"
requirements-completed: [ALG-01, ALG-02, ALG-03]
metrics:
  duration: 13m09s
  completed: "2026-04-07T09:29:19Z"
  tasks_completed: 2
  tasks_total: 2
---

# Phase 04 Plan 03: Phase 4 Testing and Verification Summary

Contract invariant tests, density rescue unit tests, Phase 4 reference baseline, and updated equivalence/validation/config tests -- 45 tests pass, 6 archived Part A tests skipped.

## Performance

- **Duration:** 13m 09s
- **Started:** 2026-04-07T09:16:10Z
- **Completed:** 2026-04-07T09:29:19Z
- **Tasks:** 2/2
- **Files modified:** 7 (2 created, 5 modified)

## Accomplishments

- 12 new tests across test_cluster_contract.py (7) and test_density_rescue.py (5)
- Phase 4 reference_v2/ baseline generated for Stages 2-4 (850 clusters, 696 supports, 62540 valid edges)
- Equivalence gate updated: Stage 1 exact-match preserved, 6 new Phase 4 v2 equivalence tests, 6 old Part A tests skipped
- Full test suite: 45 passed, 6 skipped -- all green

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test_cluster_contract.py and test_density_rescue.py** - `4d21d91` (test)
2. **Task 2: Update equivalence tests, validation tests, and generate Phase 4 baseline** - `e2dc6cd` (feat)

## Files Created/Modified

- `data_pre/bf_edge_v3/tests/test_cluster_contract.py` - Created: 7 contract invariant tests (direction, spatial, lateral, no trigger flag, non-regression, validation hook)
- `data_pre/bf_edge_v3/tests/test_density_rescue.py` - Created: 5 rescue unit tests (nearby/distant noise, empty/all-noise, scale factor)
- `data_pre/bf_edge_v3/tests/test_equivalence.py` - Rewritten: Stage 1 against reference/, Phase 4 Stages 2-4 against reference_v2/, old tests skipped
- `data_pre/bf_edge_v3/tests/test_validation.py` - Updated: trigger_flag removed, Phase 4 pipeline for local_clusters, validate_cluster_contract tests added
- `data_pre/bf_edge_v3/tests/test_config.py` - Updated: Stage2Config and Stage3Config tests match Phase 4 schemas
- `data_pre/bf_edge_v3/tests/conftest.py` - Added: phase4_stage2 fixture, REFERENCE_V2_DIR, reference_v2_dir fixture
- `data_pre/bf_edge_v3/.gitignore` - Added: tests/reference_v2/

## Decisions Made

1. **Dual reference baseline strategy:** Part A reference/ (Stage 1 only) + Phase 4 reference_v2/ (Stages 2-4). Preserves Stage 1 regression gate while establishing new Phase 4 baseline.
2. **Archived Part A tests:** Skipped with explicit reason strings rather than deleted, preserving documentation of the Part A baseline behavior.
3. **Phase 4 on-the-fly generation for validation tests:** test_validation.py generates local_clusters via Phase 4 pipeline instead of loading Part A reference (which contains cluster_trigger_flag no longer in schema).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated test_config.py for Phase 4 schemas**
- **Found during:** Task 2 (full test suite run)
- **Issue:** test_config.py referenced deleted Stage2Config trigger fields (trigger_min_cluster_size_factor, linearity_th, etc.) and old Stage3Config fields (25 DEFAULT_FIT_PARAMS). Tests fail with AttributeError.
- **Fix:** Updated TestStage2Defaults to test Phase 4 fields (rescue_knn, rescue_distance_scale, segment_* fields), TestStage3Defaults to test 7-field config, TestCustomValues to match Phase 4 Stage2Config, removed TestStage3Defaults.test_stage3_cosine_properties (no cosine properties left in Stage3Config).
- **Files modified:** data_pre/bf_edge_v3/tests/test_config.py
- **Verification:** All 10 config tests pass
- **Committed in:** e2dc6cd (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential fix -- test_config.py was not mentioned in the plan but was broken by Phase 4 schema changes. Without this fix, the full test suite cannot pass.

## Issues Encountered

- Worktree was created from pre-Phase-4 commit; required fast-forward merge from main repo HEAD to get Plans 04-01 and 04-02 changes
- Reference data and samples are gitignored; symlinks created to main repo data directories for worktree access

## User Setup Required

None - no external service configuration required.

## Test Suite Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| test_cluster_contract.py | 7 | 7 passed |
| test_config.py | 10 | 10 passed |
| test_density_rescue.py | 5 | 5 passed |
| test_equivalence.py | 15 | 9 passed, 6 skipped |
| test_validation.py | 14 | 14 passed |
| **Total** | **51** | **45 passed, 6 skipped** |

## Next Phase Readiness

- ALG-01, ALG-02, ALG-03 requirements complete
- Phase 4 testing infrastructure established for regression detection
- Contract invariant tests serve as ongoing gate for future algorithm changes
- Phase 5+ quality repair can proceed with confidence in test coverage

## Self-Check: PASSED

- test_cluster_contract.py exists: FOUND
- test_density_rescue.py exists: FOUND
- Task 1 commit 4d21d91: FOUND
- Task 2 commit e2dc6cd: FOUND
- Full test suite: 45 passed, 6 skipped

---
*Phase: 04-stage2-cluster-contract-redesign*
*Completed: 2026-04-07*
