---
phase: 03-config-injection-validation-equivalence
plan: 02
subsystem: data-pipeline
tags: [validation, numpy, cross-stage-contracts, boundary-detection]

# Dependency graph
requires:
  - phase: 03-01
    provides: frozen config dataclasses, reference data in tests/reference/
provides:
  - StageValidationError exception class
  - 4 validate_* functions covering all 7 CROSS_STAGE_CONTRACTS.md contracts
  - Validation hooks integrated into all 5 pipeline scripts
affects: [03-03-equivalence-gate, phase-04-algorithm-changes]

# Tech tracking
tech-stack:
  added: []
  patterns: [pure-inspection-hooks, fail-fast-validation, payload-dict-contract-checking]

key-files:
  created:
    - data_pre/bf_edge_v3/core/validation.py
    - data_pre/bf_edge_v3/tests/test_validation.py
  modified:
    - data_pre/bf_edge_v3/scripts/build_boundary_centers.py
    - data_pre/bf_edge_v3/scripts/build_local_clusters.py
    - data_pre/bf_edge_v3/scripts/fit_local_supports.py
    - data_pre/bf_edge_v3/scripts/build_pointwise_edge_supervision.py
    - data_pre/bf_edge_v3/scripts/build_support_dataset_v3.py

key-decisions:
  - "edge_valid dtype check accepts any integer dtype (uint8/int32) with values in {0,1} rather than requiring exactly uint8 -- both dtypes occur in the pipeline"
  - "supports validation checks only the 8-field Stage-4-minimal-read-set, not all 27 fields -- matches actual downstream consumption"

patterns-established:
  - "Validation hook pattern: validate_X(payload) -> None or raise StageValidationError, called between stage function return and export"
  - "Contract-driven testing: each test loads reference NPZ, mutates one field, asserts StageValidationError"

requirements-completed: [REF-05]

# Metrics
duration: 4min
completed: 2026-04-07
---

# Plan 03-02: Cross-Stage Validation Hooks Summary

**4 pure-inspection validation hooks covering all 7 cross-stage contracts, integrated into all 5 pipeline scripts with 12 passing tests**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-06T19:14:12Z
- **Completed:** 2026-04-06T19:18:11Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Created `core/validation.py` with StageValidationError and 4 validate_* functions implementing all 7 contracts from CROSS_STAGE_CONTRACTS.md
- All 4 validators pass on the 010101 reference data without raising -- Part A boundary rule honored
- 12 tests in `test_validation.py`: 4 valid-data tests + 8 rejection tests (missing fields, bad shapes, OOB indices, unsorted pairs, bad values)
- All 5 scripts now call validation hooks between stage function return and export/next-stage consumption
- build_support_dataset_v3.py (in-memory runner) calls 3 hooks sequentially: bc -> lc -> supports
- Full test suite: 22 tests (10 config + 12 validation) all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement validation hooks with tests (TDD)**
   - RED: `79d67c8` (test) -- 12 failing tests for validation module
   - GREEN: `1db6a90` (feat) -- core/validation.py with 4 validate_* functions, all 12 tests pass
2. **Task 2: Integrate validation hooks into all 5 scripts** - `8c38e1f` (feat)

## Files Created/Modified

- `data_pre/bf_edge_v3/core/validation.py` - StageValidationError + 4 validate_* functions (340 lines)
- `data_pre/bf_edge_v3/tests/test_validation.py` - 12 tests covering valid and malformed payloads (188 lines)
- `data_pre/bf_edge_v3/scripts/build_boundary_centers.py` - Added validate_boundary_centers() call
- `data_pre/bf_edge_v3/scripts/build_local_clusters.py` - Added validate_local_clusters() call
- `data_pre/bf_edge_v3/scripts/fit_local_supports.py` - Added validate_supports() call
- `data_pre/bf_edge_v3/scripts/build_pointwise_edge_supervision.py` - Added validate_edge_supervision() call
- `data_pre/bf_edge_v3/scripts/build_support_dataset_v3.py` - Added 3 in-memory validation calls

## Decisions Made

- edge_valid dtype check accepts any integer dtype with values in {0,1} rather than requiring exactly uint8, because the pipeline produces uint8 in Stage 4 output but int32 in some intermediate paths
- supports validation checks only the 8-field Stage-4-minimal-read-set per CROSS_STAGE_CONTRACTS.md, not all 27 fields in supports.npz -- validates what Stage 4 actually consumes

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Validation hooks provide the safety net for Plan 03-03 (equivalence gate)
- 03-03 can now run the full pipeline end-to-end with validation active and compare output against reference data
- All 22 tests (config + validation) pass, pipeline behavior unchanged

## Self-Check: PASSED

- All 8 files verified present on disk
- All 3 commits (79d67c8, 1db6a90, 8c38e1f) verified in git log
- 22/22 tests passing

---
*Phase: 03-config-injection-validation-equivalence*
*Completed: 2026-04-07*
