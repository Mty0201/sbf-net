---
phase: 03-config-injection-validation-equivalence
plan: 01
subsystem: data-preprocessing
tags: [dataclass, frozen-config, parameter-unification, bf-edge-v3, pytest]

# Dependency graph
requires:
  - phase: 02-behavioral-audit-and-module-restructure
    provides: centralized params.py with 33 parameters, modular core functions, CROSS_STAGE_CONTRACTS.md
provides:
  - 4 frozen dataclass configs (Stage1Config, Stage2Config, Stage3Config, Stage4Config)
  - Pre-change reference data for scene 010101 (NPZ + npy artifacts)
  - Stage3Config.to_runtime_dict() transition method replacing duplicated build_runtime_params()
  - Test infrastructure (conftest.py fixtures, test_config.py with 10 tests)
affects: [03-02-validation-hooks, 03-03-equivalence-gate]

# Tech tracking
tech-stack:
  added: []
  patterns: [frozen-dataclass-per-stage, build_config-from-cli-args, to_runtime_dict-transition]

key-files:
  created:
    - data_pre/bf_edge_v3/core/config.py
    - data_pre/bf_edge_v3/tests/__init__.py
    - data_pre/bf_edge_v3/tests/conftest.py
    - data_pre/bf_edge_v3/tests/test_config.py
    - data_pre/bf_edge_v3/.gitignore
  modified:
    - data_pre/bf_edge_v3/scripts/build_boundary_centers.py
    - data_pre/bf_edge_v3/scripts/build_local_clusters.py
    - data_pre/bf_edge_v3/scripts/fit_local_supports.py
    - data_pre/bf_edge_v3/scripts/build_pointwise_edge_supervision.py
    - data_pre/bf_edge_v3/scripts/build_support_dataset_v3.py

key-decisions:
  - "Hardcoded all 28 Stage3Config defaults directly (not imported from params.py) to keep config.py self-contained and freeze-safe"
  - "Kept run_scene() signatures taking args/params dicts rather than config objects, minimizing diff and preserving script-layer flexibility"
  - "Reference data generated before any code changes to ensure clean pre-change baseline for equivalence gate"

patterns-established:
  - "frozen-dataclass-per-stage: each pipeline stage has one immutable config holding all runtime parameters"
  - "build_config(args): each script has a factory function converting CLI namespace to typed config"
  - "to_runtime_dict(): transition method on Stage3Config produces the flat dict core functions expect"

requirements-completed: [REF-04, REF-06]

# Metrics
duration: 7min
completed: 2026-04-07
---

# Phase 3 Plan 01: Config Injection and Reference Data Summary

**4 frozen dataclass configs unifying 5 scattered parameter sources, with pre-change reference data captured for equivalence testing**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-06T19:04:24Z
- **Completed:** 2026-04-06T19:11:00Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- Generated pre-change reference data (11 files) from current pipeline on scene 010101 before any code changes
- Implemented 4 frozen dataclass configs in core/config.py: Stage1Config (4 fields), Stage2Config (13 fields + 2 properties), Stage3Config (28 fields + 5 cosine properties + to_runtime_dict()), Stage4Config (2 fields + sigma property)
- Integrated configs into all 5 scripts, deleting both duplicated build_runtime_params() / build_support_runtime_params() functions
- All 10 config tests pass, verifying defaults match current parameters exactly

## Task Commits

Each task was committed atomically:

1. **Task 1: Generate reference data and create test infrastructure** - `00624dd` (feat)
2. **Task 2 RED: Add failing tests for config dataclass system** - `4ea4567` (test)
3. **Task 2 GREEN: Implement 4 frozen dataclass configs** - `9c99055` (feat)
4. **Task 2 GREEN: Integrate configs into all 5 scripts** - `b98eb8f` (feat)

## Files Created/Modified
- `data_pre/bf_edge_v3/core/config.py` - 4 frozen dataclass configs with defaults matching all current parameters
- `data_pre/bf_edge_v3/tests/__init__.py` - Empty init for test package
- `data_pre/bf_edge_v3/tests/conftest.py` - Shared fixtures (sample_scene_dir, reference_dir)
- `data_pre/bf_edge_v3/tests/test_config.py` - 10 tests covering defaults, derived properties, to_runtime_dict equivalence, frozen semantics
- `data_pre/bf_edge_v3/.gitignore` - Excludes tests/reference/ and samples/
- `data_pre/bf_edge_v3/scripts/build_boundary_centers.py` - Uses Stage1Config via build_config()
- `data_pre/bf_edge_v3/scripts/build_local_clusters.py` - Uses Stage2Config via build_config()
- `data_pre/bf_edge_v3/scripts/fit_local_supports.py` - Uses Stage3Config, build_runtime_params() deleted
- `data_pre/bf_edge_v3/scripts/build_pointwise_edge_supervision.py` - Uses Stage4Config via build_config()
- `data_pre/bf_edge_v3/scripts/build_support_dataset_v3.py` - Uses Stage1/2/3Config, build_support_runtime_params() deleted

## Decisions Made
- Hardcoded all 28 Stage3Config defaults directly rather than importing from params.py -- keeps config.py self-contained and avoids circular dependency risk
- Kept run_scene() signatures unchanged where possible (still takes args or params dict) -- minimizes diff and lets the config integration be purely additive at the script layer
- Reference data generated as the very first step before any code modifications, ensuring a clean pre-change baseline

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Reference data in tests/reference/ ready for Plan 03-03 equivalence gate
- Config system ready for Plan 03-02 validation hooks (hooks can read config properties for threshold validation)
- All core function signatures unchanged -- Plan 03-02 can add validation without breaking configs

## Self-Check: PASSED

- All 10 created/modified files verified on disk
- All 4 task commits verified in git log
- All 7 reference data files verified in tests/reference/

---
*Phase: 03-config-injection-validation-equivalence*
*Completed: 2026-04-07*
