---
phase: 05-density-adaptive-supervision
plan: 02
subsystem: data-preprocessing
tags: [density-adaptive, denoise, dbscan, pipeline-verification, equivalence-gate]

# Dependency graph
requires:
  - "05-01: Config defaults and density-conditional denoise logic"
provides:
  - "DEN-02 verified: Stage 2 survival gap <5pp on 020101 (4.28pp) and 020102 (4.22pp)"
  - "DEN-03 verified: Dense-region survival rate 0.9971 on both scenes (>= 0.99)"
  - "reference_v3 baseline from 010101 for Phase 5 equivalence testing"
  - "test_density_adaptive.py: programmatic DEN-02/DEN-03 verification"
  - "Phase 5 equivalence gate in test_equivalence.py against reference_v3"
affects: [edge-supervision-pipeline, future-phase-regression-prevention]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Triple reference baseline: reference/ (Part A), reference_v2/ (Phase 4), reference_v3/ (Phase 5)"
    - "Scene-level gap measurement using P25/P75 density bucketing from diagnose_net01.py"

key-files:
  created:
    - data_pre/bf_edge_v3/tests/test_density_adaptive.py
    - data_pre/bf_edge_v3/tests/reference_v3/local_clusters.npz
    - data_pre/bf_edge_v3/tests/reference_v3/supports.npz
    - data_pre/bf_edge_v3/tests/reference_v3/edge_dist.npy
    - data_pre/bf_edge_v3/tests/reference_v3/edge_dir.npy
    - data_pre/bf_edge_v3/tests/reference_v3/edge_valid.npy
    - data_pre/bf_edge_v3/tests/reference_v3/edge_support_id.npy
  modified:
    - data_pre/bf_edge_v3/core/config.py
    - data_pre/bf_edge_v3/tests/test_config.py
    - data_pre/bf_edge_v3/tests/test_equivalence.py
    - data_pre/bf_edge_v3/tests/conftest.py

key-decisions:
  - "denoise_density_threshold corrected from 1.5 to 0.5 -- 1.5 was too permissive, only 67% of sparse-heavy clusters skipped denoise"
  - "6-column edge.npy assembled from separate Stage 4 arrays for diagnose_net01.py compatibility"
  - "Phase 4 v2 equivalence tests archived with skip markers, same pattern as Part A archival"

patterns-established:
  - "DEN-02/DEN-03 regression tests run in-memory using cluster_boundary_centers + diagnose_net01 methodology"
  - "Reference versioning: reference/ -> reference_v2/ -> reference_v3/ tracks intentional behavior changes"

requirements-completed: [DEN-02, DEN-03]

# Metrics
duration: 22min
completed: 2026-04-07
---

# Phase 05 Plan 02: Pipeline Re-run, Gap Verification, and Equivalence Gate Summary

**DEN-02/DEN-03 verified on 020101/020102 with corrected denoise threshold (0.5), programmatic test coverage, and Phase 5 equivalence gate against reference_v3**

## Performance

- **Duration:** 22 min
- **Started:** 2026-04-07T11:41:39Z
- **Completed:** 2026-04-07T12:03:14Z
- **Tasks:** 2
- **Files modified:** 10 (4 code/test + 6 reference_v3 data)

## Accomplishments
- Corrected denoise_density_threshold from 1.5 to 0.5 -- original value left 33% of sparse-heavy clusters still being denoised, producing 14.7pp gap instead of target <5pp
- Re-ran Stages 2-4 pipeline on 020101 and 020102 with corrected Phase 5 config; verified DEN-02 (gap=4.28pp / 4.22pp) and DEN-03 (dense_rate=0.9971)
- Generated reference_v3 from 010101 test scene for equivalence testing
- Created test_density_adaptive.py with 5 tests (2 DEN-02, 2 DEN-03, 1 synthetic denoise behavior)
- Updated test_equivalence.py with 6 Phase 5 v3 equivalence tests, archived 6 Phase 4 v2 tests
- Full suite: 50 passed, 12 skipped

## Task Commits

Each task was committed atomically:

1. **Task 1: Pipeline re-run with corrected threshold and reference_v3** - `c4bcac3` (feat)
2. **Task 2: Gap verification tests and equivalence update** - `64084c3` (feat)

## Files Created/Modified
- `data_pre/bf_edge_v3/core/config.py` - denoise_density_threshold corrected 1.5 -> 0.5
- `data_pre/bf_edge_v3/tests/test_config.py` - Updated assertion for new threshold
- `data_pre/bf_edge_v3/tests/test_density_adaptive.py` - NEW: DEN-02/DEN-03 verification tests
- `data_pre/bf_edge_v3/tests/test_equivalence.py` - Phase 5 v3 equivalence, v2 archived
- `data_pre/bf_edge_v3/tests/conftest.py` - REFERENCE_V3_DIR constant and fixture
- `data_pre/bf_edge_v3/tests/reference_v3/` - 6 reference files from 010101

## Decisions Made
- **Threshold correction (1.5 -> 0.5):** Research predicted 4.3pp gap with "no denoise for sparse", but Plan 01's threshold=1.5 used cluster-internal spacing as the density signal. At 1.5x global median, 33% of sparse-heavy clusters still fell below the threshold and got denoised. Lowering to 0.5x means only truly dense clusters (spacing <= half the global median) get denoised, matching the research's "skip denoise for sparse clusters" intent. Both scenes now pass DEN-02.
- **Edge.npy assembly:** diagnose_net01.py expects 6-column edge.npy [dir, dist, support, valid] but Stage 4 exports separate arrays. Assembled 6-column format from edge_dir.npy + edge_dist.npy + edge_support.npy + edge_valid.npy for diagnosis compatibility.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] denoise_density_threshold too permissive at 1.5**
- **Found during:** Task 1 (Pipeline re-run)
- **Issue:** Initial pipeline re-run showed 14.7pp gap on 020101 (target <5pp). Stage-by-stage tracing revealed denoise was still removing 33% of sparse-heavy clusters because their internal spacing fell below 1.5x global median.
- **Fix:** Lowered denoise_density_threshold from 1.5 to 0.5 after parameter sweep confirmed 0.5 achieves 4.3pp/4.2pp on both scenes while maintaining dense_rate=0.9971.
- **Files modified:** data_pre/bf_edge_v3/core/config.py, data_pre/bf_edge_v3/tests/test_config.py
- **Verification:** Full pipeline re-run + diagnose_net01.py confirms gap < 5pp on both scenes
- **Committed in:** c4bcac3

**2. [Rule 1 - Bug] Synthetic test DBSCAN spacing incompatible**
- **Found during:** Task 2 (test_density_adaptive.py creation)
- **Issue:** Initial synthetic test used 0.05 spacing for sparse cluster, but DBSCAN eps=0.08 with min_samples=5 classified all points as noise (each point had <5 neighbors within eps). Sparse cluster had 0% survival.
- **Fix:** Adjusted synthetic spacing: dense=0.005, sparse=0.03 (both comfortably within eps=0.08 for DBSCAN clustering while maintaining sufficient spacing ratio for density threshold discrimination).
- **Files modified:** data_pre/bf_edge_v3/tests/test_density_adaptive.py
- **Verification:** Synthetic test passes; sparse cluster achieves >80% survival with denoise skipped
- **Committed in:** 64084c3

**3. [Rule 1 - Bug] Scene path resolution off by one parent level**
- **Found during:** Task 2 (test_density_adaptive.py creation)
- **Issue:** Used `parents[2]` to reach repo root from test file, but `tests/` -> `bf_edge_v3/` -> `data_pre/` -> repo root requires `parents[3]`. Scenes were not found, tests incorrectly skipped.
- **Fix:** Changed to `parents[3]` for correct repo root resolution.
- **Files modified:** data_pre/bf_edge_v3/tests/test_density_adaptive.py
- **Verification:** Scene detection works; DEN-02/DEN-03 tests run and pass
- **Committed in:** 64084c3

---

**Total deviations:** 3 auto-fixed (3 Rule 1 bugs)
**Impact on plan:** The threshold correction was the most significant -- without it, DEN-02 would have failed. The other two were test-level fixes. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations.

## User Setup Required
None - no external service configuration required.

## Diagnosis Results

### 020101 (32,621 boundary centers, 367,298 scene points)
| Metric | Value |
|--------|-------|
| Stage 2 survival gap | 0.0428 (4.28pp) |
| Dense survival rate | 0.9971 |
| Sparse survival rate | 0.9543 |
| Clusters | 457 |
| Assigned centers | 32,120 |
| Denoise removed | 0 |

### 020102 (42,962 boundary centers, 513,239 scene points)
| Metric | Value |
|--------|-------|
| Stage 2 survival gap | 0.0422 (4.22pp) |
| Dense survival rate | 0.9971 |
| Sparse survival rate | 0.9549 |
| Clusters | 516 |
| Assigned centers | 42,344 |
| Denoise removed | 9 |

## Next Phase Readiness
- DEN-02 and DEN-03 are satisfied with programmatic verification
- 50 tests passing (12 archived with skip markers)
- Reference_v3 baseline established for regression prevention
- Phase 5 (density-adaptive supervision) is complete

---
*Phase: 05-density-adaptive-supervision*
*Completed: 2026-04-07*
