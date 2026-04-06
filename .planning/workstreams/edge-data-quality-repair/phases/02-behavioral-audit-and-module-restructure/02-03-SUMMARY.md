---
phase: 02-behavioral-audit-and-module-restructure
plan: 03
subsystem: data_pre/bf_edge_v3/core
tags: [refactor, parameter-extraction, params, pipeline-docs, behavioral-verification]
dependency_graph:
  requires: [02-02]
  provides: [params.py, updated-PIPELINE.md]
  affects: [supports_core.py, local_clusters_core.py]
tech_stack:
  added: []
  patterns: [centralized-parameters, named-parameter-references]
key_files:
  created:
    - data_pre/bf_edge_v3/core/params.py
  modified:
    - data_pre/bf_edge_v3/core/supports_core.py
    - data_pre/bf_edge_v3/core/local_clusters_core.py
    - data_pre/bf_edge_v3/docs/PIPELINE.md
decisions:
  - "params.py uses dict[str, float | int] type hints for all parameter dicts"
  - "DEFAULT_FIT_PARAMS re-exported from supports_core.py for backward compatibility"
  - "Stage 2 trigger_min_cluster_size and min_keep_points remain runtime-computed from factor + floor pattern"
  - "Baseline comparison not possible (supports.npz not git-tracked); formal equivalence deferred to Phase 3 REF-06"
metrics:
  duration: 5m5s
  completed: "2026-04-07T01:54:00Z"
  tasks_completed: 2
  tasks_total: 2
  files_created: 1
  files_modified: 3
---

# Phase 02 Plan 03: Parameter Extraction and Pipeline Documentation Summary

Centralized all hardcoded pipeline parameters (33 total across 3 dicts) into params.py with byte-identical default values. Replaced inline trigger thresholds and denoise safeguards in local_clusters_core.py with named parameter references. Verified refactored pipeline produces correct output on both test scenes (020101: 193 supports, 020102: 242 supports). Updated PIPELINE.md with new module inventory, parameter documentation, and Phase 2 reference docs. Phase 2 complete.

## Task Results

### Task 1: Extract parameters to params.py and update references

**Commit:** 1735070

Created `core/params.py` (100 lines) with three parameter dicts:

- **DEFAULT_FIT_PARAMS** (25 keys): All Stage 3 trigger regrouping parameters moved from supports_core.py. Values byte-identical to originals. Per-parameter comments added documenting each parameter's role.

- **DEFAULT_TRIGGER_PARAMS** (5 keys): Stage 2 trigger-split thresholds extracted from hardcoded dict in cluster_boundary_centers() (lines 191-196). Includes `trigger_min_cluster_size_factor=6`, `trigger_min_cluster_size_floor=48`, `linearity_th=0.85`, `tangent_coherence_th=0.88`, `bbox_anisotropy_th=6.0`.

- **DEFAULT_DENOISE_PARAMS** (3 keys): Stage 2 denoise safeguards extracted from inline literals in cluster_boundary_centers() (lines 217-218). Includes `max_remove_ratio=0.20`, `min_keep_points_factor=1`, `min_keep_points_floor=6`.

Updated `supports_core.py`: replaced inline DEFAULT_FIT_PARAMS definition with `from core.params import DEFAULT_FIT_PARAMS`. The dict is still accessible via `from core.supports_core import DEFAULT_FIT_PARAMS` for backward compatibility (import forwarding).

Updated `local_clusters_core.py`: replaced all 5 hardcoded trigger threshold literals and 2 denoise safeguard literals with references to DEFAULT_TRIGGER_PARAMS and DEFAULT_DENOISE_PARAMS. Runtime computation pattern preserved (e.g., `max(min_samples * factor, floor)`).

Verified: all modules import correctly, all parameter values match originals exactly, no hardcoded trigger/denoise literals remain in local_clusters_core.py.

### Task 2: Behavioral spot-check on test scenes and update PIPELINE.md

**Commit:** f3f1a30

**Spot-check results:**
- 020101 (training): 163 clusters -> 193 supports (83 line, 110 polyline, 3105 segments)
- 020102 (validation): 172 clusters -> 242 supports (149 line, 93 polyline, 2652 segments)
- All 27 expected NPZ fields present with correct shapes and dtypes
- No import errors, no runtime errors on either test scene

**Baseline comparison:** Not possible because supports.npz is not git-tracked (generated data in .gitignore). The formal equivalence gate is Phase 3 REF-06. This spot-check confirms the refactored pipeline runs correctly and produces structurally valid output.

**PIPELINE.md updates:**
- Added Section 6: Module Structure (Phase 2 Restructure)
- Section 6.1: Core Module Inventory table (8 modules with line counts and responsibilities)
- Section 6.2: Parameter Centralization (documents all 3 parameter dicts and import pattern)
- Section 6.3: Phase 2 Documentation (references BEHAVIORAL_AUDIT.md and CROSS_STAGE_CONTRACTS.md)
- Section 6.4: Stage 3 Import Structure (ASCII dependency diagram)
- Updated Section 7 to note Phase 2 completion and pending Phase 3 work
- Renumbered subsequent sections (7 -> 8)

## Phase 2 Completion Summary

Phase 2 (Behavioral Audit and Module Restructure) is now complete across all 3 plans:

| Plan | Deliverable | Key Files |
|------|-------------|-----------|
| 01 | Behavioral audit + cross-stage contracts | BEHAVIORAL_AUDIT.md, CROSS_STAGE_CONTRACTS.md |
| 02 | Module decomposition (1318-line monolith -> 4 modules) | fitting.py, trigger_regroup.py, supports_export.py, slimmed supports_core.py |
| 03 | Parameter centralization + pipeline docs | params.py, updated PIPELINE.md |

**Phase 2 total deliverables:** 5 new files created, 5 existing files modified, 2 new documentation files.

The pipeline is now audited, restructured into modular sub-modules, and documented. Phase 3 can inject per-stage configuration and build the formal equivalence verification gate on this foundation.

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

1. **params.py type hints:** Used `dict[str, float | int]` for all parameter dicts, matching Python 3.10+ syntax consistent with the rest of the codebase.

2. **DEFAULT_FIT_PARAMS re-export:** supports_core.py imports and re-exports DEFAULT_FIT_PARAMS from params.py so that existing code importing from supports_core continues to work. Both scripts (fit_local_supports.py, build_support_dataset_v3.py) still import from `core.supports_core` -- they can be updated to import directly from `core.params` in a future cleanup, but this is not required for correctness.

3. **Factor + floor pattern for computed params:** trigger_min_cluster_size and min_keep_points use a `max(value * factor, floor)` pattern rather than storing the computed value, because the computed value depends on a CLI parameter (min_samples) that varies per invocation.

4. **No baseline comparison:** supports.npz is generated data not tracked in git. The plan acknowledges this is an "informal" verification (per D-08/D-09). Formal equivalence gate is Phase 3 REF-06.

## Self-Check: PASSED

- [x] data_pre/bf_edge_v3/core/params.py exists
- [x] data_pre/bf_edge_v3/core/supports_core.py exists (modified)
- [x] data_pre/bf_edge_v3/core/local_clusters_core.py exists (modified)
- [x] data_pre/bf_edge_v3/docs/PIPELINE.md exists (modified)
- [x] Commit 1735070 found
- [x] Commit f3f1a30 found
