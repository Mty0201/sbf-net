---
phase: 02-behavioral-audit-and-module-restructure
verified: 2026-04-06T17:02:35Z
status: human_needed
score: 10/10 must-haves verified
human_verification:
  - test: "Run fit_local_supports.py on scene 020101 and compare supports.npz output fields/shapes against Phase 1 baseline"
    expected: "All 27 NPZ fields present with identical shapes and dtypes; floating-point values match within atol=1e-7"
    why_human: "Baseline supports.npz is not git-tracked (generated data), so programmatic comparison requires running the pipeline on the actual dataset. Summary claims informal spot-check passed, but bitwise equivalence was not confirmed."
  - test: "Run build_support_dataset_v3.py on samples/ directory and verify it completes without errors"
    expected: "Both training/020101 and validation/020102 produce supports.npz and support_geometry.xyz; no import errors or runtime crashes"
    why_human: "Requires actual scene data files (coord.npy, segment.npy) to exist on disk; cannot verify without running the pipeline"
---

# Phase 02: Behavioral Audit and Module Restructure Verification Report

**Phase Goal:** Audit the current pipeline to surface hidden compatibility logic, heuristics, and cross-stage behavioral contracts. Restructure into modular, independently runnable stages with clear I/O contracts, explicit behavioral documentation, and separation of core algorithm from compatibility/adaptation logic.
**Verified:** 2026-04-06T17:02:35Z
**Status:** human_needed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Every logical block in all 4 core modules is classified as core-algorithm, compatibility/adaptation, or infrastructure | VERIFIED | BEHAVIORAL_AUDIT.md contains 77 classification tags across 64 blocks in 4 stage sections (Stage 1: 13 blocks, Stage 2: 11, Stage 3: 29, Stage 4: 11) |
| 2 | Cross-stage behavioral contracts are explicitly documented with field-level precision | VERIFIED | CROSS_STAGE_CONTRACTS.md contains 7 contracts covering all stage-to-stage couplings, each with NPZ fields, semantic invariants, hidden assumptions, and risk levels; NPZ Schema Summary Table covers 39 fields across 3 NPZ files |
| 3 | A developer reading the audit can distinguish algorithm from compatibility logic without reading source code | VERIFIED | Three-way classification scheme (CORE/COMPAT/INFRA) is applied consistently with behavior descriptions; DEFAULT_FIT_PARAMS table lists all 25 parameters; trigger path architecture overview provides 9-step walkthrough |
| 4 | supports_core.py is decomposed into fitting.py, trigger_regroup.py, supports_export.py, and slimmed supports_core.py | VERIFIED | fitting.py (192 lines, 10 functions), trigger_regroup.py (686 lines, 13 functions), supports_export.py (79 lines, 4 functions), supports_core.py (400 lines, 5 defined functions + imports) all exist |
| 5 | Each sub-module has a single clear responsibility matching RESEARCH.md Section 5.1 boundaries | VERIFIED | fitting.py = core geometry/fitting; trigger_regroup.py = compatibility regrouping; supports_export.py = I/O/visualization; supports_core.py = orchestration + record assembly |
| 6 | Compatibility logic lives in trigger_regroup.py, separated from core fitting in fitting.py | VERIFIED | All COMPAT-classified trigger functions (group_tangents through absorb_sparse_endpoint_points) are in trigger_regroup.py; all CORE fitting functions (fit_line_support, fit_polyline_support, etc.) are in fitting.py; trigger_regroup.py imports from fitting.py (not vice versa) |
| 7 | Both invocation paths work: per-scene scripts AND build_support_dataset_v3.py | VERIFIED | fit_local_supports.py imports from core.supports_core, core.supports_export, core.params; build_support_dataset_v3.py imports from core.supports_core, core.supports_export; all imports resolve without ImportError (behavioral spot-check confirmed) |
| 8 | All hardcoded parameters are extracted to params.py with exact original default values | VERIFIED | params.py contains DEFAULT_FIT_PARAMS (25 keys), DEFAULT_TRIGGER_PARAMS (5 keys), DEFAULT_DENOISE_PARAMS (3 keys) = 33 total. Value spot-checks: linearity_th=0.85, tangent_coherence_th=0.88, bbox_anisotropy_th=6.0, max_remove_ratio=0.20 all exact |
| 9 | Stage 2 trigger thresholds are named parameters with defaults, not inline literals | VERIFIED | local_clusters_core.py contains zero hardcoded threshold literals (grep for = 0.85, = 0.88, = 6.0, = 0.20 returns no matches); all references use DEFAULT_TRIGGER_PARAMS["..."] and DEFAULT_DENOISE_PARAMS["..."] syntax |
| 10 | PIPELINE.md reflects the new module structure accurately | VERIFIED | Section 6 added with Core Module Inventory table (8 modules), Parameter Centralization docs, Phase 2 Documentation references, Stage 3 Import Structure diagram; fitting.py, trigger_regroup.py, params.py all referenced (10 references total) |

**Score:** 10/10 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `data_pre/bf_edge_v3/core/BEHAVIORAL_AUDIT.md` | Per-module behavioral classification | VERIFIED | 77 classification tags, 64 blocks, 4 stages, summary statistics table, DEFAULT_FIT_PARAMS table with 25 params |
| `data_pre/bf_edge_v3/core/CROSS_STAGE_CONTRACTS.md` | Cross-stage interface contracts | VERIFIED | 7 contracts, 39 NPZ fields documented, NPZ Schema Summary Table present with boundary_centers.npz, local_clusters.npz, supports.npz |
| `data_pre/bf_edge_v3/core/fitting.py` | Core fitting algorithms | VERIFIED | 192 lines, 10 functions: fit_line_support, fit_polyline_support, build_polyline_vertices, regularize_support_orientation, estimate_local_spacing, point_to_line_distance, point_to_segment_distance, point_to_polyline_distance, line_to_endpoints, segment_record_from_endpoints |
| `data_pre/bf_edge_v3/core/trigger_regroup.py` | Trigger cluster regrouping compatibility logic | VERIFIED | 686 lines, 13 functions including regroup_trigger_cluster, group_tangents, split_direction_group_into_runs, absorb_sparse_endpoint_points |
| `data_pre/bf_edge_v3/core/supports_export.py` | Support NPZ/XYZ export and visualization | VERIFIED | 79 lines, 4 functions: export_npz, sample_segment_geometry, export_support_geometry_xyz, export_trigger_group_classes_xyz |
| `data_pre/bf_edge_v3/core/supports_core.py` | Slimmed orchestration | VERIFIED | 400 lines, 5 defined functions (rebuild_cluster_records, build_standard_support_record, build_trigger_support_records, build_support_record, build_supports_payload) + re-exports from params/fitting/trigger_regroup/supports_export |
| `data_pre/bf_edge_v3/core/params.py` | Centralized parameter definitions | VERIFIED | 100 lines, 33 named parameters across 3 dicts, Stage 4 documented as comments |
| `data_pre/bf_edge_v3/docs/PIPELINE.md` | Updated pipeline documentation | VERIFIED | Section 6 added with module inventory, parameter centralization, import structure diagram |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| supports_core.py | fitting.py | `from core.fitting import` | WIRED | Imports fit_line_support, fit_polyline_support, regularize_support_orientation, estimate_local_spacing |
| supports_core.py | trigger_regroup.py | `from core.trigger_regroup import` | WIRED | Imports regroup_trigger_cluster, absorb_sparse_endpoint_points |
| supports_core.py | params.py | `from core.params import DEFAULT_FIT_PARAMS` | WIRED | DEFAULT_FIT_PARAMS imported and re-exported for backward compat |
| local_clusters_core.py | params.py | `from core.params import DEFAULT_TRIGGER_PARAMS, DEFAULT_DENOISE_PARAMS` | WIRED | Both dicts imported and used in cluster_boundary_centers() |
| trigger_regroup.py | fitting.py | `from core.fitting import` | WIRED | Imports estimate_local_spacing, fit_line_support, point_to_line_distance |
| fit_local_supports.py | supports_core.py | `from core.supports_core import build_supports_payload` | WIRED | build_supports_payload called in run_scene() |
| fit_local_supports.py | supports_export.py | `from core.supports_export import` | WIRED | export_npz, export_support_geometry_xyz, export_trigger_group_classes_xyz called in run_scene() |
| build_support_dataset_v3.py | supports_core.py | `from core.supports_core import build_supports_payload` | WIRED | build_supports_payload called in run_scene() |
| build_support_dataset_v3.py | supports_export.py | `from core.supports_export import` | WIRED | export_npz, export_support_geometry_xyz called in run_scene() |
| BEHAVIORAL_AUDIT.md | supports_core.py | Block-level references with line ranges | WIRED | All Stage 3 blocks reference specific line ranges (e.g., "lines 45-71", "lines 73-117"); ranges correspond to actual functions in the module |
| CROSS_STAGE_CONTRACTS.md | stage_io.py | NPZ schema documentation | WIRED | All NPZ field names, shapes, and dtypes documented; Stage 4 minimal read set of 8 fields documented |

### Data-Flow Trace (Level 4)

Not applicable -- this phase produces documentation and structural refactoring, not new data-rendering artifacts. The existing data pipeline was restructured without behavioral changes.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 4 new modules import without error | `python -c "from core.fitting import ...; from core.trigger_regroup import ...; from core.supports_export import ...; from core.supports_core import ..."` | All 28 exports verified importable | PASS |
| params.py parameter values are exact | `assert DEFAULT_TRIGGER_PARAMS['linearity_th'] == 0.85` (+ 3 more) | All 4 spot-checked values exact | PASS |
| No circular imports | Import chain: fitting <- trigger_regroup <- supports_core | No ImportError, no circular dependency | PASS |
| Total named parameters = 33 | `len(DEFAULT_FIT_PARAMS) + len(DEFAULT_TRIGGER_PARAMS) + len(DEFAULT_DENOISE_PARAMS)` | 25 + 5 + 3 = 33 | PASS |
| No hardcoded threshold literals in local_clusters_core.py | `grep '= 0.85\|= 0.88\|= 6.0\|= 0.20'` | 0 matches | PASS |
| BEHAVIORAL_AUDIT.md has 30+ classifications | `grep -c '[CORE]\|[COMPAT]\|[INFRA]\|[MIXED]'` | 77 classification tags | PASS |
| CROSS_STAGE_CONTRACTS.md has 6+ contracts | `grep -c '## Contract:'` | 7 contracts | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-----------|-------------|--------|----------|
| REF-01 | 02-02, 02-03 | Restructure pipeline stages into modular, independently runnable units with clear input/output contracts | SATISFIED | supports_core.py decomposed into 4 focused modules (fitting.py, trigger_regroup.py, supports_export.py, slimmed supports_core.py); params.py centralizes all parameters; CROSS_STAGE_CONTRACTS.md documents I/O contracts |
| REF-02 | 02-01, 02-03 | Audit and document all hidden compatibility logic, heuristics, and cross-stage behavioral contracts | SATISFIED | BEHAVIORAL_AUDIT.md classifies all 64 blocks with three-way scheme; CROSS_STAGE_CONTRACTS.md documents 7 contracts; PIPELINE.md updated with module inventory |
| REF-03 | 02-02, 02-03 | Separate core algorithm logic from compatibility/adaptation logic at module boundaries | SATISFIED | Core algorithms in fitting.py (192 lines); compatibility logic in trigger_regroup.py (686 lines); module boundary is the import: trigger_regroup imports from fitting, not vice versa |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No TODO, FIXME, PLACEHOLDER, or stub patterns found in any Phase 2 deliverable |

### Human Verification Required

### 1. Behavioral Equivalence on Test Scenes

**Test:** Run `python scripts/fit_local_supports.py --input samples/training/020101` and compare the resulting `supports.npz` against any pre-refactor baseline output.
**Expected:** All 27 NPZ fields present with identical shapes and dtypes. Floating-point values match within atol=1e-7 (or bitwise identical).
**Why human:** Baseline supports.npz is generated data not tracked in git. Summary claims informal spot-check passed (020101: 193 supports, 020102: 242 supports), but bitwise equivalence cannot be verified without running the pipeline and having a pre-refactor baseline available.

### 2. Dataset-Level In-Memory Path

**Test:** Run `python scripts/build_support_dataset_v3.py --input samples/` and verify it completes without errors on both training/020101 and validation/020102.
**Expected:** Both scenes produce supports.npz and support_geometry.xyz; no import errors or runtime crashes; support counts match per-scene script output.
**Why human:** Requires actual scene data files on disk and pipeline execution. The import chain was verified programmatically, but end-to-end execution requires running the actual pipeline.

### Gaps Summary

No code-level gaps were found. All 10 observable truths are verified against the codebase. All 8 required artifacts exist, are substantive, and are properly wired. All 3 requirements (REF-01, REF-02, REF-03) have clear implementation evidence. No anti-patterns or stubs were detected.

The only items requiring human attention are behavioral equivalence verification (running the pipeline on actual test scenes), which the SUMMARY acknowledges was done informally and which Phase 3 (REF-06) will formalize as an equivalence gate.

**Note:** The REQUIREMENTS.md traceability table (lines 71-73) still shows REF-01, REF-02, REF-03 as "pending" even though the checkboxes at lines 11-13 show them as complete (`[x]`). This is a minor documentation inconsistency that does not affect the code.

---

_Verified: 2026-04-06T17:02:35Z_
_Verifier: Claude (gsd-verifier)_
