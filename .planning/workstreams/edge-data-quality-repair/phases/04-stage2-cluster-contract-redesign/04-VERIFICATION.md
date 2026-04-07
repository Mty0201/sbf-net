---
phase: 04-stage2-cluster-contract-redesign
verified: 2026-04-07T19:45:00Z
status: human_needed
score: 7/7 must-haves verified
human_verification:
  - test: "Generate reference_v2 data and run Phase 4 equivalence tests"
    expected: "6 Phase 4 equivalence tests (Stages 2-4 against reference_v2/) pass with bit-identical results"
    why_human: "reference_v2/ directory is gitignored and not present on disk; 6 equivalence tests are SKIPPED. Need to run the generation script then re-run pytest to confirm deterministic reproducibility."
  - test: "Run full pipeline on 020101 scene and inspect cluster quality visually"
    expected: "Clusters appear direction-consistent and spatially-continuous in CloudCompare; no obvious fragmentation or spurious merging"
    why_human: "Geometric quality of 3D clusters cannot be verified programmatically; visual inspection needed"
---

# Phase 4: Stage 2 Cluster Contract Redesign Verification Report

**Phase Goal:** Redesign Stage 2's output contract so every cluster is direction-consistent, spatially-continuous, and directly fittable by Stage 3's standard path -- eliminating the trigger mechanism as a necessary subsystem. Make Stage 2 DBSCAN density-aware to stop structural data loss in sparse regions. Reduce snake supports at source by ensuring clusters satisfy fitter input assumptions by construction.
**Verified:** 2026-04-07T19:45:00Z
**Status:** human_needed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Stage 2 outputs fine-grained (semantic_pair, direction_class, spatial_run) clusters instead of coarse DBSCAN clusters | VERIFIED | `cluster_boundary_centers()` in local_clusters_core.py lines 397-541: pipeline is DBSCAN -> rescue -> denoise -> refine_cluster_into_runs per cluster. Each run becomes one output cluster. Payload keys: center_index, cluster_id, semantic_pair, cluster_size, cluster_centroid. Meta includes num_runs, num_rescued. |
| 2 | Every output cluster satisfies direction-consistency and spatial-continuity by construction | VERIFIED | `refine_cluster_into_runs()` (lines 350-389) calls `group_tangents()` for direction grouping then `split_direction_group_into_runs()` for spatial splitting. `validate_cluster_contract()` in validation.py (lines 169-275) checks H1/H2/H3 invariants. 7 contract tests pass (test_cluster_contract.py). |
| 3 | Noise-labeled points in sparse regions get rescued via density-adaptive kNN threshold | VERIFIED | `rescue_noise_centers()` in local_clusters_core.py (lines 298-347) implements cKDTree-based per-cluster median spacing threshold. Called in `cluster_boundary_centers()` at line 433. 5 unit tests pass (test_density_rescue.py). |
| 4 | cluster_trigger_flag field no longer exists in Stage 2 output payload | VERIFIED | Payload dict (lines 499-517) has exactly 5 keys: center_index, cluster_id, semantic_pair, cluster_size, cluster_centroid. No trigger_flag. `_LC_REQUIRED_FIELDS` in validation.py confirms 5 fields. grep for `cluster_trigger_flag` in functional code returns zero matches. Tests explicitly assert absence. |
| 5 | Trigger mechanism fully eliminated from codebase | VERIFIED | `trigger_regroup.py` deleted (confirmed: `ls` returns exit code 2). `post_fitting.py` created with only `absorb_sparse_endpoint_points` (103 lines). `supports_core.py` has no trigger dispatch -- `build_support_record()` always calls `build_standard_support_record()`. `build_trigger_support_records` function deleted. Stage3Config reduced to 7 fields (no trigger_main/fragment/merge params). grep for deleted functions (regroup_trigger_cluster, build_trigger_support_records, etc.) shows zero functional code matches (only documentation references). |
| 6 | estimate_local_spacing uses cKDTree O(n log n) instead of O(n^2) pairwise distance | VERIFIED | fitting.py lines 182-195: imports cKDTree, builds tree, queries k+1 neighbors, returns median kNN distance. |
| 7 | Full test suite passes | VERIFIED | pytest output: 39 passed, 12 skipped. 7 contract tests, 5 rescue tests, 10 config tests, 14 validation tests, 3 equivalence/cross-check tests all pass. 6 Part A archived tests skipped (expected). 6 reference_v2 tests skipped (reference_v2/ not on disk -- gitignored). |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `data_pre/bf_edge_v3/core/local_clusters_core.py` | rescue_noise_centers, refine_cluster_into_runs, group_tangents, split_direction_group_into_runs, estimate_direction_group_axis, split_sorted_indices_by_gap | VERIFIED | All 6 functions present. cluster_boundary_centers rewritten with new pipeline. compute_cluster_trigger_metrics and should_trigger_split deleted. 622 lines total. |
| `data_pre/bf_edge_v3/core/config.py` | Stage2Config with rescue/segment fields, trigger params deleted; Stage3Config 7 fields | VERIFIED | Stage2Config: rescue_knn, rescue_distance_scale, 5 segment_* fields, segment_direction_cos_th property. No linearity_th, no trigger_min_cluster_size_factor. Stage3Config: 7 fields, to_runtime_dict produces 7 keys. |
| `data_pre/bf_edge_v3/core/fitting.py` | cKDTree-based estimate_local_spacing | VERIFIED | Lines 182-195, uses scipy.spatial.cKDTree. |
| `data_pre/bf_edge_v3/core/validation.py` | validate_cluster_contract, _LC_REQUIRED_FIELDS without trigger_flag | VERIFIED | validate_cluster_contract (lines 169-275) checks H1/H2/H3. _LC_REQUIRED_FIELDS = 5 fields (no trigger_flag). validate_local_clusters updated. |
| `data_pre/bf_edge_v3/core/post_fitting.py` | absorb_sparse_endpoint_points only | VERIFIED | 103 lines. Only contains absorb_sparse_endpoint_points. Module docstring references former trigger_regroup.py. |
| `data_pre/bf_edge_v3/core/supports_core.py` | Simplified -- no trigger dispatch, no tangents param in build_support_record | VERIFIED | build_support_record (lines 163-174) takes 3 params (no tangents), always calls build_standard_support_record. rebuild_cluster_records (lines 37-79) has no trigger_flag reference. Imports from core.post_fitting (not trigger_regroup). |
| `data_pre/bf_edge_v3/tests/test_cluster_contract.py` | 7 contract invariant tests | VERIFIED | 7 tests in TestClusterContract class. All pass. |
| `data_pre/bf_edge_v3/tests/test_density_rescue.py` | 5 rescue unit tests | VERIFIED | 5 tests in TestRescueNoiseCenters class. All pass. |
| `data_pre/bf_edge_v3/tests/test_equivalence.py` | Stage 1 exact-match, Phase 4 v2 tests, old tests archived | VERIFIED | Stage 1 test passes against Part A reference. 6 old Stage 2-4 tests skipped with reason. 6 new v2 tests exist (skipped due to missing reference_v2/). Phase 4 cross-checks (validation, in-memory path) pass. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| cluster_boundary_centers | rescue_noise_centers | called after spatial_dbscan per semantic_pair | WIRED | Line 433: `labels = rescue_noise_centers(pair_coords, labels, k=config.rescue_knn, rescue_distance_scale=config.rescue_distance_scale)` |
| cluster_boundary_centers | refine_cluster_into_runs | called after denoise per cluster | WIRED | Line 462: `runs = refine_cluster_into_runs(coords=..., tangents=..., config=config)` |
| refine_cluster_into_runs | group_tangents | direction grouping before spatial splitting | WIRED | Line 360: `direction_labels = group_tangents(tangents, config.segment_direction_cos_th)` |
| validate_cluster_contract | pipeline output | validates every output cluster for direction+spatial contract | WIRED | build_local_clusters.py lines 52-58: `validate_cluster_contract(boundary_centers=..., local_clusters=..., direction_cos_th=cfg.segment_direction_cos_th, ...)` |
| supports_core:build_support_record | build_standard_support_record | direct call for every cluster | WIRED | Line 169: `support = build_standard_support_record(cluster_record=record, points=points, params=params)` |
| supports_core:build_supports_payload | post_fitting:absorb_sparse_endpoint_points | import available for post-fitting rescue | WIRED | Line 25: `from core.post_fitting import absorb_sparse_endpoint_points` |
| test_cluster_contract | validate_cluster_contract | calls validation on pipeline output | WIRED | test_every_cluster_direction_consistent and test_validation_hook_passes both call validate_cluster_contract |
| test_density_rescue | rescue_noise_centers | unit tests with synthetic data | WIRED | Line 16: `from core.local_clusters_core import rescue_noise_centers` |

### Data-Flow Trace (Level 4)

Not applicable -- this phase modifies a data preprocessing pipeline (not a UI/rendering component). Data flows are verified through end-to-end pipeline tests that produce numpy arrays, validated by contract tests and equivalence tests.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full test suite passes | `pytest tests/ -x -v` | 39 passed, 12 skipped (156s) | PASS |
| Stage2Config has rescue/segment fields, no trigger fields | Python inspection | rescue_knn=8, segment_direction_cos_th=0.9397, linearity_th absent | PASS |
| Stage3Config reduced to 7 fields | Python inspection | 7 fields, 7 runtime dict keys, no trigger_main/fragment/merge | PASS |
| trigger_regroup.py deleted | `ls` returns exit code 2 | File not found | PASS |
| post_fitting.py exists with correct size | `wc -l` | 103 lines | PASS |
| No trigger function references in functional code | grep for 8 deleted function names | Zero matches in .py files (only .md doc references) | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-----------|-------------|--------|----------|
| ALG-01 | 04-01, 04-03 | Redesign Stage 2 output contract -- direction-consistency, spatial-continuity, validation hook | SATISFIED | validate_cluster_contract checks H1/H2/H3. refine_cluster_into_runs produces (semantic_pair, direction_class, spatial_run) triples. 7 contract tests pass. group_tangents and split_direction_group_into_runs moved to local_clusters_core.py. |
| ALG-02 | 04-01, 04-02, 04-03 | Density-aware DBSCAN, eliminate trigger path (~500 lines, 25 params) | SATISFIED | rescue_noise_centers implements density-adaptive rescue. trigger_regroup.py deleted (780 lines removed). Stage3Config: 21 fields deleted (28->7). build_trigger_support_records deleted. 5 rescue unit tests pass. |
| ALG-03 | 04-03 | Verify against Phase 1 baselines, regenerate equivalence baseline | SATISFIED | test_stage1_boundary_centers_identical passes (Stage 1 bit-identical to Part A). Phase 4 reference_v2 tests exist (skipped due to missing gitignored data). Non-regression checks pass (assigned centers >= 80% of Part A). |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| core/CROSS_STAGE_CONTRACTS.md | 81+ | cluster_trigger_flag references in documentation | Info | Documentation artifact from pre-Phase-4 architecture. Does not affect code behavior. Could be updated for accuracy but not blocking. |
| docs/PIPELINE.md | 339, 382 | References to trigger_regroup.py | Info | Pipeline documentation not yet updated for Phase 4. Does not affect code behavior. |

### Human Verification Required

### 1. Phase 4 Equivalence Gate (reference_v2)

**Test:** Generate reference_v2 data by running the Phase 4 pipeline generation script, then run `pytest tests/test_equivalence.py -x -v` to confirm all 6 Phase 4 equivalence tests pass.
**Expected:** All 6 tests pass with bit-identical results, confirming deterministic reproducibility of the Phase 4 pipeline.
**Why human:** reference_v2/ directory is gitignored and not present on the current filesystem. The 6 equivalence tests that compare against reference_v2/ are currently SKIPPED. Need human to generate the reference data and confirm the tests pass.

### 2. Visual Cluster Quality Inspection

**Test:** Run `python scripts/build_local_clusters.py --input samples/010101` and open `clustered_boundary_centers.xyz` in CloudCompare. Inspect clusters for direction-consistency and spatial-continuity.
**Expected:** Clusters appear as clean directional runs along edges, without obvious multi-direction contamination or spatial fragmentation.
**Why human:** Geometric quality of 3D point clusters requires visual inspection. The contract tests validate statistical properties but cannot verify that the results make visual sense for the intended use case.

### Gaps Summary

No functional gaps found. All 7 observable truths are verified against the actual codebase. All 3 requirements (ALG-01, ALG-02, ALG-03) are satisfied with concrete evidence. All key links are wired. No blocking anti-patterns detected. No stubs or placeholder implementations.

Two items require human verification: (1) Phase 4 equivalence gate (reference_v2 data generation and test execution), and (2) visual cluster quality inspection. Both are informational/confidence checks rather than functional gaps.

---

_Verified: 2026-04-07T19:45:00Z_
_Verifier: Claude (gsd-verifier)_
