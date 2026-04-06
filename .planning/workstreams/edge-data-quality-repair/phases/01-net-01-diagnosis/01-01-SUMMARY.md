---
phase: "01"
plan: "01"
subsystem: edge-data-quality-repair
tags: [diagnosis, density-bucketing, pipeline-analysis, net-01]
dependency_graph:
  requires: [samples/training/020101, samples/validation/020102, bf_edge_v3 pipeline]
  provides: [diagnosis_output.json, diagnose_net01.py]
  affects: [01-02-PLAN edge repair strategy]
tech_stack:
  added: []
  patterns: [density-bucketed stratified analysis, kNN density estimation, cross-stage gap attribution]
key_files:
  created:
    - data_pre/bf_edge_v3/scripts/diagnose_net01.py
    - .planning/workstreams/edge-data-quality-repair/phases/01-net-01-diagnosis/diagnosis_output.json
  modified: []
key_decisions:
  - "edge.npy uses 6-column converted format: [dir_x, dir_y, dir_z, dist, support, valid]"
  - "Density thresholds derived from boundary center kNN distances, not scene-wide"
  - "Same P25/P75 thresholds applied to scene points for cross-stage comparability"
metrics:
  duration: "3m 36s"
  completed: "2026-04-06T13:57:39Z"
  tasks_completed: 2
  tasks_total: 2
---

# Phase 01 Plan 01: NET-01 Diagnosis via Density-Bucketed Pipeline Analysis

Stratified density-bucketed analysis reveals Stage 2 (DBSCAN clustering) as the dominant source of sparse-region quality loss, with survival gaps of +16-19% (dense vs sparse), while Stage 4 valid yield gaps are comparatively small (+1-7%).

## Tasks Completed

### Task 1: Regenerate pipeline intermediate outputs

Ran 3 pipeline stages (build_boundary_centers, build_local_clusters, fit_local_supports) for both test scenes. All 6 .npz intermediate files verified.

**Scene 020101 (training):**
- 367,298 points, 32,621 boundary centers, 163 clusters, 193 supports

**Scene 020102 (validation):**
- 513,239 points, 42,962 boundary centers, 172 clusters, 242 supports

No commit for this task -- output files are in gitignored `samples/` directory.

### Task 2: Write and run diagnose_net01.py

Created `data_pre/bf_edge_v3/scripts/diagnose_net01.py` implementing:
- kNN density estimation (k=10) using cKDTree
- P25/P75 percentile bucketing into dense/mid/sparse
- Stage 2 cluster survival analysis per bucket
- Stage 4 valid yield and weight analysis per bucket
- Cross-stage gap attribution

**Commit:** 293bdc3

## Diagnosis Results

### Scene 020101 (training)

| Metric | Dense | Mid | Sparse | Gap (D-S) |
|--------|-------|-----|--------|-----------|
| Stage 2 survival rate | 0.9979 | 0.9779 | 0.8107 | **+0.187** |
| Stage 2 mean cluster size | 989.5 | 971.7 | 475.5 | -- |
| Stage 4 valid rate | 0.1605 | 0.1843 | 0.1472 | +0.013 |
| Stage 4 mean weight | 0.5971 | 0.5729 | 0.5493 | +0.048 |
| Stage 4 high-weight rate | 0.0956 | 0.1059 | 0.0795 | +0.016 |

### Scene 020102 (validation)

| Metric | Dense | Mid | Sparse | Gap (D-S) |
|--------|-------|-----|--------|-----------|
| Stage 2 survival rate | 0.9992 | 0.9899 | 0.8363 | **+0.163** |
| Stage 2 mean cluster size | 967.4 | 831.4 | 482.5 | -- |
| Stage 4 valid rate | 0.1840 | 0.1732 | 0.1145 | +0.070 |
| Stage 4 mean weight | 0.5655 | 0.5786 | 0.5465 | +0.019 |
| Stage 4 high-weight rate | 0.1010 | 0.1008 | 0.0611 | +0.040 |

### Key Findings

1. **Stage 2 is the dominant quality bottleneck for sparse regions.** DBSCAN clustering drops ~16-19% of sparse boundary centers as noise, versus <1% for dense regions. This is consistent: sparse centers have fewer neighbors to form min_samples-sized clusters.

2. **Sparse clusters are significantly smaller** (~475-482 mean size vs ~967-989 for dense), which compounds downstream: smaller clusters produce less confident supports.

3. **Stage 4 valid yield gap is secondary** (+1.3% for 020101, +7.0% for 020102). Scene 020102 shows a larger stage4 gap, suggesting scene geometry also matters.

4. **Weight quality degrades moderately in sparse regions.** Mean weight drops ~0.02-0.05 for sparse vs dense; high-weight rate drops ~1.6-4.0 percentage points.

5. **Density thresholds are narrow.** P25-P75 range is only 0.007-0.011 scene units, indicating the scenes have relatively uniform density with tails.

## Deviations from Plan

None -- plan executed exactly as written.

## Known Stubs

None.
