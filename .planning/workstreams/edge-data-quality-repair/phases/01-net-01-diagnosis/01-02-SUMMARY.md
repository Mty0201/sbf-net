---
phase: 01-net-01-diagnosis
plan: 02
subsystem: edge-data-quality-repair
tags: [diagnosis, net-01, density-adaptive, dbscan, gaussian-weight]
dependency_graph:
  requires: [diagnosis_output.json, 01-CONTEXT.md]
  provides: [01-DIAGNOSIS.md]
  affects: [Phase 2 fix direction]
tech_stack:
  added: []
  patterns: [stratified-gap-attribution]
key_files:
  created:
    - .planning/workstreams/edge-data-quality-repair/phases/01-net-01-diagnosis/01-DIAGNOSIS.md
  modified: []
decisions:
  - "Stage 2 is primary bottleneck (18.7pp/16.3pp survival gap vs 1.3pp/7.0pp Stage 4 valid gap)"
  - "Phase 2 should prioritize density-adaptive eps for DBSCAN, with adaptive sigma as secondary"
metrics:
  completed: "2026-04-06T14:02:00Z"
  tasks_completed: 1
  tasks_total: 1
---

# Phase 01 Plan 02: Formal Diagnosis Synthesis Summary

Formal NET-01 diagnosis document synthesized from stratified statistics, ranking Stage 2 DBSCAN cluster loss (18.7pp survival gap) as primary bottleneck over Stage 4 Gaussian weight decay (1.3-7.0pp valid gap).

## What Was Done

Converted the raw diagnosis_output.json statistics into a structured, human-readable diagnosis document (01-DIAGNOSIS.md) that:

1. Establishes the primary/secondary bottleneck ranking with numerical evidence from both scenes (020101 and 020102)
2. Documents the density distribution thresholds (P25/P75) used for bucketing
3. Presents Stage 2 and Stage 4 metrics in comparable table format
4. Attributes the dense-sparse gap to specific pipeline stages via the gap attribution table
5. Explains the distinct failure modes per D-04 (binary segment loss vs gradual weight thinning)
6. Provides concrete Phase 2 recommendations: density-adaptive eps (primary) and density-adaptive sigma (secondary)

## Key Findings

- **Stage 2 is the primary bottleneck.** Sparse-region DBSCAN survival rate is 81.1-83.6% vs 99.8-99.9% dense, yielding an 18.7pp (020101) and 16.3pp (020102) gap. Roughly 1 in 5 sparse boundary centers are lost entirely.
- **Stage 4 is the secondary factor.** Valid yield gap is 1.3pp (020101) and 7.0pp (020102). High-weight (>=0.5) coverage in sparse regions is 6.1-7.9% vs 9.6-10.1% dense.
- **Sparse cluster size is halved.** Surviving sparse clusters average 475-482 points vs 967-989 dense, propagating quality issues downstream.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 6a55dbb | Formal diagnosis document with primary/secondary ranking |

## Deviations from Plan

None -- plan executed exactly as written.

## Self-Check: PASSED
