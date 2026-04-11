---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: — ZAHA + s3dis Handover
current_phase: 01
current_plan: 4
status: executing
stopped_at: Plan 01-03 Wave 2 complete (denoise + chunking + normals). Merged to main as 3aee88f..9d1ddfa (7 commits). Next step is Plan 01-04 (orchestrator + NPY layout).
last_updated: "2026-04-11T14:45:00Z"
last_activity: 2026-04-11
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 4
  completed_plans: 3
  percent: 75
---

# Project State

## Current Position

Phase: 01 (zaha-offline-preprocessing-pipeline) — EXECUTING
Plan: 4 of 4
**Status:** Executing Phase 01
**Current Phase:** 01
**Last Activity:** 2026-04-11
**Last Activity Description:** Plan 01-03 Wave 2 complete (denoise + chunking + normals, SOR winner approved, normals D-18 bar met)

## Progress

**Phases Complete:** 0
**Current Plan:** 4
**s3dis status:** still deferred behind ZAHA. Prior s3dis Phase 1 CONTEXT.md is now at `archive/01-s3dis-data-layout-and-edge-generation-SUPERSEDED-2026-04-11/`. It will be re-authored as Phase 5+ after ZAHA Phase 4 lands.

## Session Continuity

**Stopped At:** Plan 01-03 Wave 2 complete — denoise/chunking/normals modules landed, 7 commits merged to main (3aee88f..9d1ddfa). Next step is Plan 01-04 (pipeline orchestrator + NPY manifest + per-sample loop).
**Resume File:** `.planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/01-04-PLAN.md`

## Open Questions for ZAHA Phase 1 discuss-phase

1. **Denoising method (RESEARCH-GATED, DS-ZAHA-P1-03):** the five pathologies demand a method that tolerates the density gradient AND the scan-stripe banding AND the wall thickness. Must evaluate ≥3 candidates (SOR, radius outlier removal, and one of {bilateral, MLS, RANSAC plane residual}) against 2–3 sample chunks with before/after visualizations. This is the first gate before the pipeline is implementable.
2. **Chunking geometry (DS-ZAHA-P1-04):** axis-aligned box grid with ≥2 m overlap vs octree recursive subdivision vs facade-plane-aware partitioning. Target ≤1.0 M pts/chunk post-0.02. Determinism requirement: same input + same commit → same chunk IDs and bboxes.
3. **Normal estimation method (DS-ZAHA-P1-05):** adaptive-radius PCA (k≈30 auto-scaling) vs facade-plane RANSAC per neighborhood vs robust-weighted PCA with stripe-aware outlier rejection. User bar: "thinning the wall" is aspirational, not blocking. Must handle the density gradient.
4. **`OuterCeilingSurface` (ID 14) LoFG2 bucket (DS-ZAHA-P1-06):** paper Figure 3 does not cover it. Decide between "structural" and "other el.", document rationale in the YAML.
5. **`ignore_index` convention (DS-ZAHA-P1B-01 / DS-ZAHA-P3-01):** `0` (VOID stays as raw XML ID 0, ignore_index=0) vs `255` (remap VOID → 255 at runtime, ignore_index=255). Must be consistent between the pure-semantic track and the BF track. Driven by Pointcept trainer compatibility.
6. **Point budget per chunk (DS-ZAHA-P1-04 / DS-ZAHA-P1B-02):** the ≤1.0 M target is tentative. Planner should confirm against WSL memory and the PTv3 default batch/sphere config.
