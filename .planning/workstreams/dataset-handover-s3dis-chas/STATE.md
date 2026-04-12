---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: — ZAHA + s3dis Handover
current_phase: 01
current_plan: 4
status: phase_complete
stopped_at: "Phase 01 complete. Plan 01-04 landed: full 26-sample pipeline ran (facade-aware chunking, grid=0.04, 140 chunks, 0 errors). Output at /tmp/zaha_chunked/ pending move to /home/mty0201/data/ZAHA_chunked."
last_updated: "2026-04-12T18:00:00Z"
last_activity: 2026-04-12
progress:
  total_phases: 5
  completed_phases: 1
  total_plans: 4
  completed_plans: 4
  percent: 100
---

# Project State

## Current Position

Phase: 01 (zaha-offline-preprocessing-pipeline) — COMPLETE
Plan: 4 of 4 (all complete)
**Status:** Phase 01 complete
**Current Phase:** 01 ✅
**Last Activity:** 2026-04-12
**Last Activity Description:** Plan 01-04 Task 3 finalized — facade-aware chunking, golden snapshot, 26/26 pytest green, docs updated

## Progress

**Phases Complete:** 1
**Plans Complete:** 4/4
**s3dis status:** still deferred behind ZAHA. Prior s3dis Phase 1 CONTEXT.md is now at `archive/01-s3dis-data-layout-and-edge-generation-SUPERSEDED-2026-04-11/`. It will be re-authored as Phase 5+ after ZAHA Phase 4 lands.

## Session Continuity

**Stopped At:** Phase 01 complete. All 4 plans executed. Pipeline output at `/tmp/zaha_chunked/` (26 samples, 140 chunks). Pending: move output to `/home/mty0201/data/ZAHA_chunked`, then proceed to Phase 02+ (s3dis or ZAHA training config).
**Resume File:** `.planning/workstreams/dataset-handover-s3dis-chas/ROADMAP.md`

## Open Questions for ZAHA Phase 1 discuss-phase

1. **Denoising method (RESEARCH-GATED, DS-ZAHA-P1-03):** the five pathologies demand a method that tolerates the density gradient AND the scan-stripe banding AND the wall thickness. Must evaluate ≥3 candidates (SOR, radius outlier removal, and one of {bilateral, MLS, RANSAC plane residual}) against 2–3 sample chunks with before/after visualizations. This is the first gate before the pipeline is implementable.
2. **Chunking geometry (DS-ZAHA-P1-04):** axis-aligned box grid with ≥2 m overlap vs octree recursive subdivision vs facade-plane-aware partitioning. Target ≤1.0 M pts/chunk post-0.02. Determinism requirement: same input + same commit → same chunk IDs and bboxes.
3. **Normal estimation method (DS-ZAHA-P1-05):** adaptive-radius PCA (k≈30 auto-scaling) vs facade-plane RANSAC per neighborhood vs robust-weighted PCA with stripe-aware outlier rejection. User bar: "thinning the wall" is aspirational, not blocking. Must handle the density gradient.
4. **`OuterCeilingSurface` (ID 14) LoFG2 bucket (DS-ZAHA-P1-06):** paper Figure 3 does not cover it. Decide between "structural" and "other el.", document rationale in the YAML.
5. **`ignore_index` convention (DS-ZAHA-P1B-01 / DS-ZAHA-P3-01):** `0` (VOID stays as raw XML ID 0, ignore_index=0) vs `255` (remap VOID → 255 at runtime, ignore_index=255). Must be consistent between the pure-semantic track and the BF track. Driven by Pointcept trainer compatibility.
6. **Point budget per chunk (DS-ZAHA-P1-04 / DS-ZAHA-P1B-02):** the ≤1.0 M target is tentative. Planner should confirm against WSL memory and the PTv3 default batch/sphere config.
