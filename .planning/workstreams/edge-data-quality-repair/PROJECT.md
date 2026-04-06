# edge-data-quality-repair

## What This Is

A focused workstream to first refactor `data_pre/bf_edge_v3` into a clean, modular, instrumentable pipeline, then diagnose and fix edge data quality issues (NET-01, NET-02, NET-03) on the refactored foundation. The pipeline generates per-point edge supervision (edge.npy) used by SBF-Net training.

## Core Value

The edge supervision pipeline must have a clean structural foundation (modular stages, per-stage config, intermediate validation) before quality fixes are applied. Fixing quality issues on a tangled legacy structure creates more debt than it resolves.

## Current State

- **Parent project:** sbf-net (main workstream at v2.0)
- **Data source:** 020101 (training), 020102 (validation) edge.npy + Tier-2 visual verification
- **Pipeline:** `data_pre/bf_edge_v3/` — 4-stage pipeline (boundary centers -> clustering -> support fitting -> pointwise supervision)
- **Baseline diagnosis:** Phase 1 complete — Stage 2 DBSCAN is the primary NET-01 bottleneck (18.7pp survival gap)

## Active Milestone: v1.0 — Edge Data Pipeline Refactor and Quality Repair

**Goal:** Refactor `data_pre/bf_edge_v3` into a clean, instrumentable, verifiable pipeline structure, then fix edge data quality issues (NET-01, NET-02, NET-03) on top of the refactored foundation.

**Approach:** Refactor first, repair second.

**Target features:**
- Pipeline refactor: modular stages, per-stage configuration, intermediate validation hooks, density-adaptive parameter injection points
- NET-01 fix: density-adaptive eps/sigma (Stage 2 primary, Stage 4 secondary)
- NET-02 fix: single-side boundary center recovery + support gap fill
- NET-03 fix: support element snake smoothing / direction field quality
- Affected edge data re-generation
- Data quality verification report

## Validated

- DEN-01: NET-01 diagnosis complete — Stage 2 DBSCAN is primary bottleneck (18.7pp survival gap), Stage 4 Gaussian weighting is secondary (1.3-7.0pp valid gap) — Phase 1

## Active

- REF-01 through REF-04: Pipeline refactor requirements (to be detailed in discuss-phase)
- DEN-02, DEN-03: NET-01 density-adaptive fix and verification (post-refactor)
- SSB-01 through SSB-04: NET-02 diagnosis, fix, and verification
- SNK-01 through SNK-03: NET-03 diagnosis, fix, and verification
- VER-01 through VER-03: Re-generation and final verification

## Out of Scope

- SBF-Net retraining with repaired data — separate milestone decision
- Changes to the SBF-Net model, loss, or evaluator code
- Recon-side fixes (RECON R-01 through R-05) — separate concern
- New data format changes to edge.npy beyond what fixes require

## Constraints

- Fixes must not break the existing edge.npy (N, 5) format unless a format change is explicitly required by a fix
- Pipeline changes must be backward-compatible: existing configs should still work with default parameters
- All fixes must be verifiable with quantitative metrics (coverage rate, snake score, direction coherence)
- Refactoring must not change pipeline output for identical inputs and default parameters (behavioral equivalence)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Sequential not parallel | Issues may interact; NET-01 fix changes the baseline for NET-02/03 assessment | NET-01 -> NET-02 -> NET-03 |
| Diagnose before fix | Stage 2 cluster loss vs Stage 4 sigma decay must be disambiguated for NET-01 | Phase 1 diagnosis complete |
| Refactor before repair | Fixing NET-01/02/03 directly on old structure entangles structural debt with quality fixes; a clean pipeline foundation is prerequisite | Milestone restructured: Phase 2 = refactor, Phase 3+ = repair |

## Evolution

This document evolves at phase transitions and milestone boundaries.

---
*Last updated: 2026-04-06 after milestone restructuring (refactor-first pivot)*
