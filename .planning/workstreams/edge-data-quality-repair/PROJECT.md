# edge-data-quality-repair

## What This Is

A focused workstream to diagnose and fix edge data quality issues in the `data_pre/bf_edge_v3` pipeline. The pipeline generates per-point edge supervision (edge.npy) used by SBF-Net training. Three confirmed quality issues (NET-01, NET-02, NET-03) degrade supervision coverage, continuity, and geometric fidelity.

## Core Value

Edge supervision data must have sufficient coverage, continuity, and geometric accuracy to serve as reliable ground truth for SBF-Net boundary-aware training.

## Current State

- **Parent project:** sbf-net (main workstream at v2.0)
- **Data source:** 020101 (training), 020102 (validation) edge.npy + Tier-2 visual verification
- **Pipeline:** `data_pre/bf_edge_v3/` — 4-stage pipeline (boundary centers → clustering → support fitting → pointwise supervision)

## Active Milestone: v1.0 — Edge Data Quality Repair

**Goal:** Fix edge data quality issues in data_pre/bf_edge_v3 — restore supervision coverage, continuity, and geometric fidelity; deliver verified repaired data generation pipeline.

**Target features:**
- NET-01 diagnosis + fix: density-adaptive sigma/radius (Stage 2 vs Stage 4 root cause)
- NET-02 diagnosis + fix: single-side semantic boundary center recovery + support gap fill
- NET-03 diagnosis + fix: support element snake smoothing / direction field quality
- Affected edge data re-generation
- Data quality verification report with coverage/continuity/direction metrics

**Approach:** Sequential by priority — NET-01 first (diagnose Stage 2 vs Stage 4 bottleneck), then NET-02, then NET-03.

## Validated

_(none yet — first milestone)_

## Active

_(requirements to be defined)_

## Out of Scope

- SBF-Net retraining with repaired data — separate milestone decision
- Changes to the SBF-Net model, loss, or evaluator code
- Recon-side fixes (RECON R-01 through R-05) — separate concern
- New data format changes to edge.npy beyond what fixes require

## Constraints

- Fixes must not break the existing edge.npy (N, 5) format unless a format change is explicitly required by a fix
- Pipeline changes must be backward-compatible: existing configs should still work with default parameters
- All fixes must be verifiable with quantitative metrics (coverage rate, snake score, direction coherence)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Sequential not parallel | Issues may interact; NET-01 fix changes the baseline for NET-02/03 assessment | NET-01 → NET-02 → NET-03 |
| Diagnose before fix | Stage 2 cluster loss vs Stage 4 sigma decay must be disambiguated for NET-01 | First phase is diagnosis |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-06 after v1.0 milestone kickoff*
