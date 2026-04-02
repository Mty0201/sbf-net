# Roadmap: semantic-boundary-field

## Milestones

- [x] [`v1.0` workflow migration](/home/mty0201/Pointcept/semantic-boundary-field/.planning/milestones/v1.0-ROADMAP.md) - shipped 2026-04-02, 4 phases, 12 plans, GSD plus `.planning/` cutover complete
- [ ] `v1.1 semantic-first boundary supervision pivot` - active, phases 5-8, semantic-first supervision replacement for the old explicit field route

## Overview

Milestone `v1.1 semantic-first boundary supervision pivot` changes the active SBF direction based on experiment evidence. The old `support + axis + side` route is no longer treated as the mainline. This milestone updates repository-grounded analysis, removes direct explicit-field supervision from the active route, introduces a better semantic-first boundary-aware supervision signal, and validates the new path locally before handing off a clear full-train experiment direction.

## Phases

- [x] **Phase 5: Mainline Analysis Pivot** - Update the repo-grounded analysis, canonical guidance, and active-route references so they reflect the semantic-first pivot instead of the old explicit-field mainline. (completed 2026-04-02)
- [x] **Phase 6: Semantic-First Route Definition** - Define the support-only-first semantic-first candidate route and align the repo-local contract around it. (completed 2026-04-02)
- [x] **Phase 7: Active Route Implementation** - Implement the new active route, removing direct explicit-field supervision from the active mainline while preserving the main training architecture. (completed 2026-04-02)
- [ ] **Phase 8: Local Validation And Experiment Handoff** - Run smoke/sample validation for the new route and produce the next full-train experiment direction with clear evidence boundaries.

## Phase Details

### Phase 5: Mainline Analysis Pivot
**Goal**: Repository-grounded analysis and current-mainline references match the semantic-first supervision pivot.
**Depends on**: Phase 4 from archived milestone `v1.0`
**Requirements**: MAIN-01
**Success Criteria** (what must be TRUE):
  1. Active repo docs no longer describe `support + axis + side` as the preferred mainline.
  2. The semantic-first objective and the rejection of explicit geometric-field supervision are documented in the current repo control surface.
  3. Historical evidence is preserved without being described as the current active route.
**Plans**: 2/2 plans complete

Plans:
- [x] `05-01-PLAN.md` - Update canonical/current-mainline docs and repo analysis to reflect the semantic-first pivot.
- [x] `05-02-PLAN.md` - Align runtime guidance and active-config references with the new milestone direction while preserving historical evidence boundaries.

### Phase 6: Semantic-First Route Definition
**Goal**: The repo defines one support-only-first semantic-first candidate route and the minimal support-centric contract needed to support it.
**Depends on**: Phase 5
**Requirements**: MAIN-02, AUX-01, AUX-02
**Success Criteria** (what must be TRUE):
  1. Support-only is established as the strongest current reference baseline and support-shape is demoted to side evidence only.
  2. The candidate route is concrete enough to implement and verify without restoring explicit geometric-field supervision.
  3. The chosen route fits inside the existing architecture with only minimal supporting changes.
**Plans**: 2/2 plans complete

Plans:
- [x] `06-01-PLAN.md` - Define the support-only-first route semantics and canonical route-definition surfaces.
- [x] `06-02-PLAN.md` - Define the support-centric candidate-route contract and align runtime guidance without broad architecture churn.

### Phase 7: Active Route Implementation
**Goal**: The active training route implements the new semantic-first supervision path and removes direct explicit-field supervision from the mainline by design.
**Depends on**: Phase 6
**Requirements**: AUX-03, COMP-03
**Success Criteria** (what must be TRUE):
  1. Direct `support` / `axis-side` field supervision is no longer the active route.
  2. The active code/config path implements the new semantic-first boundary-aware supervision signal.
  3. The implementation stays inside repo-local extension boundaries and does not require Pointcept changes.
**Plans**: 4 plans

Plans:
- [ ] `07-01-PLAN.md` - Create the semantic-plus-support model path (SupportHead + SharedBackboneSemanticSupportModel + model config).
- [ ] `07-02-PLAN.md` - Create the support-guided semantic focus loss and evaluator with explicit loss math and metric contract.
- [ ] `07-03-PLAN.md` - Wire trainer plumbing and create the active-route train config with train-from-scratch and synthetic verification.
- [ ] `07-04-PLAN.md` - Update canonical docs and runtime guidance with three-category config distinction and cross-file consistency.

### Phase 8: Local Validation And Experiment Handoff
**Goal**: The new route is locally smoke/sample validated and leaves a clear next experiment direction for full training elsewhere.
**Depends on**: Phase 7
**Requirements**: VAL-01, VAL-02, COMP-04
**Success Criteria** (what must be TRUE):
  1. A local smoke/sample validation confirms the new route runs correctly.
  2. The milestone records the correct evidence boundary between local validation and full-train claims.
  3. The next full-train experiment direction is explicit and grounded in the modified route.
**Plans**: 2 plans

Plans:
- [ ] `08-01-PLAN.md` - Add or update the minimal smoke/sample validation path for the semantic-first route.
- [ ] `08-02-PLAN.md` - Record validation results, evidence limits, and the recommended next full-train experiment direction.

## Progress

**Execution Order:**
Phases execute in numeric order: 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 5. Mainline Analysis Pivot | 2/2 | Complete    | 2026-04-02 |
| 6. Semantic-First Route Definition | 2/2 | Complete | 2026-04-02 |
| 7. Active Route Implementation | 4/4 | Complete   | 2026-04-02 |
| 8. Local Validation And Experiment Handoff | 0/2 | Not started | - |
