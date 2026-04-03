# semantic-boundary-field

## What This Is

`semantic-boundary-field` is a brownfield research and training repository for SBF experiments that extends an external Pointcept checkout through project-local datasets, models, losses, evaluators, configs, and trainer wiring. The current work is no longer workflow migration; it is active SBF direction-setting and implementation under a semantic-first objective where boundary information should improve edge-region semantic quality without forcing the model to learn an explicit geometric field.

## Core Value

Semantic segmentation remains the primary objective, and any boundary-aware supervision must improve boundary-region semantic quality without dragging the semantic branch into explicit geometric-field learning.

## Current State

- **Shipped milestone:** `v1.0` on 2026-04-02
- **Workflow control path:** `GSD` -> `.planning/README.md` -> `.planning/PROJECT.md` -> `.planning/ROADMAP.md` -> `.planning/STATE.md`
- **Canonical repo facts:** `docs/canonical/README.md`
- **Legacy historical lookup:** `docs/archive/workflow-legacy/README.md`
- **Milestone archives:** `.planning/milestones/`
- **Active milestone:** `v1.1 semantic-first boundary supervision pivot`
- **Phase status:** Phase 10 complete — loss redesign with SmoothL1+Tversky support and optional Lovasz-on-boundary focus

## Current Milestone: v1.1 semantic-first boundary supervision pivot

**Goal:** Improve semantic-side performance by using boundary information more effectively near edges without forcing the model to learn an explicit geometric field.

**Target features:**
- remove direct `support` / `axis-side` field supervision from the active mainline
- stop treating explicit local field learning as a supervised target
- define and implement a better boundary-aware supervision signal under a semantic-first objective
- keep the backbone and main training architecture largely intact unless the new supervision signal clearly justifies a supporting architectural adjustment
- add repository-grounded analysis plus local smoke/sample validation for the modified path
- produce a clear next experiment direction for full training in a separate environment

## Validated

- ✓ The repo can run SBF training and smoke workflows against an external Pointcept checkout through `POINTCEPT_ROOT` and Python config entrypoints under `scripts/train/` and `configs/` — existing
- ✓ The repo already contains the current Stage-2 training architecture and route-specific runtime pieces in `project/`, including the active `axis + side + support` mainline path — existing
- ✓ The repo already enforces important training/config guardrails such as repository-bound runtime extensions, explicit dataset/config environment requirements, and no upstream Pointcept patching from the local project module path — existing
- ✓ Minimal canonical SBF-specific guidance exists under `docs/canonical/` for boundary rules, Stage-2/current-mainline facts, governing experiment evidence, and training/config guardrails — v1.0
- ✓ Default-facing docs and wrappers route maintainers to GSD, local `.planning/`, and canonical guidance instead of an old parallel control path — v1.0
- ✓ Legacy workflow material is archived under `docs/archive/workflow-legacy/` and no longer serves as the active control plane — v1.0
- ✓ Future workflow planning and execution route through GSD plus `.planning/` without relying on the archived workflow layer as the control plane — v1.0
- ✓ The repo control surface, canonical facts, and runtime guidance now describe semantic-first boundary supervision as the active direction while preserving older geometric-field routes as historical/reference evidence — Phase 5 (`MAIN-01`)
- ✓ The repo now defines a support-only-first semantic-first candidate route, records support-shape as weaker side evidence only, and documents the support-centric route contract without requiring Pointcept changes — Phase 6 (`MAIN-02`, `AUX-01`, `AUX-02`)
- ✓ The support-guided semantic focus active route is fully implemented — model (SharedBackboneSemanticSupportModel), loss (SupportGuidedSemanticFocusLoss), evaluator, trainer wiring, train config, and canonical docs all in place — Phase 7 (`AUX-03`, `COMP-03`)
- ✓ The implemented route is locally smoke/sample validated with a full pipeline check script covering forward + loss + backward + optimizer steps and focus activation verification — Phase 8 (`VAL-01`, `VAL-02`)
- ✓ The milestone leaves a clear next experiment direction for full training outside the local environment, with four documented directions (soft masking, negative sample calibration, alpha–σ coupling, adaptive inference) and support-only 74.6 mIoU as comparison baseline — Phase 8 (`COMP-04`)
- ✓ The loss redesign replaces BCE with SmoothL1+Tversky for support supervision and adds optional Lovasz-on-boundary focus, with Variant C (ablation) and Variant A (boundary focus) training configs ready — Phase 10 (`LOSS-01` through `LOSS-07`)

## Active

(No active requirements — all v1.1 requirements validated)

## Next Milestone Goals

- Establish the new active SBF direction from experiment evidence rather than continuing the old `support + axis + side` mainline.
- Remove direct explicit-field supervision from the active path and replace it with a better semantic-first boundary-aware supervision design.
- Land the smallest credible implementation and validation slice that can inform the next full-training experiment.

## Out of Scope

- Reintroducing `support + axis + side` as the active mainline — experiment evidence no longer supports it as the preferred direction
- Treating explicit local geometric-field learning as the supervised target — this now conflicts with the semantic-first objective
- Pointcept runtime, registry, trainer, or dataset protocol redesign outside this repository's allowed extension boundary — preserve host boundary
- Compatibility layers, fallback routing, or silent bypasses that hide training/runtime failures — fail fast instead
- Full-train claims from the local environment — this milestone stops at repository-grounded implementation, smoke/sample validation, and the next experiment recommendation

## Context

The repository remains a brownfield ML research codebase executed locally through Python and Conda, with Pointcept treated as an external dependency rather than something vendored or edited in place. Training and runtime behavior are assembled from Python config fragments in `configs/`, launched from `scripts/train/train.py`, and implemented through project-local registry extensions in `project/`.

The workflow migration milestone is complete and now forms validated repository infrastructure rather than the main problem. The active technical question is different: how to use boundary information to help semantic segmentation, especially near edges, without forcing the model to solve an explicit geometric prediction task that harms the main objective.

The current source of truth for this pivot is the user-confirmed experiment interpretation from this milestone kickoff:

- the old `support + axis + side` route is no longer considered viable as the mainline
- `support` only worked as a weak hint partly because it did not force explicit geometric-field learning
- `axis + side` supervision tends to create a stronger explicit geometric task that hurts semantic performance
- local boundary attraction behavior should now be treated as a possible byproduct of semantic and boundary supervision, not the direct supervised target

The preferred milestone posture is to keep the backbone and main training architecture largely intact while removing direct explicit-field supervision and exploring a better-suited boundary-aware supervision signal.

## Constraints

- **Repository Boundary**: Keep Pointcept as an external host dependency and avoid modifying code outside `semantic-boundary-field` without explicit authorization — preserves the SBF/host interface boundary
- **Semantic-First Objective**: Semantic segmentation remains the primary objective — auxiliary supervision cannot be allowed to dominate the learning problem
- **No Explicit Field Target**: Do not treat local geometric-field learning as the direct supervised target of the active mainline — current evidence says that objective pulls the model away from semantics
- **Architecture Stability**: Keep the backbone and main training architecture largely intact unless the new supervision signal clearly requires a supporting architectural change — keeps the milestone minimal and attributable
- **Validation Boundary**: Local work should stop at smoke/sample validation and experiment-direction definition — full training belongs to the separate training environment

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| GSD is the primary workflow system for future planning and execution | The goal of this project is to retire the hand-built orchestration layer and simplify workflow control | Phase 2 established GSD-first entry across root docs, formal workflow surfaces, and retained wrappers; Phase 4 turned `.planning/` into the explicit repo-local operating path |
| Legacy workflow scaffolding is a migration target, not a canonical control layer | `.codex/agents/`, `handoff/`, `project_memory/`, wrapper docs, and routing/context tools were mainly created to compensate for workflow limitations that GSD now covers | Phase 3 archived those surfaces or reduced them to compatibility redirects |
| Minimal canonical guidance must remain repository-specific | SBF/Pointcept boundaries, current-mainline architecture facts, experiment evidence, and training/config guardrails still matter for safe future work | Phases 1-3 kept canonical docs active while moving workflow history into the archive |
| Prefer physical archival over soft deprecation where practical | A clean cut is less ambiguous than leaving legacy material in default-facing paths | Phase 3 moved legacy continuity trees, manual tooling, and wrapper bodies into `docs/archive/workflow-legacy/` |
| Thin wrappers are allowed only for compatibility or transition | Any remaining wrappers must be minimal and must point to GSD rather than reintroduce manual orchestration | Phase 2 reduced retained wrappers to redirect-only compatibility surfaces |
| Live planning artifacts should stay milestone-scoped and compact | Archived roadmap and requirements history should not bloat the default control surface for future work | `v1.0` was moved into `.planning/milestones/`, leaving live `.planning/ROADMAP.md` ready for the next milestone |
| Semantic performance is the governing objective for boundary supervision | Boundary-aware signals are useful only if they help semantic quality, especially near edges | `v1.1` removes support-shape from the candidate-mainline role, keeps support-only as the strongest current reference baseline, and advances a support-centric route for implementation |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-03 after Phase 10 completion (loss redesign)*
