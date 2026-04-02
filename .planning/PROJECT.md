# semantic-boundary-field

## What This Is

`semantic-boundary-field` is a brownfield research and training repository for SBF experiments that extends an external Pointcept checkout through project-local datasets, models, losses, evaluators, configs, and trainer wiring. The repository has now completed its workflow-control migration to `GSD` plus `.planning/`, while preserving repository-specific facts and training guardrails in canonical docs and retaining legacy workflow material only as historical archive.

## Core Value

The repository must preserve correct, minimal, SBF-specific operational guidance while removing the hand-built orchestration layer as the default workflow control system.

## Current State

- **Shipped milestone:** `v1.0` on 2026-04-02
- **Workflow control path:** `GSD` -> `.planning/README.md` -> `.planning/PROJECT.md` -> `.planning/ROADMAP.md` -> `.planning/STATE.md`
- **Canonical repo facts:** `docs/canonical/README.md`
- **Legacy historical lookup:** `docs/archive/workflow-legacy/README.md`
- **Milestone archives:** `.planning/milestones/`

## Validated

- ✓ The repo can run SBF training and smoke workflows against an external Pointcept checkout through `POINTCEPT_ROOT` and Python config entrypoints under `scripts/train/` and `configs/` — existing
- ✓ The repo already contains the current Stage-2 training architecture and route-specific runtime pieces in `project/`, including the active `axis + side + support` mainline path — existing
- ✓ The repo already enforces important training/config guardrails such as repository-bound runtime extensions, explicit dataset/config environment requirements, and no upstream Pointcept patching from the local project module path — existing
- ✓ Minimal canonical SBF-specific guidance exists under `docs/canonical/` for boundary rules, Stage-2/current-mainline facts, governing experiment evidence, and training/config guardrails — v1.0
- ✓ Default-facing docs and wrappers route maintainers to GSD, local `.planning/`, and canonical guidance instead of an old parallel control path — v1.0
- ✓ Legacy workflow material is archived under `docs/archive/workflow-legacy/` and no longer serves as the active control plane — v1.0
- ✓ Future workflow planning and execution route through GSD plus `.planning/` without relying on the archived workflow layer as the control plane — v1.0

## Active

None. The next milestone has not been defined yet.

## Next Milestone Goals

- Decide whether to address accepted workflow debt first or return to repository feature and verification work.
- If workflow debt is prioritized, evaluate Nyquist validation coverage and canonical provenance cleanup.
- If feature work resumes, define the next milestone explicitly with new requirements instead of reusing archived `v1.0` scope.

## Out of Scope

- New model architecture work or feature-development on SBF training routes — this phase is workflow migration only
- Pointcept runtime, registry, trainer, or dataset protocol redesign outside this repository's allowed extension boundary — preserve host boundary
- Broad repository redesign unrelated to workflow migration — keep scope on cleanup, archival, migration boundaries, and canonical guidance minimization
- Compatibility layers, fallback routing, or new manual orchestration patches that preserve the old framework as a shadow default — this would defeat the migration goal

## Context

The repository remains a brownfield ML research codebase executed locally through Python and Conda, with Pointcept treated as an external dependency rather than something vendored or edited in place. Training and runtime behavior are still assembled from Python config fragments in `configs/`, launched from `scripts/train/train.py`, and implemented through project-local registry extensions in `project/`.

The workflow migration milestone is complete. The active control path is now small and explicit, while repository-specific knowledge remains separated from workflow policy:

- workflow control in `.planning/`
- canonical SBF facts and guardrails in `docs/canonical/`
- historical workflow continuity in `docs/archive/workflow-legacy/`

The current post-migration posture is to keep that separation intact while defining the next milestone.

## Constraints

- **Repository Boundary**: Keep Pointcept as an external host dependency and avoid modifying code outside `semantic-boundary-field` without explicit authorization — preserves the SBF/host interface boundary
- **Workflow Scope**: Phase 1 is workflow cleanup and migration only — avoids conflating process migration with model or feature development
- **Canonical Guidance**: Essential SBF-specific knowledge must remain available in minimal canonical form — future work still needs accurate architecture facts, experiment evidence, and guardrails
- **Archive Bias**: Legacy orchestration materials should be physically archived out of the default path when practical — the repo should read as GSD-centered after this phase
- **No Shadow Default**: Thin wrappers may remain temporarily, but they must not continue acting as a parallel default workflow system — prevents partial migration drift

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| GSD is the primary workflow system for future planning and execution | The goal of this project is to retire the hand-built orchestration layer and simplify workflow control | Phase 2 established GSD-first entry across root docs, formal workflow surfaces, and retained wrappers; Phase 4 turned `.planning/` into the explicit repo-local operating path |
| Legacy workflow scaffolding is a migration target, not a canonical control layer | `.codex/agents/`, `handoff/`, `project_memory/`, wrapper docs, and routing/context tools were mainly created to compensate for workflow limitations that GSD now covers | Phase 3 archived those surfaces or reduced them to compatibility redirects |
| Minimal canonical guidance must remain repository-specific | SBF/Pointcept boundaries, current-mainline architecture facts, experiment evidence, and training/config guardrails still matter for safe future work | Phases 1-3 kept canonical docs active while moving workflow history into the archive |
| Prefer physical archival over soft deprecation where practical | A clean cut is less ambiguous than leaving legacy material in default-facing paths | Phase 3 moved legacy continuity trees, manual tooling, and wrapper bodies into `docs/archive/workflow-legacy/` |
| Thin wrappers are allowed only for compatibility or transition | Any remaining wrappers must be minimal and must point to GSD rather than reintroduce manual orchestration | Phase 2 reduced retained wrappers to redirect-only compatibility surfaces |
| Live planning artifacts should stay milestone-scoped and compact | Archived roadmap and requirements history should not bloat the default control surface for future work | `v1.0` was moved into `.planning/milestones/`, leaving live `.planning/ROADMAP.md` ready for the next milestone |

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
*Last updated: 2026-04-02 after v1.0 milestone completion*
