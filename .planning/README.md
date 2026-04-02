# .planning Control Plane

This directory is the active repository control surface for workflow planning and execution.

Use it together with `GSD` commands. Do not use `handoff/`, `project_memory/`, or checkpoint artifacts under `reports/` as the default control plane.

## Default Entry Order

1. `GSD` commands such as `$gsd-progress`, `$gsd-plan-phase N`, and `$gsd-execute-phase N`
2. `.planning/PROJECT.md`
3. `.planning/ROADMAP.md`
4. `.planning/STATE.md`
5. the active phase files under `.planning/phases/`
6. `docs/canonical/README.md` for repository-specific facts and guardrails

## What Each Core File Does

- `.planning/PROJECT.md` defines project scope, current workflow-migration objective, validated requirements, and key decisions
- `.planning/ROADMAP.md` defines the phase sequence, phase goals, dependencies, and plan inventory
- `.planning/STATE.md` records the current position, active focus, recent work, and execution status
- `.planning/phases/` holds the executable phase records

## Phase Artifact Meanings

- `*-PLAN.md`: executable work instructions for a bounded plan
- `*-SUMMARY.md`: outcome record for a completed plan
- `*-VERIFICATION.md`: phase-level verification verdict and requirement coverage
- `*-UAT.md`: human verification or acceptance-testing follow-up when needed

## Normal Command Flow

Use `GSD` plus this directory in a tight loop:

1. `$gsd-progress` to see current position and next action
2. `$gsd-plan-phase N` when a roadmap phase has no executable plans yet
3. `$gsd-execute-phase N` when a phase already has approved plans

## Legacy And Evidence Boundaries

- `handoff/` and `project_memory/` are archived compatibility surfaces, not the active control plane
- `reports/` artifacts are supporting evidence only; they do not replace `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`, or active phase plans
- legacy workflow lookup belongs in `docs/archive/workflow-legacy/README.md`

## Repository Scope Reminder

This control path is repository-scope only for `semantic-boundary-field`.

Do not use it to justify Pointcept-side changes, fallback workflow layers, or host-side workaround behavior.
