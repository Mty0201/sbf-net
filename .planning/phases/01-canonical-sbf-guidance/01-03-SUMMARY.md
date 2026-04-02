---
phase: 01-canonical-sbf-guidance
plan: 03
subsystem: docs
tags: [canonical-guidance, workflow-migration, stage-2, docs]
requires:
  - phase: 01-canonical-sbf-guidance
    provides: canonical facts/evidence source in docs/canonical/sbf_facts.md
  - phase: 01-canonical-sbf-guidance
    provides: canonical training guardrails in docs/canonical/sbf_training_guardrails.md
provides:
  - canonical README index that maps the four Phase 1 maintainer questions to precise answer anchors
  - AGENTS and workflow docs that keep guardrails/lifecycle rules while pointing repo facts to canonical docs
affects: [phase-02-gsd-default-entry, phase-03-legacy-workflow-archival, workflow-entry]
tech-stack:
  added: []
  patterns:
    - repo-specific facts live under docs/canonical with README.md as the answerability index
    - workflow docs stay thin and point to canonical guidance instead of embedding scoreboards
key-files:
  created:
    - docs/canonical/README.md
    - .planning/phases/01-canonical-sbf-guidance/01-03-SUMMARY.md
  modified:
    - AGENTS.md
    - docs/workflows/sbf_net_workflow_v1.md
    - .planning/STATE.md
    - .planning/ROADMAP.md
key-decisions:
  - "docs/canonical/README.md becomes the single Phase 1 answerability index for repository-specific facts and guardrails."
  - "AGENTS.md and docs/workflows/sbf_net_workflow_v1.md keep workflow boundaries and startup rules, but point repo-specific facts to the canonical docs set."
patterns-established:
  - "Answerability index pattern: docs/canonical/README.md maps maintainer questions to exact canonical file sections."
  - "Workflow-thin pattern: AGENTS and lifecycle docs keep guardrails/flow rules and defer repo facts to docs/canonical."
requirements-completed: [GUID-01, GUID-02, GUID-03, GUID-04]
duration: 3min
completed: 2026-04-02
---

# Phase 1 Plan 3: Canonical Guidance Index Summary

**Canonical README index that answers the four Phase 1 maintainer questions and routes workflow docs to repo-specific guidance**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-02T02:53:24Z
- **Completed:** 2026-04-02T02:56:41Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Created `docs/canonical/README.md` as the single Phase 1 index into the canonical docs set, with an explicit read order and answerability mapping for the four maintainer questions.
- Added a maintainer self-check that names the exact canonical file and section anchors needed to verify Phase 1 coverage with simple file reads or `rg`.
- Slimmed `AGENTS.md` and `docs/workflows/sbf_net_workflow_v1.md` so they keep guardrails and lifecycle rules while routing repo-specific facts, evidence, and training guardrails to the canonical docs set.

## Task Commits

Each task was committed atomically:

1. **Task 1: Create the canonical guidance index and Phase 1 answerability checklist** - `728d41b` (feat)
2. **Task 2: Slim AGENTS and the formal workflow doc so they point to the canonical docs set** - `4176411` (feat)

## Files Created/Modified

- `docs/canonical/README.md` - Canonical index, four maintainer questions, read order, and Phase 1 self-check anchors.
- `AGENTS.md` - Repository guardrails and startup rules now point repo facts to the canonical docs set instead of embedding the Stage-2 scoreboard inline.
- `docs/workflows/sbf_net_workflow_v1.md` - Workflow lifecycle doc now directs maintainers to the canonical guidance set for repo-specific facts.
- `.planning/phases/01-canonical-sbf-guidance/01-03-SUMMARY.md` - Execution summary for this plan.
- `.planning/STATE.md` - Phase and plan progress, decisions, metrics, and session position after completing plan 01-03.
- `.planning/ROADMAP.md` - Phase 1 plan-progress table after completing the canonical guidance index plan.

## Decisions Made

- Used `docs/canonical/README.md` as the single answerability index rather than duplicating facts in multiple workflow surfaces.
- Kept `AGENTS.md` and `docs/workflows/sbf_net_workflow_v1.md` focused on workflow boundary and lifecycle rules while preserving direct pointers to canonical facts.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Manually synchronized STATE.md and ROADMAP.md after helper commands left stale progress text**
- **Found during:** Summary/state update step
- **Issue:** The state and roadmap helper commands updated counters, but the human-readable progress sections still showed plan 03 as incomplete.
- **Fix:** Patched `.planning/STATE.md` and `.planning/ROADMAP.md` so Phase 1 shows `3/3` complete, the plan checklist is fully checked, and the visible progress/status text matches the recorded counters.
- **Files modified:** `.planning/STATE.md`, `.planning/ROADMAP.md`
- **Verification:** Re-read both files after patching and confirmed Phase 1 is marked complete with `3/3` plans and `100%` progress.

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Corrected stale progress artifacts only. No scope creep.

## Issues Encountered

- A parallel `git add` attempt created a transient `.git/index.lock` race. Retrying the second stage operation serially resolved it without widening file scope.
- The plan's broad forbidden-path diff check surfaced pre-existing `project_memory/` working tree changes. Verification was completed with scoped checks to confirm this plan only edited the allowed canonical/workflow files.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 1 now has a complete canonical docs set plus a single index that answers the four maintainer questions without relying on legacy workflow-state files.
- Phase 2 can rewrite default-facing entry docs and thin wrappers to point at this canonical set and GSD-first workflow entry.

## Self-Check: PASSED

- Verified file exists: `docs/canonical/README.md`
- Verified file exists: `.planning/phases/01-canonical-sbf-guidance/01-03-SUMMARY.md`
- Verified commits exist: `728d41b`, `4176411`
