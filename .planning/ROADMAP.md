# Roadmap: semantic-boundary-field

## Overview

This migration first extracts the minimum SBF-specific operational knowledge into canonical guidance, then flips the repository's default workflow surfaces to GSD, archives the hand-built orchestration layer out of the default path, and finishes with a clean control-plane cutover where future planning and execution happen through GSD and `.planning/`.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Canonical SBF Guidance** - Preserve the minimum SBF-specific rules, facts, evidence, and training guardrails outside the legacy workflow scaffolding.
- [ ] **Phase 2: GSD Default Entry** - Make the repository's default-facing docs and thin wrappers point to GSD first.
- [ ] **Phase 3: Legacy Workflow Archival** - Archive the manual orchestration surfaces out of the default workflow path while keeping useful history discoverable.
- [ ] **Phase 4: Workflow Control Cutover** - Complete the transition so planning and execution proceed through GSD artifacts instead of the legacy control plane.

## Phase Details

### Phase 1: Canonical SBF Guidance
**Goal**: Maintainers can find the minimum SBF-specific boundary rules, active architecture facts, experiment evidence, and training guardrails without relying on archived workflow scaffolding.
**Depends on**: Nothing (first phase)
**Requirements**: GUID-01, GUID-02, GUID-03, GUID-04
**Success Criteria** (what must be TRUE):
  1. Maintainer can find the SBF-vs-Pointcept boundary rules in a minimal canonical guide.
  2. Maintainer can confirm the current Stage-2 status and the active `axis + side + support` mainline facts without opening archived workflow files.
  3. Maintainer can find the experiment evidence and conclusions that still govern future work in canonical guidance.
  4. Maintainer can find the training entrypoint, config usage rules, and guardrails needed to avoid invalid runs in canonical guidance.
**Plans**: 3 plans

Plans:
- [x] `01-01-PLAN.md` — Create the canonical facts/evidence source for repository boundary rules and current Stage-2 conclusions.
- [ ] `01-02-PLAN.md` — Create the canonical training guardrails source for entrypoints, configs, and fail-fast runtime rules.
- [ ] `01-03-PLAN.md` — Create the canonical guidance index and slim `AGENTS.md` plus the formal workflow doc to point to it.

### Phase 2: GSD Default Entry
**Goal**: Maintainers encounter GSD as the primary workflow system when they enter the repository through default-facing docs and thin wrappers.
**Depends on**: Phase 1
**Requirements**: FLOW-02, COMP-01
**Success Criteria** (what must be TRUE):
  1. Default-facing workflow docs identify GSD as the primary planning and execution system for the repository.
  2. Any retained thin wrapper doc redirects maintainers to GSD instead of duplicating legacy orchestration logic.
  3. A maintainer following the repository's default workflow entry no longer gets sent to a parallel hand-built control path.
**Plans**: TBD

### Phase 3: Legacy Workflow Archival
**Goal**: Legacy orchestration scaffolding is archived or removed from the default path while useful historical knowledge remains discoverable.
**Depends on**: Phase 2
**Requirements**: LEGC-01, LEGC-02, LEGC-03, LEGC-04, COMP-02
**Success Criteria** (what must be TRUE):
  1. `.codex/agents/`, `handoff/`, and `project_memory/` are archived or otherwise removed from the default workflow path.
  2. Repo-local skills, wrapper docs, hooks, and routing scripts whose main purpose was manual orchestration are archived or removed from active workflow surfaces.
  3. Any legacy workflow material that remains in place is clearly labeled as archived or non-default when opened.
  4. Useful technical project knowledge that is no longer part of the default workflow remains archived and discoverable instead of being lost.
**Plans**: TBD

### Phase 4: Workflow Control Cutover
**Goal**: Maintainers can run future repository planning and execution through GSD and `.planning/` without using the legacy workflow layer as the control plane.
**Depends on**: Phase 3
**Requirements**: FLOW-01, FLOW-03
**Success Criteria** (what must be TRUE):
  1. Maintainer can start repository planning and execution from a GSD-first default entry without consulting the legacy hand-built workflow layer.
  2. Maintainer can move from entry to active planning artifacts under `.planning/` without relying on `handoff/` or `project_memory` as the control plane.
  3. Future workflow planning for this repository can proceed through GSD artifacts and commands as the default operating path.
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Canonical SBF Guidance | 1/3 | In Progress | - |
| 2. GSD Default Entry | 0/TBD | Not started | - |
| 3. Legacy Workflow Archival | 0/TBD | Not started | - |
| 4. Workflow Control Cutover | 0/TBD | Not started | - |
