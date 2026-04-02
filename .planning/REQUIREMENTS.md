# Requirements: semantic-boundary-field

**Defined:** 2026-04-02
**Core Value:** The repository must preserve correct, minimal, SBF-specific operational guidance while removing the hand-built orchestration layer as the default workflow control system.

## v1 Requirements

Requirements for the initial workflow-migration release. Each maps to roadmap phases.

### Workflow Entry

- [x] **FLOW-01**: Maintainer can start repository planning and execution from a GSD-first default entry without consulting the legacy hand-built workflow layer.
- [x] **FLOW-02**: Default-facing workflow docs identify GSD as the primary workflow system for this repository.
- [x] **FLOW-03**: Future workflow planning for this repository can proceed through GSD artifacts and commands without relying on `handoff/` or `project_memory/` as the control plane.

### Legacy Cleanup

- [x] **LEGC-01**: Maintainer can archive `.codex/agents/` out of the default workflow path.
- [x] **LEGC-02**: Maintainer can archive `handoff/` and `project_memory/` so they no longer act as default workflow entry surfaces.
- [x] **LEGC-03**: Maintainer can archive or remove repo-local skills, wrapper docs, hooks, and routing scripts whose main purpose was manual orchestration, context-pollution management, or hand-built session continuity.
- [x] **LEGC-04**: Any legacy workflow material that temporarily remains in place is clearly marked as archived or non-default.

### Canonical Guidance

- [x] **GUID-01**: Maintainer can find canonical SBF-vs-Pointcept boundary rules in a minimal default-facing guide.
- [x] **GUID-02**: Maintainer can find the current validated Stage-2 and current-mainline architecture facts in canonical guidance without relying on archived workflow scaffolding.
- [x] **GUID-03**: Maintainer can find the key experiment evidence and conclusions that still govern future work in canonical guidance.
- [x] **GUID-04**: Maintainer can find the training entrypoint, config usage rules, and guardrails needed to avoid invalid workflow or accidental misuse in canonical guidance.

### Compatibility

- [x] **COMP-01**: Thin wrapper docs, if retained, redirect maintainers to GSD instead of duplicating orchestration logic.
- [x] **COMP-02**: Useful technical project knowledge that is no longer part of the default workflow remains archived and discoverable instead of being lost during cleanup.

## v2 Requirements

Deferred follow-on migration work that should not block the first cleanup phase.

### Follow-On Cleanup

- **FOLL-01**: Maintainer can fully delete archived legacy workflow materials after the GSD-centered flow has been validated in practice.
- **FOLL-02**: Maintainer can add automation for future archival, migration, or workflow-health checks beyond the minimum needed for the first cleanup phase.

## Out of Scope

Explicitly excluded from this migration step.

| Feature | Reason |
|---------|--------|
| New SBF model features or architecture changes | This project is workflow migration, not feature development |
| Pointcept-side redesign or host integration changes | The SBF/Pointcept boundary must stay intact |
| Broad repo redesign beyond workflow cleanup and canonical guidance minimization | Would blur the migration goal |
| New fallback or manual orchestration layers | Would preserve the old default control pattern |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FLOW-01 | Phase 4 | Complete |
| FLOW-02 | Phase 2 | Complete |
| FLOW-03 | Phase 4 | Complete |
| LEGC-01 | Phase 3 | Complete |
| LEGC-02 | Phase 3 | Complete |
| LEGC-03 | Phase 3 | Complete |
| LEGC-04 | Phase 3 | Complete |
| GUID-01 | Phase 1 | Complete |
| GUID-02 | Phase 1 | Complete |
| GUID-03 | Phase 1 | Complete |
| GUID-04 | Phase 1 | Complete |
| COMP-01 | Phase 2 | Complete |
| COMP-02 | Phase 3 | Complete |

**Coverage:**
- v1 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0

---
*Requirements defined: 2026-04-02*
*Last updated: 2026-04-02 after Phase 4 completion*
