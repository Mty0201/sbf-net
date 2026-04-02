# Workflow Legacy Archive Map

This map defines which workflow surfaces stay active, which move into the legacy archive, and which may remain only as minimal compatibility stubs.

## Active Surfaces

| Path | Status | Notes |
|------|--------|-------|
| `AGENTS.md` | Active | Repo boundary, GSD-first entry, and guardrails |
| `docs/workflows/sbf_net_workflow_v1.md` | Active | Lifecycle and closeout rules |
| `docs/canonical/README.md` | Active | Canonical SBF facts, evidence, and training guardrails |
| `docs/canonical/sbf_facts.md` | Active | Must remain active; provenance may point into the archive after Phase 3 |
| `docs/canonical/sbf_training_guardrails.md` | Active | Canonical runtime guardrails |
| `.planning/` | Active | GSD planning and execution state |
| `.codex/get-shit-done/` | Active | GSD runtime/tooling, not repo-specific legacy scaffolding |
| GSD-owned `.codex/agents/` / `.codex/skills/` assets | Active only if required | Any retained subset must be recorded in `codex/ACTIVE_TOOLING_NOTES.md` |

## Archived Trees

| Current path | Archive destination | Reason |
|--------------|---------------------|--------|
| `handoff/` | `docs/archive/workflow-legacy/handoff/` | Manual cross-window continuity layer is no longer active workflow control |
| `project_memory/` | `docs/archive/workflow-legacy/project_memory/` | Pre-GSD task and topical memory should remain discoverable but non-default |
| `scripts/agent/` | `docs/archive/workflow-legacy/codex/scripts-agent/` | Manual packet/handoff routing helpers are no longer active workflow surfaces |
| `claude/` | `docs/archive/workflow-legacy/codex/claude/` | Mirrored legacy role/skill guides are historical compatibility material |

## Archived `.codex` Subset

Archive these repo-specific legacy assets:

- `.codex/agents/architect.toml`
- `.codex/agents/worker.toml`
- `.codex/agents/maintainer.toml`
- `.codex/skills/prepare-task-brief/`
- `.codex/skills/refresh-round-artifacts/`
- `.codex/skills/update-handoff-memory/`
- `.codex/skills/workflow-consistency-smoke/`

Archive destination:

- `docs/archive/workflow-legacy/codex/agents/`
- `docs/archive/workflow-legacy/codex/skills/`

## Root Legacy-Entry Docs

These files no longer belong in the active root entry path and should be archived under `docs/archive/workflow-legacy/wrappers/` unless a thin redirect stub is still required:

- `START_HERE.md`
- `MEMORY_RULES.md`
- `docs/agents.md`
- `docs/workflow.md`
- `CLAUDE_AGENTS.md`

## Compatibility Stubs

These paths may remain only as thin redirects if old-link compatibility still matters:

- `CLAUDE.md`
- `docs/workflow.md`
- `CLAUDE_AGENTS.md`
- archived root-level `handoff/` or `project_memory/` notices only if needed for path compatibility

If retained, the stub must:

- say archived or non-default immediately
- point back to GSD and `.planning/`
- point to this archive when historical lookup is needed
- not contain a standalone startup chain

## Provenance Rule

When `project_memory/` moves into the archive, any active canonical doc that still cites `project_memory/...` must be updated to:

- point to `docs/archive/workflow-legacy/project_memory/...`, or
- point to an explicit archived stub path that preserves discoverability without restoring active workflow status

This applies to `docs/canonical/sbf_facts.md` in Phase 3.
