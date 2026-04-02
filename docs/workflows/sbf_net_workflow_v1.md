# SBF Net Workflow v1

This file is the formal lifecycle reference for the repository. Default planning and execution now start with GSD and local `.planning/` artifacts; this document remains the place for workflow boundaries, task lifecycle rules, and closeout expectations that still matter during migration.

## Default Repository Entry

Use this order when entering active work:

1. `GSD` commands such as `$gsd-progress`, `$gsd-plan-phase N`, and `$gsd-execute-phase N`
2. `.planning/PROJECT.md`
3. `.planning/ROADMAP.md`
4. `.planning/STATE.md`
5. the active plan or summary under `.planning/phases/`
6. `docs/canonical/README.md` for repository-specific facts, Stage-2 status, experiment evidence, and training guardrails

`project_memory/`, `handoff/`, `.codex/agents/`, repo-local orchestration skills, `claude/`, and checkpoint artifacts are not the default planning/execution entry. Legacy workflow lookup belongs in `docs/archive/workflow-legacy/ARCHIVE_MAP.md`.

## Repository Layers

### Default operating path

- `GSD`
- `.planning/PROJECT.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- active phase plans and summaries under `.planning/phases/`
- `docs/canonical/README.md`

### Formal workflow references

- `AGENTS.md`: repo boundary, GSD-first entry, and guardrails
- `docs/workflows/sbf_net_workflow_v1.md` (this file): lifecycle, handoff, and closeout rules

### Legacy archive lookup

- `docs/archive/workflow-legacy/README.md`: archive landing page
- `docs/archive/workflow-legacy/ARCHIVE_MAP.md`: old-path to archived-path map
- archived continuity material: `handoff/`, `project_memory/`, repo-specific legacy `.codex/agents/`, repo-local orchestration skills, `claude/`, repo-local orchestration helpers, and legacy entry docs

### Compatibility-only survivors

- any retained wrapper doc must be minimal, explicitly non-default, and point back to GSD plus the archive

### Checkpoint artifacts

- `reports/log_summaries/*.summary.md|json`
- `reports/context_packets/*.context_packet.md`
- `reports/round_updates/*.round_update.draft.md`
- `reports/workflow_smokes/*.workflow_consistency_smoke.md`

Checkpoint artifacts support a task. They do not replace GSD planning, canonical repo facts, or task closeout decisions.

## Role Boundaries

The repository still distinguishes workflow roles even though GSD now drives the default control path:

- `discussion`: clarify scope, surface constraints, and decide whether the current task is still the right boundary
- `implementation`: execute the bounded plan without expanding scope or adding fallback behavior
- `review`: verify evidence, assess risks, and decide whether the task is actually done
- `closeout`: record completed work, unresolved items, and what should carry into the next task or phase

## Task Lifecycle

One task should normally cover the full loop:

`discussion -> brief/task -> implementation -> validation -> review -> closeout`

What may happen inside one task:

- discussion and hypothesis convergence
- task clarification
- implementation
- validation and evidence collection
- review and acceptance
- summary, packet, smoke, and round-update refreshes

## Checkpoints Versus Closeout

The following are task-internal checkpoints, not closeout by themselves:

- `summary`
- `packet`
- `workflow smoke`
- `round update draft`
- `refresh`
- `preview`
- `apply`
- intermediate log analysis
- partial review

Closeout requires more than a checkpoint:

- the task's done condition is met
- required verification evidence exists
- review has passed or the remaining issue has been compressed into a new core problem
- the next step is clear: continue the same task, or start a genuinely new one

## Handoff Rules

### Web to local agent

- Prefer a single structured handoff that follows the archived continuity contract location mapped by `docs/archive/workflow-legacy/ARCHIVE_MAP.md`
- Use any retained handoff wrapper only as a thin compatibility redirect, not as a full workflow spec
- Do not use long free-form web text as the default execution entry

### Local planning to implementation

- Implementation should start from the active GSD plan and the minimum required supporting files
- Repository facts and training guardrails should come from `docs/canonical/README.md` and its linked canonical docs, not from legacy workflow scaffolding

### Implementation back to review

- Return a concise implementation summary
- Return concrete verification evidence
- Return unresolved risks or open questions
- State whether the task has actually met its done condition

## Closeout Rules

Stay in the current task when:

- only checkpoints were refreshed
- only part of the implementation or validation is done
- the core question has not been answered yet

Start a new task when:

- the current task is complete and the next step is a genuinely new core problem
- the core question changed enough that reusing the old task would blur scope, evidence, or acceptance criteria

## Documentation Placement

- Repository-specific facts, evidence, and training guardrails belong in `docs/canonical/README.md` and its linked canonical docs
- Workflow boundaries and lifecycle rules belong in `AGENTS.md` and this file
- Historical continuity state should live in the legacy archive rather than in active root workflow surfaces once Phase 3 is complete
- Compatibility docs should redirect, not reintroduce an independent workflow-control path
