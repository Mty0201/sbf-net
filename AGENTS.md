# AGENTS.md

## Scope

- The IDE workspace root may be `Pointcept`, but the only repository you should actively maintain here is `semantic-boundary-field`.
- `Pointcept` is a host dependency and interface boundary. Treat it as read-only unless the user explicitly authorizes host-side work.
- Do not modify code outside `semantic-boundary-field` without explicit approval.
- If an issue appears to come from Pointcept or the host interface, stop and report it instead of patching around it.

## Default Workflow Entry

This repository is now GSD-first.

Default planning and execution entry:

1. `GSD` commands such as `$gsd-progress`, `$gsd-plan-phase N`, and `$gsd-execute-phase N`
2. `.planning/PROJECT.md`
3. `.planning/ROADMAP.md`
4. `.planning/STATE.md`
5. the active plan or summary under `.planning/phases/`
6. `docs/canonical/README.md` for repository-specific facts and training guardrails

Do not treat `handoff/`, `project_memory/`, context packets, `.codex/agents/`, repo-local orchestration skills, `claude/`, or legacy wrapper docs as the default control plane.

## Current Stage Boundary

- Current stage: `Stage-2 architecture rollout / verification phase`
- Active mainline expression: `axis + side + support`
- In current code/document sync, the author's spoken `magnitude` maps to `support`; do not rewrite it as a separate landed branch
- Current validation center: `semseg-pt-v3m1-0-base-bf-edge-axis-side-train` and its smoke config
- Repository facts, experiment evidence, and runtime guardrails are canonicalized in `docs/canonical/README.md`, `docs/canonical/sbf_facts.md`, and `docs/canonical/sbf_training_guardrails.md`

## Canonical Repository Guidance

Use the canonical docs set for repository-specific knowledge:

- `docs/canonical/README.md`: entry index for SBF facts, evidence, and guardrails
- `docs/canonical/sbf_facts.md`: SBF-vs-Pointcept boundary rules, Stage-2 status, active mainline semantics, and governing experiment evidence
- `docs/canonical/sbf_training_guardrails.md`: training entrypoint, config roles, command patterns, and fail-fast runtime rules

Workflow control belongs to GSD and `.planning/`. Repository facts belong to the canonical docs set above.

## Legacy Archive

Legacy workflow material is being archived under `docs/archive/workflow-legacy/`.

Use:

- `docs/archive/workflow-legacy/README.md` for the archive landing page
- `docs/archive/workflow-legacy/ARCHIVE_MAP.md` for old-path to archived-path mapping

Archived or archive-bound material includes:

- `handoff/`
- `project_memory/`
- repo-specific legacy agent definitions under `.codex/agents/`
- repo-local orchestration skills that were used for manual continuity or packet refreshes
- repo-specific orchestration helpers under `scripts/agent/`
- mirrored legacy guides under `claude/`
- legacy wrapper docs such as `START_HERE.md`, `MEMORY_RULES.md`, `docs/agents.md`, `docs/workflow.md`, and `CLAUDE_AGENTS.md`

Checkpoint artifacts under `reports/` may still be used as evidence when a task needs them, but they are not canonical facts or workflow control.

If you need legacy continuity material, check the archive after the GSD and canonical entry path above.

## Guardrails

- Do not modify `semantic-boundary-field` directory code outside the current task scope.
- Do not modify Pointcept internals, registries, trainers, or dataset contracts unless explicitly authorized.
- Do not change `scripts/train/train.py`.
- Do not change `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`.
- Do not rewrite the current active mainline as already smoke-verified or full-train-verified when that evidence is absent.
- Do not introduce fallback layers, compatibility shims, automatic bypasses, swallowed errors, or other patches that hide failures.
- Do not mix author-confirmed experiment facts, current-workspace-observed artifacts, and merely-landed code state as if they were the same evidence tier.
- Do not use full `handoff/`, full `project_memory/`, or raw long logs as the default startup context when the GSD and canonical path is sufficient.

## Escalation Rule

If the work crosses repository boundaries, depends on unclear host behavior, or requires a fallback to continue, stop and report the boundary issue instead of improvising a workaround.
