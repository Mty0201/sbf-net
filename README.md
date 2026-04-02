# semantic-boundary-field

`semantic-boundary-field` is an SBF research repository built on top of Pointcept PTv3. The repository's default workflow is now GSD-first: use GSD plus local `.planning/` artifacts for planning and execution, and use the canonical docs set for repository-specific facts and training guardrails.

## Workflow Entry

Start here for active work:

1. `GSD` commands such as `$gsd-progress`, `$gsd-plan-phase N`, and `$gsd-execute-phase N`
2. `.planning/PROJECT.md`
3. `.planning/ROADMAP.md`
4. `.planning/STATE.md`
5. the active plan or summary under `.planning/phases/`

Repository-specific facts are not maintained in this README. For current SBF-vs-Pointcept boundaries, Stage-2 status, experiment evidence, and training guardrails, read [docs/canonical/README.md](docs/canonical/README.md).

## Repository Boundary

- Active maintenance scope is this repository only: `semantic-boundary-field`
- Pointcept is a host dependency and interface boundary, not the project being rewritten here
- If an issue appears to come from Pointcept or the host interface, stop and report it instead of patching around it locally

## Current Technical Context

- Current stage: `Stage-2 architecture rollout / verification phase`
- Active mainline expression: `axis + side + support`
- Current verification focus: `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py` and its smoke config

Those facts are canonicalized in [docs/canonical/sbf_facts.md](docs/canonical/sbf_facts.md) and indexed from [docs/canonical/README.md](docs/canonical/README.md).

## Runtime Docs

- [install.md](install.md): environment and Pointcept setup only
- [train.md](train.md): training entrypoint, config usage, and runtime command patterns
- [docs/canonical/README.md](docs/canonical/README.md): canonical SBF facts, evidence, and guardrails
- [docs/archive/workflow-legacy/README.md](docs/archive/workflow-legacy/README.md): legacy reference only for archived workflow material

## Repository Layout

- `project/`: project-local datasets, transforms, models, losses, evaluator, and trainer
- `configs/`: SBF configs, including the current axis-side verification configs
- `scripts/`: runnable entrypoints, including `scripts/train/train.py`
- `docs/`: canonical guidance, workflow references, and technical notes

Detailed structure notes remain in [docs/project_structure.md](docs/project_structure.md).
