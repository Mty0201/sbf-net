# sbf-net

`sbf-net` is an SBF research repository built on top of Pointcept PTv3. The repository's default workflow is now GSD-first: use `GSD` plus the `.planning` control surface for planning and execution, and use the canonical docs set for repository-specific facts and training guardrails.

## Workflow Entry

Start here for active work:

1. `GSD` commands such as `$gsd-progress`, `$gsd-plan-phase N`, and `$gsd-execute-phase N`
2. `.planning/README.md`
3. `.planning/PROJECT.md`
4. `.planning/ROADMAP.md`
5. `.planning/STATE.md`
6. the active plan or summary under `.planning/phases/`

Use [.planning/README.md](.planning/README.md) as the operational guide for the repository control path.

Repository-specific facts are not maintained in this README. For current SBF-vs-Pointcept boundaries, Stage-2 status, experiment evidence, and training guardrails, read [docs/canonical/README.md](docs/canonical/README.md).

## Repository Boundary

- Active maintenance scope is this repository only: `sbf-net`
- Pointcept is a host dependency and interface boundary, not the project being rewritten here
- If an issue appears to come from Pointcept or the host interface, stop and report it instead of patching around it locally

## Current Technical Context

- Current stage: `Stage-2 architecture rollout / verification phase`
- Active direction: semantic-first boundary supervision
- Explicit geometric-field supervision is not the preferred mainline for milestone `v1.1`
- Historical/reference evidence still includes `support`, `axis-side`, and the prior `axis + side + support` route, but those references do not describe the current active direction

The replacement semantic-first route is the current milestone target, not an already implemented or fully validated runtime path.

Those facts are canonicalized in [docs/canonical/sbf_facts.md](docs/canonical/sbf_facts.md) and indexed from [docs/canonical/README.md](docs/canonical/README.md).

## Runtime Docs

- [install.md](install.md): environment and Pointcept setup only
- [train.md](train.md): training entrypoint, config usage, and runtime command patterns
- [docs/canonical/README.md](docs/canonical/README.md): canonical SBF facts, evidence, and guardrails
- [docs/archive/workflow-legacy/README.md](docs/archive/workflow-legacy/README.md): legacy reference only for archived workflow material

## Repository Layout

- `project/`: project-local datasets, transforms, models, losses, evaluator, and trainer
- `configs/`: SBF configs, including stable runtime entry and historical reference configs
- `scripts/`: runnable entrypoints, including `scripts/train/train.py`
- `docs/`: canonical guidance, workflow references, and technical notes

Detailed structure notes remain in [docs/project_structure.md](docs/project_structure.md).
