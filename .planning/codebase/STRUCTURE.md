# Codebase Structure

**Analysis Date:** 2026-04-01

## Directory Layout

```text
semantic-boundary-field/
├── project/                 # Project-local runtime package layered over Pointcept
│   ├── datasets/            # Dataset registry extensions
│   ├── evaluator/           # Validation metric implementations
│   ├── losses/              # Loss variants for different experiment routes
│   ├── models/              # Backbone wrappers, heads, adapters
│   ├── trainer/             # Local train/validate/checkpoint loop
│   ├── transforms/          # Project-specific data transforms
│   └── utils/               # Logger and metric meter helpers
├── configs/                 # Python config fragments and runnable experiment configs
│   ├── bf/                  # Base BF dataset config
│   └── semantic_boundary/   # Semantic-only and boundary-route configs
├── scripts/                 # Runnable training, smoke, and workflow scripts
│   ├── train/               # Main training entry and train-step smoke
│   ├── check_data/          # Dataset/model/loss/validation smoke scripts
│   └── agent/               # Context-packet and round-artifact automation
├── data_pre/                # Dataset preprocessing pipeline and helpers
├── samples/                 # Local smoke-test samples
├── outputs/                 # Training logs and checkpoints (gitignored)
├── docs/                    # Project docs and workflow specs
├── project_memory/          # Canonical project state and task files
├── reports/                 # Generated summaries, packets, and workflow smokes
├── handoff/                 # Thin handoff entry files
├── .codex/                  # Codex agents, skills, templates
├── .planning/codebase/      # Generated codebase maps
├── README.md                # Project overview and run commands
├── train.md                 # Runtime and training guidance
├── install.md               # Installation guidance
└── requirements.txt         # Python dependencies
```

## Directory Purposes

**project/**
- Purpose: Hold all project-local runtime code that sits on top of Pointcept.
- Contains: `*.py` modules for datasets, models, losses, evaluator logic, trainer logic, transforms, and small utilities.
- Key files: `project/trainer/trainer.py`, `project/datasets/bf.py`, `project/models/semantic_boundary_model.py`, `project/models/heads.py`, `project/losses/axis_side_loss.py`
- Subdirectories: `project/datasets/`, `project/evaluator/`, `project/losses/`, `project/models/`, `project/trainer/`, `project/transforms/`, `project/utils/`

**configs/**
- Purpose: Store executable Python configs that compose datasets, models, and runtime knobs.
- Contains: `*.py` config fragments and full train/smoke configs.
- Key files: `configs/bf/semseg-pt-v3m1-0-base-bf.py`, `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-model.py`, `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py`
- Subdirectories: `configs/bf/` for base BF dataset config; `configs/semantic_boundary/` for semantic-only, signed-direction, support-shape, and axis-side experiment routes

**scripts/**
- Purpose: Expose runnable entry points without embedding shell logic in docs.
- Contains: `*.py` launchers and smoke checks.
- Key files: `scripts/train/train.py`, `scripts/train/check_train_step.py`, `scripts/check_data/check_model_forward.py`, `scripts/agent/refresh_round_artifacts.py`
- Subdirectories: `scripts/train/` for training entry points; `scripts/check_data/` for narrow subsystem checks; `scripts/agent/` for workflow artifact generation

**data_pre/**
- Purpose: Keep preprocessing code separate from train-time runtime code.
- Contains: the `data_pre/bf_edge_v3/` preprocessing package, scripts, and docs.
- Key files: `data_pre/bf_edge_v3/scripts/build_edge_dataset_v3.py`, `data_pre/bf_edge_v3/scripts/add_support_id_to_edge_dataset.py`, `data_pre/bf_edge_v3/docs/PIPELINE.md`
- Subdirectories: `data_pre/bf_edge_v3/core/`, `data_pre/bf_edge_v3/scripts/`, `data_pre/bf_edge_v3/utils/`, `data_pre/bf_edge_v3/docs/`

**docs/**
- Purpose: Human-facing documentation for project design, data format, workflow rules, and validation.
- Contains: Markdown documentation files.
- Key files: `docs/project_structure.md`, `docs/data_format.md`, `docs/validation_design.md`, `docs/workflows/sbf_net_workflow_v1.md`
- Subdirectories: `docs/workflows/` for formal workflow specs

**project_memory/**
- Purpose: Store canonical current state and task-oriented project knowledge.
- Contains: stable topical memory files plus task briefs under `project_memory/tasks/`.
- Key files: `project_memory/current_state.md`, `project_memory/01_current_architecture.md`, `project_memory/04_training_rules.md`, `project_memory/tasks/TASK-2026-03-31-005.md`
- Subdirectories: `project_memory/tasks/` for active and archived task briefs

**reports/**
- Purpose: Hold generated workflow artifacts derived from logs and tasks.
- Contains: `*.summary.md`, `*.summary.json`, `*.context_packet.md`, `*.round_update.draft.md`, `*.workflow_consistency_smoke.md`
- Key files: `reports/log_summaries/semantic_boundary_axis_side_train_smoke_train.summary.md`, `reports/context_packets/TASK-2026-03-31-004.codex.context_packet.md`, `reports/workflow_smokes/TASK-2026-03-31-004.codex.workflow_consistency_smoke.md`
- Subdirectories: `reports/log_summaries/`, `reports/context_packets/`, `reports/round_updates/`, `reports/workflow_smokes/`

**handoff/**
- Purpose: Provide thin entry documents for new chat windows and web-to-local handoff.
- Contains: Markdown handoff files only.
- Key files: `handoff/chat_entry.md`, `handoff/latest_round.md`, `handoff/web_to_agent_contract.md`
- Subdirectories: none

**samples/**
- Purpose: Keep small local sample directories for smoke checks and debugging.
- Contains: per-sample `.npy` arrays such as `coord.npy`, `color.npy`, `normal.npy`, `segment.npy`, `edge.npy`, and optional `edge_support_id.npy`.
- Key files: `samples/training/020101/edge.npy`, `samples/validation/020102/edge.npy`
- Subdirectories: `samples/training/`, `samples/validation/`

**outputs/**
- Purpose: Store train logs and checkpoints produced by experiment configs.
- Contains: per-run folders with `train.log` and `model/*.pth` outputs.
- Key files: `outputs/semantic_boundary_axis_side_train_smoke/train.log`, `outputs/semantic_boundary_train_smoke/model_best.pth`, `outputs/semseg-pt-v3m1-0-base-bf-edge-support-shape-train/train.log`
- Subdirectories: one directory per experiment route or smoke run

**.codex/ and .planning/codebase/:**
- Purpose: Support the repository’s agent workflow.
- Contains: agent definitions, skill templates, generated codebase maps.
- Key files: `.codex/get-shit-done/templates/codebase/architecture.md`, `.codex/get-shit-done/templates/codebase/structure.md`, `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`
- Subdirectories: `.codex/agents/`, `.codex/skills/`, `.codex/get-shit-done/`, `.planning/codebase/`

## Key File Locations

**Entry Points:**
- `scripts/train/train.py`: main training entry that loads a config and runs `SemanticBoundaryTrainer`
- `scripts/train/check_train_step.py`: forward/backward smoke test on a real sample
- `scripts/check_data/check_model_forward.py`: model-forward smoke test
- `scripts/check_data/check_validation_step.py`: validation-step smoke test
- `scripts/agent/refresh_round_artifacts.py`: single entry for workflow artifact refresh

**Configuration:**
- `configs/bf/semseg-pt-v3m1-0-base-bf.py`: base BF dataset and transform pipeline
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-model.py`: semantic-only model config
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-model.py`: shared semantic-plus-boundary model config
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-stage2-v2-model.py`: split-adapter plus support-conditioned head config
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py`: current full train config for the active axis-side route
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py`: current smoke config for the active axis-side route

**Core Logic:**
- `project/trainer/trainer.py`: local train/validate/checkpoint loop
- `project/datasets/bf.py`: dataset wrapper that loads `edge.npy` and optional `edge_support_id.npy`
- `project/transforms/index_keys.py`: transform that keeps `edge` aligned through Pointcept indexing transforms
- `project/models/semantic_boundary_model.py`: shared-backbone semantic-boundary wrapper
- `project/models/heads.py`: semantic head, edge head, support-conditioned head, residual adapter
- `project/losses/axis_side_loss.py`: active axis-side loss
- `project/evaluator/axis_side_evaluator.py`: active axis-side evaluator

**Testing / Verification:**
- `scripts/check_data/`: subsystem smoke scripts
- `scripts/train/check_train_step.py`: one-step optimizer sanity check
- `samples/training/020101/`: real training sample used by smoke checks
- `samples/validation/020102/`: real validation sample used by smoke checks

**Documentation:**
- `README.md`: project overview and main run commands
- `install.md`: environment setup instructions
- `train.md`: runtime and training notes
- `docs/workflows/sbf_net_workflow_v1.md`: formal workflow specification
- `AGENTS.md`: repo guardrails and startup rules

## Naming Conventions

**Files:**
- `snake_case.py` for Python modules under `project/`, `scripts/`, and `data_pre/bf_edge_v3/`; examples: `project/models/semantic_boundary_model.py`, `scripts/agent/build_context_packet.py`
- Long descriptive experiment config names under `configs/semantic_boundary/`; examples: `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py`, `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-stage2-support-dir-train-smoke.py`
- `*-model.py`, `*-data.py`, `*-train.py`, and `*-train-smoke.py` suffixes distinguish config role directly in the filename
- `TASK-YYYY-MM-DD-NNN.md` for task briefs in `project_memory/tasks/`

**Directories:**
- Lowercase snake_case or short lowercase names for implementation directories; examples: `project/losses/`, `project/transforms/`, `configs/semantic_boundary/`
- Plural nouns for collections of related modules; examples: `project/models/`, `project/evaluator/`, `scripts/check_data/`, `reports/log_summaries/`

**Special Patterns:**
- `__init__.py` files act as export/build hubs in `project/models/__init__.py`, `project/losses/__init__.py`, `project/evaluator/__init__.py`, and `project/trainer/__init__.py`
- Output directories mirror config intent; examples: `outputs/semantic_boundary_axis_side_train/`, `outputs/semantic_only_train_smoke/`
- Generated report filenames encode task id, actor, and artifact type; examples: `reports/context_packets/TASK-2026-03-31-004.codex.context_packet.md`, `reports/round_updates/TASK-2026-03-31-005.codex.candidate_brief.draft.md`

## Where to Add New Code

**New Training Route:**
- Primary code: add a new config in `configs/semantic_boundary/`
- Runtime reuse: point it at existing or new modules in `project/models/`, `project/losses/`, and `project/evaluator/`
- Outputs: give it a unique `work_dir` under `outputs/`

**New Model / Head Module:**
- Implementation: `project/models/`
- Export registration: update `project/models/__init__.py` if the class should be importable as part of the project package
- Config wiring: reference it from a config in `configs/semantic_boundary/`

**New Loss / Evaluator Pair:**
- Loss implementation: `project/losses/`
- Evaluator implementation: `project/evaluator/`
- Builder wiring: update `project/losses/__init__.py` and `project/evaluator/__init__.py`
- Config selection: set `loss = dict(type=...)` and `evaluator = dict(type=...)` in a train config

**New Dataset or Transform Extension:**
- Dataset wrapper: `project/datasets/`
- Transform: `project/transforms/`
- Data config wiring: usually `configs/bf/semseg-pt-v3m1-0-base-bf.py` or `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-data.py`

**New Smoke / Diagnostic Script:**
- Training-loop-adjacent check: `scripts/train/`
- Narrow subsystem check: `scripts/check_data/`
- Use `samples/` for real local example data instead of inventing another sample location

**New Workflow / Analysis Automation:**
- Script: `scripts/agent/`
- Canonical state update target: `project_memory/`
- Generated output target: `reports/`

## Special Directories

**`outputs/`:**
- Purpose: generated train logs and checkpoints
- Source: `scripts/train/train.py` and configs in `configs/semantic_boundary/`
- Committed: No; `.gitignore` excludes `outputs/`, `*.log`, and `*.pth`

**`samples/`:**
- Purpose: local smoke-test data
- Source: manually curated reference samples for scripts under `scripts/check_data/` and `scripts/train/`
- Committed: No; `.gitignore` excludes `samples/`

**`reports/`:**
- Purpose: generated workflow artifacts for task tracking
- Source: `scripts/agent/summarize_train_log.py`, `scripts/agent/build_context_packet.py`, `scripts/agent/update_round_artifacts.py`
- Committed: Yes; `reports/` is not ignored

**`project_memory/`:**
- Purpose: canonical project memory and active task definitions
- Source: maintained directly as source-of-truth documents
- Committed: Yes

**`.planning/codebase/`:**
- Purpose: generated codebase maps consumed by later GSD planning/execution steps
- Source: mapper agents
- Committed: Yes

**`data_pre/bf_edge_v3/`:**
- Purpose: standalone preprocessing pipeline for building boundary-field datasets
- Source: repository code, not train-time generated artifacts
- Committed: Yes

---

*Structure analysis: 2026-04-01*
*Update when directory structure changes*
