# Architecture

**Analysis Date:** 2026-04-01

## Pattern Overview

**Overall:** Project-local research training stack layered on external Pointcept registries

**Key Characteristics:**
- `scripts/train/train.py` boots this repo and the external Pointcept checkout together through `POINTCEPT_ROOT` instead of vendoring upstream runtime code.
- Runtime behavior is assembled from executable Python config fragments in `configs/` via `runpy.run_path(...)`, not from YAML or a service container.
- `project/trainer/trainer.py` is the single orchestration loop for semantic-only, signed-direction, support-shape, and current axis-side experiments; route changes happen through config-selected model/loss/evaluator combinations.
- The repository has a second subsystem for workflow state management in `scripts/agent/`, `project_memory/`, `reports/`, and `handoff/`; it is adjacent to the train stack but not in the train-time call path.

## Layers

**Configuration Layer:**
- Purpose: Declare runnable experiments and compose reusable config fragments.
- Location: `configs/bf/semseg-pt-v3m1-0-base-bf.py`, `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-model.py`, `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py`
- Contains: dataset dicts, model dicts, loss/evaluator selection, optimizer/scheduler settings, runtime knobs, and `work_dir` selection.
- Depends on: Python `runpy`, registry names exposed by `project/`, environment variables such as `SBF_DATA_ROOT`.
- Used by: `scripts/train/train.py` and smoke scripts under `scripts/check_data/` and `scripts/train/`.

**Bootstrapping Layer:**
- Purpose: Resolve filesystem roots, enforce required environment variables, and launch a specific workflow.
- Location: `scripts/train/train.py`, `scripts/train/check_train_step.py`, `scripts/check_data/check_model_forward.py`, `scripts/agent/refresh_round_artifacts.py`
- Contains: `POINTCEPT_ROOT` checks, `sys.path` injection, config loading, and top-level command orchestration.
- Depends on: standard library modules, `project/`, and the external Pointcept checkout.
- Used by: shell commands from `README.md`, developer smoke checks, and agent workflow refresh commands.

**Runtime Orchestration Layer:**
- Purpose: Own training, validation, checkpointing, logging, and optimizer/scheduler stepping.
- Location: `project/trainer/trainer.py`
- Contains: dataloader construction, AMP, gradient accumulation, resume/weight loading, per-epoch validation, and checkpoint selection.
- Depends on: Pointcept builders (`build_dataset`, `build_model`), local builders from `project/losses/__init__.py` and `project/evaluator/__init__.py`, plus helpers in `project/utils/logger.py` and `project/utils/meter.py`.
- Used by: `scripts/train/train.py`.

**Registry Extension Layer:**
- Purpose: Extend Pointcept with project-specific dataset, transform, and model types without patching upstream code.
- Location: `project/datasets/bf.py`, `project/transforms/index_keys.py`, `project/models/semantic_boundary_model.py`, `project/models/semantic_model.py`, `project/models/heads.py`
- Contains: `BFDataset`, `InjectIndexValidKeys`, shared-backbone model wrappers, lightweight task heads, and residual adapters.
- Depends on: Pointcept registries, PTv3 backbone construction, and PyTorch modules.
- Used by: config dictionaries resolved through Pointcept's `build_dataset` and `build_model`.

**Objective and Metric Layer:**
- Purpose: Interpret outputs and compute route-specific train and validation semantics.
- Location: `project/losses/semantic_boundary_loss.py`, `project/losses/axis_side_loss.py`, `project/losses/support_shape_loss.py`, `project/evaluator/semantic_boundary_evaluator.py`, `project/evaluator/axis_side_evaluator.py`
- Contains: semantic-only, signed-direction, Route A, support-shape, and current axis-side supervision variants.
- Depends on: PyTorch tensors, the fixed `edge.npy` label layout loaded by `project/datasets/bf.py`, and trainer-provided batch keys.
- Used by: `project/trainer/trainer.py`.

**Workflow Analysis Layer:**
- Purpose: Maintain task state, summarize evidence, and produce coordination artifacts around experiments.
- Location: `scripts/agent/build_context_packet.py`, `scripts/agent/summarize_train_log.py`, `scripts/agent/update_round_artifacts.py`, `project_memory/current_state.md`, `project_memory/tasks/`, `reports/`
- Contains: log parsing, context packet generation, workflow consistency checks, canonical current-state files, and generated summaries.
- Depends on: train logs and checkpoints under `outputs/`, plus the active task pointer in `project_memory/current_state.md`.
- Used by: human/agent workflow rather than the trainer runtime.

## Data Flow

**Training Run:**

1. A shell command invokes `scripts/train/train.py` with a config such as `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py`.
2. `scripts/train/train.py` validates `POINTCEPT_ROOT`, inserts both this repo and the Pointcept repo into `sys.path`, and loads the config with `runpy.run_path(...)`.
3. `project/trainer/trainer.py` imports `project.datasets`, `project.models`, and `project.transforms` for registration side effects, then builds datasets and the PTv3-backed model via Pointcept builders.
4. `project/datasets/bf.py` loads each sample directory, requires `edge.npy`, and optionally attaches `edge_support_id.npy`; `project/transforms/index_keys.py` ensures `edge` survives Pointcept index-based transforms.
5. `project/models/semantic_boundary_model.py` or `project/models/semantic_model.py` wraps the Pointcept backbone, extracts features, and emits `seg_logits` plus route-specific edge outputs.
6. `project/losses/*.py` turns outputs into scalar losses, while `project/evaluator/*.py` computes validation metrics and semantic per-class statistics.
7. `project/trainer/trainer.py` logs to `outputs/.../train.log`, writes checkpoints to `outputs/.../model/`, and picks `model_best.pth` strictly by semantic `val_mIoU`.

**Smoke / Diagnostic Run:**

1. A narrow script under `scripts/check_data/` or `scripts/train/check_train_step.py` bootstraps the same repo roots as the main train entry.
2. It loads one config or one real sample from `samples/` and instantiates a single subsystem: dataset, model, loss, validation step, or train step.
3. The script prints shapes or scalar metrics and exits without adding another framework layer.

**Workflow Artifact Refresh:**

1. `scripts/agent/refresh_round_artifacts.py` chains `scripts/agent/summarize_train_log.py`, `scripts/agent/build_context_packet.py`, and `scripts/agent/update_round_artifacts.py`.
2. Those scripts read canonical state from `project_memory/current_state.md` and task files in `project_memory/tasks/`.
3. Generated artifacts land in `reports/log_summaries/`, `reports/context_packets/`, `reports/round_updates/`, and `reports/workflow_smokes/`.

**State Management:**
- Runtime state is per-process inside `SemanticBoundaryTrainer`; there is no long-lived service or daemon.
- Persistent experiment state lives under `outputs/`.
- Canonical project and task state lives under `project_memory/`.
- Generated workflow checkpoints live under `reports/` and `.planning/codebase/`.

## Key Abstractions

**Registry-Backed Project Module:**
- Purpose: Extend Pointcept through registration instead of source edits.
- Examples: `project/datasets/bf.py`, `project/models/semantic_boundary_model.py`, `project/transforms/index_keys.py`
- Pattern: import-for-side-effect registration; config strings are resolved later by Pointcept builders.

**Config Fragment:**
- Purpose: Compose datasets, model variants, and runtime knobs as executable Python.
- Examples: `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-model.py`, `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-stage2-v2-model.py`, `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py`
- Pattern: `runpy.run_path(...)` imports a base fragment, mutates only the delta, and leaves top-level dicts for the trainer entry.

**Model Wrapper + Head:**
- Purpose: Reuse a Pointcept PTv3 backbone while changing only thin task heads and adapters.
- Examples: `project/models/semantic_boundary_model.py`, `project/models/heads.py`, `project/models/semantic_model.py`
- Pattern: shared-backbone wrapper with config-selected adapters and edge-head classes.

**Loss / Evaluator Pair:**
- Purpose: Keep train supervision semantics aligned with validation metrics for each experiment route.
- Examples: `project/losses/semantic_boundary_loss.py` with `project/evaluator/semantic_boundary_evaluator.py`, `project/losses/axis_side_loss.py` with `project/evaluator/axis_side_evaluator.py`
- Pattern: a config selects a matched pair; the evaluator either reuses the loss object or mirrors its target semantics.

**Current Mainline Route:**
- Purpose: Encode the active `axis + side + support` formulation without changing the trainer contract.
- Examples: `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py`, `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py`, `project/losses/axis_side_loss.py`
- Pattern: keep the existing `edge_pred` plumbing and trainer API, but reinterpret channels and swap in axis-side loss/evaluator logic.

## Entry Points

**Main Training Entry:**
- Location: `scripts/train/train.py`
- Triggers: training or smoke commands from `README.md`
- Responsibilities: validate `POINTCEPT_ROOT`, load config, apply `resume` and `weight` overrides, construct `SemanticBoundaryTrainer`, call `run()`.

**Train-Step Smoke Entry:**
- Location: `scripts/train/check_train_step.py`
- Triggers: low-cost sanity checks on a real sample before long training runs.
- Responsibilities: load `samples/training/020101`, build the edge model and loss, run one backward/optimizer step, print losses.

**Data / Model Validation Entries:**
- Location: `scripts/check_data/check_bf_dataset.py`, `scripts/check_data/check_model_forward.py`, `scripts/check_data/check_loss_forward.py`, `scripts/check_data/check_validation_step.py`, `scripts/check_data/check_validation_metrics.py`
- Triggers: targeted debugging of one pipeline stage.
- Responsibilities: validate one subsystem without running full multi-epoch training.

**Workflow Maintenance Entry:**
- Location: `scripts/agent/refresh_round_artifacts.py`
- Triggers: evidence refresh and task closeout workflow.
- Responsibilities: regenerate log summaries, context packets, and round-update drafts from current outputs and task state.

## Error Handling

**Strategy:** Fail fast at boundaries; explicit preconditions are enforced before heavy runtime work starts.

**Patterns:**
- `scripts/train/train.py`, `scripts/check_data/check_model_forward.py`, and config files under `configs/` raise `RuntimeError` when required environment variables are missing.
- `project/datasets/bf.py` raises `FileNotFoundError` if `edge.npy` is absent instead of silently dropping to semantic-only behavior.
- `project/losses/__init__.py`, `project/evaluator/__init__.py`, `project/models/semantic_boundary_model.py`, and `project/trainer/trainer.py` raise `ValueError` for unsupported types or inconsistent epoch/scheduler settings.
- `project/trainer/trainer.py` loads checkpoints with strict model-state matching and explicit path checks; there is no compatibility fallback layer.

## Cross-Cutting Concerns

**Logging:**
- `project/utils/logger.py` always installs both stream and file handlers, so runtime logs appear in the terminal and in `outputs/.../train.log`.
- `project/trainer/trainer.py` is the only place that formats train and validation summaries, best-checkpoint updates, and per-class semantic results.

**Validation:**
- Validation stays in-process inside `project/trainer/trainer.py`; every displayed epoch runs `validate()` and computes semantic per-class metrics.
- Best-checkpoint selection is semantic `val_mIoU` even for multi-head edge experiments.

**External Dependency Boundary:**
- Pointcept is treated as an external dependency; integration happens through `sys.path` bootstrapping and registry imports in `project/`.
- New runtime features should extend `project/` and `configs/`, not edit upstream Pointcept code.

**Workflow Governance:**
- Canonical state lives in `AGENTS.md`, `project_memory/current_state.md`, and `project_memory/tasks/*.md`.
- Files under `reports/` and `.planning/codebase/` are derived artifacts and should not be treated as the runtime source of truth.

---

*Architecture analysis: 2026-04-01*
*Update when major patterns change*
