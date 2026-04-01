# Coding Conventions

**Analysis Date:** 2026-04-01

## Naming Patterns

**Files:**
- Importable Python modules use `snake_case.py` under `project/`, `scripts/agent/`, `scripts/check_data/`, and `data_pre/bf_edge_v3/`; follow examples like `project/losses/axis_side_loss.py`, `project/trainer/trainer.py`, and `scripts/check_data/check_validation_step.py`.
- Package entry files use `__init__.py` with explicit re-exports in files such as `project/models/__init__.py`, `project/losses/__init__.py`, and `project/evaluator/__init__.py`.
- Training/config files under `configs/semantic_boundary/` use long lowercase hyphenated names because they are loaded via `runpy`, not imported as Python modules; follow the pattern in `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py`.
- Data-prep CLI scripts also stay lowercase and descriptive, for example `data_pre/bf_edge_v3/scripts/build_edge_dataset_v3.py` and `data_pre/bf_edge_v3/scripts/fit_local_supports.py`.

**Functions:**
- Functions and methods use `snake_case`; common entrypoint names are `parse_args`, `bootstrap_paths`, `main`, and helper names like `_build_loss_inputs` in `scripts/train/train.py`, `scripts/agent/build_context_packet.py`, and `project/trainer/trainer.py`.
- Private helper methods in classes use a leading underscore, especially in trainer/loss/evaluator modules such as `project/trainer/trainer.py` and `project/losses/semantic_boundary_loss.py`.
- CLI modules usually end with `if __name__ == "__main__": main()` as in `scripts/train/train.py`, `scripts/check_data/check_model_forward.py`, and `data_pre/bf_edge_v3/scripts/build_edge_dataset_v3.py`.

**Variables:**
- Local variables and dict keys use `snake_case`, including tensor/metric keys like `support_cover`, `dir_valid_ratio`, and `best_val_miou` in `project/trainer/trainer.py` and `project/evaluator/semantic_boundary_evaluator.py`.
- Constants use `UPPER_SNAKE_CASE`, for example `EDGE_HEAD_TYPES` in `project/models/semantic_boundary_model.py`, `GRID_SIZE` and `MAX_POINTS` in `scripts/check_data/check_validation_step.py`, and `DEFAULT_FIT_PARAMS` in `data_pre/bf_edge_v3/core/supports_core.py`.
- Configuration payloads are plain dictionaries with string keys such as `type`, `lr`, `support_cover_weight`, and `max_train_batches` in `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py`.

**Types:**
- Classes use `PascalCase`, for example `SharedBackboneSemanticBoundaryModel`, `AxisSideSemanticBoundaryLoss`, `SemanticBoundaryTrainer`, and `InjectIndexValidKeys` in `project/models/semantic_boundary_model.py`, `project/losses/axis_side_loss.py`, `project/trainer/trainer.py`, and `project/transforms/index_keys.py`.
- Type hints are common in newer code and use modern Python syntax like `dict | None`, `list[Path]`, and `dict[str, torch.Tensor]`, as seen in `project/losses/__init__.py`, `scripts/agent/build_context_packet.py`, and `project/evaluator/semantic_boundary_evaluator.py`.
- No `I*` interface prefix or custom typing aliases are used in the core runtime modules under `project/`.

## Code Style

**Formatting:**
- No formatter configuration was detected at the repo root: there is no `pyproject.toml`, `ruff.toml`, `setup.cfg`, `.flake8`, `.pylintrc`, or `.pre-commit-config.yaml`; preserve local formatting instead of assuming Black/Ruff rules.
- Use 4-space indentation and multiline trailing-comma layout for long calls and dict literals, matching `project/models/semantic_boundary_model.py`, `project/trainer/trainer.py`, and `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py`.
- Double-quoted strings dominate across runtime, config, and script code in `project/`, `scripts/`, and `data_pre/bf_edge_v3/`.
- Line length is not strictly enforced; several files keep manually wrapped blocks while others allow long lines, especially `data_pre/bf_edge_v3/core/supports_core.py` and `project/trainer/trainer.py`.

**Linting:**
- No repository-level Python lint configuration was detected.
- Local lint suppression is used only where registry side effects are intentional, for example `import pointcept.models  # noqa: F401` in `project/models/semantic_boundary_model.py` and `import project.models  # noqa: F401` in `scripts/check_data/check_model_forward.py`.
- When adding imports for Pointcept registries, prefer the existing explicit side-effect import plus a narrow `# noqa: F401` comment instead of broad suppression.

## Import Organization

**Order:**
1. `from __future__ import annotations` first when a file opts into postponed evaluation, as in `project/losses/semantic_boundary_loss.py`, `project/trainer/trainer.py`, and `scripts/train/train.py`.
2. Standard-library imports next, such as `argparse`, `os`, `runpy`, `sys`, `math`, and `pathlib` in `scripts/train/train.py`, `scripts/agent/build_context_packet.py`, and `project/trainer/trainer.py`.
3. Third-party imports follow, usually `numpy`, `torch`, and `pointcept.*`, as in `project/evaluator/semantic_boundary_evaluator.py` and `project/models/semantic_boundary_model.py`.
4. Project-local absolute imports (`project.*`) or same-package relative imports (`from .heads import ...`) come last, as in `project/trainer/trainer.py` and `project/models/semantic_boundary_model.py`.

**Grouping:**
- Blank lines separate import groups consistently in `project/` and `scripts/`.
- Relative imports are used mainly inside a package, while top-level scripts prefer absolute imports after bootstrapping `sys.path`; compare `project/models/semantic_boundary_model.py` with `scripts/check_data/check_validation_step.py`.
- Registry-populating imports are kept explicit near the top of runtime scripts and trainer code, for example `import project.datasets  # noqa: F401` and `import project.transforms  # noqa: F401` in `project/trainer/trainer.py`.

**Path Aliases:**
- No path alias system is defined.
- Runtime code relies on normal package imports from `project/` plus `sys.path.insert(0, ...)` bootstrapping in scripts such as `scripts/train/train.py`, `scripts/check_data/check_model_forward.py`, and `scripts/agent/build_context_packet.py`.

## Error Handling

**Patterns:**
- Fail fast on missing environment/config prerequisites with explicit exceptions; examples include `RuntimeError` in `scripts/train/train.py` and `scripts/check_data/check_model_forward.py`, and `FileNotFoundError` in `project/datasets/bf.py`.
- Validate tensor shapes, counts, and enum-like config values with `ValueError`, as seen in `project/models/heads.py`, `project/losses/semantic_boundary_loss.py`, `project/losses/__init__.py`, and `data_pre/bf_edge_v3/utils/stage_io.py`.
- Return safe zero-valued tensors instead of special-case `None` for degenerate metric/loss cases, following helpers such as `_weighted_mean`, `_masked_mean`, `_safe_mean`, and `_safe_zero` in `project/losses/semantic_boundary_loss.py`, `project/losses/axis_side_loss.py`, and `project/evaluator/semantic_boundary_evaluator.py`.

**Error Types:**
- Use `RuntimeError` for runtime contract violations like missing `POINTCEPT_ROOT` or zero processed batches in `scripts/train/train.py` and `project/trainer/trainer.py`.
- Use `FileNotFoundError` for missing datasets, checkpoints, or dependency roots in `project/datasets/bf.py`, `project/trainer/trainer.py`, and `scripts/agent/summarize_train_log.py`.
- Use `assert` sparingly for tight internal invariants only, as in `project/models/semantic_boundary_model.py`, `project/models/semantic_model.py`, and `project/transforms/index_keys.py`; prefer explicit exceptions for user-facing validation.

## Logging

**Framework:**
- The training runtime uses the standard-library `logging` module through `project/utils/logger.py`.
- Most utility, smoke, and agent scripts use plain `print()` output rather than a shared logger, for example `scripts/train/train.py`, `scripts/check_data/check_validation_step.py`, and `data_pre/bf_edge_v3/scripts/build_edge_dataset_v3.py`.

**Patterns:**
- Training logs are human-readable key-value strings emitted through `self.logger.info(...)` in `project/trainer/trainer.py`; they are not JSON or structured-event logs.
- `project/utils/logger.py` writes to both stderr/stdout stream and `train.log` file under the active `work_dir`.
- CLI and preprocessing scripts print concise diagnostic blocks and stop on exceptions rather than catching and muting failures.

## Comments

**When to Comment:**
- Module docstrings are standard for runtime modules and scripts; follow patterns in `project/losses/semantic_boundary_loss.py`, `project/evaluator/semantic_boundary_evaluator.py`, and `scripts/check_data/check_model_forward.py`.
- Use comments to explain supervision semantics, compatibility constraints, or math rationale, not to restate obvious tensor assignments; good examples appear in `project/losses/axis_side_loss.py`, `project/losses/semantic_boundary_loss.py`, and `project/models/semantic_boundary_model.py`.
- Bilingual explanatory comments/docstrings are acceptable in research and data-prep code, as shown in `data_pre/bf_edge_v3/core/supports_core.py`.

**JSDoc/TSDoc:**
- Not applicable in this Python repository.
- Public classes, helper functions, and CLI parsers use normal Python docstrings instead, especially in `scripts/agent/build_context_packet.py` and `data_pre/bf_edge_v3/scripts/build_edge_dataset_v3.py`.

## Function Design

**Size:**
- Simple scripts keep `main()` thin and push parsing/loading logic into helper functions like `bootstrap_paths`, `load_real_sample`, and `move_batch_to_device` in `scripts/check_data/check_validation_step.py`.
- Core runtime modules tolerate larger classes/functions when they centralize training behavior; `project/trainer/trainer.py` is the main example, and new logic there should still prefer extracted helper methods over inlining more branches into `run()`.

**Parameters:**
- Tensor-heavy code prefers explicit positional parameters and a small number of typed scalars, as in `project/losses/semantic_boundary_loss.py` and `project/evaluator/semantic_boundary_evaluator.py`.
- Flexible builder/factory boundaries accept config dicts instead of wide signatures, as in `project/losses/__init__.py`, `project/evaluator/__init__.py`, and `project/trainer/trainer.py`.

**Return Values:**
- Runtime math modules return plain dictionaries of named tensors/floats rather than tuples or custom result objects; follow `project/losses/semantic_boundary_loss.py`, `project/evaluator/axis_side_evaluator.py`, and `project/models/semantic_boundary_model.py`.
- Guard clauses and early returns are preferred in utility code, for example `project/utils/logger.py`, `project/transforms/index_keys.py`, and `data_pre/bf_edge_v3/scripts/build_edge_dataset_v3.py`.

## Module Design

**Exports:**
- Use named exports and explicit `__all__` lists in package entrypoints like `project/models/__init__.py`, `project/losses/__init__.py`, `project/evaluator/__init__.py`, and `project/utils/__init__.py`.
- Keep factory selection logic close to the package root; new loss/evaluator implementations should be wired into `project/losses/__init__.py` or `project/evaluator/__init__.py` instead of being imported ad hoc by callers.

**Barrel Files:**
- The repository uses package-level `__init__.py` files as lightweight barrel files.
- No deeper alias/barrel layer exists beyond package roots; import concrete modules directly unless the package already exports the symbol via `__all__`.
- For Pointcept registry integration, add the new class in its concrete module and ensure the corresponding package import side effect remains reachable from runtime entrypoints such as `project/trainer/trainer.py` or `scripts/check_data/check_model_forward.py`.

---

*Convention analysis: 2026-04-01*
