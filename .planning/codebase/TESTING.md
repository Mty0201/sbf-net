# Testing Patterns

**Analysis Date:** 2026-04-01

## Test Framework

**Runner:**
- No formal Python test runner is configured in this repository: there is no `pyproject.toml`, `pytest.ini`, `tox.ini`, `.coveragerc`, or `.github/workflows/` test pipeline at the repo root.
- The current testing surface is a set of executable smoke/diagnostic scripts under `scripts/check_data/` and `scripts/train/`, plus smoke training configs under `configs/semantic_boundary/`.

**Assertion Library:**
- No dedicated assertion library is used for repository-level tests.
- Current checks rely on script exit status, printed tensor shapes/metrics, and a few inline boolean guards like `loss_has_nan` / `metric_has_nan` in `scripts/check_data/check_validation_step.py`.

**Run Commands:**
```bash
export POINTCEPT_ROOT=/path/to/Pointcept
conda run --no-capture-output -n ptv3 python scripts/check_data/check_bf_dataset.py
conda run --no-capture-output -n ptv3 python scripts/check_data/check_model_forward.py
conda run --no-capture-output -n ptv3 python scripts/check_data/check_validation_step.py
conda run --no-capture-output -n ptv3 python scripts/train/check_train_step.py
conda run --no-capture-output -n ptv3 python scripts/train/train.py --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py --pointcept-root "$POINTCEPT_ROOT"
```
- Additional tensor-only diagnostics exist in `scripts/check_data/check_loss_forward.py` and `scripts/check_data/check_validation_metrics.py`, but they are still standalone scripts rather than test-runner cases.
- Watch mode: Not used.
- Coverage report: Not used.

## Test File Organization

**Location:**
- There are no co-located `test_*.py` or `*_test.py` files under `project/` or `data_pre/`.
- Manual validation scripts live in `scripts/check_data/` and `scripts/train/`.
- End-to-end smoke configs live in `configs/semantic_boundary/`, for example `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train-smoke.py` and `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py`.
- Sample fixtures are checked into `samples/training/020101` and `samples/validation/020102`.

**Naming:**
- Manual checks use verb-first filenames like `check_bf_dataset.py`, `check_model_forward.py`, `check_validation_step.py`, and `check_train_step.py`.
- Smoke configs append `-smoke.py` to the corresponding training config name in `configs/semantic_boundary/`.

**Structure:**
```text
scripts/
  check_data/
    check_bf_dataset.py
    check_loss_forward.py
    check_model_forward.py
    check_validation_metrics.py
    check_validation_step.py
  train/
    check_train_step.py
    train.py
configs/
  semantic_boundary/
    semseg-pt-v3m1-0-base-bf-edge-train-smoke.py
    semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py
samples/
  training/020101/
  validation/020102/
```

## Test Structure

**Suite Organization:**
```python
def main():
    repo_root = bootstrap_paths()

    import project.models  # noqa: F401
    from project.losses import SemanticBoundaryLoss
    from project.evaluator import SemanticBoundaryEvaluator
    from pointcept.models import build_model

    batch = move_batch_to_device(load_real_sample(sample_dir), device)

    with torch.no_grad():
        output = model(forward_input)
        loss_dict = loss_fn(...)
        metric_dict = evaluator(...)

    print(f"seg_logits_shape: {tuple(output['seg_logits'].shape)}")
    print(f"loss_has_nan: {has_nan_dict(loss_dict)}")
```
- This pattern is taken directly from `scripts/check_data/check_validation_step.py`.

**Patterns:**
- Each script is standalone and shell-invoked; there is no shared runner or discovery mechanism.
- Setup usually starts with `bootstrap_paths()` and optionally `runpy.run_path(...)` to load a config file, as seen in `scripts/check_data/check_model_forward.py`, `scripts/check_data/check_validation_step.py`, and `scripts/train/train.py`.
- Most checks run one of three paths: synthetic tensor forward pass, one real sample from `samples/`, or a one-batch smoke training config.
- Success is determined by “no exception”, expected output keys/shapes, non-NaN metrics, and for smoke training the presence of `train.log`, `model_last.pth`, or `model_best.pth` under `outputs/.../model/`.
- There is no teardown layer beyond normal process exit; scripts do not use temp directories or fixture cleanup hooks.

## Mocking

**Framework:**
- No mocking framework is used.

**Patterns:**
```python
num_points = 2048
pseudo_input = dict(
    coord=torch.randn(num_points, 3, dtype=torch.float32, device=device),
    grid_coord=torch.randint(0, 64, (num_points, 3), dtype=torch.int64, device=device),
    feat=torch.randn(num_points, 6, dtype=torch.float32, device=device),
    offset=torch.tensor([num_points], dtype=torch.int64, device=device),
)
```
- This inline pseudo-input pattern comes from `scripts/check_data/check_model_forward.py`.
- Instead of mocking internals, scripts instantiate the real model/loss/evaluator objects and feed either random tensors or checked-in sample arrays.

**What to Mock:**
- Current repository practice does not mock anything in smoke checks.
- If you introduce formal unit tests later, mock only external filesystem/environment boundaries such as missing `POINTCEPT_ROOT` or absent sample directories; keep tensor math and registry wiring real to match the current validation style.

**What NOT to Mock:**
- Do not mock the internal loss/evaluator/model interactions when following the current smoke pattern; scripts like `scripts/check_data/check_validation_step.py` and `scripts/train/check_train_step.py` intentionally exercise the real stack.
- Do not bypass Pointcept registry import side effects; scripts explicitly import `project.models`, `project.datasets`, or `project.transforms` so the real registration path is covered.

## Fixtures and Factories

**Test Data:**
```python
sample_dir = repo_root / "samples" / "training" / "020101"
batch = load_real_sample(sample_dir)
```
- Real sample fixtures live under `samples/training/020101` and `samples/validation/020102` and are loaded directly with `numpy.load(...)` in `scripts/check_data/check_validation_step.py` and `scripts/train/check_train_step.py`.
- Synthetic factories are inline functions rather than a shared fixture package; see `make_pseudo_case()` in `scripts/check_data/check_loss_forward.py` and `pseudo_input` in `scripts/check_data/check_model_forward.py`.

**Location:**
- Shared sample assets: `samples/`.
- Inline helper functions: `scripts/check_data/check_loss_forward.py`, `scripts/check_data/check_validation_step.py`, and `scripts/train/check_train_step.py`.
- No dedicated `tests/fixtures/` or `tests/factories/` directory exists.

## Coverage

**Requirements:**
- No coverage target is enforced.
- No coverage collection config or CI gate is present.

**Configuration:**
- Not applicable in the current repository state.
- The closest thing to regression coverage is manual breadth across dataset load, model forward, loss forward, validation-step, train-step, and smoke-train commands in `scripts/check_data/` and `scripts/train/`.

**View Coverage:**
```bash
# Not available: no coverage command or report output is configured.
```

## Test Types

**Unit Tests:**
- Formal unit tests are not present.
- The nearest equivalent is tensor-level script validation in `scripts/check_data/check_loss_forward.py` and `scripts/check_data/check_validation_metrics.py`, where pseudo tensors are passed through the real loss/evaluator implementations.

**Integration Tests:**
- Integration-style checks are the primary validation method.
- `scripts/check_data/check_bf_dataset.py` validates Pointcept dataset registration and dataset output keys.
- `scripts/check_data/check_model_forward.py` validates model construction and forward output structure.
- `scripts/check_data/check_validation_step.py` validates model, loss, and evaluator together on one real sample.
- `scripts/train/check_train_step.py` validates forward, backward, and optimizer step on one real sample.

**E2E Tests:**
- Full browser/UI-style E2E tests are not applicable.
- The closest end-to-end path is smoke training through `scripts/train/train.py` with a `*-smoke.py` config in `configs/semantic_boundary/`.
- README-level commands in `README.md` and `train.md` treat smoke training as the repository’s current top-level verification path.

## Common Patterns

**Async Testing:**
```python
# Not used in this repository; testing scripts are synchronous Python CLIs.
```

**Error Testing:**
```python
pointcept_root_env = os.environ.get("POINTCEPT_ROOT")
if pointcept_root_env is None:
    raise RuntimeError(
        "POINTCEPT_ROOT is required; implicit parent-directory fallback has been removed."
    )
```
- Negative-path validation is usually implemented as fail-fast precondition checks like the above from `scripts/check_data/check_model_forward.py` and `scripts/train/train.py`.

**Manual Pass Criteria:**
- For `scripts/check_data/check_bf_dataset.py`, confirm expected keys exist and `coord`/`edge` lengths align.
- For `scripts/check_data/check_model_forward.py`, confirm `output_keys` include `seg_logits` and `edge_pred` plus the task-specific heads.
- For `scripts/check_data/check_validation_step.py`, confirm printed losses/metrics are finite and both `loss_has_nan` and `metric_has_nan` are `False`.
- For `scripts/train/check_train_step.py`, confirm `backward_ok: True` and `optimizer_step_ok: True`.
- For smoke training via `scripts/train/train.py`, confirm `train.log` plus `model_last.pth` and optionally `model_best.pth` are written under the configured `work_dir`.

**Snapshot Testing:**
- Not used.

---

*Testing analysis: 2026-04-01*
