# Codebase Concerns

**Analysis Date:** 2026-04-01

## Tech Debt

**Overloaded edge route contract:**
- Issue: The same 5-channel `edge_pred` tensor and reused output keys (`dir_pred`, `dist_pred`, `support_pred`) carry different meanings across signed-direction, support-shape, and axis-side routes.
- Files: `project/models/semantic_boundary_model.py`, `project/models/heads.py`, `project/losses/semantic_boundary_loss.py`, `project/losses/support_shape_loss.py`, `project/losses/axis_side_loss.py`, `project/evaluator/semantic_boundary_evaluator.py`, `project/evaluator/axis_side_evaluator.py`, `project/trainer/trainer.py`
- Why: New research routes were added by reusing the original trainer/model plumbing instead of introducing explicit per-route contracts.
- Impact: A config can pair the wrong loss/evaluator/head semantics without an immediate schema error, and logs remain easy to misread because field names stay legacy-shaped.
- Fix approach: Introduce an explicit route schema layer or route-specific output objects, then validate `model -> loss -> evaluator -> trainer` compatibility in one place.

**Active axis-side mainline is still structurally coupled to the old shared-backbone path:**
- Issue: The current `axis + side + support` rollout still uses a shared backbone, a support-conditioned edge head, and one aggregated `loss` backward path; the code does not yet implement the stop-grad/private axis-side isolation described in current task analysis.
- Files: `project/models/heads.py`, `project/models/semantic_boundary_model.py`, `project/trainer/trainer.py`, `project_memory/tasks/TASK-2026-03-31-004.md`, `project_memory/tasks/TASK-2026-03-31-005.md`
- Why: Stage-2 rollout reused the existing stage-2 v2 structure to minimize code churn before verifying the new route.
- Impact: The repo carries code for the active route, but the main structural risk identified by current analysis is still present; semantic regression against the `support-only = 74.6` reference remains plausible until the isolation change lands.
- Fix approach: Implement the selected `support retained + axis/side private stop-grad` route in the smallest possible file boundary, then validate it against the existing smoke path and the `74.6` semantic baseline.

**Config and documentation drift around the active route:**
- Issue: The public default entry and docs still point to the older `direction + distance + support` training config, while `project_memory/current_state.md` says the active mainline is `axis + side + support`.
- Files: `scripts/train/train.py`, `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`, `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py`, `README.md`, `train.md`, `project_memory/current_state.md`
- Why: Historical experiments and the public-facing training docs were kept live while the active research route changed.
- Impact: A user following the default CLI or README can launch the old route by accident, and later analysis has to disentangle "default command" from "current active rollout."
- Fix approach: Add a single canonical active-config pointer, align docs with `current_state`, and keep historical configs clearly labeled as non-default baselines.

**Geometry preprocessing is concentrated in large heuristic-heavy modules:**
- Issue: Core BF-edge preprocessing logic is concentrated in a few large files that mix geometry heuristics, clustering, support fitting, export, and compatibility behavior.
- Files: `data_pre/bf_edge_v3/core/supports_core.py`, `data_pre/bf_edge_v3/core/local_clusters_core.py`, `data_pre/bf_edge_v3/core/boundary_centers_core.py`, `data_pre/bf_edge_v3/core/pointwise_core.py`
- Why: The pipeline evolved as research code and kept new heuristics in-place instead of splitting them into smaller verified units.
- Impact: Small changes are hard to localize, regression review is expensive, and it is difficult to tell whether a failure comes from clustering, support fitting, or pointwise supervision generation.
- Fix approach: Split the pipeline into smaller modules by responsibility, keep stable serialized fixtures for each stage, and test stage boundaries independently.

## Known Bugs

**Axis-side side-loss mask is computed but not applied:**
- Symptoms: `loss_side` backpropagates on all support-weighted valid points, even though the code comment says side supervision should only apply where axis supervision is valid and optionally above `side_support_threshold`.
- Files: `project/losses/axis_side_loss.py`
- Trigger: Any `AxisSideSemanticBoundaryLoss` run where points are valid but fail the intended side-valid region, especially near zero-distance ambiguity.
- Workaround: None in the repo; current analysis must treat side-loss behavior as broader than the documented semantics.
- Root cause: `side_valid_mask` is calculated, but `loss_side` uses `side_weight_map = support_gt * valid_gt` instead of masking by `side_valid_mask`.

**Model default output width is inconsistent with every edge head:**
- Symptoms: Model construction fails if a config relies on the class default `edge_out_channels`, because both `EdgeHead` and `SupportConditionedEdgeHead` require 5 channels.
- Files: `project/models/semantic_boundary_model.py`, `project/models/heads.py`
- Trigger: Building `SharedBackboneSemanticBoundaryModel` without explicitly setting `edge_out_channels=5`.
- Workaround: All current configs set `edge_out_channels=5` explicitly.
- Root cause: `SharedBackboneSemanticBoundaryModel.__init__` still defaults `edge_out_channels=4`.

**`check_loss_forward.py` uses a non-existent validation sample path:**
- Symptoms: The smoke script points to `samples/validation/010101`, but the repository only ships `samples/validation/020102`.
- Files: `scripts/check_data/check_loss_forward.py`, `samples/validation/020102`
- Trigger: Running `scripts/check_data/check_loss_forward.py` as-is.
- Workaround: Change the validation sample path manually before running the script.
- Root cause: The hard-coded validation sample ID drifted from the checked-in sample fixture.

**`check_validation_metrics.py` cannot print evaluator outputs safely:**
- Symptoms: The script attempts `float()` on non-scalar tensors such as `semantic_intersection`, `semantic_union`, `semantic_target`, and per-class arrays returned by the evaluator.
- Files: `scripts/check_data/check_validation_metrics.py`, `project/evaluator/semantic_boundary_evaluator.py`
- Trigger: Running `scripts/check_data/check_validation_metrics.py` as-is.
- Workaround: Restrict the script to scalar keys or add tensor-shape-aware formatting.
- Root cause: The script assumes every evaluator output is scalar.

**Standalone semantic evaluator defaults do not match standalone semantic loss defaults:**
- Symptoms: Ad hoc validation checks can report support-loss values using a different support-cover/support-reg balance than the default training loss.
- Files: `project/evaluator/semantic_boundary_evaluator.py`, `project/losses/semantic_boundary_loss.py`, `scripts/check_data/check_validation_metrics.py`, `scripts/check_data/check_validation_step.py`
- Trigger: Constructing `SemanticBoundaryEvaluator()` and `SemanticBoundaryLoss()` with defaults instead of passing explicit config values.
- Workaround: Always build the evaluator from the same config values as the loss.
- Root cause: `SemanticBoundaryEvaluator` defaults `support_cover_weight=1.0` and `support_reg_weight=0.25`, while `SemanticBoundaryLoss` defaults those weights the other way around.

**Repeated runs append into the same `train.log`:**
- Symptoms: One log file can contain multiple unrelated startup attempts; the axis-side smoke summary already reports 4 starts in one file before choosing the last valid session as authoritative.
- Files: `project/utils/logger.py`, `reports/log_summaries/semantic_boundary_axis_side_train_smoke_train.summary.md`
- Trigger: Re-running training in the same `work_dir`.
- Workaround: Use a fresh `work_dir` or delete/rotate the existing log before a new run.
- Root cause: The project logger always opens `train.log` in append mode.

## Security Considerations

**Training trusts config files and checkpoints as executable artifacts:**
- Risk: `runpy.run_path(args.config)` executes arbitrary Python from the chosen config path, and `torch.load(..., weights_only=False)` allows arbitrary pickle payloads in checkpoints.
- Files: `scripts/train/train.py`, `project/trainer/trainer.py`
- Current mitigation: The repo assumes a trusted local research environment and does not try to sandbox configs or checkpoint files.
- Recommendations: Treat configs and checkpoints as trusted-only inputs, prefer `weights_only=True` when the checkpoint format permits it, and document this trust boundary in user-facing training docs.

**`POINTCEPT_ROOT` is an unchecked import-precedence boundary:**
- Risk: The repo prepends the user-supplied Pointcept path to `sys.path`, so a stale or malicious checkout can shadow expected imports.
- Files: `scripts/train/train.py`, `scripts/check_data/check_model_forward.py`, `scripts/check_data/check_validation_step.py`, `scripts/train/check_train_step.py`, `scripts/check_data/check_bf_dataset.py`, `data_pre/bf_edge_v3/scripts/_bootstrap.py`
- Current mitigation: The path is required to exist, but there is no version or content verification.
- Recommendations: Pin a known Pointcept commit in environment docs, verify the checkout before runs, and fail fast if the expected Pointcept package layout is not present.

**Dataset rebuild deletes artifacts in place without transactional output swap:**
- Risk: `cleanup_scene_dir()` unlinks prior artifacts before new outputs are written, so an interrupted export can leave a scene partially cleaned or partially rebuilt.
- Files: `data_pre/bf_edge_v3/scripts/build_support_dataset_v3.py`
- Current mitigation: Cleanup happens after in-memory stage computation, which reduces but does not eliminate failure windows.
- Recommendations: Write outputs to a temporary directory first, validate them, and atomically replace the scene outputs only after a successful export.

## Performance Bottlenecks

**Ordinal support-shape loss uses per-scene pairwise distance matrices in the hot path:**
- Problem: The ordinal term builds a full `torch.cdist` matrix on sampled narrowband points for every scene, every train step.
- Files: `project/losses/support_shape_loss.py`, `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-support-shape-train.py`
- Measurement: No profiler is checked in, but the configured bound is still `ordinal_n_samples=512` with `ordinal_max_pairs=2048` per scene.
- Cause: Neighborhood selection is recomputed from scratch inside the loss instead of reusing a prebuilt local graph.
- Improvement path: Precompute scene-local neighborhoods or support-aware pairs and replace the full `cdist` with radius or kNN lookups.

**Support fitting and pointwise supervision contain dense pairwise and nested support loops:**
- Problem: `estimate_local_spacing()` materializes a full `N x N` distance matrix, and pointwise support assignment iterates over candidate supports and per-support segments.
- Files: `data_pre/bf_edge_v3/core/supports_core.py`, `data_pre/bf_edge_v3/core/pointwise_core.py`, `data_pre/bf_edge_v3/core/local_clusters_core.py`
- Measurement: Not instrumented in the repo; the bottleneck is visible directly in `coords[:, None, :] - coords[None, :, :]` and repeated per-support/per-segment loops.
- Cause: The preprocessing path prioritizes direct geometry code over indexed spatial queries and batched assignment.
- Improvement path: Replace dense local spacing with `cKDTree`-based kNN, pre-index support segments, and parallelize per-scene work.

**Training produces very large logs and summary inputs by default:**
- Problem: Iteration-level training and validation logging create large log files that must be reparsed by summary scripts.
- Files: `project/trainer/trainer.py`, `reports/log_summaries/semseg-pt-v3m1-0-base-bf-edge-support-shape-train_train.summary.md`
- Measurement: The checked-in support-shape full-train summary references a `42.92 MB` log with `120957` lines for the first 100 displayed epochs.
- Cause: Multiple configs use `log_freq=1` and `val_log_freq=1`, and the logger writes synchronously to both stream and file.
- Improvement path: Lower the default logging frequency, emit structured scalar snapshots per epoch, or rotate logs per run.

## Fragile Areas

**Trainer route selection depends on key presence instead of an explicit route type:**
- Files: `project/trainer/trainer.py`, `project/losses/__init__.py`, `project/evaluator/__init__.py`
- Why fragile: Logging and validation behavior switch on keys like `loss_axis`, `loss_ordinal`, and `loss_edge` rather than a stable route identifier.
- Common failures: A new loss variant that omits one expected key can silently drop metrics or hit the wrong logging branch.
- Safe modification: Add an explicit route identifier or strategy object and keep the trainer ignorant of per-loss dictionary shapes.
- Test coverage: No automated trainer-route contract tests exist.

**Axis-side semantics are layered on top of legacy output names:**
- Files: `project/models/semantic_boundary_model.py`, `project/models/heads.py`, `project/losses/axis_side_loss.py`, `project/evaluator/axis_side_evaluator.py`
- Why fragile: The model still returns `dir_pred` and `dist_pred`, while the active route interprets those channels as `axis` and `side`.
- Common failures: Wrong config pairing, misleading debug prints, and channel-order mistakes during future refactors.
- Safe modification: Change model output naming, loss parsing, evaluator parsing, and config docs together rather than one file at a time.
- Test coverage: Only smoke runs and log summaries cover this path today.

**Scene-level BF-edge rebuilds mutate outputs in place:**
- Files: `data_pre/bf_edge_v3/scripts/build_support_dataset_v3.py`, `data_pre/bf_edge_v3/utils/stage_io.py`
- Why fragile: The pipeline loads scene files, computes supports in memory, deletes old artifacts, and writes new ones back into the same scene directory.
- Common failures: Partial reruns leave inconsistent scene contents; shape mismatches only surface at runtime when later stages load the rebuilt scene.
- Safe modification: Keep old outputs until the new support bundle and exports validate successfully, then swap them in one step.
- Test coverage: No fixture-based end-to-end dataset rebuild test exists.

**Evidence generation depends on logs that can mix sessions:**
- Files: `project/utils/logger.py`, `reports/log_summaries/semantic_boundary_axis_side_train_smoke_train.summary.md`
- Why fragile: One `train.log` can contain multiple sessions, so the summary pipeline has to infer which run is authoritative.
- Common failures: Startup-only attempts pollute the same file and can confuse manual readers or downstream evidence parsing.
- Safe modification: Create one log per run or mark session boundaries explicitly in the logger.
- Test coverage: Workflow smokes check document linkage, not log-session integrity.

## Scaling Limits

**Training runtime is single-process and single-device only:**
- Current capacity: `SemanticBoundaryTrainer` chooses one device from `torch.cuda.is_available()` and runs one local process.
- Files: `project/trainer/trainer.py`, `scripts/train/train.py`, `README.md`
- Limit: There is no DDP, no multi-GPU path, and no fallback to Pointcept's distributed runtime from the project-local trainer.
- Symptoms at limit: Large experiments remain bound by one GPU's memory and throughput, and long full-train runs stay operationally expensive.
- Scaling path: Add a distributed wrapper or hand the runtime back to a stable upstream trainer while keeping project-local datasets, losses, and evaluators.

**Support-generation throughput scales poorly with scene count and scene density:**
- Current capacity: `data_pre/bf_edge_v3/scripts/build_support_dataset_v3.py` iterates scenes serially and each scene can hit dense geometry operations.
- Files: `data_pre/bf_edge_v3/scripts/build_support_dataset_v3.py`, `data_pre/bf_edge_v3/core/supports_core.py`, `data_pre/bf_edge_v3/core/pointwise_core.py`
- Limit: Serial scene processing plus per-scene dense distance work becomes slow on larger datasets or denser support sets.
- Symptoms at limit: Long preprocessing windows, avoidable memory pressure, and rebuild times that discourage verification reruns.
- Scaling path: Parallelize scene execution, cache neighbor structures, and reduce dense pairwise computations.

## Dependencies at Risk

**Pointcept is an unpinned external runtime dependency:**
- Risk: The repo imports Pointcept registries, models, transforms, and dataset utilities directly from a user-supplied checkout.
- Files: `scripts/train/train.py`, `project/models/semantic_boundary_model.py`, `project/datasets/bf.py`, `project/transforms/index_keys.py`, `README.md`, `install.md`
- Impact: Upstream API drift can break model builds, transform registration, or dataset loading without any changes inside this repo.
- Migration plan: Pin a known Pointcept commit or release, then add a compatibility smoke that validates the expected registry and backbone interfaces.

**`requirements.txt` does not cover preprocessing dependencies:**
- Risk: The declared repo requirements only list `torch` and `numpy`, while preprocessing imports `scipy` and `scikit-learn`.
- Files: `requirements.txt`, `data_pre/bf_edge_v3/core/boundary_centers_core.py`, `data_pre/bf_edge_v3/core/local_clusters_core.py`
- Impact: A fresh environment can appear "installed" for training but still fail immediately when running dataset-build scripts.
- Migration plan: Add a preprocessing extra or a separate requirements file that pins `scipy` and `scikit-learn`.

## Missing Critical Features

**No automated regression gate for the active axis-side route:**
- Problem: The active `axis + side + support` path is judged through smoke runs, task notes, and manual log summaries instead of a repeatable regression suite.
- Files: `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py`, `reports/log_summaries/semantic_boundary_axis_side_train_smoke_train.summary.md`, `project_memory/tasks/TASK-2026-03-31-005.md`
- Current workaround: Manual smoke execution plus evidence review in `reports/` and `project_memory/tasks/`.
- Blocks: Safe iteration on the active route and fast verification that a change did not slip back toward the failed signed-direction behavior.
- Implementation complexity: Medium

**No single canonical schema validator for edge-channel meaning:**
- Problem: Channel semantics live in code comments, config choices, and current task documents rather than one machine-checked schema.
- Files: `project/models/semantic_boundary_model.py`, `project/models/heads.py`, `project/losses/semantic_boundary_loss.py`, `project/losses/axis_side_loss.py`, `project_memory/current_state.md`
- Current workaround: Human discipline and task-level reminders keep the route semantics aligned.
- Blocks: Safe introduction of new edge routes and safe refactors of the model/loss/evaluator interface.
- Implementation complexity: Medium

**No reproducible environment lock for the full repo surface:**
- Problem: Training, preprocessing, and workflow tooling rely on manual environment setup rather than a pinned environment or lockfile.
- Files: `requirements.txt`, `README.md`, `install.md`
- Current workaround: Follow the docs manually and reuse an already-working local Pointcept environment.
- Blocks: Repeatable onboarding, CI, and reproducible preprocessing.
- Implementation complexity: Low to Medium

## Test Coverage Gaps

**Model/loss/evaluator route contracts are untested:**
- What's not tested: Compatibility between output tensors, route-specific loss parsing, evaluator metrics, and trainer logging for `SemanticBoundaryLoss`, `SupportShapeLoss`, and `AxisSideSemanticBoundaryLoss`.
- Files: `project/models/semantic_boundary_model.py`, `project/losses/semantic_boundary_loss.py`, `project/losses/support_shape_loss.py`, `project/losses/axis_side_loss.py`, `project/evaluator/semantic_boundary_evaluator.py`, `project/evaluator/axis_side_evaluator.py`, `project/trainer/trainer.py`
- Risk: Silent route mismatches or misleading logs can survive until a long run or manual evidence review.
- Priority: High
- Difficulty to test: Moderate; small synthetic tensor fixtures are enough for most contract checks.

**BF-edge preprocessing stages have no fixture-backed regression tests:**
- What's not tested: Boundary center extraction, clustering, support fitting, and pointwise supervision generation on degenerate or adversarial scene geometry.
- Files: `data_pre/bf_edge_v3/core/boundary_centers_core.py`, `data_pre/bf_edge_v3/core/local_clusters_core.py`, `data_pre/bf_edge_v3/core/supports_core.py`, `data_pre/bf_edge_v3/core/pointwise_core.py`
- Risk: Label-generation bugs can silently poison training data and then be misdiagnosed as model or loss failures.
- Priority: High
- Difficulty to test: High; the repo needs a few minimal deterministic fixture scenes first.

**Checkpoint/resume behavior across route changes is not covered:**
- What's not tested: Loading old checkpoints into new route semantics, resuming from reused `work_dir` directories, and verifying that logged best metrics still match the intended run.
- Files: `project/trainer/trainer.py`, `project/utils/logger.py`, `configs/semantic_boundary/`
- Risk: Stale checkpoints or appended logs can make a resumed run look valid while mixing incompatible route assumptions.
- Priority: Medium
- Difficulty to test: Moderate; it needs a small matrix of synthetic checkpoints and work directories.

**The smoke scripts themselves are not CI-verified:**
- What's not tested: End-to-end execution of `scripts/check_data/` and `scripts/train/check_train_step.py` against the checked-in sample fixtures.
- Files: `scripts/check_data/check_model_forward.py`, `scripts/check_data/check_loss_forward.py`, `scripts/check_data/check_validation_metrics.py`, `scripts/check_data/check_validation_step.py`, `scripts/train/check_train_step.py`
- Risk: Broken verification utilities create false confidence and slow down debugging when they are needed most.
- Priority: High
- Difficulty to test: Low; the repo already contains the sample fixtures needed to catch the current path and scalar-print bugs.

---

*Concerns audit: 2026-04-01*
*Update as issues are fixed or new ones discovered*
