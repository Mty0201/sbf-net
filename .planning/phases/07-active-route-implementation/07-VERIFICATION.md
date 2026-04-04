---
phase: 07-active-route-implementation
verified: 2026-04-03T12:00:00Z
status: passed
score: 3/3 must-haves verified
must_haves:
  truths:
    - "Direct support/axis-side field supervision is no longer the active route."
    - "The active code/config path implements the new semantic-first boundary-aware supervision signal."
    - "The implementation stays inside repo-local extension boundaries and does not require Pointcept changes."
  artifacts:
    - path: "project/models/semantic_support_model.py"
      provides: "SharedBackboneSemanticSupportModel class"
    - path: "project/models/heads.py"
      provides: "SupportHead class"
    - path: "project/models/__init__.py"
      provides: "Registration of SharedBackboneSemanticSupportModel and SupportHead"
    - path: "configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-model.py"
      provides: "Model-only config for the semantic-plus-support model"
    - path: "project/losses/support_guided_semantic_focus_loss.py"
      provides: "SupportGuidedSemanticFocusLoss class with three-term loss"
    - path: "project/losses/__init__.py"
      provides: "Registration of SupportGuidedSemanticFocusLoss in build_loss"
    - path: "project/evaluator/support_guided_semantic_focus_evaluator.py"
      provides: "SupportGuidedSemanticFocusEvaluator with boundary-region metrics"
    - path: "project/evaluator/__init__.py"
      provides: "Registration of SupportGuidedSemanticFocusEvaluator in build_evaluator"
    - path: "project/trainer/trainer.py"
      provides: "Updated _build_loss_inputs, _build_eval_inputs, validate(), logging for support_pred route"
    - path: "configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train.py"
      provides: "Complete active-route train config"
    - path: "docs/canonical/sbf_facts.md"
      provides: "Updated facts reflecting implemented active route"
    - path: "docs/canonical/sbf_semantic_first_route.md"
      provides: "Active Implementation Route section"
    - path: "docs/canonical/sbf_semantic_first_contract.md"
      provides: "Three-Category Config Distinction section"
    - path: "docs/canonical/sbf_training_guardrails.md"
      provides: "Three-category config roles and active route command pattern"
    - path: "docs/canonical/README.md"
      provides: "Updated index referencing active implementation route"
    - path: "train.md"
      provides: "Maintainer-facing runtime wrapper aligned to implemented route"
  key_links:
    - from: "configs/...train.py"
      to: "project/models/semantic_support_model.py"
      via: "model type registration"
    - from: "configs/...train.py"
      to: "project/losses/support_guided_semantic_focus_loss.py"
      via: "loss type in build_loss"
    - from: "configs/...train.py"
      to: "project/evaluator/support_guided_semantic_focus_evaluator.py"
      via: "evaluator type in build_evaluator"
    - from: "project/trainer/trainer.py"
      to: "project/losses/support_guided_semantic_focus_loss.py"
      via: "_build_loss_inputs forwards support_pred + edge"
    - from: "project/trainer/trainer.py"
      to: "project/evaluator/support_guided_semantic_focus_evaluator.py"
      via: "_build_eval_inputs forwards support_pred + edge"
---

# Phase 7: Active Route Implementation Verification Report

**Phase Goal:** The active training route implements the new semantic-first supervision path and removes direct explicit-field supervision from the mainline by design.
**Verified:** 2026-04-03
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Direct support/axis-side field supervision is no longer the active route | VERIFIED | `SharedBackboneSemanticSupportModel.forward()` returns only `seg_logits` and `support_pred`. No `dir_pred`, `dist_pred`, or `edge_pred` anywhere in model, loss, evaluator, or train config. |
| 2 | The active code/config path implements the new semantic-first boundary-aware supervision signal | VERIFIED | `SupportGuidedSemanticFocusLoss` has three terms (global semantic CE+Lovasz, support BCE, support-guided focus). `SupportGuidedSemanticFocusEvaluator` reports `val_mIoU`, `val_boundary_mIoU`, `support_bce`, `support_cover`. Train config wires all three primitives. Trainer forwards `support_pred`+`edge` to loss/evaluator. |
| 3 | The implementation stays inside repo-local extension boundaries and does not require Pointcept changes | VERIFIED | All new code under `project/` and `configs/`. Pointcept imports are read-only: `MODELS` registry, `build_model`, `Point`, `LovaszLoss`. No files modified outside the `sbf-net` extension. |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `project/models/semantic_support_model.py` | SharedBackboneSemanticSupportModel class | VERIFIED | 95 lines, class with backbone+semantic_head+support_head, forward returns `seg_logits`+`support_pred` only |
| `project/models/heads.py` | SupportHead class | VERIFIED | `SupportHead` at line 44, stem + support_head(1 output), no dir/dist heads |
| `project/models/__init__.py` | Registration of both new types | VERIFIED | Both `SharedBackboneSemanticSupportModel` and `SupportHead` imported and in `__all__` |
| `configs/.../...model.py` | Model-only config | VERIFIED | Sets `type="SharedBackboneSemanticSupportModel"`, no `edge_out_channels`, comment marks it model-only |
| `project/losses/support_guided_semantic_focus_loss.py` | Three-term loss class | VERIFIED | 113 lines, CE+Lovasz, support BCE, focus weighting. Module docstring documents CE overlap rationale and ground-truth guidance. |
| `project/losses/__init__.py` | Registration in build_loss | VERIFIED | Import and `if loss_type == "SupportGuidedSemanticFocusLoss"` branch present |
| `project/evaluator/support_guided_semantic_focus_evaluator.py` | Evaluator with boundary-region metrics | VERIFIED | 181 lines, reports `val_mIoU`, `val_boundary_mIoU`, `val_boundary_mAcc`, `support_bce`, `support_cover`, no `dir_cosine`/`dist_error` |
| `project/evaluator/__init__.py` | Registration in build_evaluator | VERIFIED | Import and `if evaluator_type == "SupportGuidedSemanticFocusEvaluator"` branch present |
| `project/trainer/trainer.py` | Updated plumbing for support_pred route | VERIFIED | `_build_loss_inputs` and `_build_eval_inputs` check `support_pred` first. `_loss_log_keys` has `loss_focus` branch. `validate()` has `SupportGuidedSemanticFocusLoss` branch. Logging for new metrics. Legacy branches preserved. |
| `configs/.../...train.py` | Complete active-route train config | VERIFIED | 85 lines, uses new model/loss/evaluator, `weight=None`, `resume=False`, no legacy field parameters |
| `docs/canonical/sbf_facts.md` | Updated facts | VERIFIED | "still pending later milestone phases" removed. Active route named with all three primitives. |
| `docs/canonical/sbf_semantic_first_route.md` | Active Implementation Route section | VERIFIED | `## Active Implementation Route` section present with model/loss/evaluator/config references |
| `docs/canonical/sbf_semantic_first_contract.md` | Three-Category Config Distinction | VERIFIED | `## Three-Category Config Distinction` section with stable entry, reference baseline, active route |
| `docs/canonical/sbf_training_guardrails.md` | Three-category config roles | VERIFIED | Three categories listed, active route command pattern present, no false validation claims |
| `docs/canonical/README.md` | Updated index | VERIFIED | References "active implementation route" and support-guided semantic focus route |
| `train.md` | Runtime wrapper aligned to implemented route | VERIFIED | "Active implementation route (Phase 7)" with config path, no false validation claims |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| train config | semantic_support_model.py | model type `SharedBackboneSemanticSupportModel` | WIRED | Config imports model dict via `runpy.run_path` from model-only config |
| train config | support_guided_semantic_focus_loss.py | loss type `SupportGuidedSemanticFocusLoss` | WIRED | `loss = dict(type="SupportGuidedSemanticFocusLoss", ...)` in train config, matched by `build_loss` |
| train config | support_guided_semantic_focus_evaluator.py | evaluator type `SupportGuidedSemanticFocusEvaluator` | WIRED | `evaluator = dict(type="SupportGuidedSemanticFocusEvaluator", ...)` in train config, matched by `build_evaluator` |
| trainer.py | loss module | `_build_loss_inputs` forwards `support_pred` + `edge` | WIRED | Line 311: `kwargs["support_pred"] = output["support_pred"]`, line 312: `kwargs["edge"] = batch["edge"]` |
| trainer.py | evaluator module | `_build_eval_inputs` forwards `support_pred` + `edge` | WIRED | Line 330-331: same pattern as loss inputs |
| sbf_training_guardrails.md | train config | active route config reference | WIRED | `semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train.py` appears in guardrails |
| train.md | sbf_semantic_first_contract.md | runtime contract link | WIRED | `sbf_semantic_first_contract.md` referenced in train.md |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 16 artifacts compile | `py_compile` on all files | ALL py_compile PASSED | PASS |
| Model has no dir/dist/edge_pred | Content assertion | No forbidden patterns found | PASS |
| Loss has three terms with correct math | Content assertion | `binary_cross_entropy_with_logits`, `focus_lambda`, `focus_gamma`, docstrings present | PASS |
| Evaluator has boundary metrics, no legacy metrics | Content assertion | `val_boundary_mIoU`, `support_bce`, `support_cover` present; `dir_cosine`, `dist_error` absent | PASS |
| Train config has no legacy dependencies | Content assertion | No `SemanticBoundaryLoss`, `dir_weight`, `dist_weight`; `weight=None`, `resume=False` | PASS |
| sbf_facts.md no longer says route is pending | grep for old phrasing | No matches found | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| AUX-03 | 07-01, 07-02, 07-03, 07-04 | Direct support/axis-side field supervision removed from active mainline by code/config design | SATISFIED | Model emits only `seg_logits`+`support_pred`. Loss/evaluator consume `support_pred`+`edge`, not `edge_pred`. No dir/dist anywhere in active route. Train config uses new primitives exclusively. |
| COMP-03 | 07-01, 07-02, 07-03, 07-04 | New route stays within sbf-net extension boundary, no Pointcept-side changes | SATISFIED | All new code under `project/` and `configs/`. Pointcept imports are read-only (registry, builder, Point, LovaszLoss). Trainer is project-local. No Pointcept files modified. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No TODO, FIXME, placeholder, or stub patterns found in any Phase 7 artifact |

### Human Verification Required

### 1. Synthetic Forward-Pass End-to-End

**Test:** Run the synthetic forward-pass verification from plan 07-03 that builds loss+evaluator from config, feeds dummy tensors, and checks all metric keys.
**Expected:** Loss returns all 8 expected keys (`loss`, `loss_semantic`, `loss_ce`, `loss_lovasz`, `loss_support`, `loss_focus`, `valid_ratio`, `support_positive_ratio`). Evaluator returns all 17 expected keys. No legacy keys leak through. Trainer kwargs contain `support_pred`+`edge` without `edge_pred`.
**Why human:** Requires Pointcept installation to import `LovaszLoss`, `MODELS`, `build_model`, which may not be available in the verification environment.

### 2. Visual Confirmation of Three-Category Doc Distinction

**Test:** Read `docs/canonical/sbf_training_guardrails.md` and `docs/canonical/sbf_semantic_first_contract.md`. Verify the three categories (stable entry, reference baseline, active route) are clearly distinguishable and non-contradictory.
**Expected:** A maintainer can identify all three categories and their config paths without ambiguity.
**Why human:** Readability and clarity are subjective.

### Gaps Summary

No gaps found. All three observable truths verified. All 16 artifacts exist, are substantive (no stubs), and are wired together. All key links verified. Both requirements (AUX-03, COMP-03) satisfied. No anti-patterns detected. Documentation consistently reflects the implemented route without false validation claims.

---

_Verified: 2026-04-03_
_Verifier: Claude (gsd-verifier)_
