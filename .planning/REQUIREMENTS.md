# Requirements: semantic-boundary-field

**Defined:** 2026-04-02
**Core Value:** Semantic segmentation remains the primary objective, and any boundary-aware supervision must improve boundary-region semantic quality without dragging the semantic branch into explicit geometric-field learning.

## v1 Requirements

Requirements for milestone `v1.1 semantic-first boundary supervision pivot`. Each maps to one roadmap phase.

### Mainline Pivot

- [x] **MAIN-01**: The repository-grounded analysis and active-mainline docs reflect that `support + axis + side` is no longer the preferred mainline and that semantic-first boundary supervision is now the active direction.
- [x] **MAIN-02**: The active training path definition no longer treats explicit local geometric-field learning as the supervised target.

### Boundary-Aware Supervision

- [x] **AUX-01**: The repository defines one concrete boundary-aware supervision signal that is intended to improve semantic performance near edges without direct explicit field supervision.
- [x] **AUX-02**: The new supervision route keeps the backbone and main training architecture largely intact unless a supporting architectural change is clearly required by the new signal.
- [x] **AUX-03**: Direct `support` / `axis-side` field supervision is removed from the active mainline by code/config design rather than only neutralized by zero-valued legacy weights.

### Validation

- [x] **VAL-01**: A repo-local smoke or sample validation path exists for the new semantic-first supervision route and confirms the modified path runs successfully.
- [x] **VAL-02**: The milestone records the expected comparison baseline and the next full-train experiment direction for a separate environment.

### Compatibility And Guardrails

- [x] **COMP-03**: The new route stays within the `semantic-boundary-field` extension boundary and does not require Pointcept-side changes.
- [x] **COMP-04**: The new route does not introduce fallback behavior, hidden bypasses, or documentation that overstates local smoke/sample checks as full-train validation.

## v2 Requirements

Deferred follow-on work after the semantic-first pivot lands.

### Follow-On Research

- **FOLL-03**: The repo introduces a cleaner route-schema contract so model, loss, evaluator, and trainer semantics cannot drift silently across future supervision routes.
- **FOLL-04**: The repo adds stronger automated regression coverage for route contracts and smoke validation.
- **FOLL-05**: The repo evaluates broader architectural adaptations if the semantic-first supervision signal still fails to protect semantic performance adequately.

### Full Training Analysis (Phase 9)

- [x] **ANALYSIS-01**: A log-parsing script extracts per-eval-epoch metrics from both the active route and support-only baseline train.log files into structured CSV files.
- [x] **ANALYSIS-02**: The parsing script handles both SupportGuidedSemanticFocusLoss and SemanticBoundaryLoss log formats and does not read entire log files into memory.
- [x] **ANALYSIS-03**: A structured analysis report compares active route results against the support-only baseline (val_mIoU = 74.6) across overall mIoU, per-class breakdown, boundary-region metrics, and training dynamics.
- [x] **ANALYSIS-04**: The analysis report contains concrete tuning config variant proposals with specific parameter values and expected effects, grounded in Phase 8 experiment directions.

### Loss Redesign (Phase 10)

- [ ] **LOSS-01**: Support supervision uses SmoothL1 + Tversky (not BCE) with parameters matching the support-only baseline: support_reg_weight=1.0, support_cover_weight=0.2, support_tversky_alpha=0.3, support_tversky_beta=0.7.
- [ ] **LOSS-02**: The focus term is removed in the ablation variant (Variant C) — total loss = loss_semantic + loss_support only.
- [ ] **LOSS-03**: The loss class applies sigmoid to the raw support logit before SmoothL1 and Tversky, matching the baseline pattern.
- [ ] **LOSS-04**: The evaluator reports SmoothL1 regression error and Tversky coverage (not BCE) for support metrics.
- [ ] **LOSS-05**: A Variant C training config exists with support-only-baseline loss parameters, no focus term, and ~300 eval epochs.
- [ ] **LOSS-06**: A Variant A training config exists with Lovasz-on-boundary focus (boundary_threshold=0.1, focus_weight=0.5) on top of Variant C.
- [ ] **LOSS-07**: Both configs use SharedBackboneSemanticSupportModel (not legacy EdgeHead) and inherit optimizer/scheduler from the active route.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Reinstating `support + axis + side` as the active mainline | Conflicts with the user-confirmed experiment conclusion for this milestone |
| Treating explicit local geometric-field prediction as the new main optimization target | Violates the semantic-first objective |
| Pointcept-side trainer, registry, or dataset protocol changes | Crosses the repository boundary |
| Full training proof inside the local repo environment | This milestone stops at local smoke/sample validation plus next-experiment direction |
| Broad preprocessing-pipeline redesign | Too large for the first semantic-first pivot slice |
| Implementing tuning config changes in Phase 9 | Phase 9 is analysis only; tuning implementation belongs in a follow-on phase (per D-08) |
| Variant B: Support-weighted Lovasz with per-point weighting | Deferred — only pursue if Variant A shows promise |
| Updating docs/canonical/ for Phase 10 | Only after a new config variant is validated through full training |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| MAIN-01 | Phase 5 | Complete |
| MAIN-02 | Phase 6 | Complete |
| AUX-01 | Phase 6 | Complete |
| AUX-02 | Phase 6 | Complete |
| AUX-03 | Phase 7 | Complete |
| VAL-01 | Phase 8 | Complete |
| VAL-02 | Phase 8 | Complete |
| COMP-03 | Phase 7 | Complete |
| COMP-04 | Phase 8 | Complete |
| ANALYSIS-01 | Phase 9 | Complete |
| ANALYSIS-02 | Phase 9 | Complete |
| ANALYSIS-03 | Phase 9 | Complete |
| ANALYSIS-04 | Phase 9 | Complete |
| LOSS-01 | Phase 10 | Planned |
| LOSS-02 | Phase 10 | Planned |
| LOSS-03 | Phase 10 | Planned |
| LOSS-04 | Phase 10 | Planned |
| LOSS-05 | Phase 10 | Planned |
| LOSS-06 | Phase 10 | Planned |
| LOSS-07 | Phase 10 | Planned |

**Coverage:**
- v1 requirements: 9 total (all complete)
- v2/analysis requirements: 4 total (Phase 9, complete)
- v2/loss-redesign requirements: 7 total (Phase 10, planned)
- Unmapped: 0

---
*Requirements defined: 2026-04-02*
*Last updated: 2026-04-03 after Phase 10 planning*
