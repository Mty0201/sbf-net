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

- [ ] **VAL-01**: A repo-local smoke or sample validation path exists for the new semantic-first supervision route and confirms the modified path runs successfully.
- [ ] **VAL-02**: The milestone records the expected comparison baseline and the next full-train experiment direction for a separate environment.

### Compatibility And Guardrails

- [x] **COMP-03**: The new route stays within the `semantic-boundary-field` extension boundary and does not require Pointcept-side changes.
- [ ] **COMP-04**: The new route does not introduce fallback behavior, hidden bypasses, or documentation that overstates local smoke/sample checks as full-train validation.

## v2 Requirements

Deferred follow-on work after the semantic-first pivot lands.

### Follow-On Research

- **FOLL-03**: The repo introduces a cleaner route-schema contract so model, loss, evaluator, and trainer semantics cannot drift silently across future supervision routes.
- **FOLL-04**: The repo adds stronger automated regression coverage for route contracts and smoke validation.
- **FOLL-05**: The repo evaluates broader architectural adaptations if the semantic-first supervision signal still fails to protect semantic performance adequately.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Reinstating `support + axis + side` as the active mainline | Conflicts with the user-confirmed experiment conclusion for this milestone |
| Treating explicit local geometric-field prediction as the new main optimization target | Violates the semantic-first objective |
| Pointcept-side trainer, registry, or dataset protocol changes | Crosses the repository boundary |
| Full training proof inside the local repo environment | This milestone stops at local smoke/sample validation plus next-experiment direction |
| Broad preprocessing-pipeline redesign | Too large for the first semantic-first pivot slice |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| MAIN-01 | Phase 5 | Complete |
| MAIN-02 | Phase 6 | Complete |
| AUX-01 | Phase 6 | Complete |
| AUX-02 | Phase 6 | Complete |
| AUX-03 | Phase 7 | Complete |
| VAL-01 | Phase 8 | Pending |
| VAL-02 | Phase 8 | Pending |
| COMP-03 | Phase 7 | Complete |
| COMP-04 | Phase 8 | Pending |

**Coverage:**
- v1 requirements: 9 total
- Mapped to phases: 9
- Unmapped: 0

---
*Requirements defined: 2026-04-02*
*Last updated: 2026-04-02 after Phase 6 completion*
