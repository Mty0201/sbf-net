---
phase: 11
reviewers: [codex]
reviewed_at: 2026-04-04T14:30:00Z
plans_reviewed: [11-01-PLAN.md, 11-02-PLAN.md]
---

# Cross-AI Plan Review — Phase 11

## Codex Review

### Plan 11-01: Fix trainer boundary metric registration/logging

**Summary**
This is a well-scoped plan that targets the right integration seam: evaluator keys, trainer validation dispatch, validation log formatting, and parser extraction. The main weaknesses are completeness and proof: it does not fully close the `support_positive_ratio` logging requirement, and its verification is almost entirely static for a bug that is explicitly about runtime log emission.

**Strengths**
- The plan attacks the actual broken path end-to-end: evaluator contract -> trainer metric registration -> `Val/Test` log format -> parser extraction.
- Backward compatibility is handled correctly by preserving the old `SupportGuidedSemanticFocusLoss` branch in trainer.py.
- The redesigned branch is placed before the generic boundary branch, which is the right way to avoid `support_bce`/`support_reg_error` ambiguity in trainer.py and parser run-type detection.

**Concerns**
- `HIGH`: The plan registers `support_positive_ratio` in the redesigned metric set but omits it from the proposed redesigned `Val/Test` log format, so `METRIC-01` and arguably `METRIC-02` are not fully satisfied.
- `HIGH`: The context explicitly says the fix should be verified by a short validation pass, but the plan's verification only checks string presence and syntax. That can miss exactly the kind of cross-component runtime failure this phase is about.
- `MEDIUM`: Parser verification is structural, not behavioral. parse_train_log.py now depends on log-shape assumptions, but the plan does not include even a sample-log parse check for redesigned, old focus, and support-only runs.

**Suggestions**
- Add `support_positive_ratio` to the redesigned validation log line, and decide explicitly whether it should also be exported by parse_train_log.py.
- Add one behavioral verification step: run a 1-epoch validation pass or parse a canned redesigned log snippet and assert non-empty `val_boundary_mIoU` and `support_reg_error`.
- Add parser regression checks for all three run types so the redesigned detection does not accidentally break old focus/support-only parsing.

**Risk Assessment**
`MEDIUM` because the change itself is small and localized, but the plan leaves one required metric under-logged and does not include runtime verification for a runtime logging bug.

### Plan 11-02: Create Variant A2 training config

**Summary**
This is a sensible minimal-change config plan. It keeps the experimental surface tight and reasons through the epoch math correctly. The main issues are experiment control rather than code complexity: dependency ordering is looser than the phase context, and the verification is too weak to really prove `METRIC-05`.

**Strengths**
- The plan keeps the delta intentionally narrow: `focus_weight`, epoch schedule, and `work_dir`.
- The trainer math is reasoned through explicitly and lands on the correct `total_epoch=6000, eval_epoch=300` pair.
- It preserves evaluator and support-loss settings from Variant A, which is the right choice for comparability.

**Concerns**
- `MEDIUM`: The plan says `depends_on: []`, but the phase context says Step 3 is conditional on Step 1 completion. Creating the file is independent; treating it as ready to run is not.
- `MEDIUM`: The verification checks only string presence, so it does not actually prove that model/data/optimizer/scheduler/runtime remain identical to Variant A as required by `METRIC-05`.
- `LOW`: The "match baseline duration" rationale is not anchored to the canonical baseline config on disk, which still shows `2000/100`. That makes the experiment rationale vulnerable to future confusion unless it cites the exact historical run/log instead.
- `LOW`: The plan implies a 3x training-time increase but does not mention compute/storage implications for the eventual run.

**Suggestions**
- Add an explicit launch guard: do not run A2 until Plan 11-01 passes runtime validation.
- Replace the current verification with a diff-based check against Variant A that allows only the sanctioned fields to change.
- Cite the exact baseline run or log that motivates "300 eval epochs," rather than implying the current baseline config already encodes that duration.
- Note expected runtime/checkpoint cost for the 300-epoch run.

**Risk Assessment**
`MEDIUM` because the config change is straightforward, but the plan needs tighter dependency and comparability controls to support a defensible experiment.

---

## Consensus Summary

Single reviewer (Codex). Key themes:

### Agreed Strengths
- End-to-end integration coverage: evaluator -> trainer -> log -> parser
- Backward compatibility preserved for old loss types
- Narrow, well-reasoned config delta for Variant A2 with correct epoch math

### Agreed Concerns
- **HIGH**: `support_positive_ratio` registered in metrics but missing from val batch log line
- **HIGH**: No runtime verification for a runtime logging bug — only static string checks
- **MEDIUM**: Verification for METRIC-05 (config identity) is too weak — string presence, not diff-based
- **MEDIUM**: Plan 11-02 dependency ordering is looser than CONTEXT.md specifies

### Divergent Views
N/A (single reviewer)
