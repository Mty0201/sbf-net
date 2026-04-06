# Phase 1: NET-01 diagnosis - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 01-net-01-diagnosis
**Areas discussed:** Diagnosis method, Acceptance criteria, Stage interaction

---

## Diagnosis Method

| Option | Description | Selected |
|--------|-------------|----------|
| Stratified statistics | Separately count Stage 2 cluster survival rate and Stage 4 valid yield, bucketed by sparse/dense | |
| Controlled variable experiments | Fix Stage 4, vary Stage 2 eps; then fix Stage 2, vary Stage 4 sigma | |
| Both combined | First stratified statistics to establish hypothesis, then 1-2 controlled variable experiments to confirm | ✓ |

**User's choice:** Both combined
**Notes:** Two-step approach preferred — statistics first for hypothesis, experiments for confirmation

### Follow-up: Density Bucketing

| Option | Description | Selected |
|--------|-------------|----------|
| kNN percentile | Use k=10 mean kNN distance, bucket by percentile (adapts to each scene) | ✓ |
| Absolute threshold | Fixed spacing thresholds (e.g. 0.02m / 0.04m) | |
| Claude decides | Leave to implementation based on data distribution | |

**User's choice:** kNN percentile
**Notes:** Adaptive to different scenes, recommended approach

---

## Acceptance Criteria

| Option | Description | Selected |
|--------|-------------|----------|
| Primary percentage | "Stage X contributes Y% of coverage loss" with quantified proportion | |
| Primary/secondary ranking | "Primary bottleneck is Stage X, Stage Y is secondary" — clear ordering without precise ratio | ✓ |
| Actionable conclusion | Only needs to clearly direct "Phase 2 should fix here" | |

**User's choice:** Primary/secondary ranking
**Notes:** No need for precise percentage attribution — ranking sufficient to guide Phase 2

---

## Stage Interaction

| Option | Description | Selected |
|--------|-------------|----------|
| Need isolation | Separate Stage 2-only and Stage 4-only controlled experiments | |
| Decide based on stats | Do isolation only if stratified statistics are inconclusive | |
| Not needed | Failure modes sufficiently distinct (segment loss vs coverage thinning) — statistics enough | ✓ |

**User's choice:** Not needed
**Notes:** User confirmed after receiving explanation of distinct failure modes. Stage 2 = "entire segment lost" when spacing > eps, Stage 4 = "coverage thinned" when sigma decays too fast. No isolation experiments required.

---

## Claude's Discretion

- Specific percentile thresholds for sparse/dense bucketing
- Which intermediate outputs to inspect
- Visualization or reporting format

## Deferred Ideas

None
