---
phase: 06-semantic-first-route-definition
plan: 02
subsystem: docs
tags: [runtime, docs, semantic-first, contract]
requires:
  - phase: 06-semantic-first-route-definition
    provides: support-only-first route-definition doc
provides:
  - Canonical support-centric candidate-route contract
  - Runtime guidance aligned to the support-only-first semantic-first route
affects: [phase-06-semantic-first-route-definition, runtime-guidance, canonical-docs]
key-files:
  created:
    - .planning/phases/06-semantic-first-route-definition/06-02-SUMMARY.md
    - docs/canonical/sbf_semantic_first_contract.md
  modified:
    - docs/canonical/sbf_training_guardrails.md
    - train.md
key-decisions:
  - "Support remains the only explicit boundary prediction target in the Phase 6 candidate route."
  - "The candidate route forbids direction, side, distance, coherence, and ordinal-shape pressure as mainline supervision targets."
requirements-completed: [MAIN-02, AUX-02]
duration: 10m
completed: 2026-04-02
---

# Phase 6 Plan 02: Support-Centric Contract Summary

Defined the exact support-centric contract for the Phase 6 candidate route and aligned runtime-facing guidance to that contract without overstating implementation status.

## Accomplishments

- Created `docs/canonical/sbf_semantic_first_contract.md` as the contract doc for the `support-guided semantic focus route`.
- Updated `docs/canonical/sbf_training_guardrails.md` and `train.md` so they treat support-only as the strongest current reference baseline and support-shape as side evidence only.
- Preserved the stable runtime entry config and fail-fast training rules while linking maintainers to the new semantic-first contract.

## Task Commits

1. **Task 1: Create the canonical contract for the support-guided semantic focus candidate route** - `18276fa` (docs)
2. **Task 2: Align runtime guidance to the support-only-first candidate-route contract** - `18276fa` (docs)

## Self-Check: PASSED

- Verified `docs/canonical/sbf_semantic_first_contract.md` contains the support-only baseline, support-only-explicit-target contract, prohibition list, and architecture-boundary wording required by the plan.
- Verified `train.md` and `docs/canonical/sbf_training_guardrails.md` contain the support-only baseline, support-shape side-evidence, stable runtime entry config, and contract-link wording without reintroducing an active validation center.
