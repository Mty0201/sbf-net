---
phase: 06-semantic-first-route-definition
plan: 01
subsystem: docs
tags: [canonical, docs, semantic-first, support-only]
requires:
  - phase: 05-mainline-analysis-pivot
    provides: semantic-first pivoted docs and runtime wording
provides:
  - Canonical support-only-first route-definition doc
  - Canonical entry surfaces aligned to support-only as the strongest baseline
affects: [phase-06-semantic-first-route-definition, canonical-docs, route-definition]
key-files:
  created:
    - .planning/phases/06-semantic-first-route-definition/06-01-SUMMARY.md
    - docs/canonical/sbf_semantic_first_route.md
  modified:
    - docs/canonical/README.md
    - docs/canonical/sbf_facts.md
key-decisions:
  - "Support-only is the strongest current reference baseline for the semantic-first route."
  - "Support-shape is weaker side evidence only and does not define the new mainline candidate."
requirements-completed: [MAIN-02, AUX-01]
duration: 10m
completed: 2026-04-02
---

# Phase 6 Plan 01: Support-Only-First Route Definition Summary

Defined the Phase 6 route-selection surface around the user-confirmed support-only baseline and removed support-shape from the candidate-mainline role.

## Accomplishments

- Created `docs/canonical/sbf_semantic_first_route.md` as the canonical route-definition doc for the `support-guided semantic focus route`.
- Updated `docs/canonical/README.md` so maintainers are routed to the support-only baseline and the new route-definition doc.
- Reframed `docs/canonical/sbf_facts.md` so support-only is the strongest current reference baseline and support-shape stays side evidence only.

## Task Commits

1. **Task 1: Rewrite the canonical route-definition doc around the support-only-first evidence baseline** - `d3bc432` (docs)
2. **Task 2: Align canonical entry surfaces to the support-only-first interpretation** - `d3bc432` (docs)

## Self-Check: PASSED

- Verified `docs/canonical/sbf_semantic_first_route.md` contains the exact support-only baseline, support-shape side-evidence, and support-guided candidate-route wording required by the plan.
- Verified `docs/canonical/README.md` and `docs/canonical/sbf_facts.md` both point to `docs/canonical/sbf_semantic_first_route.md` and do not promote support-shape as the new semantic-first route.
