---
phase: 05-mainline-analysis-pivot
plan: 01
subsystem: docs
tags: [canonical, docs, semantic-first, guidance]
requires:
  - phase: 04-workflow-control-cutover
    provides: active GSD and `.planning` control path
provides:
  - Current-maintainer docs aligned to the semantic-first pivot
  - Canonical facts split between active direction and historical evidence
affects: [phase-05-mainline-analysis-pivot, canonical-docs, maintainer-entry]
key-files:
  created:
    - .planning/phases/05-mainline-analysis-pivot/05-01-SUMMARY.md
  modified:
    - AGENTS.md
    - README.md
    - docs/canonical/README.md
    - docs/canonical/sbf_facts.md
key-decisions:
  - "Semantic-first boundary supervision is stated as the active repository direction without claiming implementation or validation beyond current evidence."
  - "The prior `support`, `axis-side`, and `axis + side + support` route remains visible as historical/reference evidence instead of the preferred mainline."
requirements-completed: [MAIN-01]
duration: 10m
completed: 2026-04-02
---

# Phase 5 Plan 01: Semantic-First Mainline Docs Summary

Aligned the active maintainer-facing docs and canonical facts with the semantic-first pivot while preserving the older geometric-field route as historical/reference evidence.

## Accomplishments

- Rewrote `AGENTS.md` and `README.md` so they describe semantic-first boundary supervision as the active direction and stop presenting `axis + side + support` as the preferred mainline.
- Updated `docs/canonical/README.md` to route maintainers toward active-direction answers and historical-reference semantics explicitly.
- Reframed `docs/canonical/sbf_facts.md` so active status, historical semantics, and evidence interpretation are separated cleanly.

## Task Commits

1. **Task 1: Pivot the root and guardrail entry surfaces to semantic-first status** - `9ae833a` (docs)
2. **Task 2: Reframe the canonical facts and index around active direction versus historical evidence** - `9ae833a` (docs)

## Self-Check: PASSED

- Verified the edited docs now contain `semantic-first`, `explicit geometric-field`, and `historical` or `reference` wording in the active surfaces.
- Verified the active surfaces no longer describe `axis + side + support` as the active mainline or name a current validation center for the axis-side route.
