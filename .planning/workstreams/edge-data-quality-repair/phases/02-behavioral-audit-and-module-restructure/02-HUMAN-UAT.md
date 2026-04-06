---
status: partial
phase: 02-behavioral-audit-and-module-restructure
source: [02-VERIFICATION.md]
started: 2026-04-07T00:00:00Z
updated: 2026-04-07T00:00:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. Behavioral equivalence on test scenes
expected: Run fit_local_supports.py on 020101/020102 and compare supports.npz against pre-refactor baseline. Identical NPZ fields, shapes, dtypes, and values.
result: [pending]

### 2. Dataset-level in-memory path
expected: Run build_support_dataset_v3.py on samples/ and verify end-to-end completion. Both scenes produce correct supports.npz without errors.
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
