---
status: resolved
phase: 02-behavioral-audit-and-module-restructure
source: [02-VERIFICATION.md]
started: 2026-04-07T00:00:00Z
updated: 2026-04-07T01:45:00Z
---

## Current Test

[all tests complete]

## Tests

### 1. Behavioral equivalence on test scenes
expected: Run fit_local_supports.py and compare supports.npz against pre-refactor baseline. Identical NPZ fields, shapes, dtypes, and values.
result: PASSED — 27/27 NPZ fields bit-identical, XYZ visualization file byte-identical.
method: Checked out pre-refactor code (commit 52402de, monolithic supports_core.py 1318 lines) into a git worktree. Generated Stage 1/2 intermediates from scene 010101. Ran Stage 3 (fit_local_supports.py) with both pre-refactor and refactored code on identical inputs. Compared supports.npz field-by-field via np.array_equal() — all 27 fields exact match (155 supports, 72 line, 83 polyline, 2401 segments, 2412 polyline vertices).
verified_scene: 010101

### 2. Dataset-level in-memory path
expected: Run build_support_dataset_v3.py and verify end-to-end completion. Scene produces correct supports.npz without errors.
result: PASSED — Both pre-refactor and refactored code produce bit-identical supports.npz via in-memory Stages 1→2→3 pipeline. 27/27 fields exact match, XYZ file byte-identical.
method: Created training/010101 dataset structure. Ran build_support_dataset_v3.py with pre-refactor code (worktree) and refactored code (current HEAD) on identical raw scene data (coord.npy, segment.npy, normal.npy). Compared outputs field-by-field — all 27 NPZ fields bit-identical, support_geometry.xyz byte-identical.
verified_scene: 010101

## Summary

total: 2
passed: 2
issues: 0
pending: 0
skipped: 0
blocked: 0

## Caveat

The original UAT text referenced scenes `020101` and `020102`, which do not exist in this repository. The only available test scene is `010101` (under `data_pre/bf_edge_v3/samples/`). Both tests were executed and passed on `010101`. This is "tested scope passed" — not "all originally listed scenes re-tested". The scene IDs in the original VERIFICATION.md appear to have been hallucinated from the plan template.

## Gaps
