---
phase: 01-zaha-offline-preprocessing-pipeline
plan: 03
subsystem: infra
tags: [zaha, denoise, chunking, normals, open3d, pca, research-gate, tdd-green]

requires:
  - phase: 01-zaha-offline-preprocessing-pipeline
    provides: "01-01 RED stubs in test_denoise.py + test_chunking.py + test_normals.py (8 tests); 01-02 stream_voxel_aggregate + VoxelAggregateResult; 01-CONTEXT.md D-06..D-08/D-10..D-13/D-17..D-19; 01-RESEARCH.md §D (4 denoising candidates, drop-rate table, visual protocol), §E (adaptive-radius PCA normals, §E.5 unoriented choice), §F.4 (tile-size estimator), §I.5 (open3d import-order trap)"
provides:
  - "data_pre/zaha/utils/denoise.py — DenoiseConfig/DenoiseResult/denoise_cloud + SOR/radius/MLS/ransac_plane/none dispatch, D-12 hard cap enforcement"
  - "data_pre/zaha/utils/chunking.py — ChunkingConfig/ChunkSpec/compute_chunks/chunk_name/iter_chunk_points, row-major x-outer y-inner, 3-digit zero-padded names"
  - "data_pre/zaha/utils/normals.py — NormalConfig/estimate_normals, open3d KDTreeSearchParamKNN(knn=30) adaptive-radius PCA, unit-length + no-NaN enforcement"
  - "data_pre/zaha/scripts/tools/measure_density.py — one-shot operational tool that runs parse + stream_voxel_aggregate across all 26 ZAHA samples and prints per-sample density"
  - "8 of 26 RED test stubs turned GREEN (2 denoise + 3 chunking + 3 normals) — total GREEN count in data_pre/zaha/tests/ now 17 / 26"
  - "denoising_notes.md research log (status=approved) — 4 candidates × 3 sample chunks, SOR winner (single-method, NOT sor+radius), Final Decision YAML block for Plan 04 manifest"
  - "normals_notes.md visual sanity log (status=approved) — 3 chunks, 401,878 pts, all unit-length float32, zero NaN"
affects:
  - "01-04 (orchestrator + NPY layout + manifest): consumes denoise_cloud, compute_chunks, estimate_normals, and copies denoising_notes.md Final Decision block verbatim into manifest.denoising"

tech-stack:
  added: []
  patterns:
    - "Dispatch-style denoising wrapper: SUPPORTED_METHODS frozenset + _DISPATCH table + single denoise_cloud(xyz, segment, cfg) entry point enforcing D-12 max_drop_frac cap"
    - "open3d-first import ordering in denoise.py and normals.py: `import open3d as o3d` at module top BEFORE numpy/pandas/scipy to dodge the ptv3 env `libstdc++.so.6 GLIBCXX_3.4.29 not found` trap (RESEARCH §I.5)"
    - "Adaptive-radius PCA normals via open3d.geometry.KDTreeSearchParamKNN(knn=30) — mathematically equivalent to 'adaptive radius holding exactly k=30 neighbours' per D-17"
    - "Belt-and-suspenders unit-length check: even though open3d returns unit vectors by construction, estimate_normals renormalises + raises on any row with ||n|| < unit_length_threshold (default 0.99) — catches non-finite + collinear/isolated-point degeneracies"
    - "Deterministic box-grid chunker: pure-numpy (no open3d, no pandas), grid origin = bbox_min (D-08), row-major x-outer y-inner (D-10), 3-digit zero-padded names (D-10 / §F.6)"
    - "Research-gate pattern: ship all N candidate wrappers (4 denoising methods), run visual sweep on 3 sample chunks, record Final Decision YAML block in a workstream notes file, block on human approval, then have the downstream plan consume the block verbatim"
    - "Last-tile edge-clip pattern in chunking._tile_edges: stride by (tile - overlap), clip the final tile's high edge to axis_max, pull the final tile's low edge back so width stays = tile where possible"

key-files:
  created:
    - "data_pre/zaha/utils/denoise.py"
    - "data_pre/zaha/utils/chunking.py"
    - "data_pre/zaha/utils/normals.py"
    - "data_pre/zaha/scripts/tools/__init__.py"
    - "data_pre/zaha/scripts/tools/measure_density.py"
    - ".planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/denoising_notes.md"
    - ".planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/normals_notes.md"
  modified:
    - "data_pre/zaha/tests/test_denoise.py"
    - "data_pre/zaha/tests/test_chunking.py"
    - "data_pre/zaha/tests/test_normals.py"

key-decisions:
  - "Denoising winner = single-method SOR (nb_neighbors=30, std_ratio=2.0, max_drop_frac=0.10), NOT the sor+radius sequential pipeline the plan template assumed. Radius rejected at the visual gate because it drops 28.06% on the roof-edge chunk (D-12 hard violation) — density-dependence makes it unsafe on ZAHA's sparse regions. Plan 04's manifest.denoising must record `method: sor`, not `method: [sor, radius]`."
  - "Ran 4 denoising candidates (sor, radius, mls, ransac_plane), not the plan's minimum 3. Strict superset of D-11's minimum set — the wider reject-reasoning table is more informative and D-11 is still covered."
  - "Chunk C (roof edge) bbox enlarged from the plan template's 4×4×top-4m to 8×8×top-6m because the template bbox gave a pathologically sparse 6,951-point sample unfit for a realistic visual gate. 8×8×top-6m yields 38,736 pts — actual roof-edge density on post-voxel-agg (grid=0.02) clouds."
  - "Radius filter params tuned from plan default (nb_points=8, radius=0.05) to (nb_points=4, radius=0.08). The plan defaults drop 70–98% on post-voxel-agg density — unusable. Even the tuned params fail D-12 on chunk C at 28.06%, which is the evidence used to reject the filter outright. Both the failed default and the tuned rejection are documented in denoising_notes.md."
  - "Pytest invocation must use `conda run -n ptv3 python -m pytest ...` (module form) not `conda run -n ptv3 pytest ...` (bare). Same workaround Plan 01-02 documented. The plan's literal grep acceptance line is stricter than the reliable invocation form on the ptv3 env."
  - "normals.py imports `open3d as o3d` at line 39 BEFORE `import numpy` at line 43, per RESEARCH §I.5. Verified by grep; any reorder trips the GLIBCXX_3.4.29 loader error on the ptv3 conda env."
  - "Normals sign orientation explicitly deferred (D-18 aspirational). The visual mottling on flat walls in the chunk A/C RGB views is unoriented-PCA behaviour, not a bug — downstream BFDataset treats normals as an unsigned rotation-invariant feature vector. `orient=True` would cost ~30 extra minutes across the full dataset for zero training benefit."
  - "chunking.py is pure-numpy: it does NOT import open3d, pandas, or scipy. Can be imported at any point in the pipeline without triggering the ptv3 GLIBCXX trap. D-08 origin = bbox_min makes the tile enumeration deterministic from the post-denoise cloud's bbox alone."
  - "measure_density.py is an operational one-shot tool, not part of the pipeline runtime. It runs parse + stream_voxel_aggregate across all 26 ZAHA samples and prints per-sample density — evidence for Plan 04's tile-size choice, not a dependency of the orchestrator."

requirements-completed:
  - DS-ZAHA-P1-03
  - DS-ZAHA-P1-04
  - DS-ZAHA-P1-05

duration: ~60 min
completed: 2026-04-11
---

# Phase 01 Plan 03: Denoise + chunking + normals (Wave 2) Summary

**Lands the three post-voxel transforms (SOR denoising at nb=30/std=2.0, deterministic row-major box-grid chunker at tile=4/overlap=2/full-Z, adaptive-radius PCA normals via open3d KNN k=30) plus a density measurement tool, turns 8 RED stubs GREEN, and locks the denoising winner via a 4-candidate × 3-chunk visual research gate (approved) that rejects radius on density-dependence and ransac_plane on corner failure — final decision is single-method SOR, NOT the sor+radius pipeline the plan template assumed.**

## Performance

- **Duration:** ~60 min wall clock (Tasks 1/2/3 impl + 2 human-approval research gates)
- **Started:** 2026-04-11
- **Completed:** 2026-04-11
- **Tasks:** 3 / 3
- **Files created:** 7 (3 utils + 1 scripts subpackage + 1 tool + 2 workstream notes)
- **Files modified:** 3 (3 test files turned GREEN)
- **Commits:** 6 atomic (a3dc08c, fce8006, a7a08ca, 87cb1ed, 686253d, e5340b7) + 1 pending metadata (this SUMMARY.md, orchestrator-owned)

## Accomplishments

### Task 1: denoise.py + denoising research gate

- **`data_pre/zaha/utils/denoise.py`** (250 lines). `SUPPORTED_METHODS = frozenset({'sor', 'radius', 'mls', 'ransac_plane', 'none'})`. `DenoiseConfig(method, params, max_drop_frac=0.10)` frozen dataclass with `__post_init__` validating the method and the cap. `DenoiseResult(xyz float32, segment int32, n_in, n_out, method, params)` with a `drop_frac` property. Private dispatch table `_DISPATCH` maps `sor → _sor`, `radius → _radius`, `mls → _mls`, `ransac_plane → _ransac_plane`. Public `denoise_cloud(xyz, segment, cfg)` validates shape, handles `method='none'` and empty-input passthroughs, dispatches, then hard-enforces `drop_frac > cfg.max_drop_frac → ValueError` (D-12). `import open3d as o3d` is the first import after `from __future__ import annotations` — scipy.spatial.cKDTree is imported lazily inside `_mls` AFTER open3d is loaded. `ransac_plane` pins `np.random.seed(42)` before calling `segment_plane` for bitwise reproducibility.
- **`data_pre/zaha/tests/test_denoise.py`** (2 GREEN): `test_drop_cap` builds a 1000-pt 10×10×10 cube at 0.05 m spacing + 100 uniform noise outliers, applies SOR (nb=20, std=2.0), asserts drop_frac in (0.0, 0.10]; `test_determinism` runs SOR twice on the same synthetic input and asserts bitwise-equal xyz + segment.
- **`denoising_notes.md` research log (approved):** 4 candidates (sor, radius, mls, ransac_plane) × 3 sample chunks (wall A / corner B / roof-edge C) on `DEBY_LOD2_4907179.pcd`, 39 PNGs generated under `/tmp/zaha_denoise/views/`, Final Decision block with the winner YAML.

### Task 2: chunking.py + measure_density tool

- **`data_pre/zaha/utils/chunking.py`** (345 lines, pure numpy, no open3d/pandas/scipy). `ChunkingConfig(tile_xy=4.0, overlap_xy=2.0, z_mode='full', budget_per_chunk=600_000)` frozen dataclass with validation: `tile_xy > 0`, `0 ≤ overlap_xy < tile_xy`, `z_mode in {'full', 'band:{depth}'}`, `budget_per_chunk > 0`. Derived `stride_xy` property. `ChunkSpec(chunk_idx, x_tile, y_tile, bbox_min, bbox_max)` frozen dataclass. `chunk_name(basename, idx) → f"{basename}__c{idx:03d}"` with `MAX_CHUNK_INDEX=999` (3-digit zero-pad per D-10 / §F.6). `_tile_edges(axis_min, axis_max, tile, overlap)` enumerates one-axis edges with last-tile clip + start-pullback. `compute_chunks(bbox_min, bbox_max, cfg)` enumerates row-major x-outer y-inner per D-10, supports `z_mode='full'` (single Z tile) and `z_mode='band:{depth}'` (no-overlap Z slicing). `iter_chunk_points(xyz, segment, chunk)` returns the closed-interval subset falling in the chunk bbox (so overlap regions are not deduped, per D-09).
- **`data_pre/zaha/tests/test_chunking.py`** (3 GREEN): row-major chunk_idx sequence + adjacency/overlap invariant; `chunk_name` zero-pad + out-of-range raise; `iter_chunk_points` closed-interval mask on a synthetic grid.
- **`data_pre/zaha/scripts/tools/measure_density.py`** (212 lines). One-shot operational tool: loops over all 26 `.pcd` files under `/home/mty0201/data/ZAHA_pcd/`, runs the Plan 01-02 streaming parse + voxel aggregation, prints per-sample density (points / XY footprint area) and total points. Output is informational only — Plan 04 uses this to pick the final `tile_xy` on evidence, not on the RESEARCH §F.4 ASSUMED worked example.

### Task 3: normals.py + normals visual sanity

- **`data_pre/zaha/utils/normals.py`** (179 lines). `import open3d as o3d` at line 39 BEFORE `import numpy` at line 43 (§I.5 verified by grep). `NormalConfig(knn=30, orient=False, fast=False, unit_length_threshold=0.99)` frozen dataclass with `__post_init__` validation. `estimate_normals(xyz, cfg=None)` casts input to float64 for open3d's PointCloud API, calls `pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30), fast_normal_computation=False)`, returns `np.asarray(pcd.normals, dtype=np.float32)`. D-18 hard bar enforced via (a) non-finite check (raises `ValueError` on any NaN/Inf), (b) `||n|| < unit_length_threshold` belt-and-suspenders check with a renormalise attempt + raise if still degenerate. Raises on `N < knn` — caller must widen knn or split the chunk. Returns `(0, 3) float32` on empty input.
- **`data_pre/zaha/tests/test_normals.py`** (3 GREEN): unit-length float32 invariant on a synthetic plane, degenerate-neighbourhood raise when `N < knn`, no-NaN invariant across all rows.
- **`normals_notes.md` visual sanity log (approved):** 3 chunks (wall A / corner B / roof-edge C) × 2 view types (angle-to-up RdBu on n_z + RGB `(n+1)/2`), 9 PNGs under `/tmp/zaha_normals/views/`, 401,878 total pts, all unit-length float32, zero non-finite.

## Denoising Sweep — Drop Fraction Table (Task 1)

4 candidates × 3 sample chunks of `DEBY_LOD2_4907179.pcd` (post-voxel-agg at grid=0.02). D-12 cap disabled (`max_drop_frac=1.0`) for the sweep so failure modes are visible instead of raising.

| Chunk                       | n pts    | sor   | radius (nb=4, r=0.08) | mls   | ransac_plane |
|-----------------------------|----------|-------|-----------------------|-------|--------------|
| A wall (4×4 full-Z)         | 348,018  | 4.24% | 4.93%                 | 0.00% | 92.45%       |
| B corner (4×4 full-Z)       | 42,367   | 1.35% | 4.67%                 | 0.00% | 67.65%       |
| C roof edge (8×8 top-6m)    | 38,736   | 4.05% | 28.06%                | 0.00% | 93.21%       |

**D-12 compliance (≤ 10% on all 3 chunks):** only `sor` passes. `radius` fails on C (28% > 10%). `ransac_plane` fails on all 3 (corner failure mode §D.5). `mls` is non-removing so the drop bound is moot.

**Winner = SOR single-method, NOT sor+radius sequential.** Final YAML block (quoted verbatim from denoising_notes.md `## Final Decision` for Plan 04's `manifest.denoising`):

```yaml
method: sor
sor:
  nb_neighbors: 30
  std_ratio: 2.0
max_drop_frac: 0.10
```

## Normals Visual Sanity — Unit-Length Table (Task 3)

Pipeline: `stream_voxel_aggregate(grid=0.02)` → Task-1-approved SOR denoise on the whole cloud (`n_in=1,382,868 → n_out=1,322,018`, drop 4.40%, D-13 order) → same 3 chunks from Task 1 → per-chunk `estimate_normals(NormalConfig(knn=30, orient=False, fast=False))`.

| Chunk       | n pts   | elapsed | min ‖n‖  | max ‖n‖  | NaN | n_z dist wall% / mid% / roof%  |
|-------------|---------|---------|----------|----------|-----|--------------------------------|
| A wall      | 333,392 | 0.32 s  | 1.000000 | 1.000000 | 0   | 37.3 / 40.1 / 22.6             |
| B corner    | 40,893  | 0.04 s  | 1.000000 | 1.000000 | 0   | 12.6 / 29.6 / 57.8             |
| C roof edge | 27,593  | 0.04 s  | 1.000000 | 1.000000 | 0   | 47.5 / 33.5 / 19.0             |

**Total:** 401,878 pts, all unit-length float32 (`min_norm == max_norm == 1.000000`), zero non-finite. D-18 hard bar met on every row. The mottled RGB on chunk A's flat wall is the expected §E.5 unoriented-PCA sign-arbitrariness behaviour, not a failure mode — the angle-to-up view is clean because `|cos_angle|` collapses the sign flip.

## Decisions Made

See `key-decisions` in the frontmatter for the structured list. The load-bearing choices:

1. **Denoising = single-method SOR, not `[sor, radius]` sequential.** The plan template's research block framed the sweep as picking between `sor+radius+mls`, `sor+radius+ransac_plane`, and `sor+radius+bilateral` — all variants of "SOR first, then a second-method polish". The 4-candidate sweep showed radius hard-fails D-12 on chunk C at 28.06%. Even if applied AFTER SOR (which already drops 4% on C), there's no density floor at which radius stops eating structural roof-edge points. Rejecting radius entirely is the right call; SOR alone covers the scan-stripe noise that was the original failure case for pure-semantic on ZAHA. Plan 04's `manifest.denoising` block copies the SOR-only YAML verbatim.

2. **Tested 4 candidates, not 3.** D-11's minimum set is "SOR + radius + one of {bilateral, MLS, RANSAC plane}". Running all 4 (SOR + radius + MLS + RANSAC plane) is a strict superset that produces a more informative reject-reasoning table at the cost of one extra minute of open3d kernel runtime on the smallest sample. The wider sweep is what made the "radius fails on density-heterogeneity" verdict legible.

3. **Chunk C enlarged from 4×4×top-4m to 8×8×top-6m.** The original plan template bbox (4×4×top-4m) produced 6,951 pts on `DEBY_LOD2_4907179` — pathologically sparse, unfit for evaluating a denoising filter because any filter eats most of that chunk regardless of parameters. 8×8×top-6m yields 38,736 pts, which is the realistic density of a roof-edge chunk at post-voxel-agg (grid=0.02) resolution. This is documented in denoising_notes.md Chunk C bbox + NOTE line.

4. **Radius params tuned from (nb=8, r=0.05) to (nb=4, r=0.08).** The plan default is a RESEARCH §D.2 starting point, not a tested value. On post-voxel-agg density the default drops 70–98% per chunk — completely unusable. Tuning to (nb=4, r=0.08) recovered plausible drop fractions on A/B but still fails on C at 28%, which is the evidence used to reject the filter. Both the failed default and the tuned rejection are on record in denoising_notes.md so a future reader understands why radius is out.

5. **Pytest invocation = `conda run -n ptv3 python -m pytest ...`.** Same pattern Plan 01-02 used. The plan's literal grep acceptance line (`conda run -n ptv3 pytest ...`) is stricter than what works reliably on ptv3 — bare `pytest` occasionally fails to find the correct Python interpreter under `conda run`. Module form is reliable and has the same semantics.

6. **normals.py open3d-first import order preserved.** Line 39 `import open3d as o3d`, line 43 `import numpy as np` — verified by grep after the first test run. Any attempt to import numpy or scipy before open3d on the ptv3 env trips `libstdc++.so.6 GLIBCXX_3.4.29 not found` at loader time. Callers who import pcd_parser.py (which uses pandas) before normals.py in the same process will still trip the trap; Plan 04's orchestrator must enforce the same ordering.

7. **Sign orientation deferred.** `NormalConfig.orient=False` is the D-18 aspirational choice. The visual mottling on chunk A/C flat walls in the RGB `(n+1)/2` view is sign-arbitrariness inherent to unoriented PCA, NOT noise — the same chunks are clean in the angle-to-up view because `|cos_angle|` is sign-insensitive. Downstream BFDataset concatenates `feat_keys=(color, normal)` and uses normals as an unsigned rotation-invariant feature, so orient=True costs ~20 s per 500 k chunk × ~100 chunks × 26 samples = ~30 extra minutes for zero training benefit. Future plans may revisit if a rotation-sensitive consumer lands downstream.

8. **chunking.py is pure-numpy, zero open3d imports.** The chunker takes `(N, 3) xyz + (N,) segment + ChunkSpec` and emits `(M, 3) xyz_sub + (M,) seg_sub` via a boolean mask. No PCA, no neighbour search, no file I/O. This means it can be imported at any point in the pipeline without participating in the open3d-first rule, and Plan 04 can chunk inside the orchestrator without worrying about import order for this one module.

9. **measure_density.py is a one-shot operational tool.** It is not imported anywhere — it's a standalone script that runs end-to-end parse + voxel_agg on all 26 samples and prints density. Evidence for Plan 04's tile-size choice, not a runtime dependency of the orchestrator. Placed under `data_pre/zaha/scripts/tools/` to signal "ad-hoc tools, not part of the pipeline proper".

## Task Commits

1. **Task 1a: denoise.py + test_denoise.py GREEN** — `a3dc08c` (feat)
2. **Task 1b: denoising_notes.md approved (research gate 1/2)** — `fce8006` (docs)
3. **Task 2: chunking.py + measure_density.py + test_chunking.py GREEN** — `a7a08ca` (feat)
4. **Task 3a: normals.py + test_normals.py GREEN** — `87cb1ed` (feat)
5. **Task 3b: normals_notes.md draft (pre-approval)** — `686253d` (docs)
6. **Task 3c: normals_notes.md approved (research gate 2/2)** — `e5340b7` (docs)

All commits used `--no-verify` per worktree-isolation protocol. The two research gates (1b, 3c) are separate approval commits rather than amended onto the draft, preserving the approval event in the audit trail.

## Files Created/Modified

### Created (7)

- `data_pre/zaha/utils/denoise.py` — 250 lines. SOR/radius/MLS/ransac_plane/none dispatch + D-12 hard cap.
- `data_pre/zaha/utils/chunking.py` — 345 lines. Pure-numpy deterministic box-grid chunker.
- `data_pre/zaha/utils/normals.py` — 179 lines. open3d KDTreeSearchParamKNN(knn=30) PCA normals.
- `data_pre/zaha/scripts/tools/__init__.py` — subpackage init.
- `data_pre/zaha/scripts/tools/measure_density.py` — 212 lines. 26-sample density measurement tool.
- `.planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/denoising_notes.md` — status=approved, Final Decision YAML block.
- `.planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/normals_notes.md` — status=approved, 3-chunk visual sanity log.

### Modified (3)

- `data_pre/zaha/tests/test_denoise.py` — 2 RED stubs → 2 GREEN (drop_cap + determinism).
- `data_pre/zaha/tests/test_chunking.py` — 3 RED stubs → 3 GREEN (row-major + chunk_name + iter_chunk_points).
- `data_pre/zaha/tests/test_normals.py` — 3 RED stubs → 3 GREEN (unit-length + degenerate-raise + no-NaN).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Denoising winner = single-method SOR, not `[sor, radius]` sequential pipeline.**
- **Found during:** Task 1 research sub-step (candidate sweep evaluation)
- **Issue:** The plan's `<assumptions>` block framed the sweep as picking between `sor+radius+mls`, `sor+radius+ransac_plane`, and `sor+radius+bilateral` — all variants of "SOR first, then a polish method". Running the 4-candidate sweep showed radius hard-fails D-12 on chunk C at 28.06% because it is density-dependent in a way SOR is not. Applying radius AFTER SOR (which already drops 4% on C) would still chew through > 20% of the residual roof edge. There is no tuning at which radius is safe on ZAHA's heterogeneous post-voxel-agg density.
- **Fix:** Final Decision YAML records `method: sor` (single method), not `method: [sor, radius]`. Plan 04's `manifest.denoising` block must follow the notes file verbatim.
- **Files modified:** `.planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/denoising_notes.md`
- **Verification:** `status: approved` on denoising_notes.md after user review of the 39 PNGs and the drop-frac table.
- **Committed in:** `a3dc08c` (the wrapper implementation that ships all 4 methods) + `fce8006` (the approval).

**2. [Rule 1 - Bug] Tested 4 denoising candidates, not the plan minimum 3.**
- **Found during:** Task 1 research sub-step
- **Issue:** D-11 minimum set is "SOR + radius + one of {bilateral, MLS, RANSAC plane}" — i.e. 3 methods. Running all 4 (SOR + radius + MLS + RANSAC plane) is a strict superset that produces a more complete reject-reasoning table. The plan's wording "3 candidates" was a lower bound, not an exact count.
- **Fix:** `run_denoise_sweep_v2.py` runs all 4 methods across all 3 chunks; denoising_notes.md's drop-frac table has 4 method columns. Extra runtime cost: ~1 min on the smallest sample.
- **Files modified:** `.planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/denoising_notes.md`
- **Verification:** Table in denoising_notes.md has 4 method rows.
- **Committed in:** `a3dc08c` + `fce8006`.

**3. [Rule 1 - Bug] Chunk C (roof edge) bbox enlarged from plan template 4×4×top-4m to 8×8×top-6m.**
- **Found during:** Task 1 research sub-step (first pass at the sweep)
- **Issue:** The original template bbox produced 6,951 pts on `DEBY_LOD2_4907179` — pathologically sparse because the smallest training building's roof cap is tiny at that slice. Any filter eats most of that chunk regardless of parameters, making the gate uninformative.
- **Fix:** Enlarged chunk C to 8×8×top-6m (38,736 pts), which captures the actual density of a realistic roof-edge slice on a post-voxel-agg (grid=0.02) cloud. denoising_notes.md documents the enlargement inline with a NOTE line.
- **Files modified:** `.planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/denoising_notes.md`, `/tmp/zaha_denoise/run_denoise_sweep_v2.py` (ephemeral, not committed)
- **Verification:** Chunk C row in the drop-frac table shows `n=38,736` and the `NOTE:` line under the "Sample source" section explains the enlargement.
- **Committed in:** `a3dc08c` + `fce8006`.

**4. [Rule 1 - Bug] Radius filter params tuned from plan default (nb_points=8, radius=0.05) to (nb_points=4, radius=0.08).**
- **Found during:** Task 1 research sub-step
- **Issue:** The plan's RESEARCH §D.2 default (nb=8, r=0.05) is a starting point, not a tested value. On post-voxel-agg (grid=0.02) density it drops 70–98% per chunk — completely unusable, and not representative of what a radius filter could be tuned to in principle. Reporting radius as "drops 98%" without tuning would have been misleading evidence.
- **Fix:** Tuned radius params to (nb=4, r=0.08) and reran. New drops: A 4.93%, B 4.67%, C 28.06%. C still fails D-12, which is the evidence used to reject radius — the point is that EVEN AT the best tuning the density heterogeneity kills it. denoising_notes.md explicitly says "Tuned from the plan's 8/0.05 which drops 70–98%" in the candidates table.
- **Files modified:** `.planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/denoising_notes.md`
- **Verification:** denoising_notes.md candidates table shows the tuned params inline.
- **Committed in:** `a3dc08c` + `fce8006`.

**5. [Rule 3 - Blocking] Pytest invocation = `conda run -n ptv3 python -m pytest ...` (module form) not `conda run -n ptv3 pytest ...` (bare).**
- **Found during:** Task 1 first test run
- **Issue:** The plan's literal grep-style acceptance line calls `conda run -n ptv3 pytest data_pre/zaha/tests/test_denoise.py`. That form occasionally resolves the wrong Python interpreter on the ptv3 env (same issue Plan 01-02 documented). Module form (`python -m pytest`) is reliable and has identical test-collection semantics.
- **Fix:** All local verification runs used `conda run -n ptv3 python -m pytest ...`. Test results are identical to what the plan's command would produce under a clean env.
- **Files modified:** None (local command-line change only)
- **Verification:** 8 tests GREEN across denoise (2) + chunking (3) + normals (3) confirmed via module-form invocation.
- **Committed in:** Not applicable (command-line convention, no file change).

---

**Total deviations:** 5 auto-fixed. 4 Rule 1/2 during Task 1 research (winner choice + candidate count + chunk C bbox + radius tuning), 1 Rule 3 pytest invocation form. All were user-visible and surfaced in the research gate approval flow. Tasks 2 and 3 shipped with no further deviations beyond what's documented in key-decisions (sign deferral, chunking purity, measure_density operational scope).

**Impact on plan:** The five Task 1 deviations all feed back into Plan 04 via denoising_notes.md's approved Final Decision block — Plan 04's manifest must consume `method: sor` verbatim rather than the plan's original `method: [sor, radius]` assumption. No code-surface expansion: denoise.py still ships all 4 methods (unchanged from the plan's listing), but Plan 04 only wires `method='sor'`. Tasks 2/3 shipped on schedule.

## Authentication Gates

None. Plan 03 has no external service dependencies.

## Issues Encountered

- **Plan template assumed `[sor, radius]` sequential pipeline** — resolved by the research gate rejecting radius on density-heterogeneity (Deviation #1).
- **Radius filter plan defaults (nb=8, r=0.05) are wildly wrong for post-voxel-agg density** — resolved by tuning, but the tuned params still fail D-12 on chunk C, which is the load-bearing rejection evidence (Deviation #4).
- **Template chunk C bbox (4×4×top-4m) is too sparse to evaluate filters** — resolved by enlargement (Deviation #3).
- **Pytest bare invocation unreliable under `conda run`** — resolved by module-form invocation (Deviation #5).
- **Sign-arbitrary normals on flat walls look like noise in the RGB view** — resolved by the angle-to-up view being the actual sanity channel. The mottling is expected D-18 unoriented-PCA behaviour; the visual gate approves via the angle-to-up view and leaves `orient=False`.

## Known Stubs

None introduced by this plan. The 8 tests this plan turned GREEN are real GREEN — `test_drop_cap`, `test_determinism` (denoise), `test_compute_chunks_row_major`, `test_chunk_name`, `test_iter_chunk_points` (chunking), `test_unit_length`, `test_degenerate_raises`, `test_no_nan` (normals). The remaining 9 RED stubs (2 output_layout + 1 yaml + 6 e2e) are intentional placeholders owned by Plan 01-04 (orchestrator + NPY layout + e2e).

No hardcoded empty values, no "coming soon" / placeholder strings, no wired-stub UI components. The only intentional deferral is `NormalConfig.orient=False` per §E.5 / D-18 — documented in key-decisions and normals_notes.md Final verdict.

## Next Phase Readiness

- **Plan 01-04 (Wave 3) unblocked.** The three primitives + two research logs this plan delivers are the last pre-orchestration pieces. Plan 04 consumes:
  - `denoise_cloud(xyz, segment, DenoiseConfig('sor', {'nb_neighbors': 30, 'std_ratio': 2.0}, max_drop_frac=0.10))` — called per-sample on the full post-voxel-agg cloud, BEFORE chunking, per D-13.
  - `compute_chunks(bbox_min, bbox_max, ChunkingConfig(tile_xy=…, overlap_xy=2.0, z_mode='full', budget_per_chunk=600_000))` + `iter_chunk_points(xyz, segment, chunk)` + `chunk_name(basename, idx)` — called per-sample, row-major (D-10), after denoising.
  - `estimate_normals(xyz_chunk, NormalConfig(knn=30, orient=False, fast=False))` — called per-chunk AFTER chunking (D-19 order).
- **denoising_notes.md approved** — Plan 04's `manifest.json[denoising]` block copies the Final Decision YAML verbatim. No hand-editing.
- **normals_notes.md approved** — no manifest consumption needed (pure visual sanity gate).
- **measure_density.py available** — Plan 04 runs it once across all 26 samples to measure actual D_max before locking `tile_xy`. The RESEARCH §F.4 worked example of T ≈ 4 m ± 0.5 m is the starting hypothesis; measure_density's output is the evidence.
- **Large-sample validation still deferred.** The 136.8 M-point `DEBY_LOD2_4959458.pcd` end-to-end validation is Plan 04's e2e test responsibility — Plan 03 only smoke-tested on `DEBY_LOD2_4907179` (smallest training building, 1.38 M post-voxel-agg points) because the research sweep runs 4 methods × 3 chunks in the same process and the large sample would blow the 4 GB RAM ceiling.

## Self-Check

Files listed in `key-files.created` verified on disk:

- data_pre/zaha/utils/denoise.py — FOUND (250 lines)
- data_pre/zaha/utils/chunking.py — FOUND (345 lines)
- data_pre/zaha/utils/normals.py — FOUND (179 lines)
- data_pre/zaha/scripts/tools/__init__.py — FOUND
- data_pre/zaha/scripts/tools/measure_density.py — FOUND (212 lines)
- .planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/denoising_notes.md — FOUND (status=approved)
- .planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/normals_notes.md — FOUND (status=approved)

Files listed in `key-files.modified` verified on disk:

- data_pre/zaha/tests/test_denoise.py — FOUND (2 GREEN)
- data_pre/zaha/tests/test_chunking.py — FOUND (3 GREEN)
- data_pre/zaha/tests/test_normals.py — FOUND (3 GREEN)

Commits verified in git log:

- a3dc08c — FOUND (Task 1a: denoise.py + test_denoise.py GREEN)
- fce8006 — FOUND (Task 1b: denoising_notes.md approved)
- a7a08ca — FOUND (Task 2: chunking.py + measure_density.py + test_chunking.py GREEN)
- 87cb1ed — FOUND (Task 3a: normals.py + test_normals.py GREEN)
- 686253d — FOUND (Task 3b: normals_notes.md draft)
- e5340b7 — FOUND (Task 3c: normals_notes.md approved)

Test gate verified (`conda run -n ptv3 python -m pytest data_pre/zaha/tests/test_denoise.py data_pre/zaha/tests/test_chunking.py data_pre/zaha/tests/test_normals.py`):

```
8 passed
```

Import-order gate verified (normals.py):

```
line 39: import open3d as o3d
line 43: import numpy as np
```

open3d precedes numpy — §I.5 invariant satisfied.

## Self-Check: PASSED

---
*Phase: 01-zaha-offline-preprocessing-pipeline*
*Plan: 03*
*Completed: 2026-04-11*
