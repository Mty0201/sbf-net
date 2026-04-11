---
phase: 01-zaha-offline-preprocessing-pipeline
plan: 03
task: 3
created: 2026-04-11
status: draft
---

# Normals Visual Sanity Log — 3 Sample Chunks, DEBY_LOD2_4907179

## Config tested

- `NormalConfig(knn=30, orient=False, fast=False, unit_length_threshold=0.99)`
- Pipeline: `stream_voxel_aggregate(grid=0.02)` → Task-1-approved SOR denoise
  (`nb_neighbors=30`, `std_ratio=2.0`, `max_drop_frac=0.10`) → 3 sample chunks
  cut via `data_pre/zaha/utils/chunking.iter_chunk_points` → per-chunk
  `estimate_normals`.
- `open3d` imported before `numpy` in `data_pre/zaha/utils/normals.py` per
  RESEARCH §I.5 — verified by grep (line 39 `import open3d`, line 43
  `import numpy`).
- No NaN, no non-finite, no sub-unit normals on any of the 3 chunks.

## Sample source

- Sample: `DEBY_LOD2_4907179.pcd` (1.7 M raw pts, smallest training building).
  Same sample as Task 1's denoising research — re-using the Task 1 cached
  voxel-aggregated cloud at `/tmp/zaha_denoise/full_xyz.npy`
  (`(1,382,868, 3)` float32) and `/tmp/zaha_denoise/full_seg.npy`
  (`(1,382,868,)` int32) so the chunk bounding boxes are bit-identical to the
  approved denoising research log.
- Full-cloud bbox:
  `[45.765, 264.022, 59.597] .. [55.530, 281.213, 78.834]` (roughly
  9.76 × 17.19 × 19.24 m facade envelope).
- Task 1 SOR applied to the WHOLE pre-chunk cloud (D-13 order):
  `n_in=1,382,868 → n_out=1,322,018`, drop `4.40%` (`max_drop_frac=0.10`
  cap enforced).

## Chunks

Same 3 bboxes as Task 1 `run_denoise_sweep_v2.py` — wall / corner / roof-edge
slicing of `DEBY_LOD2_4907179`, cut after the approved SOR denoise.

| Tag | Desc (size × z)            | bbox                                                                      | n_pts (post-SOR) |
|-----|----------------------------|---------------------------------------------------------------------------|------------------|
| A   | wall section (4×4×full-Z)  | `[47.765, 270.022, 59.597] .. [51.765, 274.022, 78.834]` (4 × 4 × 19.2 m) | 333,392          |
| B   | corner (4×4×full-Z)        | `[49.765, 266.022, 59.597] .. [53.765, 270.022, 78.834]` (4 × 4 × 19.2 m) | 40,893           |
| C   | roof edge (8×8×top-6m)     | `[46.765, 268.022, 72.834] .. [54.765, 276.022, 78.834]` (8 × 8 × 6 m)    | 27,593           |

## Per-chunk observations

All three chunks produced `(N, 3)` float32 normals with
`min_norm == max_norm == 1.000000` and zero non-finite entries. The D-18
hard bar (unit-length + no NaN) is met in all three.

### Chunk A (wall, n=333,392)

- **Elapsed:** 0.32 s (open3d OpenMP-parallelized KNN+PCA, §E.4 predicts
  ~2 s single-thread for 300 k pts; actual is cache-friendly at this
  density).
- **Angle-to-up split:** `|cos_angle| < 0.2` (wall-like) = 37.3 %;
  `0.2 ≤ |cos_angle| ≤ 0.8` (mixed/corner) = 40.1 %;
  `|cos_angle| > 0.8` (roof/ground-like) = 22.6 %.
- **§E.6 view 1 — angle-to-up (RdBu on `n_z`)**
  (`/tmp/zaha_normals/views/chunk_A_angle.png`): the wall slice shows a
  predominantly white/near-zero `cos_angle` band (wall normals perpendicular
  to world up) with distinct dark-red/dark-blue horizontal lines at
  z ≈ 62, 64, 67, 70, 72, 75, 77.5 — these are the slab/floor-plate
  horizontal features, correctly flagged as roof/ground-like. This is the
  textbook §E.6 signature: clean wall separation with sharp horizontal
  structural features.
- **§E.6 view 2 — RGB `(n+1)/2`**
  (`/tmp/zaha_normals/views/chunk_A_rgb.png`): the wall shows mottled
  pink/grey/purple/green because the PCA returns sign-arbitrary eigenvectors
  (§E.5: `open3d's plain PCA returns normals with arbitrary sign`). Adjacent
  flat-wall points can have opposite-sign eigenvectors, which in `(n+1)/2`
  space flips colour channels. **This is NOT a failure mode** — the same
  mottled wall in the angle-to-up view above is clean because `|cos_angle|`
  collapses the sign flip. §E.5 / D-18 explicitly leave normals unoriented
  (BFDataset treats normals as a rotation-invariant feature vector).
- **Verdict:** **pass.** Wall structure cleanly recovered; slab/plate
  horizontal features distinguishable from wall in the angle view; no NaN
  or collapse; unit-length invariant satisfied.

### Chunk B (corner, n=40,893)

- **Elapsed:** 0.04 s.
- **Angle-to-up split:** wall-like 12.6 %, mixed 29.6 %, roof/ground 57.8 %
  — dominated by horizontal features because the chunk's bbox catches a
  ground pile at z ≈ 61 and multiple slab plates at z ≈ 65, 68, 72.
- **§E.6 view 1 — angle-to-up**
  (`/tmp/zaha_normals/views/chunk_B_angle.png`): three isolated horizontal
  slab bands at z ≈ 65, 68, 72 show uniform dark/saturated colour (correct —
  roof/plate normals), and the dense ground pile at z ≈ 61–63 shows a mix of
  red/blue/white consistent with messy corner geometry (two walls meeting
  ground, with furniture/vegetation-like features). No random-speckle
  collapse on any visible wall facet.
- **§E.6 view 2 — RGB**
  (`/tmp/zaha_normals/views/chunk_B_rgb.png`): sparse coverage; the ground
  pile shows structured but diverse RGB consistent with the real corner
  geometry. No NaN, no speckle collapse, no single-color wash that would
  indicate a degenerate fallback.
- **Verdict:** **pass.** The corner failure mode §E.2 warns about (2-wall
  intersection line → ill-defined normal direction) is not visible at a
  rate that matters; the unit-length + no-NaN invariant holds for all
  40 893 points.

### Chunk C (roof edge, n=27,593)

- **Elapsed:** 0.04 s.
- **Angle-to-up split:** wall-like 47.5 %, mixed 33.5 %, roof/ground 19.0 %
  — the 8 × 8 × top-6m band above z = 72.8 catches both the main-roof top
  line at z ≈ 78.5 and a lower-roof tier at z ≈ 75.5, plus the short wall
  segments connecting them.
- **§E.6 view 1 — angle-to-up**
  (`/tmp/zaha_normals/views/chunk_C_angle.png`): two clearly visible roof
  lines at z ≈ 78.5 (main roof) and z ≈ 75.5 (lower tier) showing
  red/saturated colours (roof normals point ±up), while the wall segments
  between and below show near-white (wall-like). Sparse interior points
  (density gradient — this chunk has much lower point count than A or B)
  remain structured, not random.
- **§E.6 view 2 — RGB**
  (`/tmp/zaha_normals/views/chunk_C_rgb.png`): roof plates show a consistent
  blue/green dominant for the main-roof line and olive/yellow for the
  lower tier — each planar surface has a dominant colour, which is the
  positive §E.6 sign. The speckled RGB in the wall region is the same
  arbitrary-sign behaviour as chunk A, not noise.
- **Verdict:** **pass.** Low-density roof-edge chunk is the hardest case for
  PCA normals; SOR denoising upstream removed isolated outliers and the
  remaining 27 593 points all produced unit-length finite normals within
  0.04 s.

## Failure modes seen

1. **Arbitrary sign on flat walls** (chunks A + C RGB views): expected per
   §E.5 because the module leaves normals unoriented. Not a failure —
   downstream BFDataset concatenates `feat_keys=(color, normal)` and uses
   normals as an unsigned rotation-invariant feature. Changing to
   `orient=True` in `NormalConfig` would cost ~20 s per 500 k chunk ×
   ~100 chunks × 26 samples ≈ 30+ extra minutes for zero downstream
   benefit. **D-18 explicitly leaves this as aspirational, not blocking.**
2. **Corner facet ill-defined normals** (chunk B, 2-wall intersection
   line): not visible at a material rate. The corner chunk is dominated
   by horizontal plates and a ground pile rather than the intersection
   line itself, which is consistent with how the building samples the
   corner at 0.02 m voxel grid resolution.

## Final verdict

- [x] **Pass** — normals do not collapse to noise on any of the 3 chunks.
      All 401,878 points across the 3 chunks are unit-length float32
      (`min_norm = max_norm = 1.000000`) with zero non-finite entries. The
      `data_pre/zaha/utils/normals.py` module is ready for Plan 04 to wire
      into the per-sample loop as `estimate_normals` per chunk after
      chunking (D-19 order).
- Config for Plan 04: `NormalConfig(knn=30, orient=False, fast=False)`
  — the defaults. No override needed based on visual inspection.

## Reproduce

```bash
conda run -n ptv3 python /tmp/zaha_normals/run_normals_sanity.py
# Writes 9 PNGs to /tmp/zaha_normals/views/ + sanity_results.json
```

The script loads `/tmp/zaha_denoise/full_xyz.npy` / `full_seg.npy` (voxel-
aggregated `DEBY_LOD2_4907179.pcd` from Plan 01-02), applies the Task 1
approved SOR denoise to the whole cloud, cuts the 3 chunks above, runs
`estimate_normals` per chunk, and emits per-chunk scatter views.

## Sign-off

- Reviewer: _(pending human approval)_
- Date: _(pending)_
- Resume signal: type `approved` to advance to Plan 04, or describe a
  visual failure mode (e.g. widen knn, switch to Option B robust PCA
  §E.2) for remediation.
