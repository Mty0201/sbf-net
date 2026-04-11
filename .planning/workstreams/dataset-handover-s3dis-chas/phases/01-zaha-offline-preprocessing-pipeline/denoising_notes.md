---
phase: 01-zaha-offline-preprocessing-pipeline
plan: 03
task: 1
created: 2026-04-11
status: draft
---

# Denoising Research Log — 4 Candidates × 3 Sample Chunks

## Sample source

- Sample: `DEBY_LOD2_4907179.pcd` (1.7 M raw pts, smallest training building)
- Preprocessing applied before cutting chunks:
  - Streamed ASCII PCD parse (Plan 01-01 parser)
  - Voxel aggregation at `grid=0.02 m` (Plan 01-02 `voxel_aggregate`) → 1,382,868 centroid points
- Full-cloud bbox: `[45.765, 264.022, 59.597] .. [55.535, 281.212, 78.834]` (roughly 10 × 17 × 19 m facade envelope)
- Hand-cut 3 sample chunks from this voxel-aggregated cloud:
  - **Chunk A — wall section** (dense, vertical scan strip)
    bbox `[47.765, 270.022, 59.597] .. [51.765, 274.022, 78.834]`, n=348,018 pts (4 × 4 × full-Z)
  - **Chunk B — corner** (two wall facets meeting)
    bbox `[49.765, 266.022, 59.597] .. [53.765, 270.022, 78.834]`, n=42,367 pts (4 × 4 × full-Z)
  - **Chunk C — roof edge** (enlarged to 8 × 8 × top-6m for a usable density)
    bbox `[46.765, 268.022, 72.834] .. [54.765, 276.022, 78.834]`, n=38,736 pts
    NOTE: the original research plan used a 4 × 4 × top-4m roof chunk which turned out to be pathologically sparse (6,951 pts) — enlarged here so the density is representative of real roof edges after voxel_agg.

## Candidates tested

| Method         | Params                                         | Drop% A | Drop% B | Drop% C  | Elapsed A (s) | Elapsed B (s) | Elapsed C (s) | Notes                                                                                 |
|----------------|------------------------------------------------|---------|---------|----------|---------------|---------------|---------------|---------------------------------------------------------------------------------------|
| sor            | `nb_neighbors=30, std_ratio=2.0`               | 4.24%   | 1.35%   | 4.05%    | 0.34          | 0.03          | 0.04          | **Winner.** All three chunks ≤ 10 % D-12 cap; drops isolated noise, preserves facets. |
| radius         | `nb_points=4, radius=0.08` (tuned from 8/0.05) | 4.93%   | 4.67%   | 28.06%   | 0.38          | 0.03          | 0.02          | Fails on C (density-heterogeneity). Tuned from the plan's 8/0.05 which drops 70–98 %. |
| mls            | `knn=20`                                       | 0.00%   | 0.00%   | 0.00%    | 1.26          | 0.13          | 0.12          | Non-removing moving filter — cannot demonstrate stripe removal as a drop-frac signal. |
| ransac_plane   | `distance_threshold=0.02, ransac_n=3, iters=1000` | 92.45% | 67.65% | 93.21% | 0.36          | 0.03          | 0.06          | Corner failure mode active — latches onto one facet, kills the rest (D-12 violated).  |

**D-12 compliance:** only `sor` stays ≤ 10 % on all 3 chunks. `radius` fails on C; `ransac_plane` fails on all 3; `mls` is non-removing so the drop bound is moot.

## Visual verdict per view

All PNGs live under `/tmp/zaha_denoise/views/` (ephemeral — regenerable via `/tmp/zaha_denoise/run_denoise_sweep_v2.py`).

### Chunk A (wall, n=348,018)

- **Input XZ (`chunk_A_input_xz.png`):** dense 4 m × 19 m wall slice with clear horizontal scan bands + thin outlier shells at the top and around the window cut-outs.
- **SOR XZ (`chunk_A_sor_xz.png`) + drop mask (`chunk_A_sor_mask.png`):** 14,765 dropped (4.24 %). Red dropped points form thin shells *outside* the wall core + scattered specks in window cavities; green kept points preserve the full wall structure. **Verdict: clean scan-stripe reduction, no structural damage.**
- **Radius XZ + mask (`chunk_A_radius_*.png`):** 17,146 dropped (4.93 %) — similar shell pattern to SOR but slightly more aggressive on the top-roof transition zone where density tapers.
- **MLS facade (`chunk_A_mls_facade.png`):** visually nearly identical to input (moving-average smooths coordinates but the 0.05 m voxel-aggregated grid already smooths most of that); 0 pts dropped — invisible as a drop-frac signal.
- **RANSAC plane mask (`chunk_A_ransac_plane_mask.png`):** 321,741 dropped (92.45 %) — vast majority of wall flagged as outlier. The dominant plane is a single facet of the wall; everything else (window reveals, balconies, the second facet) lies > 0.02 m from it.

**Chunk A winner: SOR.**

### Chunk B (corner, n=42,367)

- **Input XZ (`chunk_B_input_xz.png`):** two wall facets meeting at roughly 90°, visible as an L-shape in XZ with a ground pile at z ≈ 61 m.
- **SOR mask (`chunk_B_sor_mask.png`):** 571 dropped (1.35 %) — red specks are individual outliers above the corner spine and in the ground pile, consistent with MLS-noise. Green kept points preserve both facets + the ground pile.
- **Radius mask (`chunk_B_radius_mask.png`):** 1,979 dropped (4.67 %) — similar pattern to SOR with slightly more ground-pile trimming.
- **RANSAC plane mask (`chunk_B_ransac_plane_mask.png`):** 28,661 dropped (67.65 %) — one of the two facets is killed (the corner geometry puts both facets > 0.02 m from the dominant plane). **This is the exact corner failure mode RESEARCH §D.5 warned about.**

**Chunk B winner: SOR (tightest drop).**

### Chunk C (roof edge, n=38,736)

- **Input XZ (`chunk_C_input_xz.png`):** sparse, non-planar roof geometry with 3 horizontal roof lines and a lot of cavity in between — the density is materially lower than the wall chunks (that's the whole point of this chunk).
- **SOR mask (`chunk_C_sor_mask.png`):** 1,569 dropped (4.05 %) — red points are true isolates scattered in the roof cavity (real outliers), green points preserve the structural roof edges intact. **SOR handles the density-heterogeneity without clipping structure.**
- **Radius mask (`chunk_C_radius_mask.png`):** 10,869 dropped (28.06 %) — even at the tuned `nb=4, r=0.08` radius, the filter chews through 28 % of the chunk because legitimate structural points on the roof edges have fewer neighbours than `nb_points=4` within 8 cm. **Radius filter is density-dependent in a way SOR is not.**
- **RANSAC plane mask (`chunk_C_ransac_plane_mask.png`):** 36,105 dropped (93.21 %) — roof is non-planar, so the whole chunk is flagged.

**Chunk C winner: SOR (only candidate that passes D-12 on heterogeneous density).**

### Summary of visual verdicts

| Chunk | Winner | Loser (tied bottom) |
|-------|--------|---------------------|
| A     | SOR    | RANSAC plane (92 % drop)       |
| B     | SOR    | RANSAC plane (68 % drop)       |
| C     | SOR    | Radius (28 %) + RANSAC (93 %)  |

## Final Decision

- **Winner method:** `sor` (single pass — NOT the sor+radius sequential pipeline suggested in the plan template, because radius fails D-12 on roof-edge chunks).
- **Final params:**

  ```yaml
  method: sor
  sor:
    nb_neighbors: 30
    std_ratio: 2.0
  max_drop_frac: 0.10
  ```

- **Rationale:** SOR at `(nb=30, std=2.0)` is the only candidate that:
  (a) drops ≤ 10 % on all three sample chunks (4.24 / 1.35 / 4.05 %),
  (b) preserves structural geometry without clipping legitimate low-density roof edges,
  (c) is density-agnostic in a way the radius filter is not (radius at any tuning aggressive enough to cut wall noise on A/B chews > 20 % on the sparser C chunk), and
  (d) has no corner failure mode like RANSAC plane.
  The drop on all three chunks is visibly noise (thin shells outside the wall core, isolated specks in cavities), never structural. The per-chunk drop is comfortably ≤ 10 % with the D-12 `max_drop_frac` hard cap enforced inside `denoise_cloud`.

- **Rejected candidates and why:**
  - **`radius` (nb=4, r=0.08):** hard-fails D-12 on the roof-edge chunk (28 % drop) because it is density-dependent; any tuning aggressive enough to reduce wall stripe banding on chunk A kills legitimate structural roof-edge points on chunk C. Plan's default (nb=8, r=0.05) is even worse — it drops 70–98 % on every chunk at the post-0.02 voxel density. Rejected.
  - **`mls` (knn=20):** non-removing moving filter. Drops 0 % everywhere by design — can't demonstrate scan-stripe removal as a drop-frac signal, and visually the effect is marginal on top of the 0.02 m voxel grid. If a smoother is ever needed, this remains available under `method='mls'` but it is not the Plan 03 winner.
  - **`ransac_plane` (dt=0.02):** catastrophic corner failure on B (68 %) and structural destruction on A/C (> 92 %). Confirmed the RESEARCH §D.5 failure mode: on multi-facet facades RANSAC latches onto one plane and annihilates the rest. Rejected hard.

- **Follow-up:** none. Plan 04 consumes `method: sor, nb_neighbors: 30, std_ratio: 2.0, max_drop_frac: 0.10` verbatim.

## Gate for Plan 04

Plan 04's `manifest.denoising` block is populated from this log's `## Final Decision` section:

```yaml
denoising:
  method: sor
  params:
    nb_neighbors: 30
    std_ratio: 2.0
  max_drop_frac: 0.10
```

**Do NOT hand-edit the Plan 04 manifest — it must mirror this block.**

## Reproduce the sweep

```bash
conda run -n ptv3 python /tmp/zaha_denoise/run_denoise_sweep_v2.py
# Writes 39 PNGs to /tmp/zaha_denoise/views/ and a sweep_results.json
```

The script loads `full_xyz.npy` / `full_seg.npy` (the voxel-aggregated `DEBY_LOD2_4907179.pcd` from Plan 01-02), cuts the 3 chunks above, and runs each candidate with `max_drop_frac=1.0` (cap disabled so failure modes are visible instead of raising). The D-12 cap is still enforced in production via the `DenoiseConfig.max_drop_frac` default (0.10).
