# ZAHA Offline Preprocessing Pipeline

Stage-by-stage data flow, bound to CONTEXT.md + RESEARCH.md and to the
per-module implementations under `data_pre/zaha/utils/`. Consumed by
`data_pre/zaha/scripts/build_zaha_chunks.py` — the single entry point that
turns `/home/mty0201/data/ZAHA_pcd/` into `/home/mty0201/data/ZAHA_chunked/`.

## Stage 1 — Parse (DS-ZAHA-P1-01)

- `data_pre/zaha/utils/pcd_parser.py::stream_pcd(path, chunksize=2_000_000)`
- Streaming pandas reader with `engine='c'`; yields `(x, y, z, c)` chunks of
  `float64 / float64 / float64 / int32`.
- Drops the `rgb` column unconditionally (paper §3 — not spectral).
- Rejects `DATA binary` headers and any PCD that violates the v0.7 ASCII
  header invariants in RESEARCH §A.1 (raises `PcdFormatError`).
- Peak RAM: ~250 MB at chunksize=2 M on the 136.8 M-point sample.

## Stage 2 — Voxel aggregate (DS-ZAHA-P1-02)

- `data_pre/zaha/utils/voxel_agg.py::stream_voxel_aggregate(pcd, tmp_dir, K=16)`
- Grid = 0.04 m locked module constant — aligned with main-line training
  regime (CONTEXT.md D-14 supersession 2026-04-12 — see `01-02-SUMMARY.md`
  for the original 0.02 design and Plan 01-04 Task 3 notes for the switch).
- Pass 1: streaming partition into K disk bins via `key % K`.
- Pass 2: in-RAM majority-vote aggregate per bin (stable sort + `np.unique` +
  `np.add.reduceat`).
- Pass 3: VOID drop (D-01, winner == raw class 0) + remap 1..16 → 0..15
  (D-02) + float32 cast (D-16).
- Peak RAM: ~1.5 GB at K=16 on the 136.8 M sample (RESEARCH §B.2).
- Determinism (D-16): stable-sorted input order + `np.argmax` tie-break
  (smallest class ID wins) → bitwise-identical output for the same input.

## Stage 3 — Denoise (DS-ZAHA-P1-03, pre-chunking per D-13)

- `data_pre/zaha/utils/denoise.py::denoise_cloud(xyz, segment, cfg)`
- Method + parameters pulled at build time from the `## Final Decision`
  YAML block in
  `.planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/denoising_notes.md`.
- Current approved decision: `method: sor`, `nb_neighbors: 30`,
  `std_ratio: 2.0`, `max_drop_frac: 0.10`. Radius rejected at the visual
  gate (D-12 violation on roof-edge chunks).
- Pre-chunking (whole-building cloud) per D-13 so sparse tile borders do
  not skew per-tile outlier statistics.
- Aggregate drop cap ≤ 10 % enforced in orchestration per D-12 — the
  per-step dispatch uses `max_drop_frac=1.0` so the cap is applied once on
  the cumulative drop.

## Stage 4 — Chunk (DS-ZAHA-P1-04)

- `data_pre/zaha/utils/chunking.py::compute_chunks_by_facade(xyz, segment, ...)`
- **Facade-aware occupancy chunking** (D-06/D-08/D-10 supersession 2026-04-12
  — Plan 01-04 Task 3 v3): the default path projects facade-class points
  (Wall, Window, Door, Balcony, Molding, Deco, Column, Arch, Blinds —
  remapped IDs {0..7, 12}) onto a 1 m XY occupancy grid, runs
  `scipy.ndimage.label` connected-component labelling, assigns all non-facade
  points to the nearest component by XY distance transform, then recursively
  bisects oversized components along their longest XY axis until every piece
  falls within the point budget.
- Components with < 10 000 points are dropped (small appendages that would
  produce fragments too small for meaningful training).
- Budget: 1 000 000 points per chunk (D-07 supersession 2026-04-12). Under
  grid=0.04 + facade chunking the realised distribution is p25=644k,
  median=666k, p75=909k, max=999k across all 26 samples.
- The legacy axis-aligned grid path (`compute_chunks()`) is preserved and
  activated by CLI `--tile-xy` / `--overlap-xy` / `--z-mode` flags for
  ablations.
- Naming: `<sample>__c<idx:04d>` (D-10 / RESEARCH §F.6), 4-digit zero-pad
  enforced by `chunking.chunk_name`.

## Stage 5 — Normals (DS-ZAHA-P1-05, per-chunk per D-19)

- `data_pre/zaha/utils/normals.py::estimate_normals(chunk_xyz, NormalConfig(knn=30))`
- `open3d.geometry.KDTreeSearchParamKNN(knn=30)` — adaptive-radius PCA
  equivalent per D-17.
- Unoriented (sign-arbitrary eigenvectors) per §E.5 — downstream BFDataset
  treats normals as rotation-invariant feature vector. Orientation
  propagation would cost ~30 min across the full dataset for zero training
  benefit.
- D-18 bar: unit-length `[0.99, 1.01]` + no NaN. Violation raises
  `ValueError` — caller drops the chunk or widens `knn`.

## Stage 6 — Write (DS-ZAHA-P1-06)

- `data_pre/zaha/utils/layout.py::write_chunk_npys(out_dir, coord, segment, normal)`
- Strict dtype / shape / range / finiteness / unit-length enforcement;
  raises `ValueError` on any violation.
- Directory layout:
  `<output>/<split>/<sample>__c<idx>/{coord,segment,normal}.npy`.
- Per-file SHA256 hashes returned for the manifest audit trail.

## Stage 7 — Manifest + sanity (DS-ZAHA-P1-07, D-20/D-21/D-22)

- `data_pre/zaha/utils/manifest.py::write_manifest(output_root, manifest)` —
  RESEARCH §H.2 schema (`Manifest` → `SampleEntry` → `ChunkEntry`).
- Provenance fields: `schema_version`, `pipeline_version`, `commit_hash`
  (via `git rev-parse HEAD`, with `-dirty` suffix on uncommitted trees),
  `ran_at` (UTC ISO-8601), `ran_by` (`getpass.getuser()`), `host`
  (`platform.node()`), `grid_size` (locked 0.04), `void_drop_rule`
  (`winner_eq_0_drops_voxel`).
- `run_sanity_checks(manifest, expected_samples)` — three D-21 gates:
  (a) per-class histogram drift (> 15 pp → HARD FAIL),
  (b) chunk budget (any `point_count > 1_000_000` → HARD FAIL),
  (c) readme coverage (missing or extra samples → HARD FAIL).
- Hard-failure policy (D-22): any `HARD FAIL` string from `run_sanity_checks`
  triggers exit code 2 with partial output preserved for forensics.

## Parallelism

- `ProcessPoolExecutor(max_workers=args.workers)` over small samples
  (RESEARCH §I.1 reference: `data_pre/bf_edge_v3/scripts/rebuild_edge_dataset_inplace.py`).
- Large samples (`DEBY_LOD2_4906965`, `DEBY_LOD2_4959458`) run serially at
  the end to keep WSL RAM bounded. 4 workers × ~1.5 GB peak on the 136.8 M
  sample would exceed the free-RAM budget — serializing the giants avoids
  OOM without giving up parallel throughput on the 24 small samples.
- Per-sample idempotence via `<output>/<split>/.state/<sample>.json`.
  The sidecar records the source PCD's SHA256; re-entry skips samples whose
  hash matches and `status == 'done'`. `--force` overrides.
- On failure, the sidecar is written as `status: partial` so the next run
  can resume.

## Import order discipline

`open3d` must be imported before `numpy` / `pandas` / `scipy` on the ptv3
conda env — otherwise the loader trips the `libstdc++.so.6 GLIBCXX_3.4.29
not found` trap (RESEARCH §I.5). The orchestrator imports `open3d` as the
very first line of `build_zaha_chunks.py`, and each worker process
re-imports `open3d` at the top of `process_sample` so fork-and-spawn
semantics are both safe.
