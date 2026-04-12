# Plan 01-04 Summary

## Outcome

Plan 01-04 delivered the full ZAHA offline preprocessing pipeline: 26 raw PCD files
in, 140 chunked coord/segment/normal.npy triplets out, with a provenance manifest.

## What was built

**Task 1** — `layout.py` (per-chunk NPY writer) + `manifest.py` (schema, sanity
checks, D-22 hard-failure) + `test_output_layout.py` + `test_yaml.py`.

**Task 2** — `build_zaha_chunks.py` orchestrator: parse → voxel agg → SOR denoise
→ chunk → per-chunk normals → write NPYs → manifest.json → D-21 sanity gates.
ProcessPoolExecutor with spawn context, large-sample serial fallback, per-sample
idempotence sidecar. `test_e2e.py` (6 tests including golden + determinism).

**Task 3** — Grid 0.02→0.04 switch (D-14 supersession). Facade-aware occupancy
chunking replaced the axis-aligned grid: projects facade-class points (Wall,
Window, Door, Balcony, Molding, Deco, Column, Arch, Blinds — remapped IDs
{0..7, 12}) onto a 1 m XY occupancy grid, labels connected components via
`scipy.ndimage.label`, assigns non-facade points by distance transform, bisects
oversized components recursively. Budget cap raised from 600k to 1M (D-07
supersession). Components < 10k points dropped.

## Key decisions made during execution

| Decision | Rationale |
|----------|-----------|
| grid=0.04 (was 0.02) | Aligned with main-line training regime (CONTEXT D-14 supersession) |
| Facade-aware chunking (was axis-aligned grid) | Axis-aligned grid produced 740 chunks with 166 over-budget + 126 fragments; facade geometry follows building outline |
| FACADE_CLASS_IDS = {0-7, 12} | Wall/window/door/balcony surface elements define the building footprint; floor/terrain/roof excluded |
| Budget 1M (was 600k) | Under grid=0.04, typical chunks land 200k-900k; 1M is the genuine overflow bound |
| min_pts=10000 (was 5000) | Components < 10k are small appendages that produce fragments too small for training |
| Non-facade points assigned to nearest facade component | Nothing is lost; floor/roof/interior points follow the building outline |

## Pipeline output

- 26 samples, 140 chunks (4 dropped for <10k pts), 0 errors
- Point distribution: p25=644k, median=666k, p75=909k, max=999k
- 128/140 (91%) chunks in 200k-1M range
- Output at `/tmp/zaha_chunked/` (pending move to `/home/mty0201/data/ZAHA_chunked`)

## Test results

26 tests passed (pytest), including:
- 6 e2e tests (smallest sample pipeline, manifest schema, sanity gates, determinism, golden regression, missing-sample hard failure)
- Golden snapshot: `DEBY_LOD2_4907179__goldens.json` (1 chunk, 656k pts)

## Files modified/created

- `data_pre/zaha/utils/chunking.py` — facade-aware chunking + legacy grid preserved
- `data_pre/zaha/utils/manifest.py` — budget cap 600k→1M
- `data_pre/zaha/utils/layout.py` — per-chunk NPY writer (Task 1)
- `data_pre/zaha/utils/voxel_agg.py` — grid 0.02→0.04
- `data_pre/zaha/scripts/build_zaha_chunks.py` — full orchestrator
- `data_pre/zaha/tests/test_e2e.py` — 6 e2e tests
- `data_pre/zaha/tests/test_output_layout.py`, `test_yaml.py` — layout + YAML tests
- `data_pre/zaha/tests/golden/DEBY_LOD2_4907179__goldens.json` — golden snapshot
- `data_pre/zaha/docs/PIPELINE.md`, `README.md` — updated for facade chunking
