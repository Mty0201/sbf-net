# data_pre/zaha — ZAHA offline preprocessing pipeline

**Phase 1** of the workstream `dataset-handover-s3dis-chas`. This package turns
raw ASCII PCD files from `/home/mty0201/data/ZAHA_pcd/` into per-chunk
`coord/segment/normal.npy` arrays under `/home/mty0201/data/ZAHA_chunked/`,
with a provenance manifest.

## Package layout

```
data_pre/zaha/
├── __init__.py
├── scripts/          # Entry-point scripts (parse → downsample → denoise → chunk → normals → write)
│   ├── _bootstrap.py # sys.path shim; mirrors data_pre/bf_edge_v3/scripts/_bootstrap.py
│   └── ...           # build_zaha_chunks.py added in Plan 04
├── utils/            # Reusable modules (pcd_parser, voxel_agg, denoise, chunking, normals, ...)
├── tests/            # pytest unit + integration tests (RED stubs until impl plans land)
└── docs/             # This README + denoising_notes.md + normals_notes.md (added in Plan 03)
```

## Pipeline stages (per CONTEXT.md)

1. **Parse** — streaming ASCII PCD line reader, drops `rgb` field unconditionally.
2. **grid=0.02 voxel downsample** — deterministic hash-partitioned external sort
   (plain Python dict ruled out on the 86 M / 136.8 M-point files — see Plan 02).
3. **VOID drop** — any voxel whose majority-vote label is raw class 0 (VOID)
   is removed after downsample. Remaining raw classes 1..16 are remapped to 0..15.
4. **Denoise** — research-gated method (Plan 03). Runs pre-chunking on the
   whole-building 0.02-downsampled cloud.
5. **Chunk** — axis-aligned box grid, fixed XY tile size, ≥ 2 m overlap, full Z.
   Row-major chunk ordering, deterministic bboxes. Budget ≤ 0.6 M pts/chunk.
6. **Normals** — per chunk (not whole building), adaptive-radius PCA best-effort.
   Unit-length float32 (N, 3), no NaN.
7. **Write** — `<split>/<sample>__c<idx>/{coord,segment,normal}.npy` +
   `ZAHA_chunked/manifest.json` provenance record.

## ⚠️ open3d import-order warning

**Every script in `data_pre/zaha/scripts/` and every `utils/` module that
touches `open3d` MUST start with `import open3d as o3d` BEFORE importing
`pandas`, `scipy`, or `sklearn`.**

```python
# CORRECT — open3d first
import open3d as o3d
import numpy as np
import pandas as pd
```

```python
# BROKEN — pandas/scipy first triggers GLIBCXX_3.4.29 conflict
import pandas as pd  # loads libstdc++.so.6 with an older ABI
import open3d as o3d  # fails with: version `GLIBCXX_3.4.29' not found
```

The `ptv3` conda env has a known ABI conflict where `scipy.special`'s
`libstdc++.so.6` load ordering breaks `open3d`'s import. Full failure mode:
`ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.29'
not found`. Solution: import `open3d` first, always. This pitfall is
documented in `01-RESEARCH.md §I.5 Pitfall 2`.

## Quick start

```bash
# Build chunks from the full dataset (script added in Plan 04)
python data_pre/zaha/scripts/build_zaha_chunks.py \
    --input /home/mty0201/data/ZAHA_pcd \
    --output /home/mty0201/data/ZAHA_chunked
```

## Test run

```bash
# Collect tests (Wave 0 gate — should report 26+ tests, 0 errors)
conda run -n ptv3 pytest data_pre/zaha/tests/ --collect-only -q

# Quick run (unit tests only, <10 s)
conda run -n ptv3 pytest data_pre/zaha/tests/ -x --tb=short

# Full suite
conda run -n ptv3 pytest data_pre/zaha/tests/ --tb=long
```

Until Plans 02–04 land, the tests are RED skeletons that fail with
`pytest.fail("not yet implemented — PLAN NN task K")`. This wires the
Nyquist feedback loop from Wave 0 instead of reporting "no tests ran".

## Cross-references

- `.planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/01-CONTEXT.md` — 22 locked implementation decisions (D-01 .. D-22)
- `.planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/01-RESEARCH.md` — empirical measurements + chosen implementation strategy
- `.planning/workstreams/dataset-handover-s3dis-chas/REQUIREMENTS.md` — DS-ZAHA-P1-01 .. DS-ZAHA-P1-07 acceptance criteria
- `data_pre/bf_edge_v3/scripts/_bootstrap.py` — the pattern `_bootstrap.py` mirrors
