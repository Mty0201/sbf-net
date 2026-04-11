# ZAHA_chunked Dataset Format (Phase 1 Output)

Produced by `data_pre/zaha/scripts/build_zaha_chunks.py` per Phase 1 of the
`dataset-handover-s3dis-chas` workstream. This document describes the
on-disk contract — dtype, shape, range, and manifest schema — that downstream
training configs and the verifier depend on.

## Directory layout

```
/home/mty0201/data/ZAHA_chunked/
├── manifest.json                         # global manifest (schema version 1)
├── training/
│   ├── .state/
│   │   └── <sample>.json                 # per-sample idempotence sidecar
│   ├── DEBY_LOD2_4907173__c000/
│   │   ├── coord.npy      # float32 (N, 3) — post-denoise XYZ centroids
│   │   ├── segment.npy    # int32   (N,)   — remapped LoFG3 class [0, 15]
│   │   └── normal.npy     # float32 (N, 3) — unit-length PCA normals
│   ├── DEBY_LOD2_4907173__c001/
│   │   └── ...
│   └── ...
├── validation/
│   └── ...
└── test/
    └── ...
```

Split assignments come from `/home/mty0201/data/ZAHA_pcd/readme.txt`: 19
training, 4 validation, 3 test (26 samples total per CONTEXT.md §A).

## Per-chunk NPY contract

| File          | dtype    | shape    | range / invariants                                      |
| ------------- | -------- | -------- | ------------------------------------------------------- |
| `coord.npy`   | float32  | (N, 3)   | finite; post-denoise grid-0.02 centroids                |
| `segment.npy` | int32    | (N,)     | values in [0, 15]; remapped LoFG3 (D-02), VOID dropped  |
| `normal.npy`  | float32  | (N, 3)   | unit-length `0.99 <= ||n|| <= 1.01`; finite             |

Constraints:

- Per D-03: every chunk has `N >= 1`. Empty chunks are dropped at build time.
- Per D-07: every chunk has `N <= 600_000`. Tiles that exceed the budget
  trigger `z_mode='band:{depth}'` fallback or the build fails.
- Per D-18: normals are unit-length within a ±1% tolerance. Unoriented PCA
  (sign-arbitrary eigenvectors) is the design choice — downstream BFDataset
  treats normals as an unsigned rotation-invariant feature vector (§E.5).

All three arrays in a chunk directory have matching first-axis length `N`.
The writer (`data_pre/zaha/utils/layout.py::write_chunk_npys`) raises
`ValueError` on any dtype / shape / finiteness / range violation — D-22
hard-failure policy means the orchestrator exits non-zero on any such
corruption with partial output preserved for forensics.

## Segment value space

`segment.npy` holds the LoFG3 class index in `[0, 15]` AFTER the Phase 1
voxel-aggregate stage applies:

1. VOID drop (D-01) — any voxel whose majority vote lands on raw class `0`
   is removed at the voxel-aggregate step. No VOID voxels reach the chunker.
2. Raw-to-remapped shift (D-02) — raw classes `1..16` are shifted to `0..15`
   by subtracting one.
3. Tie-break at majority vote (D-16) — smallest class ID wins on ties,
   guaranteed by `np.argmax` over a stable-sorted key space.

The LoFG2 5-bucket collapse (`lofg3_to_lofg2.yaml`) is a Phase 1b /
config-time concern and is NOT applied during Phase 1 preprocessing. See
`.planning/workstreams/dataset-handover-s3dis-chas/phases/01-zaha-offline-preprocessing-pipeline/lofg3_to_lofg2.yaml`
for the frozen YAML map; the `13 → 4` (OuterCeilingSurface → other_el) entry
is locked by CONTEXT.md D-04.

## `manifest.json`

Schema version: `1`. See
`data_pre/zaha/utils/manifest.py::Manifest` for the authoritative dataclass
layout. Top-level keys (RESEARCH §H.2):

```json
{
  "schema_version": 1,
  "pipeline_version": "1.0.0",
  "commit_hash": "<git-sha>[-dirty]",
  "ran_at": "<iso-8601-utc>",
  "ran_by": "<getpass.getuser()>",
  "host": "<platform.node()>",
  "grid_size": 0.02,
  "void_drop_rule": "winner_eq_0_drops_voxel",
  "denoising": { ... },
  "normal_estimation": { ... },
  "chunking": { ... },
  "dataset_stats": { ... },
  "samples": [ { ... } ]
}
```

Per-sample entries (`SampleEntry`) carry the raw / post-void-drop / post-denoise
counters, the axis-aligned bbox, elapsed seconds, peak RSS in MB, the raw
and final class histograms, and the list of `ChunkEntry` rows. Per-chunk
entries (`ChunkEntry`) carry the linear `chunk_idx`, the `<sample>__c<idx>`
directory name, the XY tile coordinates, the tile bbox, the point count,
the per-chunk class histogram, and the `coord_sha256 / segment_sha256 /
normal_sha256` audit hashes used by the determinism regression test.

## Reproducing from scratch

```bash
conda run -n ptv3 python data_pre/zaha/scripts/build_zaha_chunks.py \
  --input /home/mty0201/data/ZAHA_pcd \
  --output /home/mty0201/data/ZAHA_chunked \
  --workers 4
```

The pipeline is deterministic by construction (RESEARCH §B.5 / §F.1): given
the same input PCD and the same commit hash, two runs produce bitwise-equal
`coord.npy`, `segment.npy`, and `normal.npy` files across every chunk. The
determinism regression is exercised by
`data_pre/zaha/tests/test_e2e.py::test_determinism` on the smallest training
sample (`DEBY_LOD2_4907179`).
