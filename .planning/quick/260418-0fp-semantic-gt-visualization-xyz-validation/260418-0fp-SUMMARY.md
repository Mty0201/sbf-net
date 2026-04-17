---
id: 260418-0fp
type: quick
title: Semantic GT visualization for validation sets
status: complete
commit: 113e10b
completed: 2026-04-17
---

# Quick Task 260418-0fp — Summary

## What shipped

- `scripts/viz/gt_color_export.py` — standalone numpy-only script that converts validation-set semantic GT to colored XYZ point clouds.
- `scripts/viz/__init__.py` (empty, makes the directory importable).

## How it works

For each of the three datasets (BF, S3DIS, ZAHA) the script:

1. Iterates `<data_root>/validation/*` (skips hidden `.state` etc.).
2. Loads `coord.npy` (Nx3 float) and `segment.npy` (N, or Nx1 int).
3. Maps each label to RGB via a shared 32-color Kelly+Glasbey palette. Label `-1` and out-of-range labels map to black `(0,0,0)`.
4. Writes `<out_root>/validation/<sample>.xyz` with lines `x y z R G B`.
5. Emits `<out_root>/legend.txt` with the label → (class_name, RGB) table.

Outputs land next to the originals:
- `/home/mty0201/data/BF_edge_chunk_npy_gt_viz/`
- `/home/mty0201/data/s3dis_gt_viz/`
- `/home/mty0201/data/ZAHA_chunked_gt_viz/`

CLI:
```
python scripts/viz/gt_color_export.py --dataset all              # full export
python scripts/viz/gt_color_export.py --dataset zaha --limit 1   # smoke test
python scripts/viz/gt_color_export.py --dataset bf --overwrite
```

## Smoke tests run this session

| Dataset | Sample                          | Points   | XYZ written | Legend written |
|---------|---------------------------------|----------|-------------|----------------|
| ZAHA    | DEBY_LOD2_4906981__c0000        | 940 322  | ✓           | ✓              |
| BF      | 010101                          | 515 839  | ✓           | ✓              |
| S3DIS   | Area_5_WC_1                     | 719 348  | ✓           | ✓              |

Verified unique RGB triples in the ZAHA sample exactly match the legend's colors for the classes present (floor / decoration / other_el / ignore).

## Not done (intentionally)

- The full export across all validation samples was **not** run — user runs it themselves at their convenience via `--dataset all`.
- Training-split exports: out of scope.
- No PLY / PCD; XYZ only.

## Notes

- Palette is stable by class index (class 0 → palette[0], etc.); if class order in any dataset changes, colors shift accordingly — re-emit `legend.txt`.
- Shape mismatches between `coord` and `segment` produce a `[skip]` warning rather than crashing.
- Hidden dirs like `.state` are filtered out of sample iteration.
