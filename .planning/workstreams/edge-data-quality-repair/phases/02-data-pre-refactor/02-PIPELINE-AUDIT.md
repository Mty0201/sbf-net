# data_pre/bf_edge_v3 Pipeline Audit

**Date:** 2026-04-06
**Scope:** Current-state documentation of the existing pipeline — what it does, how it runs, how data flows.

---

## 1. Pipeline Overview

### Total goal

`data_pre/bf_edge_v3` generates **per-point edge supervision data** (`edge.npy`) for SBF-Net training. Given a 3D point cloud with semantic labels, it:
1. Detects semantic boundary regions
2. Clusters boundary centers into local groups
3. Fits geometric support elements (line/polyline) to each cluster
4. Computes per-point supervision: direction, distance, weight, and validity relative to the nearest support

### Stage summary

| Stage | Name | Core module | Script(s) | Input | Output |
|-------|------|-------------|-----------|-------|--------|
| 1 | Boundary centers | `boundary_centers_core.py` | `build_boundary_centers.py` | `coord.npy`, `segment.npy`, `normal.npy`(opt) | `boundary_centers.npz` |
| 2 | Local clusters | `local_clusters_core.py` | `build_local_clusters.py` | `boundary_centers.npz` | `local_clusters.npz` |
| 3 | Support fitting | `supports_core.py` | `fit_local_supports.py` | `boundary_centers.npz`, `local_clusters.npz` | `supports.npz` |
| 4 | Pointwise supervision | `pointwise_core.py` | `build_pointwise_edge_supervision.py` | `coord.npy`, `segment.npy`, `supports.npz` | `edge_*.npy` files |

**Additional scripts** (dataset-level wrappers and post-processing):
- `build_support_dataset_v3.py` — Runs stages 1-3 in-memory for all scenes, outputs only `supports.npz`
- `build_edge_dataset_v3.py` — Runs stage 4 for all scenes, produces compact `edge.npy` (5 columns)
- `convert_edge_vec_to_dir_dist.py` — Post-hoc format conversion: `edge.npy` from 5-col `[vec, support, valid]` to 6-col `[dir, dist, support, valid]`
- `add_support_id_to_edge_dataset.py` — Supplements an existing edge dataset with `edge_support_id.npy`
- `diagnose_net01.py` — Phase 1 diagnosis script (not part of the generation pipeline)

---

## 2. Entry Points

### Per-scene invocation (4 separate commands)

There is **no unified pipeline runner**. Users run each stage as a separate CLI command:

```bash
cd data_pre/bf_edge_v3/scripts

# Stage 1
python build_boundary_centers.py --scene <scene_dir> --output <out_dir>

# Stage 2
python build_local_clusters.py --input <scene_dir> --output <out_dir> \
    --eps 0.08 --min-samples 8 --denoise-knn 8

# Stage 3
python fit_local_supports.py --input <scene_dir> --output <out_dir>

# Stage 4
python build_pointwise_edge_supervision.py --input <scene_dir> --output <out_dir> \
    --support-radius 0.08
```

Note: Stage 1 uses `--scene`, stages 2-4 use `--input`. Inconsistent argument naming.

### Dataset-level invocation (2 commands)

For batch processing all scenes in a dataset:

```bash
# Stages 1-3 combined (in-memory, no intermediate .npz written)
python build_support_dataset_v3.py --input <dataset_root>

# Stage 4 separately
python build_edge_dataset_v3.py --input <dataset_root> --output <edge_dataset_root>
```

`build_support_dataset_v3.py` runs stages 1-3 **in-memory** per scene, writing only `supports.npz` and cleaning up any intermediate files. This is the "production" path.

`build_edge_dataset_v3.py` then reads `supports.npz` + scene files and produces `edge.npy` (5-col compact format: `[vec_x, vec_y, vec_z, support_weight, valid]`).

### Format conversion (optional post-step)

```bash
python convert_edge_vec_to_dir_dist.py --input <dataset_root> --output <converted_root>
```

Converts 5-col `[vec, support, valid]` to 6-col `[dir, dist, support, valid]`. This is the format used by current training configs.

---

## 3. Stage-by-Stage Breakdown

### Stage 1: Boundary Centers

**File:** `core/boundary_centers_core.py`
**Script:** `scripts/build_boundary_centers.py`

**Input:**
- `coord.npy` (N, 3) float32 — point coordinates
- `segment.npy` (N,) int32 — semantic labels
- `normal.npy` (N, 3) float32 — optional surface normals

**Processing:**
1. `build_knn_graph()` — Build kNN graph (k=32 default) using `cKDTree`
2. `detect_boundary_candidates()` — For each point, check if neighbors have different semantic labels. If cross-ratio >= `min_cross_ratio` (0.15), mark as candidate. Records `(point_index, semantic_pair, cross_ratio)`
3. `estimate_boundary_center()` — For each candidate:
   - Split neighbors into two semantic sides (label_a, label_b)
   - Require `min_side_points` (4) on each side
   - Compute center = midpoint of the two side centroids
   - Compute center_normal = separation direction
   - Compute center_tangent via PCA in the plane orthogonal to center_normal (fallback: cross product with surface normal)
   - Compute confidence = weighted combination of cross_ratio (0.45), side_balance (0.30), separation_score (0.15), tangent_score (0.10)

**Output:** `boundary_centers.npz`
- `center_coord` (M, 3) float32
- `center_normal` (M, 3) float32
- `center_tangent` (M, 3) float32
- `semantic_pair` (M, 2) int32 — sorted label pair
- `source_point_index` (M,) int32 — maps back to scene point
- `confidence` (M,) float32

**Visualization artifacts:** `boundary_centers.xyz`, `boundary_candidates.xyz`

**Key parameters:**
| Parameter | Default | Source |
|-----------|---------|--------|
| `k` | 32 | argparse `--k` |
| `min_cross_ratio` | 0.15 | argparse `--min-cross-ratio` |
| `min_side_points` | 4 | argparse `--min-side-points` |
| `ignore_index` | -1 | argparse `--ignore-index` |

---

### Stage 2: Local Clusters

**File:** `core/local_clusters_core.py`
**Script:** `scripts/build_local_clusters.py`

**Input:** `boundary_centers.npz` (all fields from Stage 1)

**Processing:**
1. `cluster_boundary_centers()` — Main entry:
   - Group boundary centers by `semantic_pair`
   - For each semantic pair group:
     a. `spatial_dbscan()` — Run sklearn DBSCAN with **fixed global eps** on center_coord. Points not reachable within eps are labeled noise (label=-1).
     b. For each surviving cluster:
        - `lightweight_denoise_cluster()` — kNN-based outlier removal within the cluster. Computes per-point mean kNN distance, removes points where spacing > `max(median * sparse_distance_ratio, median + sparse_mad_scale * MAD)`. Safeguards: max 20% removal, min `min_samples` kept.
        - `compute_cluster_trigger_metrics()` — SVD-based linearity, tangent coherence, bbox anisotropy
        - `should_trigger_split()` — Conservative flag for clusters that may need direction-based splitting in Stage 3

**Output:** `local_clusters.npz`
- `center_index` (K,) int32 — indices into boundary_centers that survived clustering
- `cluster_id` (K,) int32 — per-surviving-center cluster assignment
- `semantic_pair` (C, 2) int32 — per-cluster semantic pair
- `cluster_size` (C,) int32
- `cluster_centroid` (C, 3) float32
- `cluster_trigger_flag` (C,) uint8

**Visualization:** `clustered_boundary_centers.xyz`

**Key parameters:**
| Parameter | Default | Source |
|-----------|---------|--------|
| `eps` | 0.08 | argparse `--eps` |
| `min_samples` | 8 | argparse `--min-samples` |
| `denoise_knn` | 8 | argparse `--denoise-knn` |
| `sparse_distance_ratio` | 1.75 | argparse `--sparse-distance-ratio` |
| `sparse_mad_scale` | 3.0 | argparse `--sparse-mad-scale` |

**This is the NET-01 primary bottleneck.** Fixed `eps=0.08` causes 18.7pp survival gap between dense and sparse regions.

---

### Stage 3: Support Fitting

**File:** `core/supports_core.py` (~530 lines)
**Script:** `scripts/fit_local_supports.py`

**Input:** `boundary_centers.npz` + `local_clusters.npz`

**Processing:** `build_supports_payload()` — For each cluster:
1. **Non-trigger clusters:** Try line support first (`fit_line_support()`). If residual > `line_residual_th` (0.01), fall back to polyline support (`fit_polyline_support()`).
2. **Trigger clusters:** Direction + spatial re-grouping:
   - Split into sub-groups by direction coherence
   - Classify sub-groups: main edge / fragment / bad
   - Attempt to attach fragments to nearest main group
   - Merge compatible main groups
   - Absorb endpoint straggler points
   - Each final sub-cluster gets line or polyline support fitting

Support geometry is represented as line segments (start, end pairs). A line support = 1 segment. A polyline support = multiple segments.

**Output:** `supports.npz`
- `support_id` (S,) int32
- `semantic_pair` (S, 2) int32
- `support_type` (S,) uint8 — 0=line, 2=polyline
- `line_start` (S, 3) float32, `line_end` (S, 3) float32 — overall line endpoints
- `segment_offset` (S,) int32, `segment_length` (S,) int32 — index into segment arrays
- `segment_start` (T, 3) float32, `segment_end` (T, 3) float32 — all segment endpoints

**Visualization:** `support_geometry.xyz`, `trigger_group_classes.xyz`

**Key parameters:**
| Parameter | Default | Source |
|-----------|---------|--------|
| `line_residual_th` | 0.01 | argparse `--line-residual-th` |
| `min_cluster_size` | 8 | argparse `--min-cluster-size` |
| `max_polyline_vertices` | 32 | argparse `--max-polyline-vertices` |
| (20+ internal params) | (in `DEFAULT_FIT_PARAMS`) | hardcoded in `supports_core.py` |

`DEFAULT_FIT_PARAMS` contains ~20 internal parameters for trigger re-grouping, fragment classification, main group merging, and endpoint absorption. These are **not exposed on CLI** — they live as a dict in `supports_core.py` and are consumed by `build_runtime_params()` in the script.

---

### Stage 4: Pointwise Edge Supervision

**File:** `core/pointwise_core.py`
**Script:** `scripts/build_pointwise_edge_supervision.py`

**Input:** `coord.npy`, `segment.npy`, `supports.npz`

**Processing:** `build_pointwise_edge_supervision()`
1. `build_label_to_supports()` — Reverse index: semantic label → candidate support IDs
2. For each unique semantic label in the scene:
   - Get all scene points with that label
   - Get all supports whose `semantic_pair` includes that label
   - For each candidate support, compute closest point on support geometry (`closest_points_to_support()` → iterates segments)
   - Keep the closest support within `support_radius`
   - Record: edge_dist, edge_vec, edge_valid, edge_support_id
3. `build_edge_support()` — Gaussian weight: `exp(-dist²/(2σ²))` where `σ = support_radius / 2.0 = 0.04`

**Output (per-scene arrays):**
- `edge_dist.npy` (N,) float32 — distance to nearest support
- `edge_dir.npy` (N, 3) float32 — normalized direction to support
- `edge_valid.npy` (N,) uint8 — 1 if valid support within radius
- `edge_support_id.npy` (N,) int32 — which support (-1 if invalid)
- `edge_vec.npy` (N, 3) float32 — raw displacement vector
- `edge_support.npy` (N,) float32 — Gaussian weight
- `edge_mask.npy` (N,) uint8 — legacy alias for edge_valid
- `edge_strength.npy` (N,) float32 — legacy alias for edge_support

**Visualization:** `edge_supervision.xyz`

**Key parameters:**
| Parameter | Default | Source |
|-----------|---------|--------|
| `support_radius` | 0.08 | argparse `--support-radius` |
| `sigma` | 0.04 | **hardcoded** as `support_radius / 2.0` in `build_edge_support()` |
| `ignore_index` | -1 | argparse `--ignore-index` |

**This is the NET-01 secondary bottleneck.** Fixed `sigma=0.04` produces 1.3-7.0pp valid yield gap.

---

## 4. Data Artifacts and File Flow

### File flow diagram

```
Scene inputs                    Stage 1              Stage 2              Stage 3              Stage 4
─────────────────              ─────────            ─────────            ─────────            ─────────
coord.npy ──────────────┐
segment.npy ─────────────┼──→ boundary_centers ──→ local_clusters ──→ supports ──→ edge_*.npy
normal.npy (opt) ────────┘      .npz                  .npz               .npz       edge.npy (compact)
                                                                                   
                                  ↓                     ↓                  ↓           ↓
                            boundary_centers.xyz  clustered_bc.xyz  support_geo.xyz  edge_supervision.xyz
                            boundary_cand.xyz                      trigger_gc.xyz
```

### Cross-stage field consumption

| Producer stage | Field | Consumer stage | How consumed |
|----------------|-------|----------------|--------------|
| Scene | `coord.npy` | Stage 1 | kNN graph for boundary detection |
| Scene | `coord.npy` | Stage 4 | Point coordinates for support distance computation |
| Scene | `segment.npy` | Stage 1 | Semantic labels for boundary candidate detection |
| Scene | `segment.npy` | Stage 4 | Labels for support→point candidate matching |
| Stage 1 | `center_coord` | Stage 2 | DBSCAN clustering coordinates |
| Stage 1 | `center_tangent` | Stage 2 | Trigger metric computation |
| Stage 1 | `semantic_pair` | Stage 2 | Per-group DBSCAN |
| Stage 1 | `center_coord`, `center_tangent`, `semantic_pair`, `confidence` | Stage 3 | Support fitting uses all center fields via `center_index` |
| Stage 2 | `center_index` | Stage 3 | Which centers survived clustering |
| Stage 2 | `cluster_id`, `cluster_size` | Stage 3 | Cluster membership |
| Stage 2 | `cluster_trigger_flag` | Stage 3 | Trigger clusters get special re-grouping |
| Stage 3 | `semantic_pair` | Stage 4 | Support→point candidate matching |
| Stage 3 | `segment_offset`, `segment_length`, `segment_start`, `segment_end` | Stage 4 | Closest-point-on-support geometry |
| Stage 3 | `line_start`, `line_end` | Stage 4 | Fallback for zero-segment supports |

### Compact edge format

`build_edge_dataset_v3.py` produces `edge.npy` with 5 columns: `[vec_x, vec_y, vec_z, support_weight, valid]`.

After `convert_edge_vec_to_dir_dist.py`, it becomes 6 columns: `[dir_x, dir_y, dir_z, dist, support_weight, valid]`.

The 6-column format is what current training configs (`project/dataset/`) consume.

---

## 5. Config and Parameter Flow

### Where parameters are defined

| Location | What's defined | How consumed |
|----------|---------------|--------------|
| `scripts/*.py` argparse defaults | Stage 1-4 CLI params | Parsed at script entry, passed to core functions |
| `supports_core.py::DEFAULT_FIT_PARAMS` | 20+ internal Stage 3 params | `build_runtime_params()` in script merges with CLI params |
| `pointwise_core.py::build_edge_support()` | `sigma = support_radius / 2.0` | Hardcoded formula — no override mechanism |

### Global vs stage-specific

| Parameter | Used in | Scope |
|-----------|---------|-------|
| `ignore_index` | Stage 1, Stage 4 | Shared (but passed independently to each CLI) |
| `support_radius` | Stage 4 | Stage-specific (also determines sigma) |
| `eps` | Stage 2 | Stage-specific |
| `min_samples` | Stage 2 | Stage-specific |
| `k` (kNN) | Stage 1 | Stage-specific |
| `min_cross_ratio` | Stage 1 | Stage-specific |

**There is no config file.** All parameters are CLI arguments with hardcoded defaults. The dataset-level scripts (`build_support_dataset_v3.py`) duplicate all Stage 1-3 argparse definitions.

### Parameter passing paths

- **Per-scene scripts:** Each script gets its own argparse → passes to core function directly
- **Dataset scripts:** `build_support_dataset_v3.py` has a superset of all Stage 1-3 argparse params. `build_edge_dataset_v3.py` has Stage 4 params only.
- **No shared config object** — each script parses independently, params cannot be passed across stages

---

## 6. Structural Pain Points

### A. No unified pipeline runner

Stages must be run as separate CLI commands (or use `build_support_dataset_v3.py` which hardwires stages 1-3 in-memory). There's no way to run "the whole pipeline" with a single config.

### B. Parameter duplication

`build_support_dataset_v3.py` duplicates all Stage 1-3 argparse definitions. `build_runtime_params()` is copy-pasted between `fit_local_supports.py` and `build_support_dataset_v3.py`. If a parameter is added to one, it must be manually added to the other.

### C. No config file

Parameters are CLI args with hardcoded defaults. No YAML/JSON config. Cannot version-control a "run configuration" or compare configs between experiment runs.

### D. Hardcoded sigma

`sigma = support_radius / 2.0` in `pointwise_core.py:146` is not parameterizable. To make sigma density-adaptive, the formula must be refactored to accept an external sigma or a per-point sigma array.

### E. _bootstrap.py sys.path hack

`scripts/*.py` all start with `from _bootstrap import ensure_bf_edge_v3_root_on_path`. This modifies `sys.path` at import time to make `core/` and `utils/` importable. Breaks if scripts are invoked from a different working directory. Not a proper Python package.

### F. Inconsistent CLI argument names

Stage 1 uses `--scene`, stages 2-4 use `--input`. No consistency.

### G. No intermediate validation

No stage checks the quality or schema of its predecessor's output. If Stage 1 produces empty `boundary_centers.npz`, Stage 2 will silently produce empty clusters, Stage 3 empty supports, Stage 4 all-invalid edge data. Failures cascade silently.

### H. In-memory vs on-disk inconsistency

`build_support_dataset_v3.py` runs stages 1-3 in-memory (no intermediate .npz written — in fact it **deletes** intermediate files via `cleanup_scene_dir()`). The per-scene scripts write intermediates. The diagnosis script (`diagnose_net01.py`) requires the per-scene intermediates to exist. These two modes are not reconciled.

### I. No density information flows between stages

Stage 2 (DBSCAN) uses fixed `eps` regardless of local point density. Stage 4 uses fixed `sigma` regardless of local spacing. There is no mechanism to compute density once and pass it downstream. The diagnosis script computes density independently — this logic is not in the pipeline.

### J. Support core internal params not configurable

`DEFAULT_FIT_PARAMS` (20+ params) in `supports_core.py` is a frozen dict. Only 3 of these are exposed on CLI (`line_residual_th`, `min_cluster_size`, `max_polyline_vertices`). The rest require code changes.

---

## 7. File/Function Map

### Directory structure

```
data_pre/bf_edge_v3/
├── __init__.py                          (empty)
├── core/
│   ├── __init__.py                      (empty)
│   ├── boundary_centers_core.py         Stage 1 logic
│   ├── local_clusters_core.py           Stage 2 logic
│   ├── supports_core.py                 Stage 3 logic (~530 lines)
│   └── pointwise_core.py               Stage 4 logic
├── scripts/
│   ├── __init__.py                      (empty)
│   ├── _bootstrap.py                    sys.path hack
│   ├── build_boundary_centers.py        Stage 1 entry (per-scene)
│   ├── build_local_clusters.py          Stage 2 entry (per-scene)
│   ├── fit_local_supports.py            Stage 3 entry (per-scene)
│   ├── build_pointwise_edge_supervision.py  Stage 4 entry (per-scene)
│   ├── build_support_dataset_v3.py      Stages 1-3 batch (dataset-level)
│   ├── build_edge_dataset_v3.py         Stage 4 batch (dataset-level)
│   ├── convert_edge_vec_to_dir_dist.py  Post-hoc format conversion
│   ├── add_support_id_to_edge_dataset.py  Supplement existing dataset
│   └── diagnose_net01.py               Phase 1 diagnosis (not pipeline)
└── utils/
    ├── __init__.py                      (empty)
    ├── common.py                        EPS, normalize_vector/rows, save_xyz/npz
    └── stage_io.py                      load_scene, load_boundary_centers, load_local_clusters, collect_stage_tasks
```

### Key functions per module

**`boundary_centers_core.py`:**
- `build_knn_graph(coord, k)` → kNN index
- `detect_boundary_candidates(segment, knn_index, ignore_index, min_cross_ratio)` → candidates dict
- `estimate_boundary_center(...)` → single center record or None
- `build_boundary_centers(scene, k, min_cross_ratio, min_side_points, ignore_index)` → (candidates, centers_payload, meta)
- `export_boundary_centers_npz()`, `export_candidate_xyz()`, `export_centers_xyz()`

**`local_clusters_core.py`:**
- `spatial_dbscan(coords, eps, min_samples)` → labels
- `build_knn(coords, k)` → kNN distances
- `lightweight_denoise_cluster(coords, ...)` → (keep_mask, stats)
- `compute_cluster_trigger_metrics(coords, tangents)` → metrics dict
- `should_trigger_split(metrics, params)` → bool
- `cluster_boundary_centers(boundary_centers, eps, min_samples, denoise_knn, ...)` → (payload, meta)
- `export_npz()`, `export_clustered_boundary_centers_xyz()`

**`supports_core.py`:**
- `fit_line_support(coords, tangents)` → support record
- `fit_polyline_support(coords, tangent, ...)` → support record
- `build_supports_payload(boundary_centers, local_clusters, params)` → (payload, meta, debug)
- `export_npz()`, `export_support_geometry_xyz()`, `export_trigger_group_classes_xyz()`
- `DEFAULT_FIT_PARAMS` — dict of 20+ internal parameters

**`pointwise_core.py`:**
- `load_supports(input_dir)` → supports dict
- `closest_point_on_segment(points, seg_start, seg_end)` → (closest, dist)
- `closest_points_to_support(points, support_id, supports)` → (closest, dist)
- `build_label_to_supports(semantic_pair)` → label→support_ids dict
- `build_edge_support(edge_dist, edge_valid, support_radius)` → (weights, sigma)
- `build_pointwise_edge_supervision(scene, supports, support_radius, ignore_index)` → (payload, meta)
- `export_edge_arrays()`, `export_edge_supervision_xyz()`

**`stage_io.py`:**
- `load_scene(scene_dir, with_optional)` → scene dict
- `load_boundary_centers(input_dir)` → boundary_centers dict
- `load_local_clusters(input_dir)` → local_clusters dict
- `contains_stage_input(path, required_files)` → bool
- `collect_stage_tasks(input_path, output_path, stage_file, required_files)` → list of (in, out) pairs

**`common.py`:**
- `EPS = 1e-8`
- `normalize_vector(vec)`, `normalize_rows(arr)`
- `save_xyz(path, data, fmt)`, `save_npz(path, payload)`

---

*Audit based on current repository state as of 2026-04-06*
