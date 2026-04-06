# Cross-Stage Behavioral Contracts: bf_edge_v3 Pipeline

**Date:** 2026-04-07
**Phase:** 02-behavioral-audit-and-module-restructure, Plan 01
**Purpose:** Documents implicit cross-stage behavioral contracts -- where one stage assumes specific properties of another stage's output. These contracts are currently enforced only by convention, not by validation code.

---

## Contract: Stage 1 -> Stage 2

**Coupling type:** NPZ -> NPZ (via `boundary_centers.npz`)

**NPZ fields consumed by Stage 2:**

| Field | Shape | Dtype | Consumed by |
|---|---|---|---|
| `center_coord` | (M, 3) | float32 | `cluster_boundary_centers()` -- used for DBSCAN spatial clustering and centroid computation |
| `center_tangent` | (M, 3) | float32 | `compute_cluster_trigger_metrics()` -- used for tangent coherence metric |
| `semantic_pair` | (M, 2) | int32 | `cluster_boundary_centers()` -- used for per-pair grouping before DBSCAN |
| `confidence` | (M,) | float32 | Not consumed by Stage 2 core logic; passed through for visualization export |

**Semantic invariants:**
- All four arrays are aligned by index: row `i` of each array describes the same boundary center
- M >= 0 (Stage 2 handles M=0 gracefully by producing empty output)
- `center_coord` values are in scene-unit 3D coordinates (same coordinate system as input `coord.npy`)
- `center_tangent` vectors are normalized (unit length or zero); Stage 2's `normalize_rows()` call in `compute_cluster_trigger_metrics()` re-normalizes defensively
- `semantic_pair[i]` is a sorted pair `[label_a, label_b]` where `label_a < label_b`

**Hidden assumptions:**
- Stage 2 groups by exact `semantic_pair` match (via `np.all(semantic_pair == pair, axis=1)`), so pair sorting must be consistent
- Stage 2 assumes `center_coord` density is appropriate for the configured DBSCAN `eps` -- if Stage 1 produces centers too sparse or too dense for the eps, DBSCAN silently produces poor clusters

**Risk level:** Low. Schema is simple and shapes are validated by `stage_io.load_boundary_centers()`.

---

## Contract: Stage 1 -> Stage 3 (read-back)

**Coupling type:** NPZ -> NPZ (via `boundary_centers.npz`, accessed through Stage 2's `center_index`)

**NPZ fields consumed by Stage 3:**

| Field | Shape | Dtype | Consumed by |
|---|---|---|---|
| `center_coord` | (M, 3) | float32 | `rebuild_cluster_records()` -> indexed by `center_index` from Stage 2; used as point cloud for fitting |
| `center_tangent` | (M, 3) | float32 | `rebuild_cluster_records()` -> indexed by `center_index`; used for tangent-based regrouping |
| `confidence` | (M,) | float32 | `rebuild_cluster_records()` -> indexed by `center_index`; averaged for `cluster_confidence` |

**Critical coupling:** Stage 3 uses `center_index[k]` values from Stage 2 to index directly into Stage 1's arrays:
```
points = center_coord[record["center_indices"]]    # lines 1132
tangents = center_tangent[record["center_indices"]]  # lines 1133
```
This means `center_index[k]` must be a valid index into Stage 1's arrays -- i.e., `0 <= center_index[k] < M`.

**Preservation requirement:** Stage 1's arrays must never be reordered, filtered, or modified between Stage 1 output and Stage 3 input. The NPZ serialization format preserves array order, so this holds as long as:
1. No intermediate process modifies `boundary_centers.npz` between stages
2. The in-memory bypass path passes the same dict object (which it does -- see in-memory bypass contract below)

**Validation present:** `rebuild_cluster_records()` (lines 89-90) validates:
```python
if np.any(center_index < 0) or np.any(center_index >= center_coord.shape[0]):
    raise ValueError("center_index out of range")
```

**Risk level:** Low as long as NPZ serialization is preserved. High if any refactor introduces a different intermediate format or reorders Stage 1 arrays.

---

## Contract: Stage 2 -> Stage 3 (trigger flag semantics)

**Coupling type:** NPZ -> NPZ (via `local_clusters.npz`)

**NPZ fields consumed by Stage 3:**

| Field | Shape | Dtype | Consumed by |
|---|---|---|---|
| `center_index` | (K,) | int32 | `rebuild_cluster_records()` -- maps cluster members to Stage 1 array indices |
| `cluster_id` | (K,) | int32 | `rebuild_cluster_records()` -- assigns each center to a cluster |
| `semantic_pair` | (C, 2) | int32 | `rebuild_cluster_records()` -- per-cluster semantic pair, indexed by cluster_id |
| `cluster_trigger_flag` | (C,) | uint8 | `rebuild_cluster_records()` -> `build_support_record()` -- routes cluster to standard or trigger path |
| `cluster_size` | (C,) | int32 | `rebuild_cluster_records()` -- validated against observed member count |
| `cluster_centroid` | (C, 3) | float32 | `rebuild_cluster_records()` -- passed through for cluster record |

**Trigger flag semantics:**
- `cluster_trigger_flag[c] > 0` means: "Cluster `c` likely contains multiple edge branches. Stage 3 should use direction-based regrouping (`regroup_trigger_cluster()`) instead of direct line/polyline fitting."
- `cluster_trigger_flag[c] == 0` means: "Cluster `c` is well-behaved enough for direct fitting via `build_standard_support_record()`."

**Hidden assumptions:**
- If trigger flag is set, Stage 3 assumes the cluster's tangent vectors (from Stage 1 via `center_index`) are meaningful enough for direction-based grouping. If Stage 2's trigger criteria were loosened to flag more clusters, Stage 3's `group_tangents()` might receive tangent vectors that are too noisy for reliable direction clustering.
- The trigger threshold is conservative (AND of three conditions in Stage 2's `should_trigger_split()`), so false positives are rare. But the contract is implicit -- there is no formal definition of "what tangent quality is sufficient for regrouping."
- Stage 3 trusts that `cluster_trigger_flag` indices align with `semantic_pair`, `cluster_size`, and `cluster_centroid` arrays. Validated at line 95: `if cid < 0 or cid >= semantic_pair.shape[0]: raise ValueError(...)`.

**Risk level:** Medium. If Stage 2 trigger criteria change (e.g., different linearity threshold), Stage 3 regrouping may receive clusters it cannot handle effectively. The failure mode is not a crash but degraded fitting quality.

---

## Contract: Stage 3 -> Stage 4 (support schema)

**Coupling type:** NPZ -> NPZ (via `supports.npz`)

**NPZ fields consumed by Stage 4 (via `load_supports()`):**

| Field | Shape | Dtype | Consumed by | Role |
|---|---|---|---|---|
| `support_id` | (S,) | int32 | Shape validation and iteration | Identifies each support |
| `semantic_pair` | (S, 2) | int32 | `build_label_to_supports()` | Builds label -> support reverse index |
| `segment_offset` | (S,) | int32 | `closest_points_to_support()` | Start index into segment arrays for support `i` |
| `segment_length` | (S,) | int32 | `closest_points_to_support()` | Number of segments for support `i` |
| `segment_start` | (T, 3) | float32 | `closest_point_on_segment()` | Segment start endpoints (T = total segments across all supports) |
| `segment_end` | (T, 3) | float32 | `closest_point_on_segment()` | Segment end endpoints |
| `line_start` | (S, 3) | float32 | `closest_points_to_support()` | Fallback single-segment start for supports with segment_length <= 0 |
| `line_end` | (S, 3) | float32 | `closest_points_to_support()` | Fallback single-segment end for supports with segment_length <= 0 |

**Invariant:** Every support has EITHER:
- Valid segments: `segment_length[i] > 0` and `segment_offset[i]` through `segment_offset[i] + segment_length[i] - 1` are valid indices into `segment_start`/`segment_end`, OR
- Valid line endpoints: `line_start[i]` and `line_end[i]` are meaningful 3D coordinates

**No validation enforces this invariant.** Stage 4's `closest_points_to_support()` (lines 106-111) handles the fallback:
```python
if length <= 0:
    return closest_point_on_segment(points, line_start, line_end)
```

**Risk:** If Stage 3 produces a support with `segment_length <= 0` AND zero-valued `line_start`/`line_end` (all zeros), Stage 4 computes distances to the origin, producing incorrect supervision. The failure is silent -- no error, no warning, just wrong distances.

**Additional Stage 4 schema assumptions:**
- `segment_offset[i] + segment_length[i] <= T` for all `i` (no out-of-bounds segment access)
- `semantic_pair` labels correspond to actual labels in the scene's `segment.npy`
- `support_radius` (CLI parameter, not from NPZ) must be appropriate for the spatial scale of supports

**Risk level:** Medium. The zero-segment + zero-line-endpoint scenario is unlikely with current Stage 3 code but is not prevented by any validation.

---

## Contract: Scene -> Stage 4 (direct read)

**Coupling type:** Direct file read (skips Stages 2-3)

**Files consumed:**

| File | Shape | Dtype | Role |
|---|---|---|---|
| `coord.npy` | (N, 3) | float32 | Scene point coordinates for supervision computation |
| `segment.npy` | (N,) | int32 | Semantic labels for label -> support matching |

**No intermediate dependency.** Stage 4 reads raw scene data directly, bypassing all intermediate NPZ files. The only coupling is the scene's coordinate system and label space.

**Hidden assumption:** Scene `segment` labels match the `semantic_pair` labels in `supports.npz`. If Stages 1-3 were run with a different scene or different segmentation, the label matching would silently produce zero supervision coverage.

**Risk level:** Low (straightforward direct read with shape validation in `load_scene()`).

---

## Contract: Stage 3 internal (spacing-adaptive thresholds)

**Coupling type:** Internal to supports_core.py

**Mechanism:** All trigger regrouping thresholds in the trigger path are computed as `scale_factor * local_spacing`, where `local_spacing` comes from `estimate_local_spacing(coords, k=6)`.

**Threshold derivation:**

| Runtime Threshold | Computation | Scale Factor |
|---|---|---|
| `along_gap_th` | `segment_run_gap_scale * spacing` | 3.0 |
| `lateral_gap_th` | `segment_run_lateral_gap_scale * spacing` | 2.5 |
| `lateral_band_th` | `segment_run_lateral_band_scale * spacing` | 3.0 |
| `attach_dist_th` | `trigger_fragment_attach_dist_scale * spacing` | 2.5 |
| `attach_along_gap_th` | `trigger_fragment_attach_gap_scale * spacing` | 4.0 |
| `merge_line_dist_th` | `trigger_main_merge_dist_scale * spacing` | 1.5 |
| `merge_gap_th` | `trigger_main_merge_gap_scale * spacing` | 3.0 |
| `merge_lateral_th` | `trigger_main_merge_lateral_scale * spacing` | 1.4 |
| `endpoint_dist_th` | `trigger_endpoint_absorb_dist_scale * spacing` | 2.2 |
| `line_dist_th` | `trigger_endpoint_absorb_line_dist_scale * spacing` | 1.6 |
| `endpoint_proj_th` | `trigger_endpoint_absorb_proj_scale * spacing` | 2.6 |

**Design pattern:** This makes the regrouping behavior implicitly density-adaptive at the cluster level, even though the pipeline-level parameters (Stage 2 `eps`, Stage 4 `support_radius`) are fixed. Each cluster's spacing is estimated independently.

**`estimate_local_spacing()` behavior (lines 282-292):**
- Uses brute-force pairwise distance matrix (O(n^2) memory)
- Returns median of k-nearest-neighbor distances (k=6 default)
- Falls back to 0.01 if fewer than 2 points
- Clamps with `max(..., 1e-6)` at call sites

**Edge case:** Clusters with fewer than k=6 points get unreliable spacing estimates. The function handles this by using whatever neighbors are available, but downstream thresholds may become either too tight (false splitting) or too loose (failed splitting). The `max(..., 1e-6)` clamp prevents zero/negative thresholds.

**Risk level:** Low (safeguarded by clamping). But important to document as a design pattern for future density-adaptive work in Phase 4.

---

## Contract: In-memory bypass (build_support_dataset_v3.py)

**Coupling type:** Code-level API coupling (bypasses NPZ I/O)

**Mechanism:** `build_support_dataset_v3.py` runs Stages 1-3 in-memory by calling core functions directly, without writing or reading intermediate NPZ files.

**In-memory function call chain:**

| Step | Function Called | Module | Arguments |
|---|---|---|---|
| 1 | `load_scene(scene_dir)` | `utils.stage_io` | Direct file read |
| 2 | `build_boundary_centers(scene, k, min_cross_ratio, min_side_points, ignore_index)` | `core.boundary_centers_core` | CLI args |
| 3 | `cluster_boundary_centers(boundary_centers, eps, min_samples, denoise_knn, sparse_distance_ratio, sparse_mad_scale)` | `core.local_clusters_core` | CLI args |
| 4 | `build_supports_payload(boundary_centers, local_clusters, params)` | `core.supports_core` | `build_support_runtime_params(args)` |
| 5 | `export_npz(scene_dir / "supports.npz", supports_payload)` | `core.supports_core` | Only final output saved |
| 6 | `cleanup_scene_dir(scene_dir)` | Local function | Deletes all intermediate files |

**Per-scene script function call chain (for comparison):**

| Step | Script | Function Called | I/O |
|---|---|---|---|
| 1 | `build_boundary_centers.py` | `load_scene()`, `build_boundary_centers()` | Writes `boundary_centers.npz` |
| 2 | `build_local_clusters.py` | `load_boundary_centers()`, `cluster_boundary_centers()` | Reads `boundary_centers.npz`, writes `local_clusters.npz` |
| 3 | `fit_local_supports.py` | `load_boundary_centers()`, `load_local_clusters()`, `build_supports_payload()` | Reads both NPZs, writes `supports.npz` |
| 4 | `build_pointwise_edge_supervision.py` | `load_scene()`, `load_supports()`, `build_pointwise_edge_supervision()` | Reads `supports.npz`, writes `edge_*.npy` |

**Equivalence assumption:** The in-memory path produces identical results to the per-stage NPZ path. This holds because:
1. NPZ serialization is lossless for float32/int32 numpy arrays (exact bit-level roundtrip)
2. Both paths call the same core functions with the same arguments
3. The in-memory path passes Python dict objects directly, while the per-scene path serializes/deserializes via NPZ -- but the data types and values are identical

**Coupling risk:** Any change to core function signatures must be reflected in BOTH:
- Per-scene scripts: `build_boundary_centers.py`, `build_local_clusters.py`, `fit_local_supports.py`
- Dataset script: `build_support_dataset_v3.py`

**Parameter construction duplication:** Both `fit_local_supports.py` and `build_support_dataset_v3.py` contain independent `build_runtime_params()` / `build_support_runtime_params()` functions that convert `DEFAULT_FIT_PARAMS` angle values to cosine thresholds. These are structurally identical but maintained separately. A parameter change in one without updating the other would cause behavioral divergence between per-scene and dataset paths.

**Cleanup behavior:** `build_support_dataset_v3.py` calls `cleanup_scene_dir()` which removes all intermediate files (boundary_centers.npz, local_clusters.npz, all edge_*.npy, all xyz files). This means after a dataset run, only `supports.npz` and `support_geometry.xyz` remain.

**Risk level:** Low for correctness (lossless serialization). Medium for maintainability (function signature and parameter changes must be mirrored in two places).

---

## NPZ Schema Summary Table

### boundary_centers.npz (Stage 1 output)

| Field | Shape | Dtype | Producer | Consumers |
|---|---|---|---|---|
| `center_coord` | (M, 3) | float32 | Stage 1: `build_boundary_centers()` | Stage 2: clustering; Stage 3: fitting via read-back |
| `center_normal` | (M, 3) | float32 | Stage 1: `build_boundary_centers()` | Not consumed by later stages; kept for potential downstream use |
| `center_tangent` | (M, 3) | float32 | Stage 1: `build_boundary_centers()` | Stage 2: trigger metrics; Stage 3: tangent-based regrouping |
| `semantic_pair` | (M, 2) | int32 | Stage 1: `build_boundary_centers()` | Stage 2: per-pair grouping |
| `source_point_index` | (M,) | int32 | Stage 1: `build_boundary_centers()` | Not consumed by later stages; diagnostic |
| `confidence` | (M,) | float32 | Stage 1: `build_boundary_centers()` | Stage 2: visualization; Stage 3: cluster_confidence averaging |

### local_clusters.npz (Stage 2 output)

| Field | Shape | Dtype | Producer | Consumers |
|---|---|---|---|---|
| `center_index` | (K,) | int32 | Stage 2: `cluster_boundary_centers()` | Stage 3: index into Stage 1 arrays |
| `cluster_id` | (K,) | int32 | Stage 2: `cluster_boundary_centers()` | Stage 3: cluster membership |
| `semantic_pair` | (C, 2) | int32 | Stage 2: `cluster_boundary_centers()` | Stage 3: per-cluster pair |
| `cluster_trigger_flag` | (C,) | uint8 | Stage 2: `cluster_boundary_centers()` | Stage 3: routing to standard/trigger path |
| `cluster_size` | (C,) | int32 | Stage 2: `cluster_boundary_centers()` | Stage 3: validated against observed count |
| `cluster_centroid` | (C, 3) | float32 | Stage 2: `cluster_boundary_centers()` | Stage 3: passed through to record |

### supports.npz (Stage 3 output)

| Field | Shape | Dtype | Producer | Consumers |
|---|---|---|---|---|
| `support_id` | (S,) | int32 | Stage 3: `build_supports_payload()` | Stage 4: iteration and shape validation |
| `support_type` | (S,) | int32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 (visualization only) |
| `semantic_pair` | (S, 2) | int32 | Stage 3: `build_supports_payload()` | Stage 4: label -> support reverse index |
| `confidence` | (S,) | float32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 |
| `fit_residual` | (S,) | float32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 |
| `coverage_radius` | (S,) | float32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 |
| `cluster_id` | (S,) | int32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 |
| `origin` | (S, 3) | float32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 |
| `direction` | (S, 3) | float32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 |
| `center` | (S, 3) | float32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 (reserved, zero-filled) |
| `radius` | (S,) | float32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 (reserved, zero-filled) |
| `normal` | (S, 3) | float32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 (reserved, zero-filled) |
| `angle_min` | (S,) | float32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 (reserved, zero-filled) |
| `angle_max` | (S,) | float32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 (reserved, zero-filled) |
| `polyline_offset` | (S,) | int32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 |
| `polyline_length` | (S,) | int32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 |
| `polyline_vertices` | (V, 3) | float32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 |
| `line_start` | (S, 3) | float32 | Stage 3: `build_supports_payload()` | Stage 4: fallback endpoints when segment_length <= 0 |
| `line_end` | (S, 3) | float32 | Stage 3: `build_supports_payload()` | Stage 4: fallback endpoints when segment_length <= 0 |
| `orientation_prior_score` | (S,) | float32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 |
| `segment_offset` | (S,) | int32 | Stage 3: `build_supports_payload()` | Stage 4: index into segment arrays |
| `segment_length` | (S,) | int32 | Stage 3: `build_supports_payload()` | Stage 4: number of segments per support |
| `segment_start` | (T, 3) | float32 | Stage 3: `build_supports_payload()` | Stage 4: segment start endpoints |
| `segment_end` | (T, 3) | float32 | Stage 3: `build_supports_payload()` | Stage 4: segment end endpoints |
| `segment_origin` | (T, 3) | float32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 |
| `segment_direction` | (T, 3) | float32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 |
| `segment_point_count` | (T,) | int32 | Stage 3: `build_supports_payload()` | Not consumed by Stage 4 |

**Notation:** M = boundary centers, K = assigned center entries (after noise removal), C = clusters, S = supports, T = total segments, V = total polyline vertices.

**Stage 4 minimal read set:** Of the 27 fields in `supports.npz`, Stage 4 (`load_supports()`) reads only 8: `support_id`, `semantic_pair`, `segment_offset`, `segment_length`, `segment_start`, `segment_end`, `line_start`, `line_end`. The remaining 19 fields are diagnostic, visualization, or reserved.
