# Behavioral Audit: bf_edge_v3 Core Pipeline

**Date:** 2026-04-07
**Phase:** 02-behavioral-audit-and-module-restructure, Plan 01
**Purpose:** This document classifies every logical block in the bf_edge_v3 pipeline by behavioral role, per the three-way scheme: Core Algorithm / Compatibility-Adaptation / Infrastructure.

## Classification Scheme

- **[CORE]** -- Would exist in any clean implementation of this pipeline's purpose
- **[COMPAT]** -- Handles real-data edge cases, historical decisions, cross-stage workarounds, data-specific tuning
- **[INFRA]** -- I/O, logging, validation, debug output, visualization export

---

## Stage 1 -- boundary_centers_core.py (414 lines)

Stage 1 detects semantic boundary candidate points in a 3D point cloud using kNN-based cross-label neighbor analysis, then estimates boundary center positions, tangent directions, and confidence scores for each candidate. It is the cleanest module in the pipeline with minimal compatibility logic.

### Block-by-block classification

| Block Name | Lines | Classification | Behavior Description | Notes |
|---|---|---|---|---|
| `build_knn_graph()` | 40-56 | **[CORE]** | Builds a kNN graph via scipy cKDTree. Returns neighbor indices excluding self. | query_k = min(k+1, n_points), workers=-1 for parallel |
| Single-point / empty cloud guards in `build_knn_graph()` | 43-54 | **[COMPAT]** | Returns empty arrays when n_points==0 or query_k<=1. Handles degenerate input gracefully. | Prevents downstream crashes on empty/trivial scenes |
| `detect_boundary_candidates()` | 59-98 | **[CORE]** | Detects point-wise semantic boundary candidates from local label changes. For each point, computes cross-label neighbor ratio and identifies the dominant paired label. | Filters by ignore_index and min_cross_ratio |
| `estimate_pca_tangent()` | 101-123 | **[CORE]** | Estimates tangent direction in the plane orthogonal to boundary normal via SVD/PCA. Returns tangent vector and quality score (ratio of first singular value). | Score = S[0] / sum(S), clipped to [0,1] |
| `estimate_fallback_tangent()` | 126-144 | **[COMPAT]** | Fallback tangent via cross product of mean surface normal with center normal when PCA is unstable. Returns tangent with fixed score of 0.3. | Requires optional `normal.npy` input; score hardcoded at 0.3 |
| `estimate_boundary_center()` | 147-219 | **[CORE]** | Estimates one boundary center record: computes midpoint of two-side centroids as center_coord, separation direction as center_normal, calls PCA tangent then fallback if needed, computes composite confidence score. | Core estimation with embedded fallback call |
| Fallback tangent invocation | 192-196 | **[COMPAT]** | Within `estimate_boundary_center()`, calls `estimate_fallback_tangent()` when PCA tangent has zero norm. Falls back further to returning None if fallback also fails. | Two-stage fallback chain |
| Confidence score weighting | 200-208 | **[CORE]** | Composite confidence: `cross_ratio(0.45) + side_balance(0.30) + separation(0.15) + tangent(0.10)`. Clipped to [0,1]. | Weights sum to 1.0; side_balance = 1 - |nA-nB|/(nA+nB) |
| Empty-result zero-shape array handling | 278-286 | **[COMPAT]** | When no centers are found (center_records empty), returns zero-shape arrays with correct dtypes: `(0,3)` for coords, `(0,2)` for pairs, `(0,)` for scalars. | Prevents downstream shape errors on empty scenes |
| `build_boundary_centers()` | 222-306 | **[CORE]** | Top-level orchestration: builds kNN graph, detects candidates, estimates centers, assembles payload and metadata. | Returns (candidates, centers_payload, meta) |
| `pair_to_color()` / `build_pair_colors()` | 309-328 | **[INFRA]** | Stable pseudo-color generation for semantic pairs using hash-based seed. | Used only for visualization export |
| `export_boundary_centers_npz()` | 331-333 | **[INFRA]** | NPZ serialization of center payload via `save_npz`. | Delegates to utils.common.save_npz |
| `export_candidate_xyz()` | 336-373 | **[INFRA]** | XYZ export for boundary candidates with coord, color, pair, cross_ratio, source_index columns. | CloudCompare-style inspection |
| `export_centers_xyz()` | 376-414 | **[INFRA]** | XYZ export for boundary centers with coord, color, normal, tangent, pair, confidence, source_index columns. | 16-column output format |

---

## Stage 2 -- local_clusters_core.py (393 lines)

Stage 2 groups boundary centers by semantic pair, applies spatial DBSCAN clustering per pair, performs lightweight kNN+MAD outlier denoising per cluster, and computes trigger metrics (linearity, tangent coherence, bbox anisotropy) to flag clusters that likely contain multiple edge branches requiring Stage 3's special regrouping path.

### Block-by-block classification

| Block Name | Lines | Classification | Behavior Description | Notes |
|---|---|---|---|---|
| `spatial_dbscan()` | 37-42 | **[CORE]** | Runs sklearn DBSCAN on 3D center coordinates. Returns integer cluster labels (-1 = noise). | eps and min_samples from caller |
| Empty input guard in `spatial_dbscan()` | 39-40 | **[COMPAT]** | Returns empty array when coords.shape[0]==0. | Prevents crash on empty semantic pair group |
| `build_cluster_record()` | 45-56 | **[CORE]** | Builds minimal cluster-level statistics: cluster_id, semantic_pair, cluster_size, cluster_centroid. | Used for downstream processing |
| `compute_cluster_trigger_metrics()` | 59-93 | **[CORE]** | Computes cluster-level trigger metrics via SVD: linearity (S[0]/sum(S)), tangent coherence (mean abs dot with principal direction), bbox anisotropy (max_extent/second_extent). | Returns dict with 4 metrics |
| `should_trigger_split()` | 96-106 | **[COMPAT]** | Conservative AND-logic trigger: requires ALL three conditions -- low linearity, low tangent coherence, AND not a long strip -- plus minimum cluster size. Returns bool. | Design choice: AND reduces false positives |
| Trigger threshold hardcodes | 191-196 | **[COMPAT]** | `trigger_min_cluster_size=max(min_samples*6, 48)`, `trigger_linearity_th=0.85`, `trigger_tangent_coherence_th=0.88`, `trigger_bbox_anisotropy_th=6.0`. Hardcoded in `cluster_boundary_centers()`, not configurable. | Tuned to real data; must be made configurable in refactor |
| `build_knn()` | 109-122 | **[CORE]** | Builds local kNN distances (without self) for denoise. Uses cKDTree. | Similar pattern to Stage 1's build_knn_graph but returns distances not indices |
| `lightweight_denoise_cluster()` | 125-172 | **[CORE]** | Per-cluster denoise: computes local spacing via kNN, identifies sparse points using MAD-based threshold, removes if not too aggressive. | Core denoising algorithm |
| Denoise safeguards: `max_remove_ratio`, `min_keep_points` | 166-169 | **[COMPAT]** | `max_remove_ratio=0.20` (cancel denoise if >20% would be removed), `min_keep_points=max(min_samples, 6)` (cancel if too few would remain). Hardcoded in `cluster_boundary_centers()` call at lines 218-219. | Conservative safeguards tuned to real data |
| `cluster_boundary_centers()` | 175-310 | **[CORE]** | Top-level Stage 2 orchestration: groups by semantic_pair, runs DBSCAN per pair, denoises clusters, computes trigger metrics, assembles payload. | Returns (payload, meta) |
| `cluster_to_color()` / `build_cluster_colors()` | 313-334 | **[INFRA]** | Stable pseudo-color for cluster IDs. Noise (-1) gets gray (140,140,140). | Hash-based seed coloring |
| `export_npz()` | 337-339 | **[INFRA]** | NPZ serialization of cluster payload. | Delegates to save_npz |
| `export_clustered_boundary_centers_xyz()` | 342-393 | **[INFRA]** | XYZ export for clustered centers with coord, cluster color, cluster_id, trigger_flag, pair, cluster_size, confidence columns. Trigger clusters highlighted in red (255,90,90). | 12-column output format |

---

## Stage 3 -- supports_core.py (1318 lines)

Stage 3 is the primary complexity center of the pipeline (60%+ of total code). It fits line or polyline supports to each cluster, with a dual-path architecture: the standard path directly fits a line/polyline support, while the trigger path performs direction-based regrouping (tangent clustering, spatial run splitting, subgroup classification, fragment attachment, main bundle merging, and endpoint absorption) before reusing the standard fitting. The trigger path (~600 lines) is almost entirely compatibility logic that handles clusters where DBSCAN produced mixed-direction groupings.

### DEFAULT_FIT_PARAMS (lines 45-71)

**Classification: [COMPAT]** -- 25 hardcoded parameters for trigger regrouping, tuned to real data, with no external override mechanism.

| Parameter | Default Value | Role |
|---|---|---|
| `segment_direction_angle_deg` | 20.0 | Tangent grouping: max angle between tangents for same direction group |
| `segment_run_gap_scale` | 3.0 | Run splitting: along-axis gap threshold = scale * local_spacing |
| `segment_run_lateral_gap_scale` | 2.5 | Run splitting: lateral gap threshold = scale * local_spacing |
| `segment_run_lateral_band_scale` | 3.0 | Run splitting: lateral band width threshold = scale * local_spacing |
| `segment_min_points` | 6 | Minimum points per run/subgroup to keep |
| `trigger_main_min_points` | 12 | Main subgroup: minimum point count |
| `trigger_main_linearity_th` | 0.88 | Main subgroup: minimum PCA linearity |
| `trigger_main_tangent_angle_deg` | 20.0 | Main subgroup: max tangent angle deviation |
| `trigger_main_length_scale` | 6.0 | Main subgroup: minimum length = scale * spacing |
| `trigger_main_lateral_scale` | 2.5 | Main subgroup: max lateral spread = scale * spacing |
| `trigger_fragment_min_points` | 6 | Fragment subgroup: minimum point count |
| `trigger_fragment_linearity_th` | 0.78 | Fragment subgroup: minimum PCA linearity |
| `trigger_fragment_tangent_angle_deg` | 28.0 | Fragment subgroup: max tangent angle deviation |
| `trigger_fragment_lateral_scale` | 3.5 | Fragment subgroup: max lateral spread = scale * spacing |
| `trigger_fragment_attach_dist_scale` | 2.5 | Fragment attachment: max distance to main = scale * spacing |
| `trigger_fragment_attach_gap_scale` | 4.0 | Fragment attachment: max along-axis gap = scale * spacing |
| `trigger_fragment_attach_angle_deg` | 20.0 | Fragment attachment: max angle to main direction |
| `trigger_main_merge_angle_deg` | 10.0 | Main merge: max angle between main bundle directions |
| `trigger_main_merge_dist_scale` | 1.5 | Main merge: max line-to-centroid distance = scale * spacing |
| `trigger_main_merge_gap_scale` | 3.0 | Main merge: max along-axis endpoint gap = scale * spacing |
| `trigger_main_merge_lateral_scale` | 1.4 | Main merge: max lateral offset between centroids = scale * spacing |
| `trigger_endpoint_absorb_dist_scale` | 2.2 | Endpoint absorption: max endpoint distance = scale * spacing |
| `trigger_endpoint_absorb_line_dist_scale` | 1.6 | Endpoint absorption: max line distance = scale * spacing |
| `trigger_endpoint_absorb_proj_scale` | 2.6 | Endpoint absorption: max projection distance = scale * spacing |
| `trigger_endpoint_absorb_max_points_per_end` | 12 | Endpoint absorption: max bad points absorbed per endpoint |

### Runtime parameter derivation

The `DEFAULT_FIT_PARAMS` dict stores angle parameters in degrees, but the runtime `build_runtime_params()` (in `fit_local_supports.py` and duplicated in `build_support_dataset_v3.py`) converts them to cosine thresholds before passing to core functions. The derived runtime parameters are:

| Runtime Key | Derived From | Computation | Approximate Value |
|---|---|---|---|
| `segment_direction_cos_th` | `segment_direction_angle_deg` (20.0) | cos(deg2rad(20.0)) | 0.9397 |
| `trigger_main_tangent_cos_th` | `trigger_main_tangent_angle_deg` (20.0) | cos(deg2rad(20.0)) | 0.9397 |
| `trigger_fragment_tangent_cos_th` | `trigger_fragment_tangent_angle_deg` (28.0) | cos(deg2rad(28.0)) | 0.8829 |
| `trigger_fragment_attach_cos_th` | `trigger_fragment_attach_angle_deg` (20.0) | cos(deg2rad(20.0)) | 0.9397 |
| `trigger_main_merge_cos_th` | `trigger_main_merge_angle_deg` (10.0) | cos(deg2rad(10.0)) | 0.9848 |

Additionally, three CLI-level parameters are passed through as runtime params: `line_residual_th` (default 0.01), `min_cluster_size` (default 8), `max_polyline_vertices` (default 32).

**Important:** The angle-to-cosine conversion means that changing `DEFAULT_FIT_PARAMS` angle values without recomputing the cosine thresholds will have no effect -- the runtime params are what core functions actually consume.

### Trigger path architecture overview

The trigger path is the dominant complexity driver in Stage 3 (and the entire pipeline). It activates when Stage 2 flags a cluster as likely containing multiple edge branches (`cluster_trigger_flag > 0`). The standard path is ~80 lines; the trigger path spans ~600 lines with the following processing pipeline:

1. **Tangent grouping** (`group_tangents`, lines 295-339): Cluster boundary center tangent vectors into direction groups using sign-invariant cosine similarity. Greedy sequential algorithm with running representative update.
2. **Spatial run splitting** (`split_direction_group_into_runs`, lines 378-459): Within each direction group, separate parallel edges by lateral offset and split each lateral band by along-axis gaps. Uses three spacing-adaptive thresholds.
3. **Subgroup quality metrics** (`compute_subgroup_metrics`, lines 462-504): PCA-based metrics per run: linearity, tangent coherence, length, lateral spread.
4. **Three-way classification** (`classify_trigger_subgroup`, lines 507-536): Each run becomes main (strong edge evidence), fragment (partial edge), or bad (noise/junk).
5. **Fragment promotion** (lines 732-743): If no mains exist, the best fragment is promoted to prevent empty output.
6. **Fragment attachment** (`try_attach_group_to_main_bundle`, lines 556-600): Fragment runs are attached to compatible main bundles via cost minimization (distance + gap).
7. **Main bundle merging** (`should_merge_main_bundles`, lines 612-662): Parallel main bundles that represent the same real edge are merged via a 6-condition check.
8. **Endpoint absorption** (`absorb_sparse_endpoint_points`, lines 869-956): Bad points near bundle endpoints are scavenged within distance/projection thresholds.
9. **Standard fitting reuse** (lines 1062-1091): Each final bundle is fitted using the standard line/polyline path.

**Ratio insight:** Of the ~1318 lines in supports_core.py, the trigger path (steps 1-8 above, lines 295-956) accounts for ~660 lines. The remaining ~660 lines cover core fitting primitives (~200 lines), support assembly (~300 lines), and infrastructure/export (~160 lines).

### Block-by-block classification

| Block Name | Lines | Classification | Behavior Description | Notes |
|---|---|---|---|---|
| `SUPPORT_TYPE_LINE` / `SUPPORT_TYPE_POLYLINE` | 42-43 | **[CORE]** | Type constants: LINE=0, POLYLINE=2. | Used in payload assembly |
| `DEFAULT_FIT_PARAMS` | 45-71 | **[COMPAT]** | Frozen dict of 25 hardcoded regrouping parameters. No external override mechanism. | See table above for all values |
| `rebuild_cluster_records()` | 73-117 | **[CORE]** | Rebuilds per-cluster record dicts from compact NPZ payload. Validates center_index range, cluster_id range, cluster_size consistency. | Raises ValueError on data integrity failures |
| `point_to_line_distance()` | 120-128 | **[CORE]** | Point-to-line distance in 3D via projection. | Returns inf if direction is zero-norm |
| `point_to_segment_distance()` | 131-140 | **[CORE]** | Point-to-finite-segment distance with clamped projection parameter t in [0,1]. | Falls back to distance-to-start if seg_len2 < EPS |
| `point_to_polyline_distance()` | 143-154 | **[CORE]** | Point-to-polyline minimum distance: iterates all segments, keeps minimum. | Falls back for 0 or 1 vertex |
| `line_to_endpoints()` | 157-163 | **[CORE]** | Converts PCA line (origin+direction) to finite endpoints using point projections. | min(t) and max(t) determine extent |
| `fit_line_support()` | 166-195 | **[CORE]** | Fits a line support via PCA: SVD on centered points, converts to endpoints, computes midpoint, residual, coverage_radius, length. | Returns None if <2 points or zero direction |
| `build_polyline_vertices()` | 198-219 | **[CORE]** | Sorts points along primary PCA direction, downsamples to max_vertices via linspace. | Preserves first and last point |
| `fit_polyline_support()` | 222-239 | **[CORE]** | Fits a polyline support: builds vertices, computes residual, centroid, coverage_radius, end-to-end direction. | Always succeeds (no None return) |
| `regularize_support_orientation()` | 242-262 | **[COMPAT]** | Light orientation prior toward major axes. If direction is within 15 degrees of a coordinate axis (cos(15)=0.966), applies weighted snap: alpha = 0.3 * normalized_score, blends direction with axis. | Architectural facade prior; snap_threshold=cos(15deg), alpha capped at 0.3 |
| `segment_record_from_endpoints()` | 265-279 | **[CORE]** | Builds one segment record dict from start/end endpoints. | Helper for support assembly |
| `estimate_local_spacing()` | 282-292 | **[CORE]** | Estimates point spacing from local kNN statistics via brute-force pairwise distance. Returns median of kNN distances (k=6 default). | Falls back to 0.01 if <2 points; used by all spacing-adaptive thresholds |
| `group_tangents()` | 295-339 | **[CORE]** | Sign-invariant tangent direction clustering. Greedy sequential assignment: each point assigned to closest existing group (by abs cosine) if above threshold, otherwise creates new group. Updates group representative via running average. | direction_cos_th from params |
| `estimate_direction_group_axis()` | 342-358 | **[CORE]** | Estimates dominant axis for one direction group. Prefers mean tangent, falls back to PCA. | Used by split_direction_group_into_runs |
| `split_sorted_indices_by_gap()` | 361-375 | **[CORE]** | Splits sorted 1D values into consecutive runs by gap threshold. | Generic helper for run splitting |
| `split_direction_group_into_runs()` | 378-459 | **[COMPAT]** | Splits one tangent direction group into edge-runs by: (1) estimating dominant axis, (2) separating parallel edges by lateral offset gaps via PCA of perpendicular component, (3) splitting each lateral band by along-axis projection gaps. Three adaptive thresholds: along_gap = run_gap_scale * spacing, lateral_gap = lateral_gap_scale * spacing, lateral_band = lateral_band_scale * spacing. | Handles real-data irregularity where multiple parallel edges exist in one direction group |
| `compute_subgroup_metrics()` | 462-504 | **[CORE]** | Computes PCA quality metrics for one trigger subgroup: linearity (S[0]/(S[0]+S[1])), tangent_coherence (mean signed alignment), length (projection extent), lateral_spread (90th percentile). | Returns None if <2 points or fit fails |
| `classify_trigger_subgroup()` | 507-536 | **[COMPAT]** | Three-way subgroup classification: **main** (>=12 pts, linearity>=0.88, tangent_cos>=threshold, length>=6*spacing, lateral<=2.5*spacing), **fragment** (>=6 pts, linearity>=0.78, tangent_cos>=threshold, lateral<=3.5*spacing), **bad** (everything else). | Thresholds from params; determines subgroup fate |
| `merge_subgroup_points()` | 539-544 | **[CORE]** | Merges member indices from multiple subgroup records, deduplicates. | Helper for bundle management |
| `classify_color()` | 547-553 | **[INFRA]** | Fixed colors for trigger group visualization: main=green(80,200,90), fragment=yellow(240,190,70), bad=red(210,80,80). | Visualization only |
| `try_attach_group_to_main_bundle()` | 556-600 | **[COMPAT]** | Fragment-to-main attachment via cost minimization. For each main bundle: checks direction alignment (abs cosine >= threshold), lateral distance to line (<= attach_dist_th), along-axis gap (<= attach_along_gap_th). Picks bundle with minimum cost = dist_to_line + gap. | Recovers fragmented edge pieces by merging them into the closest compatible main bundle |
| `refresh_main_bundle_geometry()` | 603-609 | **[CORE]** | Recomputes main bundle line support after fragment/main merging. | Called after every attachment or merge |
| `should_merge_main_bundles()` | 612-662 | **[COMPAT]** | 6-condition merge check for parallel main bundles: (1) both have valid line_support, (2) both have non-zero direction, (3) direction alignment >= cos_th, (4) lateral offset between centroids <= lateral_th, (5) max bidirectional line-to-centroid distance <= line_dist_th, (6) along-axis endpoint gap <= gap_th. Returns True only if ALL conditions pass. | Handles over-splitting where DBSCAN + direction grouping creates two bundles for what is really one edge |
| `regroup_trigger_cluster()` | 665-866 | **[MIXED: orchestrates CORE + COMPAT]** | Master orchestration of trigger path. Sequence: (1) group_tangents, (2) split_direction_group_into_runs per group, (3) compute_subgroup_metrics per run, (4) classify_trigger_subgroup, (5) fragment promotion if no mains, (6) build main_bundles with line geometry, (7) attach fragments to mains, (8) merge parallel mains, (9) finalize bundles. Returns (final_bundles, visualization_rows, unused_bad_indices). | ~200 lines; the heart of trigger complexity |
| Fragment promotion when no mains | 732-743 | **[COMPAT]** | If no subgroups classified as "main", promotes the best fragment (by point_count, linearity, length) to main. Prevents trigger path from producing zero outputs. | Edge case recovery |
| `absorb_sparse_endpoint_points()` | 869-956 | **[COMPAT]** | Scavenges "bad" points near bundle endpoints. For each endpoint (start, end): selects bad points within distance, line distance, and projection thresholds, up to max_points_per_end limit. Absorbs them into the bundle. | Three spacing-adaptive thresholds: endpoint_dist=2.2*spacing, line_dist=1.6*spacing, proj=2.6*spacing. Max 12 points per end. |
| `build_standard_support_record()` | 959-1037 | **[CORE]** | Builds one line or polyline support from one cluster. Tries line fit first; if residual exceeds threshold, falls back to polyline. Applies `regularize_support_orientation()`. Assembles full support record with segments. | line_residual_th from params |
| `build_trigger_support_records()` | 1040-1091 | **[CORE]** | Trigger path entry: calls `regroup_trigger_cluster()`, then for each final bundle calls `absorb_sparse_endpoint_points()` followed by `build_standard_support_record()`. | Reuses standard fitting after regrouping |
| `build_support_record()` | 1094-1114 | **[CORE]** | Dispatcher: routes to trigger or standard path based on `cluster_trigger_flag`. | Clean dispatch logic |
| `build_supports_payload()` | 1117-1254 | **[CORE]** | Top-level assembly: rebuilds cluster records, iterates clusters, collects support records, assembles flat NPZ-ready payload with all segment/polyline offset/length arrays. | Returns (supports_payload, meta_payload, debug_payload) |
| `export_npz()` | 1257-1259 | **[INFRA]** | NPZ serialization via save_npz. | Delegates to utils |
| `sample_segment_geometry()` | 1262-1267 | **[INFRA]** | Dense segment sampling for visualization. | step-based interpolation |
| `export_support_geometry_xyz()` | 1270-1308 | **[INFRA]** | XYZ export for support geometry with sampled segment points, colors, support_id, type. | 8-column output |
| `export_trigger_group_classes_xyz()` | 1311-1318 | **[INFRA]** | XYZ export for trigger candidate subgroup classes. | 10-column output |

---

## Stage 4 -- pointwise_core.py (301 lines)

Stage 4 computes per-point edge supervision by projecting each scene point to its nearest candidate support geometry. For each point, it finds the closest support that shares a semantic label, computes the distance and direction vector, and applies a truncated Gaussian weighting. The module is structurally clean with well-localized compatibility logic.

### Block-by-block classification

| Block Name | Lines | Classification | Behavior Description | Notes |
|---|---|---|---|---|
| `FILL_VALUE` | 41 | **[CORE]** | Fill value constant: `np.inf`. Used for unmatched points. | Sentinel for invalid distances |
| `load_supports()` | 43-74 | **[INFRA]** | Loads minimal fields from supports.npz: support_id, semantic_pair, segment_offset, segment_length, segment_start, segment_end, line_start, line_end. Validates shapes. | Minimal-field parsing; reads 8 fields |
| `closest_point_on_segment()` | 77-94 | **[CORE]** | Projects points onto one finite segment. Clamps projection parameter t to [0,1]. Returns (closest_points, distances). | Core geometric projection |
| Degenerate segment fallback | 85-88 | **[COMPAT]** | If segment has zero length (seg_len2 < EPS), projects all points to segment start. | Prevents division by zero |
| `closest_points_to_support()` | 97-125 | **[CORE]** | Returns closest points on one support and distances. Iterates all segments of the support, keeps minimum distance per point. | Multi-segment support distance |
| Zero-segment fallback | 106-111 | **[COMPAT]** | If support has segment_length <= 0, uses line_start/line_end as a single segment instead of iterating the segment array. | Handles supports with no explicit segments; falls back to line endpoints |
| `build_label_to_supports()` | 128-137 | **[CORE]** | Builds reverse index from segment label to candidate support IDs. For each support, registers both labels in its semantic_pair. | Key lookup structure for per-label processing |
| `build_edge_support()` | 140-154 | **[CORE]** | Truncated Gaussian boundary support weight: `exp(-d^2 / (2*sigma^2))` where `sigma = max(support_radius/2, EPS)`. Only applied to valid points (edge_valid==1). | sigma derivation: half the support radius |
| `build_pointwise_edge_supervision()` | 157-246 | **[CORE]** | Top-level supervision builder: iterates unique labels, for each finds candidate supports via reverse index, computes nearest support per point, applies radius cutoff, normalizes direction vectors, computes Gaussian weights. | Returns (payload, meta) |
| Legacy output aliases | 228-230 | **[COMPAT]** | `edge_mask = edge_valid` (copy), `edge_strength = edge_support` (copy). Kept for backward compatibility with older training configs that reference these field names. | Comment explicitly states: "Legacy aliases kept only for downstream compatibility" |
| `export_edge_arrays()` | 249-259 | **[INFRA]** | Exports 8 .npy arrays: edge_dist, edge_dir, edge_valid, edge_support_id, edge_vec, edge_support, edge_mask, edge_strength. | Writes to individual files |
| `export_edge_supervision_xyz()` | 262-301 | **[INFRA]** | XYZ export for valid supervision points with coord, dist, support, support_id, dir, vec, segment columns. | 13-column output for visualization |

---

## Summary Statistics

| Module | CORE blocks | COMPAT blocks | INFRA blocks | MIXED blocks | Total blocks | COMPAT lines (approx) |
|---|---|---|---|---|---|---|
| Stage 1 -- boundary_centers_core.py | 6 | 3 | 4 | 0 | 13 | ~35 |
| Stage 2 -- local_clusters_core.py | 5 | 3 | 3 | 0 | 11 | ~30 |
| Stage 3 -- supports_core.py | 15 | 9 | 4 | 1 | 29 | ~600 |
| Stage 4 -- pointwise_core.py | 5 | 3 | 3 | 0 | 11 | ~20 |
| **Total** | **31** | **18** | **14** | **1** | **64** | **~685** |

**Key observations:**
- Stage 3 contains 29 of 64 blocks (45% of block count) and ~600 of ~685 COMPAT lines (88% of compatibility code)
- The trigger path (`regroup_trigger_cluster` and its sub-functions, lines 295-956) accounts for ~660 lines, nearly half of Stage 3
- Stage 1 and Stage 4 are structurally clean with minimal, well-localized compatibility logic
- Stage 2's compatibility surface is concentrated in the trigger threshold system (4 hardcoded thresholds + 2 denoise safeguards)
- All stages have clean INFRA separation (export/serialization functions are distinct from algorithm logic)
