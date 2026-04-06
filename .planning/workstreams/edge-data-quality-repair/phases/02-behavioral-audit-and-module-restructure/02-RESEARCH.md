# Phase 2: Behavioral Audit and Module Restructure — Research

**Researcher:** Claude (manual deep-code analysis)
**Date:** 2026-04-06
**Phase:** 02-behavioral-audit-and-module-restructure
**Consumed by:** `gsd-planner` via `/gsd-plan-phase 2 --ws edge-data-quality-repair`

---

## 1. Current Pipeline Architecture (Ground Truth)

### 1.1 Module inventory

| Module | File | Size | Functions | Role |
|--------|------|------|-----------|------|
| Stage 1 | `core/boundary_centers_core.py` | 13.6KB | 11 | Boundary candidate detection + center estimation |
| Stage 2 | `core/local_clusters_core.py` | 14.3KB | 11 | DBSCAN clustering + denoise + trigger marking |
| Stage 3 | `core/supports_core.py` | 53.6KB | 30+ | Support fitting (line/polyline) with trigger regrouping |
| Stage 4 | `core/pointwise_core.py` | 11KB | 8 | Per-point supervision via support projection |
| Utils | `utils/common.py` | 1.2KB | 4 | `EPS`, `normalize_vector`, `normalize_rows`, `save_xyz`, `save_npz` |
| I/O | `utils/stage_io.py` | 5.9KB | 5 | `load_scene`, `load_boundary_centers`, `load_local_clusters`, `collect_stage_tasks` |

### 1.2 Script inventory

| Script | Mode | Stages | Key behavior |
|--------|------|--------|--------------|
| `build_boundary_centers.py` | per-scene | 1 | `--scene` arg (inconsistent with others) |
| `build_local_clusters.py` | per-scene | 2 | `--input` arg |
| `fit_local_supports.py` | per-scene/batch | 3 | `--input` arg; has `build_runtime_params()` |
| `build_pointwise_edge_supervision.py` | per-scene | 4 | `--input` arg; `--support-radius`/`--max-edge-dist` alias |
| `build_support_dataset_v3.py` | dataset | 1-3 | In-memory pipeline; deletes intermediates; duplicates all Stage 1-3 argparse |
| `build_edge_dataset_v3.py` | dataset | 4 | Produces compact `edge.npy` (5-col) |
| `convert_edge_vec_to_dir_dist.py` | dataset | post | 5-col → 6-col format conversion |
| `add_support_id_to_edge_dataset.py` | dataset | post | Supplements existing edge dataset |
| `diagnose_net01.py` | diagnostic | — | Phase 1 diagnosis (not part of generation pipeline) |

### 1.3 Data flow chain

```
Scene (coord, segment, normal?, color?)
  │
  ▼ Stage 1: boundary_centers_core.build_boundary_centers()
boundary_centers.npz
  │ center_coord(M,3), center_normal(M,3), center_tangent(M,3),
  │ semantic_pair(M,2), source_point_index(M,), confidence(M,)
  │
  ▼ Stage 2: local_clusters_core.cluster_boundary_centers()
local_clusters.npz
  │ center_index(K,), cluster_id(K,), semantic_pair(C,2),
  │ cluster_trigger_flag(C,), cluster_size(C,), cluster_centroid(C,3)
  │
  ▼ Stage 3: supports_core.build_supports_payload()
  │           [reads BOTH boundary_centers.npz AND local_clusters.npz]
supports.npz
  │ support_id(S,), support_type(S,), semantic_pair(S,2),
  │ line_start(S,3), line_end(S,3),
  │ segment_offset(S,), segment_length(S,),
  │ segment_start(T,3), segment_end(T,3), ...
  │
  ▼ Stage 4: pointwise_core.build_pointwise_edge_supervision()
  │           [reads coord.npy, segment.npy, supports.npz]
edge_*.npy (8 arrays) → edge.npy (compact 5-col or 6-col)
```

### 1.4 Cross-stage dependencies (critical for refactor)

| Upstream | Downstream | Coupling type | Fields consumed |
|----------|------------|---------------|-----------------|
| Scene | Stage 1 | Direct read | coord, segment, normal(opt) |
| Scene | Stage 4 | Direct read (skips 2,3) | coord, segment |
| Stage 1 | Stage 2 | NPZ → NPZ | center_coord, center_tangent, semantic_pair, confidence |
| Stage 1 | Stage 3 | NPZ → NPZ (read-back) | All center fields (via center_index from Stage 2) |
| Stage 2 | Stage 3 | NPZ → NPZ | center_index, cluster_id, semantic_pair, cluster_trigger_flag, cluster_size |
| Stage 3 | Stage 4 | NPZ → NPZ | support_id, semantic_pair, segment_offset/length/start/end, line_start/end |

**Critical observation:** Stage 3 reads BOTH Stage 1 and Stage 2 outputs. It uses `center_index` from Stage 2 to index back into Stage 1's arrays. This means Stage 3 has an implicit dependency on the Stage 1 output schema, not just Stage 2's.

---

## 2. Behavioral Classification: Core Algorithm vs Compatibility Logic

### 2.1 Stage 1 — boundary_centers_core.py

**Core algorithm:**
- kNN graph construction (cKDTree)
- Cross-label neighbor ratio detection (candidates)
- Boundary center estimation: midpoint of two-side centroids
- PCA tangent estimation in orthogonal plane
- Confidence score: weighted `cross_ratio(0.45) + side_balance(0.30) + separation(0.15) + tangent(0.10)`

**Compatibility/adaptation:**
- Fallback tangent via cross product with surface normal when PCA fails (lines 126-144, score fixed at 0.3)
- Empty-result handling: zero-shape arrays with correct dtypes when no centers found (lines 279-286)
- Single-point / empty cloud guards in `build_knn_graph()` (lines 44-54)

**Infrastructure:**
- XYZ export for visualization (`pair_to_color`, `export_candidate_xyz`, `export_centers_xyz`)
- NPZ serialization

**Assessment:** Stage 1 is the cleanest module. Compatibility logic is minimal and well-localized. Low refactor risk.

### 2.2 Stage 2 — local_clusters_core.py

**Core algorithm:**
- Group by `semantic_pair` → per-pair DBSCAN
- Per-cluster lightweight denoise (kNN + MAD outlier detection)
- Trigger metric computation (SVD linearity, tangent coherence, bbox anisotropy)

**Compatibility/adaptation:**
- **Trigger threshold hardcodes** (lines 192-195): `trigger_min_cluster_size = max(min_samples * 6, 48)`, `linearity_th = 0.85`, `tangent_coherence_th = 0.88`, `bbox_anisotropy_th = 6.0`. These are NOT configurable — hardcoded in `cluster_boundary_centers()`.
- **Denoise safeguards** (lines 166-168): `max_remove_ratio=0.20`, `min_keep_points=max(min_samples, 6)`. These are data-adaptation heuristics.
- **Conservative AND-logic for trigger** (lines 96-106): requires ALL three metrics below threshold. This is a design choice to avoid false positives — real-data adaptation.

**Infrastructure:**
- XYZ export, NPZ serialization, cluster coloring

**Assessment:** The trigger system is the main compatibility surface. It was designed to flag "complex" clusters for Stage 3's special regrouping path. The trigger thresholds are tuned to real data but completely non-configurable. Refactor must make these configurable without changing defaults.

### 2.3 Stage 3 — supports_core.py (primary complexity)

**Core algorithm:**
- Line fitting via PCA (`fit_line_support`, lines 165-194)
- Polyline fitting via PCA-ordered vertex downsampling (`fit_polyline_support`, lines 221-238)
- Support type decision: line if residual ≤ threshold, else polyline

**Compatibility/adaptation (this is the bulk of the 53KB):**

| Block | Lines | Behavior | Classification |
|-------|-------|----------|----------------|
| `DEFAULT_FIT_PARAMS` | 44-70 | 20+ hardcoded parameters for trigger regrouping | Compatibility: tuned to real data |
| `group_tangents()` | 294-338 | Sign-invariant tangent direction clustering | Core algorithm |
| `split_direction_group_into_runs()` | 377-458 | Spatial run segmentation by lateral offset + along-axis gaps | Compatibility: handles real-data irregularity |
| `compute_subgroup_metrics()` | 461-504 | PCA quality metrics for subgroups | Core algorithm |
| `classify_trigger_subgroup()` | 507-536 | Three-way classification: main/fragment/bad | Compatibility: real-data edge case handling |
| `try_attach_group_to_main_bundle()` | 556-600 | Fragment-to-main attachment via cost minimization | Compatibility: recovers fragmented edges |
| `should_merge_main_bundles()` | 612-662 | 6-condition merge check for parallel mains | Compatibility: handles over-splitting |
| Fragment promotion | 732-743 | If no mains exist, promote best fragment | Compatibility: edge case recovery |
| `absorb_sparse_endpoint_points()` | 869-956 | Scavenge "bad" points near bundle endpoints | Compatibility: endpoint cleanup |
| `regularize_support_orientation()` | 241-261 | Snap toward axis-aligned directions (15° threshold, 0.3 weight) | Compatibility: architectural facade prior |
| `regroup_trigger_cluster()` | 665-868 | Master orchestration of trigger path | Mixed: orchestrates core + compatibility |

**Key insight:** The trigger cluster regrouping path (lines 665-956, ~300 lines) is almost entirely compatibility logic — it exists to handle clusters where DBSCAN produced mixed-direction groupings. The standard (non-trigger) path is only ~80 lines. The 6:1 ratio of compatibility-to-core reflects the real-data adaptation cost.

**Infrastructure:**
- NPZ/XYZ serialization, debug payload for trigger visualization

**Assessment:** Stage 3 is where refactor pays off most. The trigger path is a single monolithic function (`regroup_trigger_cluster`, 200+ lines) that mixes direction grouping, spatial splitting, classification, attachment, merging, and absorption. Splitting this into sub-modules with clear contracts is the primary structural win.

### 2.4 Stage 4 — pointwise_core.py

**Core algorithm:**
- Label-to-supports reverse index
- Per-point closest-point-on-segment projection
- Multi-segment support distance computation (iterate all segments, keep minimum)
- Gaussian weight: `exp(-d²/(2σ²))` where `σ = max(support_radius/2, EPS)`

**Compatibility/adaptation:**
- **Degenerate segment fallback** (lines 85-88): if segment has zero length, project all points to segment start
- **Zero-segment fallback** (lines 106-111): if support has `segment_length <= 0`, use `line_start/line_end` instead of iterating segments
- **Legacy output aliases** (lines 228-230): `edge_mask` = copy of `edge_valid`, `edge_strength` = copy of `edge_support`. These exist for backward compatibility with older training configs.

**Infrastructure:**
- `load_supports()` with minimal-field parsing
- XYZ/NPY export

**Assessment:** Stage 4 is clean. The support schema coupling is the main concern — it assumes specific field names (`segment_offset`, `segment_length`, `segment_start`, `segment_end`, `line_start`, `line_end`). Refactor must preserve this schema or update all consumers.

---

## 3. Cross-Stage Behavioral Contracts (Hidden Couplings)

These are behaviors where one stage implicitly assumes specific properties of another stage's output, but the contract is not documented or enforced.

### 3.1 Stage 2 → Stage 3: Trigger flag semantics

**Contract:** `cluster_trigger_flag > 0` means "this cluster likely contains multiple edge branches; Stage 3 should use direction-based regrouping instead of direct fitting."

**Hidden assumption:** Stage 3's `regroup_trigger_cluster()` assumes that if the trigger flag is set, the cluster's tangent vectors are meaningful enough to perform direction-based grouping. If Stage 2's trigger criteria change (e.g., different linearity threshold), Stage 3's regrouping may get clusters it can't handle.

**Risk level:** Medium. The trigger threshold is conservative (AND of three conditions), so false positives are rare. But the contract is implicit.

### 3.2 Stage 1 → Stage 3: Array index contract

**Contract:** Stage 3 uses `center_index` from Stage 2 to index into Stage 1's `center_coord`, `center_tangent`, `confidence` arrays. This requires that Stage 2's `center_index` values are valid indices into Stage 1's arrays.

**Hidden assumption:** Stage 1's arrays are never reordered or filtered between Stage 1 output and Stage 3 input. The NPZ format preserves array order, so this holds — but only because the file format is coincidentally stable.

**Risk level:** Low as long as NPZ serialization is preserved. High if any refactor introduces a different intermediate format.

### 3.3 Stage 3 → Stage 4: Support schema contract

**Contract:** Stage 4's `closest_points_to_support()` expects specific fields:
- `segment_offset[i]` and `segment_length[i]` to index into `segment_start`/`segment_end`
- `line_start[i]`/`line_end[i]` as fallback when `segment_length[i] <= 0`
- `semantic_pair[i]` to build the label→support reverse index

**Hidden assumption:** Every support has either valid segments OR valid line endpoints. There is no validation of this invariant at the Stage 3/Stage 4 boundary.

**Risk level:** Medium. If Stage 3 produces a support with neither segments nor valid line endpoints, Stage 4 will silently produce infinite distances for all points near that support.

### 3.4 Stage 3 internal: spacing-adaptive thresholds

**Contract:** All trigger regrouping thresholds in `supports_core.py` are computed as `scale_factor × local_spacing`, where `local_spacing` comes from `estimate_local_spacing(coords, k=6)`. This means the regrouping behavior is implicitly density-adaptive at the cluster level — even though the overall pipeline parameters (eps, sigma) are fixed.

**Hidden assumption:** `estimate_local_spacing()` returns a meaningful positive value. If a cluster has fewer than k=6 points, the spacing estimate may be unreliable. The function handles this by clamping to `EPS`, but the downstream thresholds may become either too tight or too loose.

**Risk level:** Low (safeguarded). But important to document as a design pattern: Stage 3 already has a local density-adaptation mechanism, separate from the Stage 2 fixed-eps problem.

### 3.5 Dataset-level scripts: in-memory bypass

**Contract:** `build_support_dataset_v3.py` runs Stages 1-3 in-memory, calling core functions directly without intermediate NPZ I/O. It then deletes any leftover intermediate files via `cleanup_scene_dir()`.

**Hidden assumption:** The in-memory path produces identical results to the per-stage NPZ path. This is true if and only if the NPZ serialization is lossless (which it is for float32/int32 numpy arrays). But the in-memory path skips all `stage_io.load_*` validation.

**Risk level:** Low for correctness. But creates a maintainability coupling: any change to core function signatures must be reflected in both per-scene scripts AND `build_support_dataset_v3.py`.

---

## 4. Structural Problems That Phase 2 Must Address

### 4.1 Critical (must fix for REF-01, REF-02, REF-03)

**P1: `supports_core.py` monolithic complexity (53KB, 1318 lines)**
- The trigger path (`regroup_trigger_cluster`, 200+ lines) is a single function mixing 6 distinct operations
- `DEFAULT_FIT_PARAMS` (20+ parameters) is a frozen dict with no external override mechanism
- All trigger regrouping parameters are entangled: you can't tune fragment attachment without also affecting main bundle merging

**P2: No per-stage I/O contracts enforced**
- `stage_io.py` validates shapes on load but does not validate semantic invariants (e.g., "all cluster_ids are valid", "all center_indices reference existing centers")
- Stages fail silently on bad input — errors cascade to later stages

**P3: No behavioral documentation for compatibility logic**
- The trigger system, denoise safeguards, orientation regularization, and fragment attachment are all undocumented design decisions
- A developer reading the code cannot distinguish "this is the algorithm" from "this handles a real-data edge case"

**P4: Parameter system is ad-hoc**
- Stage 1-2 params: CLI argparse defaults
- Stage 3 params: `DEFAULT_FIT_PARAMS` dict + 3 CLI params
- Stage 4 params: CLI + hardcoded `sigma = support_radius / 2`
- Cross-stage params: duplicated in `build_support_dataset_v3.py`
- No config file, no config validation, no config versioning

### 4.2 Important (should fix for clean module boundaries)

**P5: `_bootstrap.py` sys.path hack**
- All scripts modify `sys.path` at import time to make `core/` and `utils/` importable
- Breaks if scripts are invoked from non-standard working directory
- Should be replaced by proper Python package structure

**P6: Inconsistent CLI argument naming**
- Stage 1: `--scene`, Stages 2-4: `--input`
- Stage 4: `--support-radius` aliased as `--max-edge-dist`

**P7: In-memory vs on-disk code duplication**
- `build_support_dataset_v3.py` duplicates parameter construction logic from `fit_local_supports.py`
- `build_runtime_params()` exists in two places with subtle differences

### 4.3 Deferred (NOT Phase 2 — belongs in Phase 3 or later)

- Config injection system implementation → Phase 3 (REF-04)
- Intermediate validation hooks → Phase 3 (REF-05)
- Formal behavioral equivalence gate → Phase 3 (REF-06)
- Density-adaptive eps/sigma → Phase 4 (ALG-01)
- Algorithm improvements → Phase 4 (ALG-02)

---

## 5. Recommended Module Boundary Cuts

### 5.1 Stage 3 internal decomposition (primary target)

Current `supports_core.py` (1318 lines, 30+ functions) should be split into:

| New module | Contents | Lines (approx) | Why separate |
|------------|----------|-----------------|--------------|
| `fitting.py` | `fit_line_support`, `fit_polyline_support`, `build_polyline_vertices`, `regularize_support_orientation`, geometry primitives (`point_to_line_distance`, etc.) | ~200 | Core fitting algorithms — rarely change |
| `trigger_regroup.py` | `group_tangents`, `split_direction_group_into_runs`, `compute_subgroup_metrics`, `classify_trigger_subgroup`, `try_attach_group_to_main_bundle`, `should_merge_main_bundles`, `absorb_sparse_endpoint_points`, `regroup_trigger_cluster` | ~600 | Compatibility logic — changes when data characteristics change |
| `supports_core.py` (slimmed) | `rebuild_cluster_records`, `build_standard_support_record`, `build_trigger_support_records`, `build_support_record`, `build_supports_payload` | ~300 | Orchestration + assembly — stable API surface |
| `supports_export.py` | `export_npz`, `export_support_geometry_xyz`, `export_trigger_group_classes_xyz`, visualization helpers | ~200 | I/O — independent of algorithm |

### 5.2 Parameter extraction

`DEFAULT_FIT_PARAMS` should be moved from `supports_core.py` to a dedicated location where Phase 3 can wire it into the config system:

- Extract to `params.py` or a params section in a future config module
- All 20+ parameters become named, documented, with type hints
- Default values preserved exactly (behavioral equivalence)

### 5.3 Stage 2 trigger params

The trigger threshold hardcodes in `cluster_boundary_centers()` (lines 192-195) should be extracted to named parameters with defaults, parallel to Stage 3's `DEFAULT_FIT_PARAMS` pattern. This prepares Phase 3's config injection without changing behavior.

### 5.4 Cross-stage contract enforcement

Add schema validation functions in `stage_io.py`:
- `validate_boundary_centers(payload) → bool` — check required fields, shapes, dtypes
- `validate_local_clusters(payload, boundary_centers) → bool` — check center_index bounds
- `validate_supports(payload) → bool` — check segment offset/length consistency

These are infrastructure, not algorithm — they enforce the hidden contracts identified in Section 3 above.

---

## 6. Ordering Constraints and Dependencies

### 6.1 What must happen first

1. **Behavioral audit documentation** — Before any code moves, document what each compatibility block does and why. This is the foundation for verifying behavioral preservation.
2. **Parameter extraction** — Move hardcoded parameters to named, documented locations (but keep default values). This is zero-risk and unblocks both module splitting and Phase 3's config system.

### 6.2 What can happen in parallel (after audit + param extraction)

- Stage 3 module splitting (`supports_core.py` → `fitting.py` + `trigger_regroup.py` + slimmed `supports_core.py` + `supports_export.py`)
- Stage 2 trigger param extraction
- Cross-stage contract validation functions in `stage_io.py`
- `_bootstrap.py` replacement with proper package structure
- CLI argument normalization

### 6.3 What must happen last

- Informal behavioral equivalence spot-check: run refactored pipeline on 020101/020102, diff against pre-refactor outputs
- Update `PIPELINE.md` and `REFACTOR_TARGET.md` to reflect new structure

### 6.4 What must NOT happen in Phase 2

- Changing any default parameter value
- Changing any algorithm behavior
- Removing any compatibility logic (even if it looks "wrong")
- Adding density-adaptive parameters (Phase 3 injection points only)
- Modifying the NPZ schema (field names, shapes, dtypes)
- Changing output file names or formats

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Stage 3 split introduces import error | Medium | Low (caught immediately) | Run all 4 stages on test scene after each split |
| Moving trigger_regroup breaks internal state | Medium | Medium | Test with trigger-heavy scenes (020101 has trigger clusters) |
| Parameter extraction changes float precision | Low | High (breaks equivalence) | Use exact same literal values, same types |
| `_bootstrap.py` removal breaks script invocation | Low | Low | Test all script entry points |
| build_support_dataset_v3.py breaks after core restructure | Medium | Medium | Update in-memory call sites in same PR |
| Missing an implicit cross-stage contract | Medium | High (silent data corruption) | Section 3 documents all known contracts; add validation |

---

## 8. Test Scenes and Verification Data

**Available scenes:**
- `samples/training/020101/` — training sample with existing edge data
- `samples/validation/020102/` — validation sample with existing edge data

**Pre-refactor baseline outputs to capture:**
- `boundary_centers.npz` — Stage 1 output
- `local_clusters.npz` — Stage 2 output
- `supports.npz` — Stage 3 output
- `edge_*.npy` (all 8 arrays) — Stage 4 output

**Verification approach for Phase 2 (informal, per D-08/D-09 from CONTEXT.md):**
- Run refactored pipeline on both test scenes
- Binary diff of all NPZ/NPY outputs against pre-refactor baseline
- Any difference = refactor introduced a behavioral change = must be investigated and either fixed or explicitly flagged as deferred to Phase 3

---

## 9. Planner Guardrails

The planner MUST respect these constraints when creating tasks:

1. **A/B boundary rule:** No task may change default algorithm output. Every code change must be structure-only.
2. **Behavioral documentation before code changes:** Audit docs must exist before any module is split or moved.
3. **Parameter extraction preserves exact values:** Default parameter values are behavioral contracts. Changing `0.85` to `0.85001` breaks equivalence.
4. **Stage 3 is the primary target:** 60%+ of effort should focus on `supports_core.py` decomposition. Stage 1 and Stage 4 need minimal restructuring.
5. **Cross-stage contracts must be explicitly documented:** The contracts in Section 3 must become code-level documentation or validation, not just planning artifacts.
6. **Both invocation paths must work:** Per-scene scripts AND `build_support_dataset_v3.py` must both work after refactor.
7. **Test on real data, not just code review:** Every structural change must be verified against 020101/020102 baseline outputs.
8. **No NPZ schema changes:** Field names, shapes, and dtypes in all `.npz` files must remain identical.

---

*Research completed: 2026-04-06*
*Scope: Phase 2 (behavioral audit and module restructure) of edge-data-quality-repair workstream*
