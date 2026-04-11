# Phase 1: ZAHA Offline Preprocessing Pipeline — Context

**Gathered:** 2026-04-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Produce `/home/mty0201/data/ZAHA_chunked/{training,validation,test}/<sample>__c<idx>/{coord,segment,normal}.npy` from raw PCD via a deterministic offline pipeline:

1. Parse raw PCD (ASCII, fields `x y z classification rgb`, drop `rgb`)
2. grid=0.02 voxel downsample (centroid + majority-vote segment)
3. Drop VOID points (raw ID 0)
4. Denoise (method RESEARCH-GATED)
5. Building-level chunking into sub-chunks ≤ 0.6 M pts, axis-aligned box grid + ≥2 m overlap
6. Robust normal estimation per chunk
7. Remap raw IDs 1–16 → remapped classes 0–15
8. Write NPY layout + `manifest.json` + `lofg3_to_lofg2.yaml`

Phase 1 is a pure data producer. No training, no config authoring, no BFDataset edits, no boundary/edge generation, no LoFG3→LoFG2 runtime wiring.

Downstream consumers of this output:
- **Phase 1b** — pure-semantic PTv3 on stock `DefaultDataset`
- **Phase 2** — edge.npy + boundary_mask_r060.npy generation via `rebuild_edge_dataset_inplace.py`
- **Phases 3–4** — BFDataset edge-enabled track on the same chunks

Hard rule: raw PCD is touched only in step 1 of Phase 1 and becomes provenance-only after this phase ships.

</domain>

<decisions>
## Implementation Decisions

### Ignore-index convention + VOID handling
- **D-01:** **Drop VOID points entirely in Phase 1 after 0.02 downsample, before denoise/chunk/normals.** Raw classification=0 never reaches downstream phases. Downsample runs on the full cloud first so voxel majority vote gets honest input, then any voxel whose winning label is 0 is removed.
- **D-02:** **Remap raw IDs 1–16 → remapped classes 0–15.** `segment.npy` is written with int32 values strictly in `[0, 15]`. `num_classes=16` for LoFG3. No `ignore_index` is required (no VOID remains); Pointcept configs can set `ignore_index=-1` or `255` as a defensive no-op.
- **D-03:** Because VOID is dropped, coord/segment/normal arrays in every chunk have identical length `(N,)` / `(N, 3)` where N counts only real-class points. The per-chunk point budget below applies to this post-drop count.

### OuterCeilingSurface (raw ID 14) LoFG2 bucket
- **D-04:** **Raw ID 14 (remapped class 13, `OuterCeilingSurface`) → LoFG2 bucket `'other el.'`.** Rationale recorded in `lofg3_to_lofg2.yaml`: it is a facade envelope element bridging outdoor/indoor space (loggia/passage ceiling), not a load-bearing structural member, and paper Figure 3's `'other el.'` catch-all is the closest semantic fit. Placing it in `'structural'` would risk polluting the dominant Wall/Roof class's precision without a reasoned basis.
- **D-05:** `lofg3_to_lofg2.yaml` is authored at `phases/01-zaha-offline-preprocessing-pipeline/lofg3_to_lofg2.yaml` (workstream-local) and maps **remapped classes 0–15** (NOT raw XML IDs 1–16) to LoFG2 bucket indices. The remap from raw → remapped happens first; the LoFG3→LoFG2 bucket map happens on top of the remapped space. The YAML header comment explicitly documents this ordering to prevent off-by-one bugs downstream.

### Chunking strategy
- **D-06:** **Axis-aligned box grid + fixed XY tile size + ≥2 m overlap + full Z extent** (Z-banding only if a specific building is measured to exceed budget; planner calls). Deterministic: grid origin = each building's bbox min, tile stride = (tile_size − overlap). Same inputs + same commit → identical chunk IDs and bboxes.
- **D-07:** **Point budget per chunk: ≤ 0.6 M points** (tighter than the roadmap's initial 1.0 M target). Driven by WSL ≈4 GB free RAM headroom for Phase 2's edge rebuild (which assumes it holds a whole chunk in memory) and PTv3 sphere-crop batch memory in Phase 1b/4.
- **D-08:** **Fixed tile size** across the dataset, not per-building adaptive. Planner measures post-0.02 post-VOID-drop density across all 26 samples during planning and picks the single XY tile size that keeps worst-case density under 0.6 M. This maximises determinism and makes cross-chunk audits trivial.
- **D-09:** **Keep overlap, no dedup flags in manifest.** Overlap regions may carry the same point in two chunks; training sphere-crop doesn't care. Any eval-time exact-mIoU dedup is a Phase 1b/4 concern, not Phase 1's.
- **D-10:** Chunk directory naming: `<sample>__c<idx>/` where `idx` is a zero-padded integer reflecting row-major order over the (x-tile, y-tile) grid. `_A/_B/_C` suffixes on `DEBY_LOD2_4907207` are preserved in the parent `sample` name and carry their split assignment (A→val, B→test, C→train) from the readme manifest.

### Denoising research gate
- **D-11:** **Fixed effort budget + pick-best.** Researcher evaluates exactly **3 candidates** — minimum set: SOR + radius outlier removal + one of {bilateral filter, MLS smoothing, RANSAC plane-residual filter} — on **3 fixed sample chunks** (small / medium / one slice from the 86M-point `DEBY_LOD2_4906965` sample). Ships `phases/01-zaha-offline-preprocessing-pipeline/denoising_notes.md` with per-candidate params, visual before/after per chunk, points-dropped fraction, and a one-paragraph verdict picking the winner.
- **D-12:** **Acceptance rubric for the winner:** (a) ≤ 10% of points dropped on any sample chunk (hard cap from DS-ZAHA-P1-03), (b) visible scan-stripe reduction on at least one inspected chunk, (c) visual inspection passes on all three test chunks. No quantitative stripe-energy threshold — visual-inspection bar is sufficient for Phase 1's scope.
- **D-13:** **Denoising runs pre-chunking** on the full 0.02-downsampled + VOID-dropped building cloud, once per building. This avoids chunk-boundary artifacts where a point is removed in chunk A but retained in the overlap with chunk B. Peak memory is bounded by whole-building point count post-downsample post-VOID-drop, which for typical ZAHA buildings is well under 10 M points and fits comfortably.

### Memory strategy for the 4.6 GB PCD
- **D-14:** **Streaming ASCII line-reader + chunked voxel accumulation** for raw PCD parse. Parse line-by-line; for each point, compute integer voxel key `(floor(x/0.02), floor(y/0.02), floor(z/0.02))` and aggregate into a `dict[voxel_key] → (running_coord_sum, point_count, label_histogram)`. After the whole file is streamed, finalise each voxel's centroid coord and majority-vote label in one pass. Peak memory is bounded by (a) one PCD line at a time + (b) the growing voxel dict, which for ~86 M points at grid=0.02 on a facade is likely ≤ 3 M unique voxels (~200 MB). WSL-safe even for `DEBY_LOD2_4906965.pcd`.
- **D-15:** **Do NOT use `open3d.io.read_point_cloud` for the 4.6 GB sample** — it loads the whole cloud into memory and would OOM on 4 GB free RAM. Smaller samples MAY use open3d for convenience, but the streaming path must be available and exercised on at least the 86M-point sample before the phase ships. Planner may choose to use the streaming parser uniformly across all samples for code simplicity.
- **D-16:** Voxel-dict determinism: voxel keys are integer tuples, so insertion order of equal keys is irrelevant. Majority vote ties are broken by picking the smallest raw class ID (deterministic). Centroid coord is computed as `sum / count` in float64, cast to float32 at write time.
- **D-14/D-15 supersession (2026-04-11, per RESEARCH §A/§B):** The original
  hypothesis of "~3 M unique voxels / ~200 MB" via plain Python dict
  aggregation was empirically invalidated — actual voxel retention on ZAHA
  facade data is 62-67%, which produces ~92 M voxels (17-34 GB dict state)
  for the largest file. **The implementation path is hash-partitioned
  external sort (K=16 disk bins) per RESEARCH §B.2, peak RAM ~1.5 GB.**
  Also, the actual largest sample is `DEBY_LOD2_4959458.pcd`
  (136,830,225 pts, 6.9 GB ASCII, validation split), NOT
  `DEBY_LOD2_4906965.pcd` (86 M). Streaming-path validation targets and
  memory budgets key off BOTH files. D-14's intent (streaming,
  deterministic, memory-bounded, hash-stable voxel centroids) remains
  locked; only the concrete data structure changes.

### Normal estimation
- **D-17:** **Best-effort + visual sanity + short note** (not a multi-method bake-off). Planner picks ONE method — default strong candidate: **adaptive-radius PCA** with radius auto-scaling to hold roughly k ≈ 30 neighbors — and the researcher visualizes normals on 2–3 sample chunks (small / medium / one chunk from the 86M-point building), writing `normals_notes.md` documenting the method + params + observed failure modes.
- **D-18:** **Acceptance bar:** normals are unit-length float32 `(N, 3)` with correct shape and no NaN. "Thinning the wall" is explicitly aspirational, not blocking — if thick-wall regions still show noisy normals after the best-effort method, the phase ships anyway, and the failure mode is recorded in `normals_notes.md`.
- **D-19:** Normals are computed **per chunk** after chunking (not on the whole building), because the chunk is the downstream consumer and per-chunk normals respect whatever edge effects the chunking introduces.

### Pipeline orchestration + manifest
- **D-20:** Single entry-point script under `data_pre/zaha/` that runs parse → downsample → VOID-drop → denoise → chunk → normals → write, then emits `ZAHA_chunked/manifest.json` with per-chunk metadata: `{sample, chunk_idx, bbox, point_count, denoising_method, denoising_params, normal_method, normal_params, source_pcd, commit_hash}`.
- **D-21:** **Sanity checks that gate phase completion:** (a) total per-class point histogram across chunks roughly matches raw-PCD class histogram (within VOID-drop + downsampling variance); (b) no chunk exceeds the 0.6 M point budget; (c) every chunk has `coord/segment/normal` with matching `(N,)` / `(N, 3)` shapes and unit-length normals; (d) `segment.npy` values are in `[0, 15]` with no value ≥ 16 or < 0; (e) the readme manifest's 19/4/3 sample list is fully covered (hard failure if any sample is missing from its split).
- **D-22:** **Hard-failure policy.** Any raw PCD that crashes the parser, any chunk that violates a sanity check, any file in the readme manifest that cannot be produced → Phase 1 does not complete. No skip-on-error.

### Claude's Discretion (planner may decide)
- Exact SOR/radius/bilateral/MLS parameter sweeps within the 3-candidate rubric — the researcher picks reasonable starting values per candidate, not the user.
- Adaptive-radius PCA implementation specifics (k value, initial radius, max radius cap) — default k ≈ 30, planner refines against a sanity viz.
- XY tile size (the exact meter value) — planner measures density and picks a value during planning such that the 0.6 M budget is never violated.
- Chunk index zero-padding width (3 or 4 digits depending on max chunk count per building).
- Voxel-dict implementation: plain Python dict vs `numpy.unique` batched approach — planner picks based on streaming parser performance on the 86M sample.
- Exact `manifest.json` schema fields beyond the minimum set listed in D-20.

</decisions>

<specifics>
## Specific Ideas

- User emphasised "determinism" more than once: rerunning Phase 1 with the same inputs + same commit hash MUST produce identical chunk IDs, identical bbox values, and identical normal arrays. The manifest's `commit_hash` field is the audit hook.
- User's mental model of chunking: "each building is already a natural top-level chunk; we just sub-tile it". Building-level chunking is NOT re-examining the readme split; it's sub-tiling inside each building with the building's official split assignment inherited unchanged.
- User accepts that Phase 1 is a slow, I/O-heavy one-shot — no per-phase re-runs expected once `ZAHA_chunked/` lands. Design for correctness over speed.
- User has explicitly set normals as "ship even if walls stay thick" — do not block Phase 1 on a perfect normal estimation algorithm.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Workstream planning
- `.planning/workstreams/dataset-handover-s3dis-chas/ROADMAP.md` §"Phase 1: ZAHA offline preprocessing pipeline" — phase goal, scope rules, hard constraints, reference material
- `.planning/workstreams/dataset-handover-s3dis-chas/REQUIREMENTS.md` §"ZAHA Phase 1 — Offline Preprocessing Pipeline" — requirement IDs DS-ZAHA-P1-01..07 with acceptance criteria
- `.planning/workstreams/dataset-handover-s3dis-chas/PROJECT.md` — constraints (§"Constraints"), key decisions (§"Key Decisions"), active milestone scope

### ZAHA dataset source of truth (local)
- `/home/mty0201/data/ZAHA_pcd/readme.txt` — official 19/4/3 split manifest, UTM32 shift `[690826, 5335877, 500]`, author contact `olaf.wysocki@tum.de`
- `/home/mty0201/data/ZAHA_pcd/settings_labeling.xml` — 17-class authoritative schema (0=VOID, 1–16=real classes). Supersedes paper Table 2 which is incomplete.
- `/home/mty0201/data/ZAHA_pcd/{training,validation,test}/*.pcd` — raw input, fields `x y z classification rgb`, `SIZE 4 4 4 1 4`, `TYPE F F F U U`, DATA ascii

### ZAHA paper + repo
- Wysocki et al., WACV 2025 — arxiv `2411.04865` — §3 "no spectral information" (RGB drop rationale), Table 3 LoFG3 baselines, Table 4 LoFG2 baselines, Figure 3 LoFG3→LoFG2 mapping
- Repo `https://github.com/OloOcki/zaha` — documentation-only; no loader/config/YAML. Any tooling is built locally.
- Benchmark `https://tum2t.win/benchmarks/pc-fac` — leaderboard reference only

### Related existing code (read for patterns, not reused directly)
- `data_pre/bf_edge_v3/scripts/rebuild_edge_dataset_inplace.py` — Phase 2 consumer; establishes the directory-walking contract Phase 1 must match (`<root>/<split>/<sample>/`)
- `data_pre/chas/explore_chas*.py` — prior open3d usage in this repo (example of how o3d is imported/used in the `data_pre/` tree)
- `project/datasets/bf.py` — BFDataset that will consume the chunks in Phase 3 via `edge.npy`; Phase 1 does NOT touch this file

### Prior workstream archaeology
- `.planning/workstreams/dataset-handover-s3dis-chas/archive/01-s3dis-data-layout-and-edge-generation-SUPERSEDED-2026-04-11/` — the superseded s3dis-first Phase 1. Read only for the edge-rebuild + BFDataset contract patterns; ZAHA-specific decisions do NOT apply.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **open3d** already imported under `data_pre/chas/explore_chas*.py` — reusable for voxel ops and PCA normal estimation on **smaller** samples. NOT safe for the 4.6 GB sample (see D-15).
- **`data_pre/bf_edge_v3/scripts/rebuild_edge_dataset_inplace.py`** — Phase 2 consumer; directory-walking contract `<root>/<split>/<sample>/` is what Phase 1's output MUST match.
- **Pointcept `GridSample` transform pattern** (referenced in `configs/semantic_boundary/clean_reset_s38873367/clean_reset_data_g04.py`) — for runtime voxelization, NOT for offline downsampling. Phase 1 implements its own deterministic streaming voxel aggregation; the runtime GridSample is a separate concern.

### Established Patterns
- Dataset producers in this repo (`data_pre/bf_edge_v3/`, `data_pre/chas/`) live under `data_pre/<dataset_name>/` with subdirectories like `scripts/`, `utils/`, `docs/`. Phase 1 should follow the same layout: `data_pre/zaha/{scripts/,utils/,docs/}`.
- NPY-per-field layout `<sample>/{coord,segment,normal}.npy` is the stock Pointcept `DefaultDataset` contract; Phase 1's output slots in without any new dataset class.
- `segment.npy` is int32 in the existing BFDataset track (verifiable via `project/datasets/bf.py`); Phase 1 should write int32 to match.

### Integration Points
- Output root `/home/mty0201/data/ZAHA_chunked/` is consumed by Phase 1b's `DefaultDataset` config and Phase 2's edge rebuild. The readme split manifest is the ground truth for which samples land in which split directory.
- `lofg3_to_lofg2.yaml` lives in the phase dir and is referenced at runtime by Phase 1b (DS-ZAHA-P1B-01) and Phase 3 (DS-ZAHA-P3-01) configs. The remap-at-runtime wiring is NOT a Phase 1 concern; Phase 1 only authors the YAML.
- `manifest.json` is consumed by human audits (not by any training code) — it is the provenance surface for Phase 1 reproducibility.
- No `CLAUDE.md` update is expected from Phase 1. Workstream docs (ROADMAP/REQUIREMENTS/PROJECT/STATE) will be updated by the planner/executor at the usual GSD checkpoints.

</code_context>

<deferred>
## Deferred Ideas

- **edge.npy generation** → Phase 2 (DS-ZAHA-P2-01). Phase 1 does not touch the edge supervision path.
- **boundary_mask_r060.npy generation** → Phase 2 (DS-ZAHA-P2-03). Uses r=0.06 m absolute (BFANet truth), computed on the chunked clouds.
- **Pure-semantic ZAHA PTv3 config + training** → Phase 1b (DS-ZAHA-P1B-01..03).
- **BF-style ZAHA config + training** → Phases 3–4 (DS-ZAHA-P3-01, DS-ZAHA-P4-01).
- **Email to olaf.wysocki@tum.de** asking how `OuterCeilingSurface` was bucketed in their paper's LoFG2 numbers — backlog item. Phase 1 ships with the `'other el.'` decision (D-04) and revisits only if Phase 4's per-class IoU reveals a clear anomaly.
- **LoFG3 training (16 classes)** → deferred behind Phase 4 (LoFG2 first). Not a Phase 1 concern; the raw 16-class label space is already what `segment.npy` carries.
- **Quantitative stripe-energy denoising bar** — considered and explicitly rejected in D-12 in favor of visual inspection. If Phase 1b/4 shows denoising is a mIoU-limiter, revisit as a follow-up phase.
- **Multi-method normal estimation bake-off** — considered and explicitly rejected in D-17 (aspirational, not blocking). Revisit only if Phase 4 results show normal quality is a mIoU-limiter.
- **Eval-time overlap de-duplication** — considered and explicitly deferred in D-09. A Phase 1b/4 eval concern, not Phase 1's.
- **Per-building adaptive tile size** — considered and explicitly rejected in D-08 in favor of a single fixed tile size for determinism. Revisit only if worst-case building density forces excessive over-chunking.
- **Z-banding inside each tile** — available as a planner fallback in D-06 if some buildings' full Z extent still exceeds the 0.6 M budget. Planner decides during density measurement.
- **s3dis handover** — deferred behind ZAHA Phase 4 (workstream-level decision, not a Phase 1 concern).

</deferred>

---

*Phase: 01-zaha-offline-preprocessing-pipeline*
*Context gathered: 2026-04-11*
