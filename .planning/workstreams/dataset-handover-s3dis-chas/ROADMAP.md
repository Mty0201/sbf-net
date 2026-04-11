# Roadmap: dataset-handover-s3dis-chas

**Main line:** ZAHA (raw PCD facade dataset). s3dis deferred.
**Data root:** `/home/mty0201/data/` (moved 2026-04-11 from `/mnt/e/WSL/data/`).
**Route (2026-04-11):** Option C — dedicated offline preprocessing phase producing chunked NPY, followed by a parallel pure-semantic baseline (PTv3 + `DefaultDataset`) *and* the edge-enabled BF track (`BFDataset`) both consuming the same chunked dataset.

## v1.0 — ZAHA + s3dis Handover

**Goal:** Land ZAHA as a trainable dataset through two complementary tracks on a single shared offline-preprocessed chunk layout:
1. A **pure-semantic reference** — vanilla PTv3 with `DefaultDataset`, no edge supervision. Establishes the simplest possible sbf-net baseline on facade data.
2. An **edge-enabled track** — the existing `BFDataset` path (edge.npy hard-required) reusing the same chunks once Phase 1 lands.

s3dis handover is deferred behind ZAHA P4 and will be re-authored in a later phase block.

**Approach:** Preprocessing is treated as its own phase because the five known ZAHA pathologies (scale, density gradient, thick walls, scan stripes, noise) are all preprocessing concerns and belong in one place. The pure-semantic baseline is split into its own phase so the project has a clean PTv3 number on facade data *before* anything edge-specific is touched.

**Scope rules:**
- `BFDataset` must remain unchanged. The pure-semantic track uses Pointcept's stock `DefaultDataset`, not a modified BFDataset — this is the cleanest way to get a PTv3 baseline without loosening BFDataset's hard-required `edge.npy` contract.
- All four ZAHA phases consume the **same** Phase 1 chunked output. Phase 1 is the single source of truth for preprocessed ZAHA.
- Raw 136.8 M-point samples MUST NOT be touched by any training or evaluation path. Only chunks leave Phase 1.

**Reference material (ZAHA):**
- Paper: Wysocki et al., WACV 2025, arxiv 2411.04865
- Repo: https://github.com/OloOcki/zaha (documentation-only; no loader, no config, no split file)
- Leaderboard: https://tum2t.win/benchmarks/pc-fac
- Published LoFG2 baselines (paper Table 4 µIoU): PointNet 55.8, PointNet++ 59.8, PT 63.9, **DGCNN 68.5** (winner), KPConv 52.3
- Published LoFG3 baselines (paper Table 3 µIoU): PointNet 26.4, PointNet++ 25.6, **PT 41.6** (winner), DGCNN 33.4, KPConv 28.6
- Training budget in paper: fixed 100 epochs, batch 32 × 1024 pts/sample (matches our v2.0 global epoch limit)
- **Authoritative dataset files (local):** `/home/mty0201/data/ZAHA_pcd/readme.txt` (split manifest + UTM32 shift), `/home/mty0201/data/ZAHA_pcd/settings_labeling.xml` (17-class schema: 0=VOID, 1–16 real classes — supersedes paper Table 2 which is incomplete)

**Known ZAHA pathologies driving this roadmap (2026-04-11):**
- **Scale:** largest sample ~136.8 M points (`DEBY_LOD2_4959458.pcd`, superseded prior claim of `DEBY_LOD2_4906965` ~86 M on 2026-04-11; both files are 4+ GB ASCII). Training-time random sphere cropping against raw samples is not viable — memory will fail.
- **Density gradient:** ground/low regions are extremely dense, high facade regions are sparse. Any fixed-radius neighborhood method misbehaves on at least one end.
- **Thick walls:** facades have 10–30 cm MLS thickness. Small-radius PCA normals collapse into noise directions. Prior sbf-net experiments already confirmed this failure mode.
- **Scan stripes:** the raw PCD contains visible stripe patterns from MLS scan lines. These amplify density gradient and bias neighborhood estimators.
- **Noise:** raw data has non-trivial outliers. Untreated, these poison both normals and chunking statistics.

The user has explicitly authorized: (a) grid=0.02 voxel downsampling as the authoritative input for all downstream work (raw PCD becomes source-of-truth only for provenance), (b) building-level offline chunking, (c) "thinning the wall if possible; acceptable if not" as an aspirational — not blocking — goal for normals/denoising.

---

### — ZAHA Handover (Phases 1–4B) —

*Goal: Land ZAHA as a trainable dataset and record BOTH a pure-semantic PTv3 reference and an edge-enabled BF baseline under a shared offline-preprocessed chunk layout.*

### Phase 1: ZAHA offline preprocessing pipeline

**Goal:** Produce `/home/mty0201/data/ZAHA_chunked/{training,validation,test}/<sample_chunk>/` containing `coord.npy`, `segment.npy`, and `normal.npy` for every building-level chunk of every ZAHA sample. Derived from raw PCD via a deterministic, reproducible offline pipeline: (1) parse raw PCD → (2) grid=0.02 voxel downsample → (3) denoise → (4) building-level chunking → (5) robust normal estimation → (6) write NPY layout. Deliver a filesystem state where `ZAHA_chunked/` is the single authoritative ZAHA input for every downstream phase in this workstream.

**Requires:** DS-ZAHA-P1-01 (raw PCD parse), DS-ZAHA-P1-02 (0.02 downsample), DS-ZAHA-P1-03 (denoise research + method selection), DS-ZAHA-P1-04 (building-level chunking), DS-ZAHA-P1-05 (robust normals), DS-ZAHA-P1-06 (NPY layout + LoFG3→LoFG2 YAML), DS-ZAHA-P1-07 (pipeline orchestration script + reproducibility)
**Depends on:** Nothing — first phase in the workstream
**Status:** Planned (2026-04-11)
**Role in milestone:** Single source of truth for preprocessed ZAHA. Every subsequent phase consumes `ZAHA_chunked/`. Raw `ZAHA_pcd/` is frozen after Phase 1 lands.

**Plans:** 4 plans

Plans:
- [x] 01-01-PLAN.md — Wave 0: package skeleton, pytest install, RED test stubs, lofg3_to_lofg2.yaml, upstream doc-drift fixes (DS-ZAHA-P1-06)
- [x] 01-02-PLAN.md — Wave 1: streaming PCD parser + hash-partitioned external-sort voxel aggregation with deterministic tie-break and VOID drop (DS-ZAHA-P1-01, DS-ZAHA-P1-02)
- [x] 01-03-PLAN.md — Wave 2: denoise (SOR winner, nb=30/std=2.0) + box-grid chunker + open3d-first KNN=30 normals, 2 human-verify checkpoints approved (DS-ZAHA-P1-03, DS-ZAHA-P1-04, DS-ZAHA-P1-05)
- [ ] 01-04-PLAN.md — Wave 3: layout/manifest writers + build_zaha_chunks.py orchestrator + golden file + 26-sample full run + 136.8 M-point DEBY_LOD2_4959458 dry-run checkpoint (DS-ZAHA-P1-06, DS-ZAHA-P1-07)

Scope notes:
- **Input:** `/home/mty0201/data/ZAHA_pcd/{training,validation,test}/*.pcd`, fields `x y z classification rgb`, SIZE `4 4 4 1 4`, TYPE `F F F U U`. Drop `rgb` entirely (paper §3: no spectral info). `classification` → `segment.npy`. Coordinates in meters, local CRS (do NOT apply UTM32 shift).
- **Downsampling (step 2):** grid=0.02 voxel downsampling, centroid average, before any other preprocessing step. Rationale: raw PCD has extreme density pathologies that break downstream neighborhood stats; 0.02 is the smallest grid the user has validated as "acceptable loss" for facade detail. After this step, sample sizes drop from ~136.8 M to a tractable range and all later statistics are computed on the downsampled cloud. Raw PCD remains source-of-truth only for provenance auditing. Implementation uses hash-partitioned external sort (K=16 disk bins) to hold peak RAM ~1.5 GB per worker on the 136.8 M-point sample — supersedes the earlier plain-Python-dict plan in CONTEXT D-14 per RESEARCH §B.2.
- **Denoising (step 3, RESEARCH-GATED):** method explicitly TBD and treated as a Phase 1 research sub-task, not an implementation detail. Candidates include statistical outlier removal (SOR), radius outlier removal, bilateral filtering, or MLS smoothing. The density gradient and the scan-stripe signature together mean a single-parameter SOR will likely fail at one end of the density spectrum. Deliverable: a short research note (`denoising_notes.md`) under the phase directory documenting the chosen method and its parameter selection rationale, before the method is wired into the pipeline. Human-verify checkpoint required before Wave 3 runs.
- **Chunking (step 4):** building-level chunking — each ZAHA sample is already a full building; splits already honor `_A/_B/_C` chunking for `DEBY_LOD2_4907207`. Within a building, offline chunks are produced by axis-aligned spatial partitioning (box grid with overlap) sized so that no single chunk exceeds a trainable point budget (target: ≤600 k pts/chunk post-0.02 per CONTEXT D-07). Chunking is deterministic row-major x-outer y-inner with 4 m tile + 2 m overlap; Z is full range. Hard rule: rerunning Phase 1 with the same inputs and commit hash must produce the exact same chunk layout (determinism regression is enforced via golden file).
- **Normal estimation (step 5):** method must be robust to (a) 10–30 cm wall thickness, (b) density gradient, (c) scan-stripe banding. Prior sbf-net experience flagged small-radius PCA as unreliable; Phase 1 uses open3d `KDTreeSearchParamKNN(knn=30)` per-chunk after chunking (adaptive radius via fixed k) with degenerate fallback for rank-deficient neighborhoods. User has explicitly set the expectation that "thinning the wall" is a nice-to-have, not a Phase 1 blocker — if normals remain noisy after a best-effort method, the phase still ships. Deliverable: a short note (`normals_notes.md`) documenting the chosen method and a sanity check (normals visualized on 2–3 sample chunks). Human-verify checkpoint required before Wave 3 runs.
- **Output layout (step 6):** `/home/mty0201/data/ZAHA_chunked/{training,validation,test}/<sample>__c<idx>/{coord,segment,normal}.npy`. `segment.npy` is written as `int32` strictly in `[0, 15]` (raw XML IDs 1–16 remapped to 0–15, VOID class 0 dropped at voxel stage). Runtime LoFG2 folding is a config concern. A workstream-local `lofg3_to_lofg2.yaml` is authored in Phase 1 alongside the data, transcribed from paper Figure 3 and explicitly documenting the `OuterCeilingSurface` (ID 14) LoFG2 bucket decision with rationale.
- **Pipeline orchestration (step 7):** single entry-point script `data_pre/zaha/scripts/build_zaha_chunks.py` that reads raw PCD, runs all five preprocessing steps, writes NPY output, and emits a manifest (`ZAHA_chunked/manifest.json`) with per-chunk metadata: source building, chunk index, bbox, post-0.02 point count, denoising method + params, normal method + params, sha256 of coord/segment/normal, commit hash. Sanity checks (D-21): (a) total class histogram across chunks within warn/fail drift thresholds of raw histogram, (b) no chunk exceeds the point budget, (c) every chunk has coord/segment/normal with matching `(N,)` / `(N,3)` shapes, (d) readme coverage is 100 %. Parallelism: `ProcessPoolExecutor(max_workers=4)` for the 24 small samples, serial for the 2 large samples (DEBY_LOD2_4906965, DEBY_LOD2_4959458) to stay within the WSL RAM cap. Content-hash idempotence via `<output>/<split>/.state/<sample>.json` sidecars.
- **Out of scope for Phase 1:** edge.npy generation, boundary_mask, any training-time transform, any config change, any model change, BFDataset touches. Phase 1 is purely a data-producer phase.
- **Failure policy:** hard-failure on any sample. If even one raw PCD crashes the pipeline, Phase 1 does not complete. The 136.8 M-point sample must be handled with streaming IO (chunksize=2 M) and external-sort voxel aggregation.

---

### Phase 1b: ZAHA pure-semantic PTv3 baseline (DefaultDataset track)

**Goal:** Register a ZAHA-specific `DefaultDataset` config pointing at Phase 1's `ZAHA_chunked/` layout and record a pure-semantic PTv3 LoFG2 baseline mIoU — no edge loss, no boundary_mask, no BFDataset. Deliver a training log, best val mIoU, per-class IoU at best epoch, and a short canonical note comparing against paper Table 4 LoFG2 baselines. This is the simplest end-to-end reference on facade data and the reference number the edge-enabled BF track is later evaluated against.

**Requires:** DS-ZAHA-P1B-01 (DefaultDataset config), DS-ZAHA-P1B-02 (smoke validation on large chunk + val dataloader), DS-ZAHA-P1B-03 (short-training run + baseline record)
**Depends on:** Phase 1 (needs `ZAHA_chunked/` populated)
**Status:** Not started
**Role in milestone:** Pure-semantic reference. Establishes that PTv3 on offline-chunked ZAHA actually trains and that there is a meaningful number to improve against before the BF edge-loss machinery is introduced. Without this, any later BF result on ZAHA would be impossible to attribute.

Scope notes:
- **Dataset class:** stock Pointcept `DefaultDataset` (via `project/datasets/` reusing Pointcept's `DefaultDataset` directly — no new dataset class). Phase 1 chunks are already in the expected `<root>/<split>/<sample>/{coord,segment,normal}.npy` layout, which is `DefaultDataset`-compatible without any change.
- **Config:** `project/configs/zaha/pure_semantic_ptv3_lofg2.py` — clone of the existing `configs/bf/semseg-pt-v3m1-0-base-bf.py` with: dataset class swapped to `DefaultDataset`, data root pointed at `ZAHA_chunked/`, `num_classes=5` (LoFG2), `ignore_index` aligned with planner decision, `feat_keys` = `coord/normal` (no color), LoFG3→LoFG2 remap applied via the Phase 1 YAML, edge-related loss weights set to zero / removed. No loss customization beyond semantic CE.
- **Smoke validation:** one full forward+backward+optimizer step on both train and val dataloaders, including one large-building chunk, before the short training is kicked off.
- **Training budget:** ≤100 epochs (global v2.0 limit), single seed, default optimizer schedule.
- **Deliverables:** `outputs/zaha_pure_semantic_ptv3_lofg2/` training log, best val mIoU + epoch, per-class IoU, short canonical note under `docs/canonical/` comparing against paper Table 4 (PT 63.9, DGCNN 68.5). The sbf-net pure-semantic baseline is NOT required to match or beat the paper — goal is a reproducible reference.
- **Out of scope:** edge.npy, boundary_mask, BFDataset, LoFG3 training, any loss tuning.

---

### Phase 2: ZAHA edge generation (edge-enabled track foundation)

**Goal:** Generate `edge.npy` for every chunk under `/home/mty0201/data/ZAHA_chunked/{training,validation}/` using `data_pre/bf_edge_v3/scripts/rebuild_edge_dataset_inplace.py`. Deliver a filesystem state where every training + validation chunk has a valid `edge.npy` of shape `(N, 5)`.

**Requires:** DS-ZAHA-P2-01 (edge rebuild on chunked layout), DS-ZAHA-P2-02 (sanity pass + Stage 4 retune), DS-ZAHA-P2-03 (boundary_mask_r060.npy generation for the chunks, matching the BFANet r=0.06m absolute truth)
**Depends on:** Phase 1 (chunks must exist) AND Phase 1b's pure-semantic baseline number (to have a reference to compare the BF track against later)
**Status:** Not started
**Role in milestone:** Edge-supervision foundation for the BF track. Phase 3 cannot start until BFDataset can find `edge.npy` for every training + validation chunk. Chunks (not raw samples) are the target because edge rebuild assumes it can hold the whole cloud in memory for stage 1–4, which rules out 136.8 M-point raw samples.

Scope notes:
- `rebuild_edge_dataset_inplace.py` iterates `<input>/training/*/` and `<input>/validation/*/` — test split is intentionally excluded (BFDataset does not load it). The script is consumed unchanged; the only Phase 2 concern is input path + Stage 4 retuning.
- Stage 1–3 parameters are scale-adaptive (`scale × global_median_spacing`) and should work out-of-box on 0.02-downsampled ZAHA. Stage 4 absolute-distance params (`support_radius=0.08 m`, `support_sigma=0.02 m`) were tuned for BF indoor millimeter scale and very likely need retuning for ZAHA facade meter scale. Retuning is allowed as config, not code changes.
- Sanity pass first: 3 training + 1 validation chunk + the `DEBY_LOD2_4907207_*` chunks. Measure `edge_valid` ratio distribution. If median < 5%, halt and retune Stage 4 before the full pass.
- `boundary_mask_r060.npy`: BFDataset reads this optional file. Must be generated against the chunked clouds using r=0.06m absolute (BFANet truth — do NOT derive from voxel ratio). This is the `boundary_mask` hook the user explicitly listed as required.
- Hard-failure policy: any chunk that errors or fails the `(N, 5)` shape check blocks phase completion.

---

### Phase 3: ZAHA BF-style config and smoke validation

**Goal:** Author a BF-style ZAHA training config pointing at the Phase 2 chunked + edge-enriched layout and smoke-validate the full load + forward + backward + optimizer path on both train and val dataloaders. Deliver a config that can be handed to `tools/train.py` and runs at least one complete training step without error.

**Requires:** DS-ZAHA-P3-01 (BF-style ZAHA config), DS-ZAHA-P3-02 (smoke validation on large chunk + val dataloader)
**Depends on:** Phase 2 (edge.npy + boundary_mask must exist before BFDataset will load)
**Status:** Not started
**Role in milestone:** Config-side contract for the edge-enabled track. Establishes that `BFDataset` consumes ZAHA chunks correctly and that the edge-enabled training loop is wired end-to-end before any real training is attempted.

Scope notes:
- Config is a minimal clone of the existing BF config with dataset-specific fields changed: `num_classes=5` (LoFG2 first), `ignore_index` consistent with Phase 1b, split roots under `ZAHA_chunked/`, `feat_keys` = `coord/normal` (no color), dataset class = `BFDataset`, segment remapping via the Phase 1 YAML.
- No loss, trainer, or evaluator customizations beyond the existing BF defaults. The goal is to prove the data plumbing, not to tune anything.
- Smoke validation must cover (a) a large-building chunk to verify memory behavior at chunked ZAHA scale, and (b) the validation dataloader to catch val-only failures early.
- LoFG3 config (16 real classes + VOID) is deferred to a future sub-phase after LoFG2 works end-to-end on both tracks.

---

### Phase 4: ZAHA short-training LoFG2 baseline (edge-enabled BF track)

**Goal:** Run a short ZAHA training under the Phase 3 config and record the edge-enabled BF track's baseline validation mIoU. Deliver a training log, best val mIoU, per-class IoU at best epoch, and a short canonical note comparing against (a) paper Table 4 LoFG2 baselines and (b) the Phase 1b pure-semantic reference.

**Requires:** DS-ZAHA-P4-01 (baseline training + record)
**Depends on:** Phase 3 (smoke must pass) AND Phase 1b (pure-semantic reference must exist so the BF number is attributable)
**Status:** Not started
**Role in milestone:** Edge-enabled reference baseline. Completes the handover by delivering the paired (pure-semantic, edge-enabled) reference numbers on chunked ZAHA. Future work (edge-loss ablations on facade data, LoFG3 training, s3dis handover) compares against this pair.

Scope notes:
- Training budget: ≤100 epochs (matches paper's fixed 100-epoch recipe and our v2.0 global limit). No tuning beyond what the BF-style config defaults provide.
- Deliverables: `outputs/zaha_bf_ptv3_lofg2/` training log, best val mIoU with epoch number, per-class IoU table across 5 LoFG2 classes, and a short note under `docs/canonical/` recording the baseline with a three-way comparison (sbf-net pure-semantic, sbf-net BF track, paper Table 4).
- The sbf-net BF track is not required to beat the pure-semantic reference or the paper — the goal is a reproducible paired reference on facade data. A meaningful gap (either direction) should be investigated in a follow-up phase, not within this one.

---

### — s3dis Handover (Phases 5+, Deferred) —

*s3dis phases will be appended here after ZAHA Phase 4 lands. Prior scope notes (to be refined when s3dis is authored):*
- *In-place reorg: `/home/mty0201/data/s3dis/Area_{1..6}/<room>/` → `training/Area_{N}_{room}/` + `validation/Area_5_{room}/`. Room-name collisions handled via `Area_{N}_` prefix.*
- *edge.npy generation via the same `rebuild_edge_dataset_inplace.py` tool with default Stage 1–4 params (indoor scale matches BF better than facades do).*
- *BF-style config with `num_classes=13`, `feat_keys` = `coord/color/normal`.*
- *Short-training baseline on Area_5.*
- *The prior s3dis Phase 1 CONTEXT.md is archived at `archive/01-s3dis-data-layout-and-edge-generation-SUPERSEDED-2026-04-11/`; paths will need updating to `/home/mty0201/data/s3dis/` when s3dis is re-authored.*
