---
phase: 1
slug: zaha-offline-preprocessing-pipeline
status: active
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-11
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

Derived from `01-RESEARCH.md` §J (Validation Architecture). Planner will wire per-task `<automated>` verify commands to the entries below.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (Wave 0 installs if missing from `ptv3` conda env) |
| **Config file** | `data_pre/zaha/pyproject.toml` (Wave 0 creates; may reuse repo-root if present) |
| **Quick run command** | `pytest data_pre/zaha/tests/ -x --tb=short` |
| **Full suite command** | `pytest data_pre/zaha/tests/ --tb=long` |
| **End-to-end smoke command** | `python data_pre/zaha/scripts/build_zaha_chunks.py --input /home/mty0201/data/ZAHA_pcd --output /tmp/zaha_chunked_smoke --samples DEBY_LOD2_4907179` |
| **Estimated runtime** | Quick: <10 s · Full: <2 min · E2E smoke: ~30–70 s on `DEBY_LOD2_4907179` (1.7 M pts) |

---

## Sampling Rate

- **After every task commit:** Run `pytest data_pre/zaha/tests/test_pcd_parser.py data_pre/zaha/tests/test_voxel_agg.py data_pre/zaha/tests/test_chunking.py -x` (unit-only, <10 s)
- **After every plan wave:** Run `pytest data_pre/zaha/tests/ -x` (full unit + integration on smallest sample, <2 min)
- **Before `/gsd-verify-work`:** Full suite green + end-to-end on smallest sample green + human sign-off on `denoising_notes.md` and `normals_notes.md` visual inspections + 136.8 M-point dry-run on `DEBY_LOD2_4959458` completes.
- **Max feedback latency:** 10 s per commit, 120 s per wave.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| P01-parser-01 | 01-02 | 1 | DS-ZAHA-P1-01 | — | N/A | unit | `pytest data_pre/zaha/tests/test_pcd_parser.py::test_header_roundtrip -x` | ✅ W0 | ⬜ pending |
| P01-parser-02 | 01-02 | 1 | DS-ZAHA-P1-01 | — | N/A | unit | `pytest data_pre/zaha/tests/test_pcd_parser.py::test_count_match -x` | ✅ W0 | ⬜ pending |
| P01-parser-03 | 01-02 | 1 | DS-ZAHA-P1-01 | — | N/A | unit | `pytest data_pre/zaha/tests/test_pcd_parser.py::test_binary_rejected -x` | ✅ W0 | ⬜ pending |
| P01-parser-04 | 01-02 | 1 | DS-ZAHA-P1-01 | — | N/A | unit | `pytest data_pre/zaha/tests/test_pcd_parser.py::test_rgb_dropped -x` | ✅ W0 | ⬜ pending |
| P01-parser-05 | 01-02 | 1 | DS-ZAHA-P1-01 | — | N/A | unit | `pytest data_pre/zaha/tests/test_pcd_parser.py::test_header_field_types -x` | ✅ W0 | ⬜ pending |
| P01-voxel-01 | 01-02 | 1 | DS-ZAHA-P1-02 | — | N/A | unit | `pytest data_pre/zaha/tests/test_voxel_agg.py::test_determinism -x` | ✅ W0 | ⬜ pending |
| P01-voxel-02 | 01-02 | 1 | DS-ZAHA-P1-02 | — | N/A | unit | `pytest data_pre/zaha/tests/test_voxel_agg.py::test_tie_break -x` | ✅ W0 | ⬜ pending |
| P01-voxel-03 | 01-02 | 1 | DS-ZAHA-P1-02 | — | N/A | unit | `pytest data_pre/zaha/tests/test_voxel_agg.py::test_hash_partition -x` | ✅ W0 | ⬜ pending |
| P01-voxel-04 | 01-02 | 1 | DS-ZAHA-P1-02 | — | N/A | unit | `pytest data_pre/zaha/tests/test_voxel_agg.py::test_void_drop_order -x` | ✅ W0 | ⬜ pending |
| P01-denoise-01 | 01-03 | 2 | DS-ZAHA-P1-03 | — | N/A | integration | `pytest data_pre/zaha/tests/test_denoise.py::test_drop_cap -x` | ✅ W0 | ⬜ pending |
| P01-denoise-02 | 01-03 | 2 | DS-ZAHA-P1-03 | — | N/A | unit | `pytest data_pre/zaha/tests/test_denoise.py::test_determinism -x` | ✅ W0 | ⬜ pending |
| P01-denoise-manual | 01-03 | 2 | DS-ZAHA-P1-03 | — | N/A | **manual** | see `denoising_notes.md` | — | ⬜ pending |
| P01-chunk-01 | 01-03 | 2 | DS-ZAHA-P1-04 | — | N/A | unit | `pytest data_pre/zaha/tests/test_chunking.py::test_deterministic -x` | ✅ W0 | ⬜ pending |
| P01-chunk-02 | 01-03 | 2 | DS-ZAHA-P1-04 | — | N/A | integration | `pytest data_pre/zaha/tests/test_chunking.py::test_budget -x` | ✅ W0 | ⬜ pending |
| P01-chunk-03 | 01-03 | 2 | DS-ZAHA-P1-04 | — | N/A | unit | `pytest data_pre/zaha/tests/test_chunking.py::test_overlap -x` | ✅ W0 | ⬜ pending |
| P01-normal-01 | 01-03 | 2 | DS-ZAHA-P1-05 | — | N/A | unit | `pytest data_pre/zaha/tests/test_normals.py::test_unit_length -x` | ✅ W0 | ⬜ pending |
| P01-normal-02 | 01-03 | 2 | DS-ZAHA-P1-05 | — | N/A | unit | `pytest data_pre/zaha/tests/test_normals.py::test_no_nan -x` | ✅ W0 | ⬜ pending |
| P01-normal-03 | 01-03 | 2 | DS-ZAHA-P1-05 | — | N/A | unit | `pytest data_pre/zaha/tests/test_normals.py::test_degenerate_fallback -x` | ✅ W0 | ⬜ pending |
| P01-normal-manual | 01-03 | 2 | DS-ZAHA-P1-05 | — | N/A | **manual** | see `normals_notes.md` | — | ⬜ pending |
| P01-layout-01 | 01-04 | 3 | DS-ZAHA-P1-06 | — | N/A | integration | `pytest data_pre/zaha/tests/test_output_layout.py::test_file_structure -x` | ✅ W0 | ⬜ pending |
| P01-layout-02 | 01-04 | 3 | DS-ZAHA-P1-06 | — | N/A | integration | `pytest data_pre/zaha/tests/test_output_layout.py::test_segment_range -x` | ✅ W0 | ⬜ pending |
| P01-yaml-01 | 01-01 | 0 | DS-ZAHA-P1-06 | — | N/A | unit | `pytest data_pre/zaha/tests/test_yaml.py::test_schema -x` | ✅ W0 | ⬜ pending |
| P01-e2e-01 | 01-04 | 3 | DS-ZAHA-P1-07 | — | N/A | integration | `pytest data_pre/zaha/tests/test_e2e.py::test_smallest_sample -x` | ✅ W0 | ⬜ pending |
| P01-e2e-02 | 01-04 | 3 | DS-ZAHA-P1-07 | — | N/A | integration | `pytest data_pre/zaha/tests/test_e2e.py::test_manifest_schema -x` | ✅ W0 | ⬜ pending |
| P01-e2e-03 | 01-04 | 3 | DS-ZAHA-P1-07 | — | N/A | integration | `pytest data_pre/zaha/tests/test_e2e.py::test_sanity_gates -x` | ✅ W0 | ⬜ pending |
| P01-e2e-04 | 01-04 | 3 | DS-ZAHA-P1-07 | — | N/A | integration | `pytest data_pre/zaha/tests/test_e2e.py::test_determinism -x` | ✅ W0 | ⬜ pending |
| P01-e2e-05 | 01-04 | 3 | DS-ZAHA-P1-07 | — | N/A | integration | `pytest data_pre/zaha/tests/test_e2e.py::test_hard_failure_on_missing_sample -x` | ✅ W0 | ⬜ pending |
| P01-e2e-golden | 01-04 | 3 | DS-ZAHA-P1-07 | — | N/A | integration | `pytest data_pre/zaha/tests/test_e2e.py::test_golden -x` | ✅ W0 | ⬜ pending |
| P01-largefile-manual | 01-04 | 3 | DS-ZAHA-P1-07 | — | N/A | **manual** | Full pipeline on `DEBY_LOD2_4959458.pcd` (136.8 M pts) — record peak RAM + elapsed time in `manifest.json` | — | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Task IDs rewritten from placeholder → `P01-*` in Plan 01-01 Task 2. Plan column shows
which downstream plan (01-02 / 01-03 / 01-04) owns the GREEN transition. `P01-yaml-01`
is owned by Plan 01-01 itself because it asserts on the YAML artifact authored here
in Task 3. Two stubs added beyond the original 24 to hit the ≥ 26 acceptance bar:
`P01-parser-05` (PCL header FIELDS/SIZE/TYPE validation) and `P01-e2e-05` (D-22
hard-failure policy on missing readme-manifested sample).*

---

## Wave 0 Requirements

- [ ] `data_pre/zaha/tests/conftest.py` — shared fixtures (test PCD paths, 1000-point synthetic PCD fixture)
- [ ] `data_pre/zaha/tests/__init__.py`
- [ ] `data_pre/zaha/tests/test_pcd_parser.py` — covers DS-ZAHA-P1-01
- [ ] `data_pre/zaha/tests/test_voxel_agg.py` — covers DS-ZAHA-P1-02 + VOID drop ordering
- [ ] `data_pre/zaha/tests/test_denoise.py` — covers DS-ZAHA-P1-03 (parametrized over 3 methods)
- [ ] `data_pre/zaha/tests/test_chunking.py` — covers DS-ZAHA-P1-04
- [ ] `data_pre/zaha/tests/test_normals.py` — covers DS-ZAHA-P1-05
- [ ] `data_pre/zaha/tests/test_output_layout.py` — covers DS-ZAHA-P1-06
- [ ] `data_pre/zaha/tests/test_yaml.py` — covers DS-ZAHA-P1-06 YAML
- [ ] `data_pre/zaha/tests/test_e2e.py` — covers DS-ZAHA-P1-07 (smallest sample + golden)
- [ ] Framework install: verify `pip list | grep pytest` in `ptv3` env; `pip install pytest` if missing
- [ ] `data_pre/zaha/tests/golden/DEBY_LOD2_4907179__goldens.json` — frozen SHA256 hashes of chunk outputs for determinism regression (filled after first green E2E run)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Denoising visibly reduces scan-stripe banding on ≥1 of 3 sample chunks | DS-ZAHA-P1-03 | Visual-inspection bar per CONTEXT D-12; no quantitative stripe-energy threshold | 3 sample chunks × 3 views (XZ slice, facade projection, colored-by-residual) saved as PNGs embedded in `denoising_notes.md` with per-candidate verdict |
| Normals do not collapse to noise on 2–3 sample chunks | DS-ZAHA-P1-05 | Visual sanity bar per CONTEXT D-18; "thinning the wall" is aspirational | Color normals by angle-to-global-up, render per chunk, embed PNGs in `normals_notes.md`, note failure modes |
| 136.8 M-point sample (`DEBY_LOD2_4959458`) completes end-to-end without OOM | DS-ZAHA-P1-01 / -02 / -07 | Integration-level concern; can only be tested on real hardware + real file; unit tests use synthetic fixtures | Run full pipeline on `DEBY_LOD2_4959458.pcd`, capture peak RAM + elapsed wall-clock, record in final `manifest.json` under `per_sample_stats`; hard-fail if RAM exceeds WSL cap |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (conftest + synthetic PCD fixture + golden file)
- [ ] No watch-mode flags (pytest runs are one-shot `-x`)
- [ ] Feedback latency: <10 s per task commit, <120 s per wave
- [ ] `nyquist_compliant: true` set in frontmatter after planner fills per-task IDs

**Approval:** pending
