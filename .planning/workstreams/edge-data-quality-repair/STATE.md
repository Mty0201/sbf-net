---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: — Edge Data Pipeline Refactor and Quality Repair
status: Phase 4 in progress — Plan 01 complete
stopped_at: Phase 4 Plan 01 complete
last_updated: "2026-04-07T08:52:16Z"
last_activity: 2026-04-07
progress:
  total_phases: 8
  completed_phases: 3
  total_plans: 11
  completed_plans: 9
---

# Project State

## Current Position

Phase: 4 — Stage 2 cluster contract redesign
Plan: 01 complete, 02 next
Status: Plan 04-01 complete. Stage 2 algorithm redesigned with noise rescue, direction grouping, spatial splitting. 850 fine-grained clusters (was 135 coarse). Trigger mechanism eliminated from Stage 2 output.
Last activity: 2026-04-07

## Recent Context

- **[2026-04-06]** Workstream created for edge data quality repair
- **[2026-04-06]** Milestone v1.0 kicked off — 3 issues (NET-01, NET-02, NET-03) scoped
- **[2026-04-06]** Phase 1 complete — NET-01 baseline diagnosis delivered:
  - **Primary bottleneck:** Stage 2 DBSCAN (18.7pp / 16.3pp survival gap)
  - **Secondary factor:** Stage 4 Gaussian weighting (1.3pp / 7.0pp valid yield gap)
  - Sparse clusters ~50% smaller than dense (475-482 vs 967-989 mean cluster size)
- **[2026-04-06]** Milestone direction changed: refactor `data_pre` first, then repair on refactored pipeline
- **[2026-04-06]** Milestone restructured with Part A/B split:
  - **Part A (Phases 2-3):** Algorithm-preserving refactor — behavioral audit, module restructure, config injection, validation hooks, equivalence gate. No semantic changes to algorithm output.
  - **Part B (Phase 4):** Algorithm improvement — density-adaptive parameters, improved splitting/fitting, intentional redesign. Every behavioral change explicitly marked.
  - **Quality repair (Phases 5-8):** NET-01/02/03 fixes + final re-generation on improved pipeline.
  - **A/B boundary rule:** Any change that alters default output semantics belongs in Part B or later, not Part A.
  - Rationale: the current pipeline contains substantial compatibility logic for real project data. Separating "make current behavior explicit and stable" from "change behavior to improve results" prevents silent algorithmic drift during structural refactor.

## Decisions

- Sequential priority: NET-01 → NET-02 → NET-03 (issues may interact)
- **[Phase 1]** Stage 2 is the primary NET-01 bottleneck — density-adaptive eps needed
- **[Milestone pivot]** Refactor `data_pre` before repairing edge quality — prevents structural debt from entangling with quality fixes
- **[A/B split]** Part A preserves behavior; Part B changes behavior. The boundary is semantic, not structural — if a refactor changes what the algorithm produces, it's Part B work.
- **[A/B split]** Part B may later be promoted to a separate milestone if scope/experimental load warrants it
- [Phase 02]: Stage 3 trigger path classified as MIXED (orchestrates CORE + COMPAT); runtime parameter derivation documented as hidden behavioral contract
- [Phase 02]: params.py centralizes all pipeline parameters (33 total); DEFAULT_FIT_PARAMS re-exported from supports_core.py for backward compat
- [Phase 03-01]: 4 frozen dataclass configs (Stage1-4Config) unify 5 scattered parameter sources; defaults hardcoded in config.py (not imported from params.py) for self-containment
- [Phase 03-01]: Both duplicated build_runtime_params() functions deleted; Stage3Config.to_runtime_dict() is the single source of truth for runtime param dict
- [Phase 03-01]: Reference data captured before code changes for equivalence gate in Plan 03-03
- [Phase 03-02]: 4 validation hooks (bc, lc, supports, edge) cover all 7 cross-stage contracts; pure inspection, no output modification
- [Phase 03-02]: edge_valid dtype check accepts uint8 or int32 with values in {0,1}; supports validation checks only 8-field Stage-4 minimal read set

## Blockers / Concerns

- ~~NET-01 root cause ambiguity~~ — **RESOLVED:** Stage 2 primary, Stage 4 secondary
- ~~Phase 2 scope needs discuss-phase to define~~ — **RESOLVED:** Phase 2 complete, all 3 plans executed
- **[2026-04-07]** Phase 3 planned: 3 plans (config dataclasses, validation hooks, equivalence gate). Research + planning complete.
- **[2026-04-07]** Plan 03-01 executed: 4 frozen dataclass configs, reference data generated, 10 tests passing, both build_runtime_params() deleted
- **[2026-04-07]** Plan 03-02 executed: 4 validation hooks in core/validation.py, integrated into all 5 scripts, 12 new tests (22 total) passing
- **[2026-04-07]** Plan 03-03 executed: equivalence gate — 9 tests, all bit-identical (np.array_equal). Human-approved. 31 total tests passing.
- **[2026-04-07]** Phase 3 complete. Part A (algorithm-preserving refactor) is done. Phase 4 (Part B: algorithm improvement) can begin.
- [Phase 03-03]: Equivalence gate covers all 4 stages + in-memory path; uses np.array_equal exclusively; serves as regression gate for Phase 4
- **[2026-04-07]** Plan 04-01 executed: Stage 2 algorithm redesigned.
  - estimate_local_spacing replaced with O(n log n) cKDTree implementation
  - Stage2Config: 5 trigger fields deleted, 7 new fields (5 direction/spatial + 2 rescue)
  - rescue_noise_centers: 74 noise points rescued on 010101
  - refine_cluster_into_runs: each cluster split by direction then spatial continuity
  - cluster_boundary_centers: 135 -> 850 fine-grained clusters, trigger_flag eliminated
  - validate_cluster_contract: H1/H2/H3 checks (fallback clusters excluded)
  - [Phase 04-01]: validate_cluster_contract uses count-based threshold, not per-cluster strict raise; 49 fallback clusters skipped
  - [Phase 04-01]: H1 check uses group_tangents re-check, not pairwise cosine (matches actual grouping algorithm contract)

## Session Continuity

Last session: 2026-04-07T08:52:16Z
Stopped at: Phase 4 Plan 01 complete
Resume file: None
