---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: — Edge Data Pipeline Refactor and Quality Repair
status: executing
stopped_at: Milestone restructured — Part A/B split applied
last_updated: "2026-04-06T15:00:00Z"
last_activity: 2026-04-06 -- Milestone restructured with Part A/B split
progress:
  total_phases: 8
  completed_phases: 1
  total_plans: 2
  completed_plans: 2
  percent: 13
---

# Project State

## Current Position

Phase: 2 — Behavioral audit and module restructure (next)
Plan: Not yet planned
Status: Phase 1 complete. Milestone restructured into Part A (behavior-preserving refactor, Phases 2-3) and Part B (algorithm improvement, Phase 4), followed by quality repair (Phases 5-8).
Last activity: 2026-04-06 -- Part A/B split applied

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

## Blockers / Concerns

- ~~NET-01 root cause ambiguity~~ — **RESOLVED:** Stage 2 primary, Stage 4 secondary
- Phase 2 scope needs discuss-phase to define: behavioral audit method, module boundary cuts, compatibility logic classification criteria

## Session Continuity

Last session: 2026-04-06
Stopped at: Milestone restructured with Part A/B split — awaiting Phase 2 discuss/plan
Resume file: .planning/workstreams/edge-data-quality-repair/ROADMAP.md
