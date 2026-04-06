# Phase 2: Behavioral audit and module restructure - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Audit the current `data_pre/bf_edge_v3` pipeline to surface hidden compatibility logic, heuristics, and cross-stage behavioral contracts. Restructure into modular, independently runnable stages with clear I/O contracts, explicit behavioral documentation, and separation of core algorithm from compatibility/adaptation logic.

**Part A boundary rule:** No changes to default algorithm output semantics. This phase restructures and documents — it does not alter what the pipeline produces.

</domain>

<decisions>
## Implementation Decisions

### Behavioral Audit Method
- **D-01:** Per-block annotation of each logical block in the 4 core modules, classifying behavioral role. Not per-line, not per-function-only — the granularity that captures compatibility logic buried inside large functions.
- **D-02:** Three-way classification scheme: **Core algorithm** (would exist in any clean implementation), **Compatibility/adaptation** (handles real-data edge cases, historical decisions, cross-stage workarounds), **Infrastructure** (I/O, logging, validation, debug output).
- **D-03:** Audit produces a structured document per module, plus a cross-module behavioral contracts summary.

### Module Boundary Cuts
- **D-04:** Follow REFACTOR_TARGET.md direction: separate clustering from refinement/splitting in `local_clusters_core.py`, and decouple pointwise field generation from support-specific assumptions in `pointwise_core.py`. Concrete cut points determined during implementation based on audit findings.
- **D-05:** Keep the current 4-stage pipeline shape as the top-level orchestration. Internal module splits add sub-modules under each stage, not new top-level stages. The 8-module vision from REFACTOR_TARGET.md guides direction but is not the Phase 2 deliverable.

### Compatibility Logic Placement
- **D-06:** Compatibility/adaptation logic identified during audit gets separated into clearly marked sections or helper functions within each module — not scattered inline, not a separate adapter layer. The goal is visibility and isolability, not a new architectural layer.
- **D-07:** Cross-stage behavioral contracts (where one stage assumes specific properties of another stage's output) get documented as explicit interface contracts in I/O type definitions or docstrings.

### Behavioral Preservation
- **D-08:** Phase 2 focuses on structural changes with behavioral preservation as a guiding principle. Formal equivalence verification (bitwise/tolerance, intermediate matching, non-determinism handling) is Phase 3's REF-06 deliverable.
- **D-09:** During Phase 2, behavioral preservation is validated informally: run the refactored pipeline on test scenes (020101/020102) and spot-check that outputs are unchanged. Formal equivalence gate is Phase 3.

### Claude's Discretion
- Specific module split points within `local_clusters_core.py` and `supports_core.py` — determined by audit findings
- Naming conventions for separated compatibility functions
- Documentation format for behavioral contracts (inline docstrings vs separate doc files)
- Order of module refactoring (which module to tackle first)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architectural Direction
- `data_pre/bf_edge_v3/REFACTOR_TARGET.md` — 8-module target architecture, design philosophy, refactoring principles

### Pipeline Documentation
- `data_pre/bf_edge_v3/docs/PIPELINE.md` — current 4-stage pipeline documentation
- `data_pre/bf_edge_v3/docs/DATASET_FORMAT.md` — output format specification

### Core Modules (audit targets)
- `data_pre/bf_edge_v3/core/boundary_centers_core.py` — Stage 1: boundary candidate detection + center construction (13.6KB)
- `data_pre/bf_edge_v3/core/local_clusters_core.py` — Stage 2: DBSCAN clustering + denoise (14.3KB, primary complexity target)
- `data_pre/bf_edge_v3/core/supports_core.py` — Stage 3: support fitting (53.6KB, largest module)
- `data_pre/bf_edge_v3/core/pointwise_core.py` — Stage 4: pointwise edge supervision (11KB)

### Script Entry Points
- `data_pre/bf_edge_v3/scripts/build_boundary_centers.py` — Stage 1 entry
- `data_pre/bf_edge_v3/scripts/build_local_clusters.py` — Stage 2 entry
- `data_pre/bf_edge_v3/scripts/fit_local_supports.py` — Stage 3 entry
- `data_pre/bf_edge_v3/scripts/build_pointwise_edge_supervision.py` — Stage 4 entry

### Diagnosis Evidence
- `.planning/workstreams/edge-data-quality-repair/phases/01-net-01-diagnosis/01-CONTEXT.md` — Phase 1 diagnosis context (Stage 2 primary bottleneck, extension points needed)

### Issue Documentation
- `docs/NET-ISSUES.md` — NET-01/02/03 issue descriptions with quantitative evidence

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `data_pre/bf_edge_v3/utils/stage_io.py` — existing stage I/O utilities (load/save intermediates)
- `data_pre/bf_edge_v3/utils/common.py` — shared utility functions

### Key Complexity Points
- `local_clusters_core.py`: mixes DBSCAN clustering, stable kernel analysis, connected component splitting, transition attach, cluster merge, and post-RANSAC split in a single control flow
- `supports_core.py` (53KB): mixes cluster reconstruction, support type selection, geometric fitting, direction regularization, and format export — the largest single file
- `pointwise_core.py`: tightly coupled to support schema — `build_edge_support()` assumes specific support representation

### Integration Points
- Script entry points (`scripts/`) call into core modules — these are the orchestration layer
- `build_edge_dataset_v3.py` and `build_support_dataset_v3.py` are higher-level runners that chain stages
- `utils/stage_io.py` handles intermediate file I/O between stages

### Key Parameters (from Phase 1 diagnosis)
- Stage 2: `eps=0.08`, `min_samples=8`, `denoise_knn=8`, `sparse_distance_ratio=1.75`, `sparse_mad_scale=3.0`
- Stage 4: `support_radius=0.08`, `sigma = support_radius / 2.0 = 0.04`
- These must become config-injectable in Phase 3, but Phase 2 just identifies where they live

</code_context>

<specifics>
## Specific Ideas

- User explicitly wants: "make the current behavior explicit and stable" — this is the primary deliverable
- User identified that the pipeline "contains substantial compatibility logic for real project data" that is currently implicit and cross-stage
- REFACTOR_TARGET.md section 3 identifies the key structural problems: support dependency, clustering complexity, "for fitting not for field quality" orientation
- The audit should distinguish "behaviors that are true algorithmic requirements" from "compatibility patches and data-adaptation logic"
- User wants a plain-language summary after implementation: what changed, behavioral assumptions identified, where compatibility logic lives, what remains for Phase 3, risks/ambiguities

</specifics>

<deferred>
## Deferred Ideas

- Full 8-module restructure (REFACTOR_TARGET.md sections 5.4-5.8) — deferred to future phases if needed
- Algorithm reorientation toward "field quality first" — explicitly Part B (Phase 4)
- Formal behavioral equivalence gate — Phase 3 (REF-06)
- Config injection system implementation — Phase 3 (REF-04)
- Validation hooks — Phase 3 (REF-05)

</deferred>

---

*Phase: 02-behavioral-audit-and-module-restructure*
*Context gathered: 2026-04-06*
