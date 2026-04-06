# Phase 1: NET-01 diagnosis - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Determine whether sparse-region edge supervision coverage loss (NET-01) originates primarily in Stage 2 (DBSCAN cluster loss with fixed eps=0.08), Stage 4 (fixed sigma=0.04 Gaussian decay), or both. Produce a clear primary/secondary ranking with supporting evidence.

This phase is diagnosis only — no code fixes.

</domain>

<decisions>
## Implementation Decisions

### Diagnosis Method
- **D-01:** Two-step approach: first do stratified statistics (Stage 2 cluster survival rate + Stage 4 valid yield, bucketed by sparse/dense) to establish hypothesis, then run 1-2 controlled variable experiments to confirm.
- **D-02:** Density bucketing uses kNN percentile method: compute mean kNN distance (k=10) per point, bucket by percentile (e.g. P25-below = dense, P75-above = sparse). This adapts to each scene's actual density distribution.

### Acceptance Criteria
- **D-03:** Diagnosis conclusion needs clear primary/secondary ranking ("primary bottleneck is Stage X, Stage Y is secondary factor"). No precise percentage attribution required — just enough clarity to direct Phase 2 fix effort.

### Stage Interaction
- **D-04:** No isolation experiments needed. The two stages have sufficiently distinct failure modes (Stage 2 = "entire segment lost" when point spacing > eps, Stage 4 = "coverage thinned" when sigma decays too fast) that stratified statistics can distinguish primary vs secondary without controlled variable isolation.

### Claude's Discretion
- Specific percentile thresholds for sparse/dense bucketing (P25/P75 suggested, adjust based on actual distribution)
- Which intermediate outputs to inspect (boundary_centers.npz, local_clusters.npz, supports.npz, edge.npy)
- Visualization or reporting format for the diagnosis

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Pipeline Code (data_pre/bf_edge_v3)
- `data_pre/bf_edge_v3/core/boundary_centers_core.py` — Stage 1: boundary candidate detection + center construction
- `data_pre/bf_edge_v3/core/local_clusters_core.py` — Stage 2: DBSCAN clustering (eps=0.08) + denoise (NET-01 cluster loss location)
- `data_pre/bf_edge_v3/core/supports_core.py` — Stage 3: support fitting
- `data_pre/bf_edge_v3/core/pointwise_core.py` — Stage 4: pointwise edge supervision (sigma=0.04 decay, NET-01 valid yield location)

### Pipeline Scripts
- `data_pre/bf_edge_v3/scripts/build_boundary_centers.py` — Stage 1 entry
- `data_pre/bf_edge_v3/scripts/build_local_clusters.py` — Stage 2 entry (--eps 0.08, --min-samples 8)
- `data_pre/bf_edge_v3/scripts/fit_local_supports.py` — Stage 3 entry
- `data_pre/bf_edge_v3/scripts/build_pointwise_edge_supervision.py` — Stage 4 entry (--support-radius 0.08)

### Issue Documentation
- `docs/NET-ISSUES.md` — NET-01/02/03 issue descriptions with quantitative evidence from 020101/020102

</canonical_refs>

<code_context>
## Existing Code Insights

### Key Parameters (NET-01 scope)
- Stage 2: `eps=0.08` (DBSCAN radius), `min_samples=8`, `denoise_knn=8`, `sparse_distance_ratio=1.75`, `sparse_mad_scale=3.0`
- Stage 4: `support_radius=0.08`, `sigma = support_radius / 2.0 = 0.04`

### Stage 2 Failure Mechanism
- `spatial_dbscan()` at line 37-42 of local_clusters_core.py: fixed eps=0.08m means points with spacing > 0.08m cannot cluster
- `lightweight_denoise_cluster()` at lines 125-172: kNN-based outlier removal further reduces sparse clusters

### Stage 4 Failure Mechanism
- `build_edge_support()` at lines 140-154 of pointwise_core.py: `sigma = max(support_radius / 2.0, EPS)` is fixed at 0.04
- Gaussian weight = exp(-dist²/(2×0.04²)) — at dist=0.06m weight is ~0.32, at dist=0.08m weight is ~0.14

### Intermediate Outputs Available for Analysis
- `boundary_centers.npz` — Stage 1 output (before clustering)
- `local_clusters.npz` — Stage 2 output (after clustering + denoise)
- `supports.npz` — Stage 3 output (fitted support geometry)
- `edge_*.npy` — Stage 4 output (per-point valid/weight/direction)

### Test Scenes
- 020101 (training), 020102 (validation) — both have existing edge data for before/after comparison

</code_context>

<specifics>
## Specific Ideas

- NET-ISSUES.md reports sparse region valid edge coverage at 9-14% vs dense region ~18% — this is the gap to diagnose
- NET-ISSUES.md reports weight ≥ 0.5 in sparse regions at only 5-7% vs dense ~10%
- The diagnosis should clarify whether the gap is primarily "no support exists" (Stage 2) or "support exists but Gaussian is too narrow" (Stage 4)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-net-01-diagnosis*
*Context gathered: 2026-04-06*
