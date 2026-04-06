# NET-01 Diagnosis: Sparse-Region Coverage Loss

**Date:** 2026-04-06
**Status:** Complete
**Requirement:** DEN-01

## Conclusion

**Primary bottleneck:** Stage 2 (DBSCAN clustering) -- fixed eps=0.08 causes 18.7-16.3pp survival gap between dense and sparse regions, losing entire boundary segments where point spacing exceeds eps.
**Secondary factor:** Stage 4 (Gaussian weighting) -- fixed sigma=0.04 contributes an additional 1.3-7.0pp valid yield gap, thinning coverage on boundary segments that survive Stage 2.

## Evidence Summary

### Density Distribution

- P25 threshold (dense/mid boundary): 0.025278 m (scene 020101) / 0.024665 m (020102)
- P75 threshold (mid/sparse boundary): 0.036738 m (020101) / 0.031747 m (020102)
- Interpretation: boundary centers with mean kNN spacing above P75 are classified as "sparse". The P75 values (0.032-0.037 m) are well below eps=0.08, yet sparse-region boundary centers have significantly worse outcomes at both Stage 2 and Stage 4. This is because the P75 threshold marks the *onset* of low-density regions -- actual point spacing in the sparse tail extends substantially beyond 0.037 m.

### Stage 2: DBSCAN Cluster Survival (eps=0.08, min_samples=8)

| Scene | Bucket | Centers | Survived | Survival Rate |
|-------|--------|---------|----------|---------------|
| 020101 | dense | 8156 | 8139 | 99.8% |
| 020101 | mid | 16309 | 15948 | 97.8% |
| 020101 | sparse | 8156 | 6612 | 81.1% |
| 020102 | dense | 10741 | 10732 | 99.9% |
| 020102 | mid | 21480 | 21262 | 99.0% |
| 020102 | sparse | 10741 | 8983 | 83.6% |

Stage 2 dense-sparse survival gap: 18.7pp (020101) / 16.3pp (020102)

The fixed eps=0.08 DBSCAN systematically fails in sparse regions: roughly 1 in 5 sparse-region boundary centers are lost to noise classification (noise rate 18.9% in 020101, 16.4% in 020102). In contrast, dense regions lose fewer than 0.3% of centers. This is the single largest source of coverage loss in the pipeline. Furthermore, sparse clusters that do survive are substantially smaller -- mean cluster size drops from 967-989 points (dense) to 475-482 points (sparse), a reduction of roughly 50%. Smaller clusters mean less geometric context for Stage 3 support fitting, propagating quality degradation downstream even for surviving clusters.

### Stage 4: Valid Yield and Weight (sigma=0.04, radius=0.08)

| Scene | Bucket | Points | Valid | Valid Rate | Mean Weight | Weight>=0.5 Rate |
|-------|--------|--------|-------|------------|-------------|------------------|
| 020101 | dense | 121561 | 19505 | 16.0% | 0.597 | 9.6% |
| 020101 | mid | 181172 | 33388 | 18.4% | 0.573 | 10.6% |
| 020101 | sparse | 64565 | 9502 | 14.7% | 0.549 | 7.9% |
| 020102 | dense | 156464 | 28796 | 18.4% | 0.565 | 10.1% |
| 020102 | mid | 242736 | 42036 | 17.3% | 0.579 | 10.1% |
| 020102 | sparse | 114039 | 13059 | 11.5% | 0.547 | 6.1% |

Stage 4 dense-sparse valid gap: 1.3pp (020101) / 7.0pp (020102)
Stage 4 dense-sparse weight gap: 1.6pp (020101) / 4.0pp (020102)

Stage 4's fixed sigma=0.04 produces a measurable but secondary density bias. The valid yield gap is modest in 020101 (1.3pp) and more pronounced in 020102 (7.0pp). The high-weight (>=0.5) rate drops from 9.6-10.1% (dense) to 6.1-7.9% (sparse), indicating that sparse-region points that do receive valid supervision tend to get lower-confidence weights. However, these gaps are consistently smaller than Stage 2's survival gap by a factor of 2-14x. Note also that sparse regions contribute fewer total points to Stage 4 (64K-114K vs 122K-156K dense), partly because Stage 2 already eliminated many sparse boundary centers before they could generate Stage 4 points.

### Gap Attribution

| Metric | 020101 | 020102 |
|--------|--------|--------|
| Stage 2 survival gap (dense - sparse) | 18.7pp | 16.3pp |
| Stage 4 valid gap (dense - sparse) | 1.3pp | 7.0pp |
| Stage 4 weight gap (dense - sparse) | 1.6pp | 4.0pp |

Stage 2's survival gap dominates in both scenes, exceeding the Stage 4 valid gap by 14.4x in 020101 and 2.3x in 020102. Even in the more Stage-4-affected scene (020102), Stage 2 still contributes more than twice the gap. The ranking is unambiguous: **Stage 2 is the primary bottleneck.**

## Diagnosis Logic

Per D-04, the two stages have distinct failure modes that are distinguishable from stratified statistics alone:

- **Stage 2 failure = "entire boundary segment lost."** When local point spacing exceeds eps=0.08, DBSCAN cannot form a cluster. The boundary center is classified as noise and the entire local boundary segment disappears from downstream processing. This is a binary, catastrophic failure: the segment either clusters or it does not. The 18.7pp and 16.3pp survival gaps confirm that this binary failure disproportionately hits sparse regions, where ~17-19% of boundary centers are lost entirely.

- **Stage 4 failure = "coverage thinned."** Even when a cluster survives Stage 2 and a support is fitted in Stage 3, the fixed sigma=0.04 Gaussian decay assigns low weights to points that are far from their support center. In sparse regions where average point-to-support distances are larger, more points fall below validity thresholds or receive low weights. This is a gradual degradation, not a binary failure. The 1.3-7.0pp valid gaps and 1.6-4.0pp weight gaps confirm this secondary thinning effect.

The statistics clearly show that the dominant pattern is Stage 2's binary segment loss (primary), with Stage 4's weight thinning as a compounding secondary factor. Fixing Stage 2 alone would recover ~17-19% of currently-lost sparse boundary centers. Fixing Stage 4 in addition would improve weight quality for both the recovered centers and the existing sparse-region survivors.

## Phase 2 Recommendation

Based on this diagnosis, Phase 2 (DEN-02, DEN-03) should:

1. **Primary target -- density-adaptive eps for Stage 2 DBSCAN.** Replace the fixed eps=0.08 with a locally computed eps derived from each boundary center's neighborhood density. Candidate formula: `eps_local = max(eps_base, alpha * mean_knn_distance(k=10))` with alpha in the range 1.5-2.0. This directly addresses the primary bottleneck by allowing DBSCAN to form clusters in sparse regions where point spacing exceeds the current fixed threshold. Expected impact: recover the majority of the 17-19% lost sparse-region boundary centers.

2. **Secondary target -- density-adaptive sigma for Stage 4 Gaussian weighting.** Replace the fixed sigma=0.04 with `sigma_local = max(sigma_base, beta * local_spacing)`. This addresses the secondary thinning effect by widening the Gaussian window in sparse regions, increasing both valid yield and mean weight. Expected impact: reduce the 1.3-7.0pp valid yield gap and improve high-weight coverage from the current 6.1-7.9% toward the dense-region level of 9.6-10.1%.

3. **Verification plan:** After implementing density-adaptive parameters, rerun the same stratified diagnosis on scenes 020101 and 020102. Success criteria: sparse-region survival rate above 95% (from current 81-84%), and Stage 4 valid gap reduced below 3pp in both scenes.
