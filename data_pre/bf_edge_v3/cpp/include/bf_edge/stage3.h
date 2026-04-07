#pragma once

#include <bf_edge/common.h>
#include <cstdint>
#include <vector>

namespace bf_edge {

struct Stage3Params {
    float line_residual_th = 0.01f;
    int min_cluster_size = 4;
    int max_polyline_vertices = 32;
    float polyline_residual_th = 0.04f;
    float min_cluster_density = 15.0f;
};

struct SupportsResult {
    std::vector<int32_t> support_id;
    std::vector<int32_t> support_type;
    std::vector<int32_t> semantic_pair;        // (n * 2)
    std::vector<float> confidence;
    std::vector<float> fit_residual;
    std::vector<float> coverage_radius;
    std::vector<int32_t> cluster_id;
    std::vector<float> origin;                 // (n * 3)
    std::vector<float> direction;              // (n * 3)
    std::vector<float> line_start;             // (n * 3)
    std::vector<float> line_end;               // (n * 3)
    std::vector<float> orientation_prior_score;

    // Segments (flattened)
    std::vector<int32_t> segment_offset;
    std::vector<int32_t> segment_length;
    std::vector<float> segment_start;          // (total_segs * 3)
    std::vector<float> segment_end;            // (total_segs * 3)
};

void build_supports(
    const float* center_coord, int n_centers,
    const float* center_confidence,
    const int32_t* cluster_center_index, int n_entries,
    const int32_t* cluster_cluster_id,
    const int32_t* cluster_semantic_pair,  // (n_clusters, 2)
    const int32_t* cluster_size,
    const float* cluster_centroid,         // (n_clusters, 3)
    int n_clusters,
    const Stage3Params& params,
    SupportsResult& result
);

}  // namespace bf_edge
