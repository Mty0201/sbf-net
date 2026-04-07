#pragma once

#include <bf_edge/common.h>
#include <cstdint>
#include <vector>

namespace bf_edge {

struct ClusterResult {
    std::vector<int32_t> center_index;
    std::vector<int32_t> cluster_id;
    std::vector<int32_t> semantic_pair;    // (n_clusters * 2)
    std::vector<int32_t> cluster_size;
    std::vector<float> cluster_centroid;   // (n_clusters * 3)
};

struct Stage2Params {
    float micro_eps_scale = 3.5f;
    int micro_min_samples = 3;
    float split_lateral_threshold_scale = 5.0f;
    float merge_radius_scale = 8.0f;
    float merge_direction_cos_th = 0.7071f;  // cos(45 deg)
    float merge_lateral_scale = 5.0f;
    float rescue_radius_scale = 10.0f;
    int min_cluster_points = 4;
};

void cluster_boundary_centers(
    const float* center_coord, int n_centers,
    const float* center_tangent,
    const int32_t* semantic_pair,   // (n_centers, 2)
    const Stage2Params& params,
    ClusterResult& result
);

}  // namespace bf_edge
