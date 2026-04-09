#pragma once

#include <bf_edge/common.h>
#include <cstdint>
#include <set>
#include <unordered_map>
#include <vector>

namespace bf_edge {

struct SupportsData {
    int n_supports;
    const int32_t* support_id;
    const int32_t* semantic_pair;   // (n_supports, 2) row-major
    const int32_t* segment_offset;
    const int32_t* segment_length;
    const float* segment_start;     // (n_segments, 3) row-major
    const float* segment_end;       // (n_segments, 3) row-major
    const float* line_start;        // (n_supports, 3) row-major
    const float* line_end;          // (n_supports, 3) row-major
    const int32_t* cluster_id;
    const int32_t* support_type;
};

struct BoundaryCentersData {
    int n_centers;
    const float* center_coord;    // (n_centers, 3)
    const float* center_tangent;  // (n_centers, 3)
};

struct LocalClustersData {
    int n_entries;
    const int32_t* cluster_id;
    const int32_t* center_index;
};

struct PointwiseResult {
    std::vector<float> edge_dist;
    std::vector<float> edge_dir;      // (N*3)
    std::vector<uint8_t> edge_valid;
    std::vector<int32_t> edge_support_id;
    std::vector<float> edge_vec;      // (N*3)
    std::vector<float> edge_support;  // gaussian weights
};

void build_pointwise_edge_supervision(
    const float* coord, int n_points,
    const int32_t* segment,
    const SupportsData& supports,
    float support_radius,
    int ignore_index,
    const std::set<int>& skip_supports,
    PointwiseResult& result,
    float sigma = -1.0f
);

std::set<int> find_bad_supports(
    const SupportsData& supports,
    const BoundaryCentersData& bc,
    const LocalClustersData& lc,
    float min_length = 0.05f,
    float middle_lo = 0.2f,
    float middle_hi = 0.8f,
    float max_middle_fraction = 0.15f,
    float min_tangent_alignment = 0.5f
);

}  // namespace bf_edge
