#pragma once

#include <bf_edge/common.h>
#include <cstdint>
#include <vector>

namespace bf_edge {

struct BoundaryCenterRecord {
    Vec3f center_coord;
    Vec3f center_normal;
    Vec3f center_tangent;
    int32_t semantic_pair[2];
    int32_t source_point_index;
    float confidence;
};

struct BoundaryCentersResult {
    // Candidates
    std::vector<int32_t> cand_point_index;
    std::vector<int32_t> cand_semantic_pair;  // (n_cand * 2)
    std::vector<float> cand_cross_ratio;

    // Centers
    std::vector<float> center_coord;    // (n * 3)
    std::vector<float> center_normal;   // (n * 3)
    std::vector<float> center_tangent;  // (n * 3)
    std::vector<int32_t> semantic_pair; // (n * 2)
    std::vector<int32_t> source_point_index;
    std::vector<float> confidence;
};

void build_boundary_centers(
    const float* coord, int n_points,
    const int32_t* segment,
    const float* normal,   // may be nullptr
    int k,
    float min_cross_ratio,
    int min_side_points,
    int ignore_index,
    BoundaryCentersResult& result
);

}  // namespace bf_edge
