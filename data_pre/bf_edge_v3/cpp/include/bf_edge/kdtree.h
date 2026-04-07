#pragma once

#include <nanoflann.hpp>
#include <cstdint>
#include <vector>

namespace bf_edge {

// Adapter for nanoflann over a flat float* buffer of (N,3) points
struct PointCloud3f {
    const float* pts;
    int n;

    PointCloud3f(const float* data, int count) : pts(data), n(count) {}

    inline size_t kdtree_get_point_count() const { return static_cast<size_t>(n); }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx * 3 + dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree3f = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud3f>,
    PointCloud3f,
    3,
    int32_t
>;

// build_kdtree removed — KDTree3f is not copyable/movable.
// Construct directly at use site:
//   PointCloud3f cloud(data, n);
//   KDTree3f tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(16));
//   tree.buildIndex();

// KNN query returning (N, k) indices. Self-excluded: caller should skip idx==query_idx.
inline void knn_query(
    const KDTree3f& tree,
    const float* query, int n, int k,
    std::vector<int32_t>& indices,
    std::vector<float>& dists
) {
    indices.resize(static_cast<size_t>(n) * k);
    dists.resize(static_cast<size_t>(n) * k);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        std::vector<int32_t> ret_idx(k);
        std::vector<float> ret_dist(k);
        size_t found = tree.knnSearch(&query[i * 3], k, ret_idx.data(), ret_dist.data());
        for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
            size_t offset = static_cast<size_t>(i) * k + j;
            if (j < found) {
                indices[offset] = ret_idx[j];
                dists[offset] = ret_dist[j];
            } else {
                indices[offset] = -1;
                dists[offset] = std::numeric_limits<float>::max();
            }
        }
    }
}

// Radius query for a single point. Returns indices within radius.
inline std::vector<int32_t> radius_query(
    const KDTree3f& tree,
    const float* query,
    float radius
) {
    float radius_sq = radius * radius;
    std::vector<nanoflann::ResultItem<int32_t, float>> matches;
    tree.radiusSearch(query, radius_sq, matches);
    std::vector<int32_t> result;
    result.reserve(matches.size());
    for (auto& m : matches) {
        result.push_back(m.first);
    }
    return result;
}

}  // namespace bf_edge
