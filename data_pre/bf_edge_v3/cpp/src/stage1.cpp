#include <bf_edge/stage1.h>
#include <bf_edge/kdtree.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bf_edge {

// ---- Internal helpers ----

struct CandidateInfo {
    int32_t point_idx;
    int32_t pair[2];
    float cross_ratio;
};

static Vec3f estimate_pca_tangent(
    const float* points, int n,
    const Vec3f& center_coord,
    const Vec3f& center_normal,
    float& score_out
) {
    score_out = 0.0f;
    if (n < 2) return Vec3f::Zero();

    // Project to plane orthogonal to center_normal
    Eigen::MatrixXf planar(n, 3);
    int valid = 0;
    for (int i = 0; i < n; ++i) {
        Vec3f p = Eigen::Map<const Vec3f>(&points[i * 3]) - center_coord;
        Vec3f proj = p - p.dot(center_normal) * center_normal;
        if (proj.norm() > EPS) {
            planar.row(valid++) = proj.transpose();
        }
    }
    if (valid < 2) return Vec3f::Zero();

    planar.conservativeResize(valid, 3);
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(planar, Eigen::ComputeThinV);
    Vec3f tangent = normalize(svd.matrixV().col(0));
    if (tangent.norm() < EPS) return Vec3f::Zero();

    auto sv = svd.singularValues();
    float denom = sv.sum() + EPS;
    score_out = std::clamp(sv(0) / denom, 0.0f, 1.0f);
    return tangent;
}

static Vec3f estimate_fallback_tangent(
    const float* normal, // may be null
    const int32_t* local_index, int n_local,
    const Vec3f& center_normal,
    float& score_out
) {
    score_out = 0.0f;
    if (normal == nullptr || n_local == 0) return Vec3f::Zero();

    Vec3f sum_normal = Vec3f::Zero();
    int valid = 0;
    for (int i = 0; i < n_local; ++i) {
        Vec3f n = Eigen::Map<const Vec3f>(&normal[local_index[i] * 3]);
        if (n.norm() > EPS) {
            sum_normal += n;
            valid++;
        }
    }
    if (valid == 0) return Vec3f::Zero();

    Vec3f surface_normal = normalize(sum_normal / valid);
    Vec3f tangent = normalize(surface_normal.cross(center_normal));
    if (tangent.norm() < EPS) return Vec3f::Zero();
    score_out = 0.3f;
    return tangent;
}

// ---- Public API ----

void build_boundary_centers(
    const float* coord, int n_points,
    const int32_t* segment,
    const float* normal,
    int k,
    float min_cross_ratio,
    int min_side_points,
    int ignore_index,
    BoundaryCentersResult& result
) {
    if (n_points == 0) return;

    // Build KD-tree for kNN
    PointCloud3f cloud(coord, n_points);
    KDTree3f tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(16));
    tree.buildIndex();

    int query_k = std::min(k + 1, n_points);

    // Phase 1: detect boundary candidates (parallel)
    struct CandResult {
        int32_t point_idx;
        int32_t pair[2];
        float cross_ratio;
    };

    std::vector<std::vector<CandResult>> thread_cands;
    int n_threads = 1;
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    #endif
    thread_cands.resize(n_threads);

    #pragma omp parallel
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        auto& local = thread_cands[tid];

        std::vector<int32_t> nn_idx(query_k);
        std::vector<float> nn_dist(query_k);

        #pragma omp for schedule(static)
        for (int i = 0; i < n_points; ++i) {
            int32_t label_self = segment[i];
            if (label_self == static_cast<int32_t>(ignore_index)) continue;

            size_t found = tree.knnSearch(&coord[i * 3], query_k, nn_idx.data(), nn_dist.data());

            int n_valid = 0, n_diff = 0;
            std::unordered_map<int32_t, int> diff_counts;

            for (size_t j = 0; j < found; ++j) {
                if (nn_idx[j] == i) continue;  // skip self
                int32_t nl = segment[nn_idx[j]];
                if (nl == static_cast<int32_t>(ignore_index)) continue;
                n_valid++;
                if (nl != label_self) {
                    n_diff++;
                    diff_counts[nl]++;
                }
            }

            if (n_valid == 0 || n_diff == 0) continue;
            float ratio = static_cast<float>(n_diff) / std::max(n_valid, 1);
            if (ratio < min_cross_ratio) continue;

            // Find dominant paired label
            int32_t paired_label = -1;
            int max_count = 0;
            for (auto& [lab, cnt] : diff_counts) {
                if (cnt > max_count) {
                    max_count = cnt;
                    paired_label = lab;
                }
            }

            CandResult c;
            c.point_idx = i;
            c.pair[0] = std::min(label_self, paired_label);
            c.pair[1] = std::max(label_self, paired_label);
            c.cross_ratio = ratio;
            local.push_back(c);
        }
    }

    // Merge thread results
    std::vector<CandResult> candidates;
    for (auto& tc : thread_cands) {
        candidates.insert(candidates.end(), tc.begin(), tc.end());
    }

    // Store candidate results
    result.cand_point_index.resize(candidates.size());
    result.cand_semantic_pair.resize(candidates.size() * 2);
    result.cand_cross_ratio.resize(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        result.cand_point_index[i] = candidates[i].point_idx;
        result.cand_semantic_pair[i * 2 + 0] = candidates[i].pair[0];
        result.cand_semantic_pair[i * 2 + 1] = candidates[i].pair[1];
        result.cand_cross_ratio[i] = candidates[i].cross_ratio;
    }

    // Phase 2: build boundary centers (parallel)
    std::vector<std::vector<BoundaryCenterRecord>> thread_records;
    thread_records.resize(n_threads);

    #pragma omp parallel
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        auto& local = thread_records[tid];

        std::vector<int32_t> nn_idx(query_k);
        std::vector<float> nn_dist(query_k);

        #pragma omp for schedule(dynamic, 64)
        for (int ci = 0; ci < static_cast<int>(candidates.size()); ++ci) {
            auto& cand = candidates[ci];
            int pi = cand.point_idx;
            int32_t label_a = cand.pair[0];
            int32_t label_b = cand.pair[1];

            // Get neighbors
            size_t found = tree.knnSearch(&coord[pi * 3], query_k, nn_idx.data(), nn_dist.data());

            // Build local index (unique, includes self)
            std::set<int32_t> local_set;
            local_set.insert(pi);
            for (size_t j = 0; j < found; ++j) {
                if (nn_idx[j] != pi) local_set.insert(nn_idx[j]);
            }

            std::vector<int32_t> index_a, index_b;
            std::vector<int32_t> pair_index;
            for (int32_t idx : local_set) {
                int32_t lab = segment[idx];
                if (lab == label_a) { index_a.push_back(idx); pair_index.push_back(idx); }
                else if (lab == label_b) { index_b.push_back(idx); pair_index.push_back(idx); }
            }

            if (static_cast<int>(index_a.size()) < min_side_points ||
                static_cast<int>(index_b.size()) < min_side_points) continue;

            // Compute centroids
            Vec3f centroid_a = Vec3f::Zero(), centroid_b = Vec3f::Zero();
            for (int32_t idx : index_a) centroid_a += Eigen::Map<const Vec3f>(&coord[idx * 3]);
            for (int32_t idx : index_b) centroid_b += Eigen::Map<const Vec3f>(&coord[idx * 3]);
            centroid_a /= static_cast<float>(index_a.size());
            centroid_b /= static_cast<float>(index_b.size());

            Vec3f separation = centroid_b - centroid_a;
            float sep_norm = separation.norm();
            if (sep_norm < EPS) continue;

            Vec3f center_c = 0.5f * (centroid_a + centroid_b);
            Vec3f center_n = separation / sep_norm;

            // Gather pair points for PCA
            int n_pair = static_cast<int>(pair_index.size());
            std::vector<float> pair_pts(n_pair * 3);
            for (int i = 0; i < n_pair; ++i) {
                for (int d = 0; d < 3; ++d) {
                    pair_pts[i * 3 + d] = coord[pair_index[i] * 3 + d];
                }
            }

            float tangent_score = 0.0f;
            Vec3f tangent = estimate_pca_tangent(
                pair_pts.data(), n_pair, center_c, center_n, tangent_score
            );

            if (tangent.norm() < EPS) {
                tangent = estimate_fallback_tangent(
                    normal, pair_index.data(), n_pair, center_n, tangent_score
                );
            }
            if (tangent.norm() < EPS) continue;

            // Confidence
            float side_balance = 1.0f - std::abs(
                static_cast<float>(index_a.size()) - static_cast<float>(index_b.size())
            ) / std::max(static_cast<float>(index_a.size() + index_b.size()), 1.0f);
            float sep_score = sep_norm / (sep_norm + 1.0f);
            float confidence = std::clamp(
                0.45f * cand.cross_ratio
                + 0.30f * side_balance
                + 0.15f * sep_score
                + 0.10f * tangent_score,
                0.0f, 1.0f
            );

            BoundaryCenterRecord rec;
            rec.center_coord = center_c;
            rec.center_normal = center_n;
            rec.center_tangent = tangent;
            rec.semantic_pair[0] = label_a;
            rec.semantic_pair[1] = label_b;
            rec.source_point_index = pi;
            rec.confidence = confidence;
            local.push_back(rec);
        }
    }

    // Merge and normalize
    std::vector<BoundaryCenterRecord> all_records;
    for (auto& tr : thread_records) {
        all_records.insert(all_records.end(), tr.begin(), tr.end());
    }

    int n_centers = static_cast<int>(all_records.size());
    result.center_coord.resize(n_centers * 3);
    result.center_normal.resize(n_centers * 3);
    result.center_tangent.resize(n_centers * 3);
    result.semantic_pair.resize(n_centers * 2);
    result.source_point_index.resize(n_centers);
    result.confidence.resize(n_centers);

    for (int i = 0; i < n_centers; ++i) {
        auto& r = all_records[i];
        // Normalize normal and tangent
        Vec3f cn = normalize(r.center_normal);
        Vec3f ct = normalize(r.center_tangent);

        for (int d = 0; d < 3; ++d) {
            result.center_coord[i * 3 + d] = r.center_coord[d];
            result.center_normal[i * 3 + d] = cn[d];
            result.center_tangent[i * 3 + d] = ct[d];
        }
        result.semantic_pair[i * 2 + 0] = r.semantic_pair[0];
        result.semantic_pair[i * 2 + 1] = r.semantic_pair[1];
        result.source_point_index[i] = r.source_point_index;
        result.confidence[i] = r.confidence;
    }
}

}  // namespace bf_edge
