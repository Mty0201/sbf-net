#include <bf_edge/stage3.h>
#include <bf_edge/kdtree.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace bf_edge {

// ---- Local spacing estimator (matches Python's estimate_local_spacing) ----

static float estimate_local_spacing(const float* coords, int n, int k = 6) {
    if (n < 2) return 0.01f;

    PointCloud3f cloud(coords, n);
    KDTree3f tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(16));
    tree.buildIndex();

    int query_k = std::min(k + 1, n);
    std::vector<float> all_dists;
    all_dists.reserve(n * k);

    for (int i = 0; i < n; ++i) {
        std::vector<int32_t> idx(query_k);
        std::vector<float> dist(query_k);
        tree.knnSearch(&coords[i * 3], query_k, idx.data(), dist.data());
        for (int j = 1; j < query_k; ++j) {
            all_dists.push_back(std::sqrt(dist[j]));
        }
    }
    if (all_dists.empty()) return 0.01f;

    std::sort(all_dists.begin(), all_dists.end());
    return all_dists[all_dists.size() / 2];
}

// ---- Point-to-line distance ----

static float point_to_line_dist_mean(
    const float* points, int n,
    const Vec3f& origin, const Vec3f& direction
) {
    Vec3f dir = normalize(direction);
    if (dir.norm() < EPS) return std::numeric_limits<float>::infinity();

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        Vec3f p = Eigen::Map<const Vec3f>(&points[i * 3]);
        Vec3f diff = p - origin;
        Vec3f proj = diff.dot(dir) * dir;
        Vec3f perp = diff - proj;
        sum += perp.norm();
    }
    return static_cast<float>(sum / n);
}

// ---- Point-to-segment distance ----

static float point_to_segment_dist(const Vec3f& p, const Vec3f& s, const Vec3f& e) {
    Vec3f sv = e - s;
    float len2 = sv.squaredNorm();
    if (len2 < EPS) return (p - s).norm();
    float t = std::clamp((p - s).dot(sv) / len2, 0.0f, 1.0f);
    return (p - (s + t * sv)).norm();
}

// ---- Point-to-polyline distance (mean) ----

static float point_to_polyline_dist_mean(
    const float* points, int n,
    const std::vector<Vec3f>& vertices
) {
    if (vertices.empty()) return std::numeric_limits<float>::infinity();
    if (vertices.size() == 1) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            Vec3f p = Eigen::Map<const Vec3f>(&points[i * 3]);
            sum += (p - vertices[0]).norm();
        }
        return static_cast<float>(sum / n);
    }

    double sum = 0.0;
    int n_segs = static_cast<int>(vertices.size()) - 1;
    for (int i = 0; i < n; ++i) {
        Vec3f p = Eigen::Map<const Vec3f>(&points[i * 3]);
        float best = std::numeric_limits<float>::infinity();
        for (int s = 0; s < n_segs; ++s) {
            best = std::min(best, point_to_segment_dist(p, vertices[s], vertices[s + 1]));
        }
        sum += best;
    }
    return static_cast<float>(sum / n);
}

// ---- Fit line support ----

struct LineFit {
    Vec3f origin, direction, start, end;
    float residual, coverage_radius, length;
    bool valid;
};

static LineFit fit_line(const float* points, int n) {
    LineFit result;
    result.valid = false;
    if (n < 2) return result;

    Vec3f centroid = Vec3f::Zero();
    for (int i = 0; i < n; ++i) centroid += Eigen::Map<const Vec3f>(&points[i * 3]);
    centroid /= static_cast<float>(n);

    Eigen::MatrixXf centered(n, 3);
    for (int i = 0; i < n; ++i)
        centered.row(i) = (Eigen::Map<const Vec3f>(&points[i * 3]) - centroid).transpose();

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(centered, Eigen::ComputeThinV);
    Vec3f dir = normalize(svd.matrixV().col(0));
    if (dir.norm() < EPS) return result;

    // Project to get endpoints
    Eigen::VectorXf t = centered * dir;
    float t_min = t.minCoeff();
    float t_max = t.maxCoeff();
    result.start = centroid + t_min * dir;
    result.end = centroid + t_max * dir;

    Vec3f midpoint = 0.5f * (result.start + result.end);
    Vec3f final_dir = normalize(result.end - result.start);
    if (final_dir.norm() < EPS) return result;

    result.origin = midpoint;
    result.direction = final_dir;
    result.residual = point_to_line_dist_mean(points, n, midpoint, final_dir);

    float max_r = 0.0f;
    for (int i = 0; i < n; ++i) {
        float r = (Eigen::Map<const Vec3f>(&points[i * 3]) - midpoint).norm();
        max_r = std::max(max_r, r);
    }
    result.coverage_radius = max_r;
    result.length = (result.end - result.start).norm();
    result.valid = true;
    return result;
}

// ---- Build polyline vertices ----

static std::vector<Vec3f> build_polyline_vertices(
    const float* points, int n, int max_vertices
) {
    if (n == 0) return {};
    if (n == 1) return {Eigen::Map<const Vec3f>(&points[0])};

    Vec3f centroid = Vec3f::Zero();
    for (int i = 0; i < n; ++i) centroid += Eigen::Map<const Vec3f>(&points[i * 3]);
    centroid /= static_cast<float>(n);

    Eigen::MatrixXf centered(n, 3);
    for (int i = 0; i < n; ++i)
        centered.row(i) = (Eigen::Map<const Vec3f>(&points[i * 3]) - centroid).transpose();

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(centered, Eigen::ComputeThinV);
    Vec3f dir = normalize(svd.matrixV().col(0));
    Eigen::VectorXf t = centered * dir;

    float t_min = t.minCoeff();
    float t_max = t.maxCoeff();
    float extent = t_max - t_min;
    if (extent < 1e-12f) {
        return {centroid};
    }

    float spacing = std::max(estimate_local_spacing(points, n), 1e-6f);
    int n_bins = std::clamp(static_cast<int>(std::round(extent / (5.0f * spacing))), 2, max_vertices);
    n_bins = std::min(n_bins, n);

    if (n_bins == n) {
        // Sort by t and return
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return t(a) < t(b); });
        std::vector<Vec3f> verts(n);
        for (int i = 0; i < n; ++i)
            verts[i] = Eigen::Map<const Vec3f>(&points[order[i] * 3]);
        return verts;
    }

    // Bin points
    float bin_width = extent / n_bins;
    std::vector<Vec3f> vertices;
    for (int b = 0; b < n_bins; ++b) {
        float lo = t_min + b * bin_width;
        float hi = (b == n_bins - 1) ? t_max + 1.0f : t_min + (b + 1) * bin_width;
        Vec3f sum = Vec3f::Zero();
        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (t(i) >= lo && t(i) < hi) {
                sum += Eigen::Map<const Vec3f>(&points[i * 3]);
                count++;
            }
        }
        if (count > 0) vertices.push_back(sum / static_cast<float>(count));
    }
    if (vertices.empty()) return {centroid};
    return vertices;
}

// ---- Orientation regularization ----

static std::pair<Vec3f, float> regularize_orientation(const Vec3f& direction) {
    Vec3f dir = normalize(direction);
    if (dir.norm() < EPS) return {dir, 0.0f};

    Eigen::Matrix3f axes = Eigen::Matrix3f::Identity();
    Vec3f dots = axes * dir;
    int best_axis = 0;
    float best_dot = dots(0);
    float best_score = std::abs(best_dot);
    for (int i = 1; i < 3; ++i) {
        if (std::abs(dots(i)) > best_score) {
            best_axis = i;
            best_dot = dots(i);
            best_score = std::abs(dots(i));
        }
    }

    float snap_th = std::cos(15.0f * M_PI / 180.0f);
    if (best_score < snap_th) return {dir, best_score};

    Vec3f target = axes.col(best_axis) * (best_dot >= 0.0f ? 1.0f : -1.0f);
    float alpha = std::clamp((best_score - snap_th) / (1.0f - snap_th + EPS), 0.0f, 1.0f);
    alpha *= 0.3f;
    Vec3f reg = normalize((1.0f - alpha) * dir + alpha * target);
    return {reg, best_score};
}

// ---- Split spatial gaps ----

static void split_spatial_gaps(
    const float* points, int n,
    float gap_ratio,
    int min_frag,
    std::vector<std::vector<int>>& fragments
) {
    if (n < 2 * min_frag) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        fragments.push_back(all);
        return;
    }

    float spacing = std::max(estimate_local_spacing(points, n), 1e-6f);
    float threshold = gap_ratio * spacing;

    Vec3f centroid = Vec3f::Zero();
    for (int i = 0; i < n; ++i) centroid += Eigen::Map<const Vec3f>(&points[i * 3]);
    centroid /= static_cast<float>(n);

    Eigen::MatrixXf centered(n, 3);
    for (int i = 0; i < n; ++i)
        centered.row(i) = (Eigen::Map<const Vec3f>(&points[i * 3]) - centroid).transpose();

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(centered, Eigen::ComputeThinV);
    Eigen::VectorXf t = centered * svd.matrixV().col(0);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return t(a) < t(b); });

    float max_gap = 0.0f;
    int max_gap_idx = 0;
    for (int i = 0; i < n - 1; ++i) {
        float gap = t(order[i + 1]) - t(order[i]);
        if (gap > max_gap) {
            max_gap = gap;
            max_gap_idx = i;
        }
    }

    if (max_gap < threshold) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        fragments.push_back(all);
        return;
    }

    // Split
    std::vector<int> left_idx, right_idx;
    for (int i = 0; i <= max_gap_idx; ++i) left_idx.push_back(order[i]);
    for (int i = max_gap_idx + 1; i < n; ++i) right_idx.push_back(order[i]);

    auto make_sub = [&](const std::vector<int>& idx) {
        if (static_cast<int>(idx.size()) < min_frag) return;
        int sn = static_cast<int>(idx.size());
        std::vector<float> sub_pts(sn * 3);
        for (int i = 0; i < sn; ++i)
            for (int d = 0; d < 3; ++d)
                sub_pts[i * 3 + d] = points[idx[i] * 3 + d];

        std::vector<std::vector<int>> sub_frags;
        split_spatial_gaps(sub_pts.data(), sn, gap_ratio, min_frag, sub_frags);
        for (auto& sf : sub_frags) {
            std::vector<int> mapped;
            for (int si : sf) mapped.push_back(idx[si]);
            fragments.push_back(mapped);
        }
    };

    make_sub(left_idx);
    make_sub(right_idx);
    if (fragments.empty()) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        fragments.push_back(all);
    }
}

// ---- Public API ----

void build_supports(
    const float* center_coord, int n_centers,
    const float* center_confidence,
    const int32_t* cluster_center_index, int n_entries,
    const int32_t* cluster_cluster_id,
    const int32_t* cluster_semantic_pair,
    const int32_t* cluster_size,
    const float* cluster_centroid,
    int n_clusters,
    const Stage3Params& params,
    SupportsResult& result
) {
    int total_segments = 0;

    for (int ci = 0; ci < n_clusters; ++ci) {
        int csize = cluster_size[ci];
        if (csize < params.min_cluster_size) continue;

        // Gather member center indices
        std::vector<int32_t> member_indices;
        for (int i = 0; i < n_entries; ++i) {
            if (cluster_cluster_id[i] == ci) {
                member_indices.push_back(cluster_center_index[i]);
            }
        }
        int np = static_cast<int>(member_indices.size());
        if (np < params.min_cluster_size) continue;

        // Gather points
        std::vector<float> points(np * 3);
        for (int i = 0; i < np; ++i)
            for (int d = 0; d < 3; ++d)
                points[i * 3 + d] = center_coord[member_indices[i] * 3 + d];

        // Density check
        float min_cluster_density = params.min_cluster_density;
        if (np > 1) {
            float max_pairwise = 0.0f;
            for (int i = 0; i < np; ++i) {
                for (int j = i + 1; j < np; ++j) {
                    float d = (Eigen::Map<const Vec3f>(&points[i * 3]) -
                               Eigen::Map<const Vec3f>(&points[j * 3])).norm();
                    max_pairwise = std::max(max_pairwise, d);
                }
            }
            if (max_pairwise > 0.01f) {
                float density = static_cast<float>(np) / max_pairwise;
                if (density < min_cluster_density) continue;
            }
        }

        // Cluster confidence
        float cluster_conf = 0.0f;
        for (int32_t mi : member_indices) cluster_conf += center_confidence[mi];
        cluster_conf /= np;

        // Split spatial gaps
        std::vector<std::vector<int>> fragments;
        split_spatial_gaps(points.data(), np, 8.0f, params.min_cluster_size, fragments);

        for (auto& frag : fragments) {
            int fn = static_cast<int>(frag.size());
            if (fn < params.min_cluster_size) continue;

            std::vector<float> frag_pts(fn * 3);
            for (int i = 0; i < fn; ++i)
                for (int d = 0; d < 3; ++d)
                    frag_pts[i * 3 + d] = points[frag[i] * 3 + d];

            // Try line fit first
            auto lf = fit_line(frag_pts.data(), fn);

            if (lf.valid && lf.residual <= params.line_residual_th) {
                auto [dir, ori_score] = regularize_orientation(lf.direction);

                int sid = static_cast<int>(result.support_id.size());
                result.support_id.push_back(sid);
                result.support_type.push_back(0);  // LINE
                result.semantic_pair.push_back(cluster_semantic_pair[ci * 2 + 0]);
                result.semantic_pair.push_back(cluster_semantic_pair[ci * 2 + 1]);
                result.confidence.push_back(cluster_conf);
                result.fit_residual.push_back(lf.residual);
                result.coverage_radius.push_back(lf.coverage_radius);
                result.cluster_id.push_back(ci);
                for (int d = 0; d < 3; ++d) result.origin.push_back(lf.origin[d]);
                for (int d = 0; d < 3; ++d) result.direction.push_back(dir[d]);
                for (int d = 0; d < 3; ++d) result.line_start.push_back(lf.start[d]);
                for (int d = 0; d < 3; ++d) result.line_end.push_back(lf.end[d]);
                result.orientation_prior_score.push_back(ori_score);

                // Line has 1 segment = the line itself
                result.segment_offset.push_back(total_segments);
                result.segment_length.push_back(1);
                for (int d = 0; d < 3; ++d) result.segment_start.push_back(lf.start[d]);
                for (int d = 0; d < 3; ++d) result.segment_end.push_back(lf.end[d]);
                total_segments++;
                continue;
            }

            // Polyline fallback
            auto vertices = build_polyline_vertices(frag_pts.data(), fn, params.max_polyline_vertices);
            float residual = point_to_polyline_dist_mean(frag_pts.data(), fn, vertices);

            if (residual > params.polyline_residual_th) continue;

            Vec3f origin = Vec3f::Zero();
            for (int i = 0; i < fn; ++i) origin += Eigen::Map<const Vec3f>(&frag_pts[i * 3]);
            origin /= static_cast<float>(fn);

            Vec3f dir = (vertices.size() >= 2) ?
                normalize(vertices.back() - vertices.front()) : Vec3f::Zero();
            auto [reg_dir, ori_score] = regularize_orientation(dir);

            float max_r = 0.0f;
            for (int i = 0; i < fn; ++i) {
                float r = (Eigen::Map<const Vec3f>(&frag_pts[i * 3]) - origin).norm();
                max_r = std::max(max_r, r);
            }

            int sid = static_cast<int>(result.support_id.size());
            result.support_id.push_back(sid);
            result.support_type.push_back(2);  // POLYLINE
            result.semantic_pair.push_back(cluster_semantic_pair[ci * 2 + 0]);
            result.semantic_pair.push_back(cluster_semantic_pair[ci * 2 + 1]);
            result.confidence.push_back(cluster_conf);
            result.fit_residual.push_back(residual);
            result.coverage_radius.push_back(max_r);
            result.cluster_id.push_back(ci);
            for (int d = 0; d < 3; ++d) result.origin.push_back(origin[d]);
            for (int d = 0; d < 3; ++d) result.direction.push_back(reg_dir[d]);
            // line_start/end = zeros for polyline
            for (int d = 0; d < 3; ++d) result.line_start.push_back(0.0f);
            for (int d = 0; d < 3; ++d) result.line_end.push_back(0.0f);
            result.orientation_prior_score.push_back(ori_score);

            // Segments from polyline vertices
            int n_segs = static_cast<int>(vertices.size()) - 1;
            result.segment_offset.push_back(total_segments);
            result.segment_length.push_back(std::max(n_segs, 0));
            for (int s = 0; s < n_segs; ++s) {
                for (int d = 0; d < 3; ++d) result.segment_start.push_back(vertices[s][d]);
                for (int d = 0; d < 3; ++d) result.segment_end.push_back(vertices[s + 1][d]);
            }
            total_segments += n_segs;
        }
    }
}

}  // namespace bf_edge
