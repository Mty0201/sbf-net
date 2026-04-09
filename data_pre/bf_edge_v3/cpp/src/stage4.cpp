#include <bf_edge/stage4.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bf_edge {

// ---- Internal helpers ----

static inline void closest_point_on_segment(
    const float* points, int n,
    const float* seg_start, const float* seg_end,
    float* out_closest, float* out_dist
) {
    Vec3f ss = Eigen::Map<const Vec3f>(seg_start);
    Vec3f se = Eigen::Map<const Vec3f>(seg_end);
    Vec3f sv = se - ss;
    float seg_len2 = sv.squaredNorm();

    if (seg_len2 < EPS) {
        for (int i = 0; i < n; ++i) {
            Vec3f p = Eigen::Map<const Vec3f>(points + i * 3);
            Eigen::Map<Vec3f>(out_closest + i * 3) = ss;
            out_dist[i] = (p - ss).norm();
        }
        return;
    }

    float inv_len2 = 1.0f / seg_len2;
    for (int i = 0; i < n; ++i) {
        Vec3f p = Eigen::Map<const Vec3f>(points + i * 3);
        float t = std::clamp((p - ss).dot(sv) * inv_len2, 0.0f, 1.0f);
        Vec3f closest = ss + t * sv;
        Eigen::Map<Vec3f>(out_closest + i * 3) = closest;
        out_dist[i] = (p - closest).norm();
    }
}

static void closest_points_to_support(
    const float* points, int n,
    int support_id,
    const SupportsData& sup,
    float* out_closest, float* out_dist
) {
    int offset = sup.segment_offset[support_id];
    int length = sup.segment_length[support_id];

    if (length <= 0) {
        closest_point_on_segment(
            points, n,
            &sup.line_start[support_id * 3],
            &sup.line_end[support_id * 3],
            out_closest, out_dist
        );
        return;
    }

    // First segment initializes
    closest_point_on_segment(
        points, n,
        &sup.segment_start[offset * 3],
        &sup.segment_end[offset * 3],
        out_closest, out_dist
    );

    if (length == 1) return;

    // Remaining segments: keep best
    std::vector<float> tmp_closest(n * 3);
    std::vector<float> tmp_dist(n);

    for (int si = 1; si < length; ++si) {
        int seg_idx = offset + si;
        closest_point_on_segment(
            points, n,
            &sup.segment_start[seg_idx * 3],
            &sup.segment_end[seg_idx * 3],
            tmp_closest.data(), tmp_dist.data()
        );
        for (int i = 0; i < n; ++i) {
            if (tmp_dist[i] < out_dist[i]) {
                out_dist[i] = tmp_dist[i];
                out_closest[i * 3 + 0] = tmp_closest[i * 3 + 0];
                out_closest[i * 3 + 1] = tmp_closest[i * 3 + 1];
                out_closest[i * 3 + 2] = tmp_closest[i * 3 + 2];
            }
        }
    }
}

// Build label -> vector of support ids
static std::unordered_map<int32_t, std::vector<int32_t>> build_label_to_supports(
    const int32_t* semantic_pair, int n_supports
) {
    std::unordered_map<int32_t, std::vector<int32_t>> result;
    for (int i = 0; i < n_supports; ++i) {
        int32_t a = semantic_pair[i * 2 + 0];
        int32_t b = semantic_pair[i * 2 + 1];
        result[a].push_back(i);
        if (b != a) {
            result[b].push_back(i);
        }
    }
    return result;
}

// ---- Public API ----

void build_pointwise_edge_supervision(
    const float* coord, int n_points,
    const int32_t* segment,
    const SupportsData& supports,
    float support_radius,
    int ignore_index,
    const std::set<int>& skip_supports,
    PointwiseResult& result,
    float sigma
) {
    result.edge_dist.assign(n_points, std::numeric_limits<float>::infinity());
    result.edge_dir.assign(n_points * 3, 0.0f);
    result.edge_valid.assign(n_points, 0);
    result.edge_support_id.assign(n_points, -1);
    result.edge_vec.assign(n_points * 3, 0.0f);
    result.edge_support.assign(n_points, 0.0f);

    auto label_to_supports = build_label_to_supports(
        supports.semantic_pair, supports.n_supports
    );

    // Collect unique labels
    std::vector<int32_t> unique_labels;
    {
        std::set<int32_t> label_set(segment, segment + n_points);
        unique_labels.assign(label_set.begin(), label_set.end());
    }

    // Per-label processing (parallelizable: each label writes to disjoint point indices)
    #pragma omp parallel for schedule(dynamic)
    for (int li = 0; li < static_cast<int>(unique_labels.size()); ++li) {
        int32_t label = unique_labels[li];
        if (label == static_cast<int32_t>(ignore_index)) continue;

        auto it = label_to_supports.find(label);
        if (it == label_to_supports.end()) continue;
        const auto& cand_supports = it->second;

        // Gather point indices for this label
        std::vector<int32_t> point_index;
        point_index.reserve(n_points / unique_labels.size());
        for (int i = 0; i < n_points; ++i) {
            if (segment[i] == label) {
                point_index.push_back(i);
            }
        }
        int np = static_cast<int>(point_index.size());
        if (np == 0) continue;

        // Gather points
        std::vector<float> points(np * 3);
        for (int i = 0; i < np; ++i) {
            points[i * 3 + 0] = coord[point_index[i] * 3 + 0];
            points[i * 3 + 1] = coord[point_index[i] * 3 + 1];
            points[i * 3 + 2] = coord[point_index[i] * 3 + 2];
        }

        std::vector<float> best_dist(np, std::numeric_limits<float>::infinity());
        std::vector<float> best_q(np * 3, 0.0f);
        std::vector<int32_t> best_support(np, -1);

        std::vector<float> tmp_closest(np * 3);
        std::vector<float> tmp_dist(np);

        for (int32_t sid : cand_supports) {
            if (skip_supports.count(sid)) continue;

            closest_points_to_support(
                points.data(), np, sid, supports,
                tmp_closest.data(), tmp_dist.data()
            );

            for (int i = 0; i < np; ++i) {
                if (tmp_dist[i] < best_dist[i]) {
                    best_dist[i] = tmp_dist[i];
                    best_q[i * 3 + 0] = tmp_closest[i * 3 + 0];
                    best_q[i * 3 + 1] = tmp_closest[i * 3 + 1];
                    best_q[i * 3 + 2] = tmp_closest[i * 3 + 2];
                    best_support[i] = sid;
                }
            }
        }

        // Write results
        for (int i = 0; i < np; ++i) {
            if (!std::isfinite(best_dist[i]) || best_support[i] < 0 ||
                best_dist[i] > support_radius) {
                continue;
            }

            int gi = point_index[i];
            result.edge_dist[gi] = best_dist[i];
            result.edge_valid[gi] = 1;
            result.edge_support_id[gi] = best_support[i];

            float vx = best_q[i * 3 + 0] - points[i * 3 + 0];
            float vy = best_q[i * 3 + 1] - points[i * 3 + 1];
            float vz = best_q[i * 3 + 2] - points[i * 3 + 2];
            result.edge_vec[gi * 3 + 0] = vx;
            result.edge_vec[gi * 3 + 1] = vy;
            result.edge_vec[gi * 3 + 2] = vz;

            float vn = std::sqrt(vx * vx + vy * vy + vz * vz);
            if (vn > EPS) {
                result.edge_dir[gi * 3 + 0] = vx / vn;
                result.edge_dir[gi * 3 + 1] = vy / vn;
                result.edge_dir[gi * 3 + 2] = vz / vn;
            }
        }
    }

    // Gaussian support weights
    float eff_sigma = (sigma > 0.0f) ? sigma : std::max(support_radius / 2.0f, EPS);
    float inv_2sigma2 = 1.0f / (2.0f * eff_sigma * eff_sigma);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_points; ++i) {
        if (result.edge_valid[i] == 1) {
            float d = result.edge_dist[i];
            result.edge_support[i] = std::exp(-d * d * inv_2sigma2);
        }
    }
}

// ---- Arc position helper for find_bad_supports ----

static void arc_position_on_support(
    const float* points, int n,
    int support_id,
    const SupportsData& sup,
    std::vector<float>& t_norm_out,
    float& total_length_out
) {
    int offset = sup.segment_offset[support_id];
    int length = sup.segment_length[support_id];

    const float* seg_starts;
    const float* seg_ends;
    int n_segs;

    // Temporary storage for line support case
    float line_ss[3], line_se[3];

    if (length <= 0) {
        for (int d = 0; d < 3; ++d) {
            line_ss[d] = sup.line_start[support_id * 3 + d];
            line_se[d] = sup.line_end[support_id * 3 + d];
        }
        seg_starts = line_ss;
        seg_ends = line_se;
        n_segs = 1;
    } else {
        seg_starts = &sup.segment_start[offset * 3];
        seg_ends = &sup.segment_end[offset * 3];
        n_segs = length;
    }

    // Compute cumulative arc lengths
    std::vector<double> cum_len(n_segs + 1, 0.0);
    std::vector<float> seg_lens(n_segs);
    for (int i = 0; i < n_segs; ++i) {
        Vec3f ss = Eigen::Map<const Vec3f>(&seg_starts[i * 3]);
        Vec3f se = Eigen::Map<const Vec3f>(&seg_ends[i * 3]);
        seg_lens[i] = (se - ss).norm();
        cum_len[i + 1] = cum_len[i] + seg_lens[i];
    }
    double total_len = cum_len[n_segs];
    total_length_out = static_cast<float>(total_len);

    t_norm_out.resize(n);
    if (total_len < EPS) {
        std::fill(t_norm_out.begin(), t_norm_out.end(), 0.0f);
        return;
    }

    std::vector<double> best_dist(n, std::numeric_limits<double>::infinity());
    std::vector<double> best_arc(n, 0.0);

    for (int si = 0; si < n_segs; ++si) {
        Vec3f ss = Eigen::Map<const Vec3f>(&seg_starts[si * 3]);
        Vec3f se = Eigen::Map<const Vec3f>(&seg_ends[si * 3]);
        Vec3f sv = se - ss;
        float sl = seg_lens[si];

        for (int i = 0; i < n; ++i) {
            Vec3f p = Eigen::Map<const Vec3f>(&points[i * 3]);
            Vec3f diff = p - ss;
            double d, arc;
            if (sl < EPS) {
                d = diff.cast<double>().norm();
                arc = cum_len[si];
            } else {
                float t = std::clamp(diff.dot(sv) / (sl * sl), 0.0f, 1.0f);
                Vec3f cp = ss + t * sv;
                d = (p - cp).cast<double>().norm();
                arc = cum_len[si] + static_cast<double>(t) * sl;
            }
            if (d < best_dist[i]) {
                best_dist[i] = d;
                best_arc[i] = arc;
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        t_norm_out[i] = static_cast<float>(best_arc[i] / total_len);
    }
}

std::set<int> find_bad_supports(
    const SupportsData& supports,
    const BoundaryCentersData& bc,
    const LocalClustersData& lc,
    float min_length,
    float middle_lo,
    float middle_hi,
    float max_middle_fraction,
    float min_tangent_alignment
) {
    std::set<int> bad_ids;

    for (int sid = 0; sid < supports.n_supports; ++sid) {
        int cid = supports.cluster_id[sid];

        // Gather center indices belonging to this cluster
        std::vector<int32_t> idx;
        for (int i = 0; i < lc.n_entries; ++i) {
            if (lc.cluster_id[i] == cid) {
                idx.push_back(lc.center_index[i]);
            }
        }
        int n = static_cast<int>(idx.size());
        if (n < 2) continue;

        // Gather points and tangents
        std::vector<float> pts(n * 3);
        std::vector<float> tgts(n * 3);
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < 3; ++d) {
                pts[i * 3 + d] = bc.center_coord[idx[i] * 3 + d];
                tgts[i * 3 + d] = bc.center_tangent[idx[i] * 3 + d];
            }
        }

        // Hollow check
        std::vector<float> t_norm;
        float total_len;
        arc_position_on_support(pts.data(), n, sid, supports, t_norm, total_len);

        if (total_len >= min_length) {
            int n_middle = 0;
            for (int i = 0; i < n; ++i) {
                if (t_norm[i] >= middle_lo && t_norm[i] <= middle_hi) {
                    n_middle++;
                }
            }
            if (static_cast<float>(n_middle) / n <= max_middle_fraction) {
                bad_ids.insert(sid);
                continue;
            }
        }

        // Diagonal check
        int stype = supports.support_type[sid];
        Vec3f sup_dir;
        float sup_len;

        if (stype == 0) {
            Vec3f ls = Eigen::Map<const Vec3f>(&supports.line_start[sid * 3]);
            Vec3f le = Eigen::Map<const Vec3f>(&supports.line_end[sid * 3]);
            Vec3f diff = le - ls;
            sup_len = diff.norm();
            if (sup_len < EPS) continue;
            sup_dir = diff / sup_len;
        } else {
            int off = supports.segment_offset[sid];
            int length = supports.segment_length[sid];
            if (length == 0) continue;
            Vec3f s0 = Eigen::Map<const Vec3f>(&supports.segment_start[off * 3]);
            Vec3f sn = Eigen::Map<const Vec3f>(&supports.segment_end[(off + length - 1) * 3]);
            Vec3f overall = sn - s0;
            sup_len = overall.norm();
            if (sup_len < EPS) continue;
            sup_dir = overall / sup_len;
        }

        // Mean |cos| alignment
        double sum_align = 0.0;
        for (int i = 0; i < n; ++i) {
            Vec3f t = Eigen::Map<const Vec3f>(&tgts[i * 3]);
            float tn = t.norm();
            if (tn > 1e-9f) {
                t /= tn;
            }
            sum_align += std::abs(t.dot(sup_dir));
        }
        float mean_align = static_cast<float>(sum_align / n);
        if (mean_align < min_tangent_alignment) {
            bad_ids.insert(sid);
        }
    }

    return bad_ids;
}

}  // namespace bf_edge
