#include <bf_edge/stage2.h>
#include <bf_edge/kdtree.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bf_edge {

// ---- Union-Find ----

struct UnionFind {
    std::vector<int> parent;
    explicit UnionFind(int n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a != b) parent[a] = b;
    }
};

// ---- DBSCAN (spatial, using nanoflann) ----

static std::vector<int32_t> spatial_dbscan(
    const float* coords, int n,
    float eps, int min_samples
) {
    std::vector<int32_t> labels(n, -1);
    if (n == 0) return labels;

    PointCloud3f cloud(coords, n);
    KDTree3f tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(16));
    tree.buildIndex();

    float eps_sq = eps * eps;
    int cluster_id = 0;

    std::vector<bool> visited(n, false);

    for (int i = 0; i < n; ++i) {
        if (visited[i]) continue;

        std::vector<nanoflann::ResultItem<int32_t, float>> neighbors;
        tree.radiusSearch(&coords[i * 3], eps_sq, neighbors);

        if (static_cast<int>(neighbors.size()) < min_samples) {
            visited[i] = true;
            continue;  // noise
        }

        int cid = cluster_id++;
        labels[i] = cid;
        visited[i] = true;

        std::vector<int32_t> queue;
        for (auto& nb : neighbors) {
            if (nb.first != i) queue.push_back(nb.first);
        }

        for (size_t qi = 0; qi < queue.size(); ++qi) {
            int j = queue[qi];
            if (!visited[j]) {
                visited[j] = true;
                std::vector<nanoflann::ResultItem<int32_t, float>> j_neighbors;
                tree.radiusSearch(&coords[j * 3], eps_sq, j_neighbors);
                if (static_cast<int>(j_neighbors.size()) >= min_samples) {
                    for (auto& nb : j_neighbors) {
                        if (labels[nb.first] < 0 && !visited[nb.first]) {
                            queue.push_back(nb.first);
                        }
                    }
                }
            }
            if (labels[j] < 0) {
                labels[j] = cid;
            }
        }
    }
    return labels;
}

// ---- Tangent grouping ----

static std::vector<int32_t> group_tangents(
    const float* tangents, int n,
    float cos_th
) {
    std::vector<int32_t> labels(n, -1);
    if (n == 0) return labels;

    std::vector<Vec3f> reps;
    std::vector<Vec3f> accum;

    for (int i = 0; i < n; ++i) {
        Vec3f t = normalize(Eigen::Map<const Vec3f>(&tangents[i * 3]));

        int best_group = -1;
        float best_score = -1.0f;
        float best_sign = 1.0f;
        for (int g = 0; g < static_cast<int>(reps.size()); ++g) {
            float score_signed = t.dot(reps[g]);
            float score = std::abs(score_signed);
            if (score > best_score) {
                best_score = score;
                best_group = g;
                best_sign = score_signed >= 0.0f ? 1.0f : -1.0f;
            }
        }
        if (best_group >= 0 && best_score >= cos_th) {
            labels[i] = best_group;
            accum[best_group] = accum[best_group] + best_sign * t;
            reps[best_group] = normalize(accum[best_group]);
        } else {
            labels[i] = static_cast<int32_t>(reps.size());
            reps.push_back(t);
            accum.push_back(t);
        }
    }
    return labels;
}

// ---- Lateral split detection ----

static float lateral_bimodal_split_1d(
    const float* sorted_lat, int n,
    float threshold,
    bool& found
) {
    found = false;
    if (n < 2) return 0.0f;
    float max_gap = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < n - 1; ++i) {
        float gap = sorted_lat[i + 1] - sorted_lat[i];
        if (gap > max_gap) {
            max_gap = gap;
            max_idx = i;
        }
    }
    if (max_gap < threshold) return 0.0f;
    found = true;
    return (sorted_lat[max_idx] + sorted_lat[max_idx + 1]) / 2.0f;
}

// Recursive lateral split: returns list of local index arrays
static void recursive_lateral_split(
    const std::vector<int>& g_local,
    const float* pts, const float* tgts,
    float split_threshold, int min_pts,
    std::vector<std::vector<int>>& fragments
) {
    int n = static_cast<int>(g_local.size());
    if (n < 2 * min_pts) {
        fragments.push_back(g_local);
        return;
    }

    // Mean tangent
    Vec3f ref = Eigen::Map<const Vec3f>(&tgts[g_local[0] * 3]);
    Vec3f mt = Vec3f::Zero();
    for (int idx : g_local) {
        Vec3f t = Eigen::Map<const Vec3f>(&tgts[idx * 3]);
        float sign = t.dot(ref) >= 0.0f ? 1.0f : -1.0f;
        mt += sign * t;
    }
    float mt_norm = mt.norm();
    if (mt_norm < 1e-9f) { fragments.push_back(g_local); return; }
    mt /= mt_norm;

    // Lateral projection
    Vec3f centroid = Vec3f::Zero();
    for (int idx : g_local) centroid += Eigen::Map<const Vec3f>(&pts[idx * 3]);
    centroid /= static_cast<float>(n);

    Eigen::MatrixXf lateral(n, 3);
    float max_lat_norm = 0.0f;
    for (int i = 0; i < n; ++i) {
        Vec3f d = Eigen::Map<const Vec3f>(&pts[g_local[i] * 3]) - centroid;
        Vec3f lat = d - d.dot(mt) * mt;
        lateral.row(i) = lat.transpose();
        max_lat_norm = std::max(max_lat_norm, lat.norm());
    }
    if (max_lat_norm < split_threshold) { fragments.push_back(g_local); return; }

    // 1D lateral axis via SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(lateral, Eigen::ComputeThinV);
    Eigen::VectorXf lat_1d = lateral * svd.matrixV().col(0);

    // Sort
    std::vector<float> sorted(n);
    for (int i = 0; i < n; ++i) sorted[i] = lat_1d(i);
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return sorted[a] < sorted[b]; });
    std::vector<float> sorted_vals(n);
    for (int i = 0; i < n; ++i) sorted_vals[i] = sorted[order[i]];

    bool found;
    float sp = lateral_bimodal_split_1d(sorted_vals.data(), n, split_threshold, found);
    if (!found) { fragments.push_back(g_local); return; }

    std::vector<int> left, right;
    for (int i = 0; i < n; ++i) {
        if (lat_1d(i) < sp) left.push_back(g_local[i]);
        else right.push_back(g_local[i]);
    }

    if (static_cast<int>(left.size()) >= min_pts) {
        recursive_lateral_split(left, pts, tgts, split_threshold, min_pts, fragments);
    }
    if (static_cast<int>(right.size()) >= min_pts) {
        recursive_lateral_split(right, pts, tgts, split_threshold, min_pts, fragments);
    }
    if (fragments.empty()) fragments.push_back(g_local);
}

// ---- Check merged lateral gap ----

static bool merged_has_lateral_gap(
    const std::vector<std::vector<int>>& member_groups,
    const float* pts, const float* tgts,
    float cos_th, float split_threshold, int min_pts
) {
    // Gather all points
    std::vector<int> all_idx;
    for (auto& g : member_groups) all_idx.insert(all_idx.end(), g.begin(), g.end());
    int n = static_cast<int>(all_idx.size());
    if (n < 2 * min_pts) return false;

    // Group by tangent
    std::vector<float> all_tgts(n * 3);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d)
            all_tgts[i * 3 + d] = tgts[all_idx[i] * 3 + d];
    }
    auto dir_labels = group_tangents(all_tgts.data(), n, cos_th);

    std::set<int32_t> dir_ids(dir_labels.begin(), dir_labels.end());
    dir_ids.erase(-1);

    for (int32_t gid : dir_ids) {
        std::vector<int> g_idx;
        for (int i = 0; i < n; ++i) {
            if (dir_labels[i] == gid) g_idx.push_back(i);
        }
        int gn = static_cast<int>(g_idx.size());
        if (gn < 2 * min_pts) continue;

        // Mean tangent
        Vec3f ref(all_tgts[g_idx[0] * 3], all_tgts[g_idx[0] * 3 + 1], all_tgts[g_idx[0] * 3 + 2]);
        Vec3f mt_vec = Vec3f::Zero();
        for (int i : g_idx) {
            Vec3f t(all_tgts[i * 3], all_tgts[i * 3 + 1], all_tgts[i * 3 + 2]);
            float sign = t.dot(ref) >= 0.0f ? 1.0f : -1.0f;
            mt_vec += sign * t;
        }
        float mt_n = mt_vec.norm();
        if (mt_n < 1e-9f) continue;
        mt_vec /= mt_n;

        // Lateral proj
        Vec3f centroid_g = Vec3f::Zero();
        for (int i : g_idx) {
            int ri = all_idx[i];
            centroid_g += Eigen::Map<const Vec3f>(&pts[ri * 3]);
        }
        centroid_g /= static_cast<float>(gn);

        Eigen::MatrixXf lateral(gn, 3);
        float max_ln = 0.0f;
        for (int j = 0; j < gn; ++j) {
            int ri = all_idx[g_idx[j]];
            Vec3f d = Eigen::Map<const Vec3f>(&pts[ri * 3]) - centroid_g;
            Vec3f lat = d - d.dot(mt_vec) * mt_vec;
            lateral.row(j) = lat.transpose();
            max_ln = std::max(max_ln, lat.norm());
        }
        if (max_ln < split_threshold) continue;

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(lateral, Eigen::ComputeThinV);
        Eigen::VectorXf lat_1d = lateral * svd.matrixV().col(0);
        std::vector<float> sorted(gn);
        for (int j = 0; j < gn; ++j) sorted[j] = lat_1d(j);
        std::sort(sorted.begin(), sorted.end());

        bool found;
        lateral_bimodal_split_1d(sorted.data(), gn, split_threshold, found);
        if (found) return true;
    }
    return false;
}

// ---- Compute per-micro-cluster mean tangent ----

static Vec3f compute_mean_tangent(const float* tgts, const std::vector<int>& indices) {
    if (indices.empty()) return Vec3f::Zero();
    Vec3f ref = normalize(Eigen::Map<const Vec3f>(&tgts[indices[0] * 3]));
    Vec3f acc = Vec3f::Zero();
    for (int i : indices) {
        Vec3f t = normalize(Eigen::Map<const Vec3f>(&tgts[i * 3]));
        float sign = t.dot(ref) >= 0.0f ? 1.0f : -1.0f;
        acc += sign * t;
    }
    return normalize(acc);
}

// ---- Cluster one semantic pair ----

static void cluster_one_pair(
    const float* pair_coords, const float* pair_tangents, int n,
    const Stage2Params& params, float global_median_spacing,
    std::vector<int32_t>& out_labels, int& num_rescued
) {
    float eps_micro = params.micro_eps_scale * global_median_spacing;
    float merge_radius = params.merge_radius_scale * global_median_spacing;
    float rescue_radius = params.rescue_radius_scale * global_median_spacing;
    float cos_th = params.merge_direction_cos_th;
    float lateral_max = params.merge_lateral_scale * global_median_spacing;
    float split_threshold = params.split_lateral_threshold_scale * global_median_spacing;
    int min_pts = params.min_cluster_points;

    // Step 1: Micro-cluster via DBSCAN
    auto labels = spatial_dbscan(pair_coords, n, eps_micro, params.micro_min_samples);
    std::set<int32_t> micro_id_set;
    for (auto l : labels) if (l >= 0) micro_id_set.insert(l);
    std::vector<int32_t> micro_ids(micro_id_set.begin(), micro_id_set.end());

    if (micro_ids.empty()) {
        out_labels.assign(n, -1);
        num_rescued = 0;
        return;
    }

    // Step 1.5: Bimodal lateral split
    {
        int next_id = *std::max_element(micro_ids.begin(), micro_ids.end()) + 1;
        for (int32_t mid : std::vector<int32_t>(micro_ids)) {
            std::vector<int> global_idx;
            for (int i = 0; i < n; ++i) if (labels[i] == mid) global_idx.push_back(i);
            if (static_cast<int>(global_idx.size()) < 2 * min_pts) continue;

            // Group by tangent
            int gn = static_cast<int>(global_idx.size());
            std::vector<float> g_tgts(gn * 3);
            for (int i = 0; i < gn; ++i)
                for (int d = 0; d < 3; ++d)
                    g_tgts[i * 3 + d] = pair_tangents[global_idx[i] * 3 + d];

            auto dir_labels = group_tangents(g_tgts.data(), gn, cos_th);
            std::set<int32_t> dir_set(dir_labels.begin(), dir_labels.end());
            dir_set.erase(-1);

            bool any_split = false;
            std::vector<std::vector<int>> fragments;

            for (int32_t gid : dir_set) {
                std::vector<int> g_local;
                for (int i = 0; i < gn; ++i) if (dir_labels[i] == gid) g_local.push_back(i);

                std::vector<std::vector<int>> sub_frags;
                // Convert to global indices for pts/tgts lookup
                std::vector<int> g_local_global;
                for (int li : g_local) g_local_global.push_back(global_idx[li]);

                recursive_lateral_split(g_local_global, pair_coords, pair_tangents,
                                        split_threshold, min_pts, sub_frags);
                if (sub_frags.size() > 1) any_split = true;
                for (auto& f : sub_frags) fragments.push_back(f);
            }

            if (!any_split || fragments.size() < 2) continue;

            // Reassign
            for (int gi : global_idx) labels[gi] = -1;
            for (size_t fi = 0; fi < fragments.size(); ++fi) {
                if (static_cast<int>(fragments[fi].size()) < min_pts) continue;
                int32_t assign_id = (fi == 0) ? mid : next_id++;
                if (fi > 0) micro_ids.push_back(assign_id);
                for (int gi : fragments[fi]) labels[gi] = assign_id;
            }
        }
        // Prune empty
        micro_ids.erase(
            std::remove_if(micro_ids.begin(), micro_ids.end(),
                [&](int32_t m) { return std::none_of(labels.begin(), labels.end(),
                    [m](int32_t l) { return l == m; }); }),
            micro_ids.end()
        );
    }

    if (micro_ids.empty()) {
        out_labels.assign(n, -1);
        num_rescued = 0;
        return;
    }

    // Step 2: Compute centroids and mean tangents
    int nm = static_cast<int>(micro_ids.size());
    std::vector<Vec3f> centroids(nm);
    std::vector<Vec3f> mean_tangents(nm);
    std::vector<std::vector<int>> micro_members(nm);

    for (int mi = 0; mi < nm; ++mi) {
        int32_t mid = micro_ids[mi];
        Vec3f sum = Vec3f::Zero();
        for (int i = 0; i < n; ++i) {
            if (labels[i] == mid) {
                micro_members[mi].push_back(i);
                sum += Eigen::Map<const Vec3f>(&pair_coords[i * 3]);
            }
        }
        centroids[mi] = sum / static_cast<float>(micro_members[mi].size());
        mean_tangents[mi] = compute_mean_tangent(pair_tangents, micro_members[mi]);
    }

    // Step 3: Direction-aware merge via union-find
    UnionFind uf(nm);
    // Component tracking for lateral gap check
    std::unordered_map<int, std::vector<int>> components;
    for (int i = 0; i < nm; ++i) components[i] = {i};

    // Build centroid KD-tree for radius query
    std::vector<float> centroid_flat(nm * 3);
    for (int i = 0; i < nm; ++i)
        for (int d = 0; d < 3; ++d)
            centroid_flat[i * 3 + d] = centroids[i][d];

    if (nm >= 2) {
        PointCloud3f ccloud(centroid_flat.data(), nm);
        KDTree3f ctree(3, ccloud, nanoflann::KDTreeSingleIndexAdaptorParams(16));
        ctree.buildIndex();

        float mr_sq = merge_radius * merge_radius;

        for (int i = 0; i < nm; ++i) {
            std::vector<nanoflann::ResultItem<int32_t, float>> neighbors;
            ctree.radiusSearch(&centroid_flat[i * 3], mr_sq, neighbors);

            for (auto& nb : neighbors) {
                int j = nb.first;
                if (j <= i) continue;
                if (uf.find(i) == uf.find(j)) continue;

                float cos_sim = std::abs(mean_tangents[i].dot(mean_tangents[j]));
                if (cos_sim < cos_th) continue;

                Vec3f avg_t = mean_tangents[i] + mean_tangents[j];
                float tn = avg_t.norm();
                if (tn < 1e-9f) continue;
                avg_t /= tn;

                Vec3f diff = centroids[j] - centroids[i];
                float along = diff.dot(avg_t);
                float lateral = std::sqrt(std::max(diff.squaredNorm() - along * along, 0.0f));
                if (lateral > lateral_max) continue;

                // Lateral gap check
                int ri = uf.find(i), rj = uf.find(j);
                auto& ci = components[ri];
                auto& cj = components[rj];
                std::vector<std::vector<int>> merged_groups;
                for (int mi : ci) merged_groups.push_back(micro_members[mi]);
                for (int mi : cj) merged_groups.push_back(micro_members[mi]);

                if (merged_has_lateral_gap(merged_groups, pair_coords, pair_tangents,
                                           cos_th, split_threshold, min_pts)) continue;

                // Merge
                int old_ri = ri;
                uf.unite(i, j);
                int new_root = uf.find(i);
                if (new_root != old_ri) {
                    components[new_root].insert(components[new_root].end(),
                                                 components[old_ri].begin(),
                                                 components[old_ri].end());
                    components.erase(old_ri);
                }
                if (new_root != rj && components.count(rj)) {
                    components[new_root].insert(components[new_root].end(),
                                                 components[rj].begin(),
                                                 components[rj].end());
                    components.erase(rj);
                }
            }
        }
    }

    // Map to contiguous merged IDs
    std::vector<int32_t> roots(nm);
    for (int i = 0; i < nm; ++i) roots[i] = uf.find(i);
    std::set<int32_t> unique_roots(roots.begin(), roots.end());
    std::unordered_map<int32_t, int32_t> root_to_merged;
    int32_t merged_id = 0;
    for (int32_t r : unique_roots) root_to_merged[r] = merged_id++;

    std::vector<int32_t> merged_labels(n, -1);
    for (int mi = 0; mi < nm; ++mi) {
        int32_t ml = root_to_merged[roots[mi]];
        for (int pi : micro_members[mi]) {
            merged_labels[pi] = ml;
        }
    }

    // Step 4: Rescue noise
    num_rescued = 0;
    std::vector<int> noise_idx;
    std::vector<int> cluster_idx;
    for (int i = 0; i < n; ++i) {
        if (merged_labels[i] < 0) noise_idx.push_back(i);
        else cluster_idx.push_back(i);
    }

    if (!noise_idx.empty() && !cluster_idx.empty()) {
        int nc = static_cast<int>(cluster_idx.size());
        std::vector<float> cluster_pts(nc * 3);
        for (int i = 0; i < nc; ++i)
            for (int d = 0; d < 3; ++d)
                cluster_pts[i * 3 + d] = pair_coords[cluster_idx[i] * 3 + d];

        PointCloud3f rcloud(cluster_pts.data(), nc);
        KDTree3f rtree(3, rcloud, nanoflann::KDTreeSingleIndexAdaptorParams(16));
        rtree.buildIndex();

        for (int ni : noise_idx) {
            int32_t nn_idx;
            float nn_dist;
            rtree.knnSearch(&pair_coords[ni * 3], 1, &nn_idx, &nn_dist);
            if (std::sqrt(nn_dist) <= rescue_radius) {
                merged_labels[ni] = merged_labels[cluster_idx[nn_idx]];
                num_rescued++;
            }
        }
    }

    out_labels = std::move(merged_labels);
}

// ---- Public API ----

void cluster_boundary_centers(
    const float* center_coord, int n_centers,
    const float* center_tangent,
    const int32_t* semantic_pair,
    const Stage2Params& params,
    ClusterResult& result
) {
    if (n_centers == 0) return;

    // Global median 1-NN spacing
    float global_median_spacing = 0.01f;
    if (n_centers >= 2) {
        PointCloud3f cloud(center_coord, n_centers);
        KDTree3f tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(16));
        tree.buildIndex();

        std::vector<float> nn1_dists(n_centers);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_centers; ++i) {
            int32_t idx[2];
            float dist[2];
            tree.knnSearch(&center_coord[i * 3], 2, idx, dist);
            nn1_dists[i] = std::sqrt(dist[1]);  // dist[0] is self
        }
        std::sort(nn1_dists.begin(), nn1_dists.end());
        global_median_spacing = nn1_dists[n_centers / 2];
    }

    // Group by semantic pair
    struct PairKey {
        int32_t a, b;
        bool operator==(const PairKey& o) const { return a == o.a && b == o.b; }
    };
    struct PairHash {
        size_t operator()(const PairKey& k) const {
            return std::hash<int64_t>()(static_cast<int64_t>(k.a) << 32 | static_cast<uint32_t>(k.b));
        }
    };

    std::unordered_map<PairKey, std::vector<int>, PairHash> pair_groups;
    for (int i = 0; i < n_centers; ++i) {
        PairKey pk{semantic_pair[i * 2], semantic_pair[i * 2 + 1]};
        pair_groups[pk].push_back(i);
    }

    // Process each pair
    struct RunRecord {
        int32_t pair[2];
        std::vector<int32_t> global_indices;
    };
    std::vector<RunRecord> runs;
    int total_rescued = 0;

    for (auto& [pk, indices] : pair_groups) {
        int np = static_cast<int>(indices.size());
        if (np < params.micro_min_samples) continue;

        // Gather pair data
        std::vector<float> pair_coords_local(np * 3);
        std::vector<float> pair_tgts(np * 3);
        for (int i = 0; i < np; ++i) {
            for (int d = 0; d < 3; ++d) {
                pair_coords_local[i * 3 + d] = center_coord[indices[i] * 3 + d];
                pair_tgts[i * 3 + d] = center_tangent[indices[i] * 3 + d];
            }
        }

        std::vector<int32_t> merged_labels;
        int rescued;
        cluster_one_pair(
            pair_coords_local.data(), pair_tgts.data(), np,
            params, global_median_spacing,
            merged_labels, rescued
        );
        total_rescued += rescued;

        std::set<int32_t> cids(merged_labels.begin(), merged_labels.end());
        cids.erase(-1);

        for (int32_t cid : cids) {
            RunRecord rr;
            rr.pair[0] = pk.a;
            rr.pair[1] = pk.b;
            for (int i = 0; i < np; ++i) {
                if (merged_labels[i] == cid) {
                    rr.global_indices.push_back(indices[i]);
                }
            }
            if (static_cast<int>(rr.global_indices.size()) >= params.min_cluster_points) {
                runs.push_back(std::move(rr));
            }
        }
    }

    // Build output
    int n_clusters = static_cast<int>(runs.size());
    result.semantic_pair.resize(n_clusters * 2);
    result.cluster_size.resize(n_clusters);
    result.cluster_centroid.resize(n_clusters * 3);

    for (int ci = 0; ci < n_clusters; ++ci) {
        auto& rr = runs[ci];
        result.semantic_pair[ci * 2 + 0] = rr.pair[0];
        result.semantic_pair[ci * 2 + 1] = rr.pair[1];
        result.cluster_size[ci] = static_cast<int32_t>(rr.global_indices.size());

        Vec3f centroid = Vec3f::Zero();
        for (int32_t gi : rr.global_indices) {
            centroid += Eigen::Map<const Vec3f>(&center_coord[gi * 3]);
        }
        centroid /= static_cast<float>(rr.global_indices.size());
        for (int d = 0; d < 3; ++d) {
            result.cluster_centroid[ci * 3 + d] = centroid[d];
        }

        for (int32_t gi : rr.global_indices) {
            result.center_index.push_back(gi);
            result.cluster_id.push_back(ci);
        }
    }
}

}  // namespace bf_edge
