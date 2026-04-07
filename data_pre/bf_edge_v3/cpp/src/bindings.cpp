#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <bf_edge/common.h>
#include <bf_edge/stage1.h>
#include <bf_edge/stage2.h>
#include <bf_edge/stage3.h>
#include <bf_edge/stage4.h>

namespace py = pybind11;
using namespace bf_edge;

// ---- Helper: ensure contiguous float32/int32 arrays ----

template <typename T>
static py::array_t<T> ensure_c_contiguous(py::array_t<T> arr) {
    if (arr.ndim() == 0) return arr;
    auto info = arr.request();
    bool contiguous = true;
    ssize_t expected = sizeof(T);
    for (int i = info.ndim - 1; i >= 0; --i) {
        if (info.strides[i] != expected) { contiguous = false; break; }
        expected *= info.shape[i];
    }
    if (contiguous) return arr;
    return py::array_t<T>::ensure(arr.attr("copy")());
}

// ---- Stage 1: build_boundary_centers ----

static py::dict py_build_boundary_centers(
    py::array_t<float> coord,
    py::array_t<int32_t> segment,
    py::object normal_obj,
    int k, float min_cross_ratio, int min_side_points, int ignore_index
) {
    auto c = ensure_c_contiguous(coord);
    auto s = ensure_c_contiguous(segment);
    int n_points = static_cast<int>(c.shape(0));

    const float* normal_ptr = nullptr;
    py::array_t<float> normal_arr;
    if (!normal_obj.is_none()) {
        normal_arr = ensure_c_contiguous(normal_obj.cast<py::array_t<float>>());
        normal_ptr = normal_arr.data();
    }

    BoundaryCentersResult result;
    build_boundary_centers(
        c.data(), n_points, s.data(), normal_ptr,
        k, min_cross_ratio, min_side_points, ignore_index,
        result
    );

    int nc = static_cast<int>(result.confidence.size());
    int ncand = static_cast<int>(result.cand_point_index.size());

    py::dict out;

    // Candidates
    out["cand_point_index"] = py::array_t<int32_t>({ncand}, result.cand_point_index.data());
    out["cand_semantic_pair"] = py::array_t<int32_t>({ncand, 2}, result.cand_semantic_pair.data());
    out["cand_cross_ratio"] = py::array_t<float>({ncand}, result.cand_cross_ratio.data());

    // Centers
    out["center_coord"] = py::array_t<float>({nc, 3}, result.center_coord.data());
    out["center_normal"] = py::array_t<float>({nc, 3}, result.center_normal.data());
    out["center_tangent"] = py::array_t<float>({nc, 3}, result.center_tangent.data());
    out["semantic_pair"] = py::array_t<int32_t>({nc, 2}, result.semantic_pair.data());
    out["source_point_index"] = py::array_t<int32_t>({nc}, result.source_point_index.data());
    out["confidence"] = py::array_t<float>({nc}, result.confidence.data());

    return out;
}

// ---- Stage 2: cluster_boundary_centers ----

static py::dict py_cluster_boundary_centers(
    py::array_t<float> center_coord,
    py::array_t<float> center_tangent,
    py::array_t<int32_t> semantic_pair,
    float micro_eps_scale,
    int micro_min_samples,
    float split_lateral_threshold_scale,
    float merge_radius_scale,
    float merge_direction_cos_th,
    float merge_lateral_scale,
    float rescue_radius_scale,
    int min_cluster_points
) {
    auto cc = ensure_c_contiguous(center_coord);
    auto ct = ensure_c_contiguous(center_tangent);
    auto sp = ensure_c_contiguous(semantic_pair);
    int n = static_cast<int>(cc.shape(0));

    Stage2Params params;
    params.micro_eps_scale = micro_eps_scale;
    params.micro_min_samples = micro_min_samples;
    params.split_lateral_threshold_scale = split_lateral_threshold_scale;
    params.merge_radius_scale = merge_radius_scale;
    params.merge_direction_cos_th = merge_direction_cos_th;
    params.merge_lateral_scale = merge_lateral_scale;
    params.rescue_radius_scale = rescue_radius_scale;
    params.min_cluster_points = min_cluster_points;

    ClusterResult result;
    cluster_boundary_centers(cc.data(), n, ct.data(), sp.data(), params, result);

    int ne = static_cast<int>(result.center_index.size());
    int ncl = static_cast<int>(result.cluster_size.size());

    py::dict out;
    out["center_index"] = py::array_t<int32_t>({ne}, result.center_index.data());
    out["cluster_id"] = py::array_t<int32_t>({ne}, result.cluster_id.data());
    out["semantic_pair"] = py::array_t<int32_t>({ncl, 2}, result.semantic_pair.data());
    out["cluster_size"] = py::array_t<int32_t>({ncl}, result.cluster_size.data());
    out["cluster_centroid"] = py::array_t<float>({ncl, 3}, result.cluster_centroid.data());

    return out;
}

// ---- Stage 3: build_supports ----

static py::dict py_build_supports(
    py::array_t<float> center_coord,
    py::array_t<float> center_confidence,
    py::array_t<int32_t> cluster_center_index,
    py::array_t<int32_t> cluster_cluster_id,
    py::array_t<int32_t> cluster_semantic_pair,
    py::array_t<int32_t> cluster_size,
    py::array_t<float> cluster_centroid,
    float line_residual_th,
    int min_cluster_size,
    int max_polyline_vertices,
    float polyline_residual_th,
    float min_cluster_density
) {
    auto cc = ensure_c_contiguous(center_coord);
    auto conf = ensure_c_contiguous(center_confidence);
    auto cci = ensure_c_contiguous(cluster_center_index);
    auto ccid = ensure_c_contiguous(cluster_cluster_id);
    auto csp = ensure_c_contiguous(cluster_semantic_pair);
    auto cs = ensure_c_contiguous(cluster_size);
    auto ccent = ensure_c_contiguous(cluster_centroid);

    int n_centers = static_cast<int>(cc.shape(0));
    int n_entries = static_cast<int>(cci.shape(0));
    int n_clusters = static_cast<int>(cs.shape(0));

    Stage3Params params;
    params.line_residual_th = line_residual_th;
    params.min_cluster_size = min_cluster_size;
    params.max_polyline_vertices = max_polyline_vertices;
    params.polyline_residual_th = polyline_residual_th;
    params.min_cluster_density = min_cluster_density;

    SupportsResult result;
    build_supports(
        cc.data(), n_centers, conf.data(),
        cci.data(), n_entries, ccid.data(),
        csp.data(), cs.data(), ccent.data(), n_clusters,
        params, result
    );

    int ns = static_cast<int>(result.support_id.size());
    int ts = static_cast<int>(result.segment_start.size()) / 3;

    py::dict out;
    out["support_id"] = py::array_t<int32_t>({ns}, result.support_id.data());
    out["support_type"] = py::array_t<int32_t>({ns}, result.support_type.data());
    out["semantic_pair"] = py::array_t<int32_t>({ns, 2}, result.semantic_pair.data());
    out["confidence"] = py::array_t<float>({ns}, result.confidence.data());
    out["fit_residual"] = py::array_t<float>({ns}, result.fit_residual.data());
    out["coverage_radius"] = py::array_t<float>({ns}, result.coverage_radius.data());
    out["cluster_id"] = py::array_t<int32_t>({ns}, result.cluster_id.data());
    out["origin"] = py::array_t<float>({ns, 3}, result.origin.data());
    out["direction"] = py::array_t<float>({ns, 3}, result.direction.data());
    out["line_start"] = py::array_t<float>({ns, 3}, result.line_start.data());
    out["line_end"] = py::array_t<float>({ns, 3}, result.line_end.data());
    out["orientation_prior_score"] = py::array_t<float>({ns}, result.orientation_prior_score.data());
    out["segment_offset"] = py::array_t<int32_t>({ns}, result.segment_offset.data());
    out["segment_length"] = py::array_t<int32_t>({ns}, result.segment_length.data());
    out["segment_start"] = py::array_t<float>({ts, 3}, result.segment_start.data());
    out["segment_end"] = py::array_t<float>({ts, 3}, result.segment_end.data());

    return out;
}

// ---- Stage 4: build_pointwise_edge_supervision ----

static py::dict py_build_pointwise_edge_supervision(
    py::array_t<float> coord,
    py::array_t<int32_t> segment,
    py::array_t<int32_t> support_id,
    py::array_t<int32_t> semantic_pair,
    py::array_t<int32_t> segment_offset,
    py::array_t<int32_t> segment_length,
    py::array_t<float> segment_start,
    py::array_t<float> segment_end,
    py::array_t<float> line_start,
    py::array_t<float> line_end,
    py::array_t<int32_t> cluster_id_arr,
    py::array_t<int32_t> support_type,
    float support_radius,
    int ignore_index,
    py::object skip_supports_obj
) {
    auto c = ensure_c_contiguous(coord);
    auto seg = ensure_c_contiguous(segment);
    int n_points = static_cast<int>(c.shape(0));
    int n_supports = static_cast<int>(support_id.size());

    SupportsData sup;
    sup.n_supports = n_supports;
    sup.support_id = ensure_c_contiguous(support_id).data();
    sup.semantic_pair = ensure_c_contiguous(semantic_pair).data();
    sup.segment_offset = ensure_c_contiguous(segment_offset).data();
    sup.segment_length = ensure_c_contiguous(segment_length).data();
    sup.segment_start = ensure_c_contiguous(segment_start).data();
    sup.segment_end = ensure_c_contiguous(segment_end).data();
    sup.line_start = ensure_c_contiguous(line_start).data();
    sup.line_end = ensure_c_contiguous(line_end).data();
    sup.cluster_id = ensure_c_contiguous(cluster_id_arr).data();
    sup.support_type = ensure_c_contiguous(support_type).data();

    // Keep contiguous arrays alive
    auto sp_c = ensure_c_contiguous(semantic_pair);
    auto so_c = ensure_c_contiguous(segment_offset);
    auto sl_c = ensure_c_contiguous(segment_length);
    auto ss_c = ensure_c_contiguous(segment_start);
    auto se_c = ensure_c_contiguous(segment_end);
    auto ls_c = ensure_c_contiguous(line_start);
    auto le_c = ensure_c_contiguous(line_end);
    auto ci_c = ensure_c_contiguous(cluster_id_arr);
    auto st_c = ensure_c_contiguous(support_type);
    auto si_c = ensure_c_contiguous(support_id);

    sup.support_id = si_c.data();
    sup.semantic_pair = sp_c.data();
    sup.segment_offset = so_c.data();
    sup.segment_length = sl_c.data();
    sup.segment_start = ss_c.data();
    sup.segment_end = se_c.data();
    sup.line_start = ls_c.data();
    sup.line_end = le_c.data();
    sup.cluster_id = ci_c.data();
    sup.support_type = st_c.data();

    std::set<int> skip;
    if (!skip_supports_obj.is_none()) {
        auto skip_set = skip_supports_obj.cast<std::set<int>>();
        skip = skip_set;
    }

    PointwiseResult result;
    build_pointwise_edge_supervision(
        c.data(), n_points, seg.data(),
        sup, support_radius, ignore_index, skip,
        result
    );

    py::dict out;
    out["edge_dist"] = py::array_t<float>({n_points}, result.edge_dist.data());
    out["edge_dir"] = py::array_t<float>({n_points, 3}, result.edge_dir.data());
    out["edge_valid"] = py::array_t<uint8_t>({n_points}, result.edge_valid.data());
    out["edge_support_id"] = py::array_t<int32_t>({n_points}, result.edge_support_id.data());
    out["edge_vec"] = py::array_t<float>({n_points, 3}, result.edge_vec.data());
    out["edge_support"] = py::array_t<float>({n_points}, result.edge_support.data());

    return out;
}

// ---- Stage 4: find_bad_supports ----

static py::set py_find_bad_supports(
    py::array_t<int32_t> support_id,
    py::array_t<int32_t> semantic_pair,
    py::array_t<int32_t> segment_offset,
    py::array_t<int32_t> segment_length,
    py::array_t<float> segment_start,
    py::array_t<float> segment_end,
    py::array_t<float> line_start,
    py::array_t<float> line_end,
    py::array_t<int32_t> cluster_id_arr,
    py::array_t<int32_t> support_type,
    py::array_t<float> center_coord,
    py::array_t<float> center_tangent,
    py::array_t<int32_t> lc_cluster_id,
    py::array_t<int32_t> lc_center_index,
    float min_length,
    float middle_lo,
    float middle_hi,
    float max_middle_fraction,
    float min_tangent_alignment
) {
    int n_supports = static_cast<int>(support_id.size());
    auto si_c = ensure_c_contiguous(support_id);
    auto sp_c = ensure_c_contiguous(semantic_pair);
    auto so_c = ensure_c_contiguous(segment_offset);
    auto sl_c = ensure_c_contiguous(segment_length);
    auto ss_c = ensure_c_contiguous(segment_start);
    auto se_c = ensure_c_contiguous(segment_end);
    auto ls_c = ensure_c_contiguous(line_start);
    auto le_c = ensure_c_contiguous(line_end);
    auto ci_c = ensure_c_contiguous(cluster_id_arr);
    auto st_c = ensure_c_contiguous(support_type);

    SupportsData sup;
    sup.n_supports = n_supports;
    sup.support_id = si_c.data();
    sup.semantic_pair = sp_c.data();
    sup.segment_offset = so_c.data();
    sup.segment_length = sl_c.data();
    sup.segment_start = ss_c.data();
    sup.segment_end = se_c.data();
    sup.line_start = ls_c.data();
    sup.line_end = le_c.data();
    sup.cluster_id = ci_c.data();
    sup.support_type = st_c.data();

    auto cc_c = ensure_c_contiguous(center_coord);
    auto ct_c = ensure_c_contiguous(center_tangent);
    BoundaryCentersData bc;
    bc.n_centers = static_cast<int>(cc_c.shape(0));
    bc.center_coord = cc_c.data();
    bc.center_tangent = ct_c.data();

    auto lcid_c = ensure_c_contiguous(lc_cluster_id);
    auto lcci_c = ensure_c_contiguous(lc_center_index);
    LocalClustersData lc;
    lc.n_entries = static_cast<int>(lcid_c.shape(0));
    lc.cluster_id = lcid_c.data();
    lc.center_index = lcci_c.data();

    auto bad = find_bad_supports(
        sup, bc, lc,
        min_length, middle_lo, middle_hi,
        max_middle_fraction, min_tangent_alignment
    );

    py::set result;
    for (int id : bad) result.add(py::int_(id));
    return result;
}

// ---- Module definition ----

PYBIND11_MODULE(bf_edge_cpp, m) {
    m.doc() = "C++ accelerated bf_edge_v3 pipeline";

    m.def("build_boundary_centers", &py_build_boundary_centers,
        py::arg("coord"), py::arg("segment"), py::arg("normal"),
        py::arg("k"), py::arg("min_cross_ratio"),
        py::arg("min_side_points"), py::arg("ignore_index"),
        "Stage 1: detect boundary centers");

    m.def("cluster_boundary_centers", &py_cluster_boundary_centers,
        py::arg("center_coord"), py::arg("center_tangent"),
        py::arg("semantic_pair"),
        py::arg("micro_eps_scale") = 3.5f,
        py::arg("micro_min_samples") = 3,
        py::arg("split_lateral_threshold_scale") = 5.0f,
        py::arg("merge_radius_scale") = 8.0f,
        py::arg("merge_direction_cos_th") = 0.7071f,
        py::arg("merge_lateral_scale") = 5.0f,
        py::arg("rescue_radius_scale") = 10.0f,
        py::arg("min_cluster_points") = 4,
        "Stage 2: cluster boundary centers");

    m.def("build_supports", &py_build_supports,
        py::arg("center_coord"), py::arg("center_confidence"),
        py::arg("cluster_center_index"), py::arg("cluster_cluster_id"),
        py::arg("cluster_semantic_pair"), py::arg("cluster_size"),
        py::arg("cluster_centroid"),
        py::arg("line_residual_th") = 0.01f,
        py::arg("min_cluster_size") = 4,
        py::arg("max_polyline_vertices") = 32,
        py::arg("polyline_residual_th") = 0.04f,
        py::arg("min_cluster_density") = 15.0f,
        "Stage 3: build support elements");

    m.def("build_pointwise_edge_supervision", &py_build_pointwise_edge_supervision,
        py::arg("coord"), py::arg("segment"),
        py::arg("support_id"), py::arg("semantic_pair"),
        py::arg("segment_offset"), py::arg("segment_length"),
        py::arg("segment_start"), py::arg("segment_end"),
        py::arg("line_start"), py::arg("line_end"),
        py::arg("cluster_id"), py::arg("support_type"),
        py::arg("support_radius"), py::arg("ignore_index"),
        py::arg("skip_supports") = py::none(),
        "Stage 4: build pointwise edge supervision");

    m.def("find_bad_supports", &py_find_bad_supports,
        py::arg("support_id"), py::arg("semantic_pair"),
        py::arg("segment_offset"), py::arg("segment_length"),
        py::arg("segment_start"), py::arg("segment_end"),
        py::arg("line_start"), py::arg("line_end"),
        py::arg("cluster_id"), py::arg("support_type"),
        py::arg("center_coord"), py::arg("center_tangent"),
        py::arg("lc_cluster_id"), py::arg("lc_center_index"),
        py::arg("min_length") = 0.05f,
        py::arg("middle_lo") = 0.2f,
        py::arg("middle_hi") = 0.8f,
        py::arg("max_middle_fraction") = 0.15f,
        py::arg("min_tangent_alignment") = 0.5f,
        "Stage 4: find bad/hollow supports");
}
