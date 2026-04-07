#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <vector>

namespace bf_edge {

constexpr float EPS = 1e-8f;

using Vec3f = Eigen::Vector3f;
using Vec3d = Eigen::Vector3d;
using MatX3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
using VecXf = Eigen::VectorXf;
using VecXi = Eigen::VectorXi;

// Map numpy (N,3) float32 row-major to Eigen
inline Eigen::Map<const MatX3f> map_points(const float* data, int n) {
    return Eigen::Map<const MatX3f>(data, n, 3);
}

inline Eigen::Map<MatX3f> map_points_mut(float* data, int n) {
    return Eigen::Map<MatX3f>(data, n, 3);
}

inline Vec3f normalize(const Vec3f& v) {
    float n = v.norm();
    if (n > EPS) return Vec3f(v / n);
    return Vec3f::Zero();
}

// Normalize each row of an (N,3) matrix in-place
inline void normalize_rows_inplace(MatX3f& mat) {
    for (int i = 0; i < mat.rows(); ++i) {
        float n = mat.row(i).norm();
        if (n > EPS) {
            mat.row(i) /= n;
        } else {
            mat.row(i).setZero();
        }
    }
}

}  // namespace bf_edge
