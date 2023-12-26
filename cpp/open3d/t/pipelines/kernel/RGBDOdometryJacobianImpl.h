// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Private header. Do not include in Open3d.h.

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

using t::geometry::kernel::NDArrayIndexer;
using t::geometry::kernel::TransformIndexer;

#ifndef __CUDACC__
using std::abs;
using std::isnan;
using std::max;
#endif

inline OPEN3D_HOST_DEVICE float HuberDeriv(float r, float delta) {
    float abs_r = abs(r);
    return abs_r < delta ? r : delta * Sign(r);
}

inline OPEN3D_HOST_DEVICE float HuberLoss(float r, float delta) {
    float abs_r = abs(r);
    return abs_r < delta ? 0.5 * r * r : delta * abs_r - 0.5 * delta * delta;
}

OPEN3D_HOST_DEVICE inline bool GetJacobianPointToPoint(
        int x,
        int y,
        const float square_dist_thr,
        const NDArrayIndexer& source_vertex_indexer,
        const NDArrayIndexer& target_vertex_indexer,
        const TransformIndexer& ti,
        float* J_x,
        float* J_y,
        float* J_z,
        float& rx,
        float& ry,
        float& rz) {
    float* source_v = source_vertex_indexer.GetDataPtr<float>(x, y);
    if (isnan(source_v[0])) {
        return false;
    }

    // Transform source points to the target camera's coordinate space.
    float T_source_to_target_v[3], u, v;
    ti.RigidTransform(source_v[0], source_v[1], source_v[2],
                      &T_source_to_target_v[0], &T_source_to_target_v[1],
                      &T_source_to_target_v[2]);
    ti.Project(T_source_to_target_v[0], T_source_to_target_v[1],
               T_source_to_target_v[2], &u, &v);
    u = roundf(u);
    v = roundf(v);

    if (T_source_to_target_v[2] < 0 ||
        !target_vertex_indexer.InBoundary(u, v)) {
        return false;
    }

    int ui = static_cast<int>(u);
    int vi = static_cast<int>(v);
    float* target_v = target_vertex_indexer.GetDataPtr<float>(ui, vi);
    if (isnan(target_v[0])) {
        return false;
    }

    rx = (T_source_to_target_v[0] - target_v[0]);
    ry = (T_source_to_target_v[1] - target_v[1]);
    rz = (T_source_to_target_v[2] - target_v[2]);
    float r2 = rx * rx + ry * ry + rz * rz;
    if (r2 > square_dist_thr) {
        return false;
    }

    // T s - t
    J_x[0] = J_x[4] = J_x[5] = 0.0;
    J_x[1] = T_source_to_target_v[2];
    J_x[2] = -T_source_to_target_v[1];
    J_x[3] = 1.0;

    J_y[1] = J_y[3] = J_y[5] = 0.0;
    J_y[0] = -T_source_to_target_v[2];
    J_y[2] = T_source_to_target_v[0];
    J_y[4] = 1.0;

    J_z[2] = J_z[3] = J_z[4] = 0.0;
    J_z[0] = T_source_to_target_v[1];
    J_z[1] = -T_source_to_target_v[0];
    J_z[5] = 1.0;

    return true;
}

OPEN3D_HOST_DEVICE inline bool GetJacobianPointToPlane(
        int x,
        int y,
        const float depth_outlier_trunc,
        const NDArrayIndexer& source_vertex_indexer,
        const NDArrayIndexer& target_vertex_indexer,
        const NDArrayIndexer& target_normal_indexer,
        const TransformIndexer& ti,
        float* J_ij,
        float& r) {
    float* source_v = source_vertex_indexer.GetDataPtr<float>(x, y);
    if (isnan(source_v[0])) {
        return false;
    }

    // Transform source points to the target camera's coordinate space.
    float T_source_to_target_v[3], u, v;
    ti.RigidTransform(source_v[0], source_v[1], source_v[2],
                      &T_source_to_target_v[0], &T_source_to_target_v[1],
                      &T_source_to_target_v[2]);
    ti.Project(T_source_to_target_v[0], T_source_to_target_v[1],
               T_source_to_target_v[2], &u, &v);
    u = roundf(u);
    v = roundf(v);

    if (T_source_to_target_v[2] < 0 ||
        !target_vertex_indexer.InBoundary(u, v)) {
        return false;
    }

    int ui = static_cast<int>(u);
    int vi = static_cast<int>(v);
    float* target_v = target_vertex_indexer.GetDataPtr<float>(ui, vi);
    float* target_n = target_normal_indexer.GetDataPtr<float>(ui, vi);
    if (isnan(target_v[0]) || isnan(target_n[0])) {
        return false;
    }

    r = (T_source_to_target_v[0] - target_v[0]) * target_n[0] +
        (T_source_to_target_v[1] - target_v[1]) * target_n[1] +
        (T_source_to_target_v[2] - target_v[2]) * target_n[2];
    if (abs(r) > depth_outlier_trunc) {
        return false;
    }

    J_ij[0] = -T_source_to_target_v[2] * target_n[1] +
              T_source_to_target_v[1] * target_n[2];
    J_ij[1] = T_source_to_target_v[2] * target_n[0] -
              T_source_to_target_v[0] * target_n[2];
    J_ij[2] = -T_source_to_target_v[1] * target_n[0] +
              T_source_to_target_v[0] * target_n[1];
    J_ij[3] = target_n[0];
    J_ij[4] = target_n[1];
    J_ij[5] = target_n[2];

    return true;
}

OPEN3D_HOST_DEVICE inline bool GetJacobianIntensity(
        int x,
        int y,
        const float depth_outlier_trunc,
        const NDArrayIndexer& source_depth_indexer,
        const NDArrayIndexer& target_depth_indexer,
        const NDArrayIndexer& source_intensity_indexer,
        const NDArrayIndexer& target_intensity_indexer,
        const NDArrayIndexer& target_intensity_dx_indexer,
        const NDArrayIndexer& target_intensity_dy_indexer,
        const NDArrayIndexer& source_vertex_indexer,
        const TransformIndexer& ti,
        float* J_I,
        float& r_I) {
    const float sobel_scale = 0.125;

    float* source_v = source_vertex_indexer.GetDataPtr<float>(x, y);
    if (isnan(source_v[0])) {
        return false;
    }

    // Transform source points to the target camera's coordinate space.
    float T_source_to_target_v[3], u_tf, v_tf;
    ti.RigidTransform(source_v[0], source_v[1], source_v[2],
                      &T_source_to_target_v[0], &T_source_to_target_v[1],
                      &T_source_to_target_v[2]);
    ti.Project(T_source_to_target_v[0], T_source_to_target_v[1],
               T_source_to_target_v[2], &u_tf, &v_tf);
    int u_t = int(roundf(u_tf));
    int v_t = int(roundf(v_tf));

    if (T_source_to_target_v[2] < 0 ||
        !target_depth_indexer.InBoundary(u_t, v_t)) {
        return false;
    }

    float fx, fy;
    ti.GetFocalLength(&fx, &fy);

    float depth_t = *target_depth_indexer.GetDataPtr<float>(u_t, v_t);
    float diff_D = depth_t - T_source_to_target_v[2];
    if (isnan(depth_t) || abs(diff_D) > depth_outlier_trunc) {
        return false;
    }

    float diff_I = *target_intensity_indexer.GetDataPtr<float>(u_t, v_t) -
                   *source_intensity_indexer.GetDataPtr<float>(x, y);
    float dIdx = sobel_scale *
                 (*target_intensity_dx_indexer.GetDataPtr<float>(u_t, v_t));
    float dIdy = sobel_scale *
                 (*target_intensity_dy_indexer.GetDataPtr<float>(u_t, v_t));

    float invz = 1 / T_source_to_target_v[2];
    float c0 = dIdx * fx * invz;
    float c1 = dIdy * fy * invz;
    float c2 = -(c0 * T_source_to_target_v[0] + c1 * T_source_to_target_v[1]) *
               invz;

    J_I[0] = (-T_source_to_target_v[2] * c1 + T_source_to_target_v[1] * c2);
    J_I[1] = (T_source_to_target_v[2] * c0 - T_source_to_target_v[0] * c2);
    J_I[2] = (-T_source_to_target_v[1] * c0 + T_source_to_target_v[0] * c1);
    J_I[3] = (c0);
    J_I[4] = (c1);
    J_I[5] = (c2);
    r_I = diff_I;

    return true;
}

OPEN3D_HOST_DEVICE inline bool GetJacobianHybrid(
        int x,
        int y,
        const float depth_outlier_trunc,
        const NDArrayIndexer& source_depth_indexer,
        const NDArrayIndexer& target_depth_indexer,
        const NDArrayIndexer& source_intensity_indexer,
        const NDArrayIndexer& target_intensity_indexer,
        const NDArrayIndexer& target_depth_dx_indexer,
        const NDArrayIndexer& target_depth_dy_indexer,
        const NDArrayIndexer& target_intensity_dx_indexer,
        const NDArrayIndexer& target_intensity_dy_indexer,
        const NDArrayIndexer& source_vertex_indexer,
        const TransformIndexer& ti,
        float* J_I,
        float* J_D,
        float& r_I,
        float& r_D) {
    // sqrt 0.5, according to
    // http://redwood-data.org/indoor_lidar_rgbd/supp.pdf
    const float sqrt_lambda_intensity = 0.707;
    const float sqrt_lambda_depth = 0.707;
    const float sobel_scale = 0.125;

    float* source_v = source_vertex_indexer.GetDataPtr<float>(x, y);
    if (isnan(source_v[0])) {
        return false;
    }

    // Transform source points to the target camera coordinate space.
    float T_source_to_target_v[3], u_tf, v_tf;
    ti.RigidTransform(source_v[0], source_v[1], source_v[2],
                      &T_source_to_target_v[0], &T_source_to_target_v[1],
                      &T_source_to_target_v[2]);
    ti.Project(T_source_to_target_v[0], T_source_to_target_v[1],
               T_source_to_target_v[2], &u_tf, &v_tf);
    int u_t = int(roundf(u_tf));
    int v_t = int(roundf(v_tf));

    if (T_source_to_target_v[2] < 0 ||
        !target_depth_indexer.InBoundary(u_t, v_t)) {
        return false;
    }

    float fx, fy;
    ti.GetFocalLength(&fx, &fy);

    float depth_t = *target_depth_indexer.GetDataPtr<float>(u_t, v_t);
    float diff_D = depth_t - T_source_to_target_v[2];
    if (isnan(depth_t) || abs(diff_D) > depth_outlier_trunc) {
        return false;
    }

    float dDdx = sobel_scale *
                 (*target_depth_dx_indexer.GetDataPtr<float>(u_t, v_t));
    float dDdy = sobel_scale *
                 (*target_depth_dy_indexer.GetDataPtr<float>(u_t, v_t));
    if (isnan(dDdx) || isnan(dDdy)) {
        return false;
    }

    float diff_I = *target_intensity_indexer.GetDataPtr<float>(u_t, v_t) -
                   *source_intensity_indexer.GetDataPtr<float>(x, y);
    float dIdx = sobel_scale *
                 (*target_intensity_dx_indexer.GetDataPtr<float>(u_t, v_t));
    float dIdy = sobel_scale *
                 (*target_intensity_dy_indexer.GetDataPtr<float>(u_t, v_t));

    float invz = 1 / T_source_to_target_v[2];
    float c0 = dIdx * fx * invz;
    float c1 = dIdy * fy * invz;
    float c2 = -(c0 * T_source_to_target_v[0] + c1 * T_source_to_target_v[1]) *
               invz;
    float d0 = dDdx * fx * invz;
    float d1 = dDdy * fy * invz;
    float d2 = -(d0 * T_source_to_target_v[0] + d1 * T_source_to_target_v[1]) *
               invz;

    J_I[0] = sqrt_lambda_intensity *
             (-T_source_to_target_v[2] * c1 + T_source_to_target_v[1] * c2);
    J_I[1] = sqrt_lambda_intensity *
             (T_source_to_target_v[2] * c0 - T_source_to_target_v[0] * c2);
    J_I[2] = sqrt_lambda_intensity *
             (-T_source_to_target_v[1] * c0 + T_source_to_target_v[0] * c1);
    J_I[3] = sqrt_lambda_intensity * (c0);
    J_I[4] = sqrt_lambda_intensity * (c1);
    J_I[5] = sqrt_lambda_intensity * (c2);
    r_I = sqrt_lambda_intensity * diff_I;

    J_D[0] = sqrt_lambda_depth *
             ((-T_source_to_target_v[2] * d1 + T_source_to_target_v[1] * d2) -
              T_source_to_target_v[1]);
    J_D[1] = sqrt_lambda_depth *
             ((T_source_to_target_v[2] * d0 - T_source_to_target_v[0] * d2) +
              T_source_to_target_v[0]);
    J_D[2] = sqrt_lambda_depth *
             ((-T_source_to_target_v[1] * d0 + T_source_to_target_v[0] * d1));
    J_D[3] = sqrt_lambda_depth * (d0);
    J_D[4] = sqrt_lambda_depth * (d1);
    J_D[5] = sqrt_lambda_depth * (d2 - 1.0f);

    r_D = sqrt_lambda_depth * diff_D;

    return true;
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
