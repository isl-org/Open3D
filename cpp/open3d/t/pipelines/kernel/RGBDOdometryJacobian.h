// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
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

using NDArrayIndexer = t::geometry::kernel::NDArrayIndexer;

OPEN3D_HOST_DEVICE inline bool GetJacobianPointToPlane(
        int64_t workload_idx,
        int64_t cols,
        float depth_diff,
        const NDArrayIndexer& source_vertex_indexer,
        const NDArrayIndexer& target_vertex_indexer,
        const NDArrayIndexer& source_normal_indexer,
        const t::geometry::kernel::TransformIndexer& ti,
        float* J_ij,
        float& r) {
    int64_t y = workload_idx / cols;
    int64_t x = workload_idx % cols;

    float* target_v = target_vertex_indexer.GetDataPtrFromCoord<float>(x, y);
    if (__ISNAN(target_v[0])) {
        return false;
    }

    float T_target_v[3], u, v;
    ti.RigidTransform(target_v[0], target_v[1], target_v[2], &T_target_v[0],
                      &T_target_v[1], &T_target_v[2]);
    ti.Project(T_target_v[0], T_target_v[1], T_target_v[2], &u, &v);
    u = round(u);
    v = round(v);

    if (T_target_v[2] < 0 || !source_vertex_indexer.InBoundary(u, v)) {
        return false;
    }

    int64_t ui = static_cast<int64_t>(u);
    int64_t vi = static_cast<int64_t>(v);
    float* source_v = source_vertex_indexer.GetDataPtrFromCoord<float>(ui, vi);
    float* source_n = source_normal_indexer.GetDataPtrFromCoord<float>(ui, vi);
    if (__ISNAN(source_v[0]) || __ISNAN(source_n[0])) {
        return false;
    }

    r = (T_target_v[0] - source_v[0]) * source_n[0] +
        (T_target_v[1] - source_v[1]) * source_n[1] +
        (T_target_v[2] - source_v[2]) * source_n[2];
    if (abs(r) > depth_diff) {
        return false;
    }

    J_ij[0] = -T_target_v[2] * source_n[1] + T_target_v[1] * source_n[2];
    J_ij[1] = T_target_v[2] * source_n[0] - T_target_v[0] * source_n[2];
    J_ij[2] = -T_target_v[1] * source_n[0] + T_target_v[0] * source_n[1];
    J_ij[3] = source_n[0];
    J_ij[4] = source_n[1];
    J_ij[5] = source_n[2];

    return true;
}

OPEN3D_HOST_DEVICE inline bool GetJacobianHybrid(
        int64_t workload_idx,
        int64_t cols,
        float depth_diff,
        const NDArrayIndexer& source_depth_indexer,
        const NDArrayIndexer& target_depth_indexer,
        const NDArrayIndexer& source_intensity_indexer,
        const NDArrayIndexer& target_intensity_indexer,
        const NDArrayIndexer& source_depth_dx_indexer,
        const NDArrayIndexer& source_depth_dy_indexer,
        const NDArrayIndexer& source_intensity_dx_indexer,
        const NDArrayIndexer& source_intensity_dy_indexer,
        const NDArrayIndexer& target_vertex_indexer,
        const t::geometry::kernel::TransformIndexer& ti,
        float* J_I,
        float* J_D,
        float& r_I,
        float& r_D) {
    // sqrt 0.5, according to
    // http://redwood-data.org/indoor_lidar_rgbd/supp.pdf
    const float sqrt_lambda_intensity = 0.707;
    const float sqrt_lambda_depth = 0.707;
    const float sobel_scale = 0.125;

    int v_d = workload_idx / cols;
    int u_d = workload_idx % cols;

    float* target_v =
            target_vertex_indexer.GetDataPtrFromCoord<float>(u_d, v_d);
    if (__ISNAN(target_v[0])) {
        return false;
    }

    // target on source: T_target_v
    float T_target_v[3], u_sf, v_sf;
    ti.RigidTransform(target_v[0], target_v[1], target_v[2], &T_target_v[0],
                      &T_target_v[1], &T_target_v[2]);
    ti.Project(T_target_v[0], T_target_v[1], T_target_v[2], &u_sf, &v_sf);
    int u_s = int(round(u_sf));
    int v_s = int(round(v_sf));

    if (T_target_v[2] < 0 || !source_depth_indexer.InBoundary(u_s, v_s)) {
        return false;
    }

    // TODO: customize filter depth grads

    const double fx = ti.fx_;
    const double fy = ti.fy_;

    // TODO: depth scale
    float depth_s =
            *source_depth_indexer.GetDataPtrFromCoord<float>(u_s, v_s) / 1000.0;
    if (__ISNAN(depth_s)) {
        return false;
    }

    float diff_D = depth_s - T_target_v[2];
    if (abs(diff_D) > depth_diff) {
        return false;
    }
    float dDdx =
            sobel_scale *
            (*source_depth_dx_indexer.GetDataPtrFromCoord<float>(u_s, v_s)) /
            1000.0;
    float dDdy =
            sobel_scale *
            (*source_depth_dy_indexer.GetDataPtrFromCoord<float>(u_s, v_s)) /
            1000.0;
    if (__ISNAN(dDdx) || __ISNAN(dDdy)) {
        return false;
    }

    float diff_I =
            *source_intensity_indexer.GetDataPtrFromCoord<float>(u_s, v_s) -
            *target_intensity_indexer.GetDataPtrFromCoord<float>(u_d, v_d);
    float dIdx =
            sobel_scale *
            (*source_intensity_dx_indexer.GetDataPtrFromCoord<float>(u_s, v_s));
    float dIdy =
            sobel_scale *
            (*source_intensity_dy_indexer.GetDataPtrFromCoord<float>(u_s, v_s));

    // printf("%ld: (%d %d %f) -> (%f %f %f) -> (%d %d) -> depth diff: %f,
    // color
    // "
    //        "diff: %f\n",
    //        workload_idx, u_d, v_d,
    //        *target_depth_indexer.GetDataPtrFromCoord<float>(u_d, v_d),
    //        target_v[0], target_v[1], target_v[2], u_s, v_s, depth_s -
    //        T_target_v[2],
    //        *source_intensity_indexer.GetDataPtrFromCoord<float>(u_s, v_s) -
    //                *target_intensity_indexer.GetDataPtrFromCoord<float>(u_d,
    //                v_d));

    float invz = 1 / T_target_v[2];
    float c0 = dIdx * fx * invz;
    float c1 = dIdy * fy * invz;
    float c2 = -(c0 * T_target_v[0] + c1 * T_target_v[1]) * invz;
    float d0 = dDdx * fx * invz;
    float d1 = dDdy * fy * invz;
    float d2 = -(d0 * T_target_v[0] + d1 * T_target_v[1]) * invz;

    J_I[0] = sqrt_lambda_intensity * (-T_target_v[2] * c1 + T_target_v[1] * c2);
    J_I[1] = sqrt_lambda_intensity * (T_target_v[2] * c0 - T_target_v[0] * c2);
    J_I[2] = sqrt_lambda_intensity * (-T_target_v[1] * c0 + T_target_v[0] * c1);
    J_I[3] = sqrt_lambda_intensity * (c0);
    J_I[4] = sqrt_lambda_intensity * (c1);
    J_I[5] = sqrt_lambda_intensity * (c2);
    r_I = sqrt_lambda_intensity * diff_I;

    J_D[0] = sqrt_lambda_depth *
             ((-T_target_v[2] * d1 + T_target_v[1] * d2) - T_target_v[1]);
    J_D[1] = sqrt_lambda_depth *
             ((T_target_v[2] * d0 - T_target_v[0] * d2) + T_target_v[0]);
    J_D[2] = sqrt_lambda_depth * ((-T_target_v[1] * d0 + T_target_v[0] * d1));
    J_D[3] = sqrt_lambda_depth * (d0);
    J_D[4] = sqrt_lambda_depth * (d1);
    J_D[5] = sqrt_lambda_depth * (d2 - 1.0f);

    // printf("D: (%d %d) -> (%f %f %f) -> (%d %d), (%f %f %f %f %f %f, %f,
    // %f)
    // "
    //        "I: (%f %f %f %f %f %f)\n",
    //        u_d, v_d, target_v[0], target_v[1], target_v[2], u_s, v_s,
    //        J_D[0], J_D[1], J_D[2], J_D[3], J_D[4], J_D[5], dDdx, dDdy,
    //        J_I[0], J_I[1], J_I[2],
    //            J_I[3], J_I[4], J_I[5]);
    r_D = sqrt_lambda_depth * diff_D;

    return true;
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
