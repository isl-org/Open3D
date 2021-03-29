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

void CreateVertexMapCPU(const core::Tensor& depth_map,
                        const core::Tensor& intrinsics,
                        core::Tensor& vertex_map,
                        float depth_scale,
                        float depth_max);

void CreateNormalMapCPU(const core::Tensor& vertex_map,
                        core::Tensor& normal_map);

void ComputePosePointToPlaneCPU(const core::Tensor& source_vertex_map,
                                const core::Tensor& target_vertex_map,
                                const core::Tensor& source_normal_map,
                                const core::Tensor& intrinsics,
                                const core::Tensor& init_source_to_target,
                                core::Tensor& delta,
                                core::Tensor& residual,
                                float depth_diff);

void ComputePoseDirectHybridCPU(const core::Tensor& source_depth,
                                const core::Tensor& target_depth,
                                const core::Tensor& source_intensity,
                                const core::Tensor& target_intensity,
                                const core::Tensor& source_depth_dx,
                                const core::Tensor& source_depth_dy,
                                const core::Tensor& source_intensity_dx,
                                const core::Tensor& source_intensity_dy,
                                const core::Tensor& target_vtx_map,
                                const core::Tensor& intrinsics,
                                const core::Tensor& init_source_to_target,
                                core::Tensor& delta,
                                core::Tensor& residual,
                                float depth_diff);
#ifdef BUILD_CUDA_MODULE

void CreateVertexMapCUDA(const core::Tensor& depth_map,
                         const core::Tensor& intrinsics,
                         core::Tensor& vertex_map,
                         float depth_scale,
                         float depth_max);

void CreateNormalMapCUDA(const core::Tensor& vertex_map,
                         core::Tensor& normal_map);

void ComputePosePointToPlaneCUDA(const core::Tensor& source_vertex_map,
                                 const core::Tensor& target_vertex_map,
                                 const core::Tensor& source_normal_map,
                                 const core::Tensor& intrinsics,
                                 const core::Tensor& init_source_to_target,
                                 core::Tensor& delta,
                                 core::Tensor& residual,
                                 float depth_diff);

void ComputePoseDirectHybridCUDA(const core::Tensor& source_depth,
                                 const core::Tensor& target_depth,
                                 const core::Tensor& source_intensity,
                                 const core::Tensor& target_intensity,
                                 const core::Tensor& source_depth_dx,
                                 const core::Tensor& source_depth_dy,
                                 const core::Tensor& source_intensity_dx,
                                 const core::Tensor& source_intensity_dy,
                                 const core::Tensor& target_vtx_map,
                                 const core::Tensor& intrinsics,
                                 const core::Tensor& init_source_to_target,
                                 core::Tensor& delta,
                                 core::Tensor& residual,
                                 float depth_diff);
#endif

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void PreprocessDepthCUDA
#else
void PreprocessDepthCPU
#endif
        (const core::Tensor& depth,
         core::Tensor& depth_processed,
         float depth_scale,
         float depth_max) {
    depth.AssertDtype(core::Dtype::Float32);

    t::geometry::kernel::NDArrayIndexer depth_in_indexer(depth, 2);

    depth_processed = core::Tensor::EmptyLike(depth);
    t::geometry::kernel::NDArrayIndexer depth_out_indexer(depth_processed, 2);

    // Output
    int64_t rows = depth_in_indexer.GetShape(0);
    int64_t cols = depth_in_indexer.GetShape(1);

    int64_t n = rows * cols;
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher::LaunchGeneralKernel(
#else
    core::kernel::CPULauncher::LaunchGeneralKernel(
#endif
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                int64_t y = workload_idx / cols;
                int64_t x = workload_idx % cols;

                float* d_in_ptr =
                        depth_in_indexer.GetDataPtrFromCoord<float>(x, y);
                float* d_out_ptr =
                        depth_out_indexer.GetDataPtrFromCoord<float>(x, y);

                float d = *d_in_ptr / depth_scale;
                bool valid = (d > 0 && d < depth_max);
                *d_out_ptr = valid ? *d_in_ptr : NAN;
            });
}

OPEN3D_HOST_DEVICE inline bool GetJacobianLocal(
        int64_t workload_idx,
        int64_t cols,
        float depth_diff,
        const t::geometry::kernel::NDArrayIndexer& source_vertex_indexer,
        const t::geometry::kernel::NDArrayIndexer& target_vertex_indexer,
        const t::geometry::kernel::NDArrayIndexer& source_normal_indexer,
        const t::geometry::kernel::TransformIndexer& ti,
        float* J_ij,
        float& r) {
    int64_t y = workload_idx / cols;
    int64_t x = workload_idx % cols;

    float* dst_v = target_vertex_indexer.GetDataPtrFromCoord<float>(x, y);
    if (dst_v[0] == NAN) {
        return false;
    }

    float T_dst_v[3], u, v;
    ti.RigidTransform(dst_v[0], dst_v[1], dst_v[2], &T_dst_v[0], &T_dst_v[1],
                      &T_dst_v[2]);
    ti.Project(T_dst_v[0], T_dst_v[1], T_dst_v[2], &u, &v);
    u = round(u);
    v = round(v);

    if (T_dst_v[2] < 0 || !source_vertex_indexer.InBoundary(u, v)) {
        return false;
    }

    int64_t ui = static_cast<int64_t>(u);
    int64_t vi = static_cast<int64_t>(v);
    float* src_v = source_vertex_indexer.GetDataPtrFromCoord<float>(ui, vi);
    float* src_n = source_normal_indexer.GetDataPtrFromCoord<float>(ui, vi);
    if (src_v[0] == NAN || src_n[0] == NAN) {
        return false;
    }

    r = (T_dst_v[0] - src_v[0]) * src_n[0] +
        (T_dst_v[1] - src_v[1]) * src_n[1] + (T_dst_v[2] - src_v[2]) * src_n[2];
    if (abs(r) > depth_diff) {
        return false;
    }

    J_ij[0] = -T_dst_v[2] * src_n[1] + T_dst_v[1] * src_n[2];
    J_ij[1] = T_dst_v[2] * src_n[0] - T_dst_v[0] * src_n[2];
    J_ij[2] = -T_dst_v[1] * src_n[0] + T_dst_v[0] * src_n[1];
    J_ij[3] = src_n[0];
    J_ij[4] = src_n[1];
    J_ij[5] = src_n[2];

    return true;
}

OPEN3D_HOST_DEVICE inline bool GetJacobianDirectHybridLocal(
        int64_t workload_idx,
        int64_t cols,
        float depth_diff,
        const t::geometry::kernel::NDArrayIndexer& src_depth_indexer,
        const t::geometry::kernel::NDArrayIndexer& dst_depth_indexer,
        const t::geometry::kernel::NDArrayIndexer& src_intensity_indexer,
        const t::geometry::kernel::NDArrayIndexer& dst_intensity_indexer,
        const t::geometry::kernel::NDArrayIndexer& src_depth_dx_indexer,
        const t::geometry::kernel::NDArrayIndexer& src_depth_dy_indexer,
        const t::geometry::kernel::NDArrayIndexer& src_intensity_dx_indexer,
        const t::geometry::kernel::NDArrayIndexer& src_intensity_dy_indexer,
        const t::geometry::kernel::NDArrayIndexer& dst_vertex_indexer,
        const t::geometry::kernel::TransformIndexer& ti,
        float* J_I,
        float* J_D,
        float& r_I,
        float& r_D) {
    // sqrt 0.5, according to http://redwood-data.org/indoor_lidar_rgbd/supp.pdf
    const float sqrt_lambda_img = 0.707;
    const float sqrt_lambda_dep = 0.707;
    const float sobel_scale = 0.125;

    int v_d = workload_idx / cols;
    int u_d = workload_idx % cols;

    float* dst_v = dst_vertex_indexer.GetDataPtrFromCoord<float>(u_d, v_d);
    if (__ISNAN(dst_v[0])) {
        return false;
    }

    // dst on source: T_dst_v
    float T_dst_v[3], u_sf, v_sf;
    ti.RigidTransform(dst_v[0], dst_v[1], dst_v[2], &T_dst_v[0], &T_dst_v[1],
                      &T_dst_v[2]);
    ti.Project(T_dst_v[0], T_dst_v[1], T_dst_v[2], &u_sf, &v_sf);
    int u_s = int(round(u_sf));
    int v_s = int(round(v_sf));

    if (T_dst_v[2] < 0 || !src_depth_indexer.InBoundary(u_s, v_s)) {
        return false;
    }

    // TODO: customize filter depth grads

    const double fx = ti.fx_;
    const double fy = ti.fy_;

    // TODO: depth scale
    float depth_s =
            *src_depth_indexer.GetDataPtrFromCoord<float>(u_s, v_s) / 1000.0;
    if (__ISNAN(depth_s)) {
        return false;
    }

    float diff_D = depth_s - T_dst_v[2];
    if (abs(diff_D) > depth_diff) {
        return false;
    }
    float dDdx = sobel_scale *
                 (*src_depth_dx_indexer.GetDataPtrFromCoord<float>(u_s, v_s)) /
                 1000.0;
    float dDdy = sobel_scale *
                 (*src_depth_dy_indexer.GetDataPtrFromCoord<float>(u_s, v_s)) /
                 1000.0;
    if (__ISNAN(dDdx) || __ISNAN(dDdy)) {
        return false;
    }

    float diff_I = *src_intensity_indexer.GetDataPtrFromCoord<float>(u_s, v_s) -
                   *dst_intensity_indexer.GetDataPtrFromCoord<float>(u_d, v_d);
    float dIdx =
            sobel_scale *
            (*src_intensity_dx_indexer.GetDataPtrFromCoord<float>(u_s, v_s));
    float dIdy =
            sobel_scale *
            (*src_intensity_dy_indexer.GetDataPtrFromCoord<float>(u_s, v_s));

    // printf("%ld: (%d %d %f) -> (%f %f %f) -> (%d %d) -> depth diff: %f, color
    // "
    //        "diff: %f\n",
    //        workload_idx, u_d, v_d,
    //        *dst_depth_indexer.GetDataPtrFromCoord<float>(u_d, v_d), dst_v[0],
    //        dst_v[1], dst_v[2], u_s, v_s, depth_s - T_dst_v[2],
    //        *src_intensity_indexer.GetDataPtrFromCoord<float>(u_s, v_s) -
    //                *dst_intensity_indexer.GetDataPtrFromCoord<float>(u_d,
    //                v_d));

    float invz = 1 / T_dst_v[2];
    float c0 = dIdx * fx * invz;
    float c1 = dIdy * fy * invz;
    float c2 = -(c0 * T_dst_v[0] + c1 * T_dst_v[1]) * invz;
    float d0 = dDdx * fx * invz;
    float d1 = dDdy * fy * invz;
    float d2 = -(d0 * T_dst_v[0] + d1 * T_dst_v[1]) * invz;

    J_I[0] = sqrt_lambda_img * (-T_dst_v[2] * c1 + T_dst_v[1] * c2);
    J_I[1] = sqrt_lambda_img * (T_dst_v[2] * c0 - T_dst_v[0] * c2);
    J_I[2] = sqrt_lambda_img * (-T_dst_v[1] * c0 + T_dst_v[0] * c1);
    J_I[3] = sqrt_lambda_img * (c0);
    J_I[4] = sqrt_lambda_img * (c1);
    J_I[5] = sqrt_lambda_img * (c2);
    r_I = sqrt_lambda_img * diff_I;

    J_D[0] = sqrt_lambda_dep *
             ((-T_dst_v[2] * d1 + T_dst_v[1] * d2) - T_dst_v[1]);
    J_D[1] = sqrt_lambda_dep *
             ((T_dst_v[2] * d0 - T_dst_v[0] * d2) + T_dst_v[0]);
    J_D[2] = sqrt_lambda_dep * ((-T_dst_v[1] * d0 + T_dst_v[0] * d1));
    J_D[3] = sqrt_lambda_dep * (d0);
    J_D[4] = sqrt_lambda_dep * (d1);
    J_D[5] = sqrt_lambda_dep * (d2 - 1.0f);

    // printf("D: (%d %d) -> (%f %f %f) -> (%d %d), (%f %f %f %f %f %f, %f, %f)
    // "
    //        "I: (%f %f %f %f %f %f)\n",
    //        u_d, v_d, dst_v[0], dst_v[1], dst_v[2], u_s, v_s, J_D[0], J_D[1],
    //        J_D[2], J_D[3], J_D[4], J_D[5], dDdx, dDdy, J_I[0], J_I[1],
    //        J_I[2],
    //            J_I[3], J_I[4], J_I[5]);
    r_D = sqrt_lambda_dep * diff_D;

    return true;
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
