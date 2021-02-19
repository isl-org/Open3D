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

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {
namespace odometry {

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void CreateVertexMapCUDA
#else
void CreateVertexMapCPU
#endif
        (const core::Tensor& depth_map,
         const core::Tensor& intrinsics,
         core::Tensor& vertex_map,
         float depth_scale,
         float depth_max) {

    t::geometry::kernel::NDArrayIndexer depth_indexer(depth_map, 2);
    t::geometry::kernel::TransformIndexer ti(intrinsics);

    // Output
    int64_t rows = depth_indexer.GetShape(0);
    int64_t cols = depth_indexer.GetShape(1);

    vertex_map = core::Tensor::Zeros({rows, cols, 3}, core::Dtype::Float32,
                                     depth_map.GetDevice());
    t::geometry::kernel::NDArrayIndexer vertex_indexer(vertex_map, 2);

    int64_t n = rows * cols;
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
#else
    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
#endif
                int64_t y = workload_idx / cols;
                int64_t x = workload_idx % cols;

                float d = *depth_indexer.GetDataPtrFromCoord<float>(x, y) /
                          depth_scale;
                if (d > 0 && d < depth_max) {
                    float* vertex =
                            vertex_indexer.GetDataPtrFromCoord<float>(x, y);
                    ti.Unproject(static_cast<float>(x), static_cast<float>(y),
                                 d, vertex + 0, vertex + 1, vertex + 2);
                }
            });
}

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void CreateNormalMapCUDA
#else
void CreateNormalMapCPU
#endif
        (const core::Tensor& vertex_map,
         core::Tensor& normal_map,
         float depth_scale,
         float depth_max,
         float depth_diff) {

    t::geometry::kernel::NDArrayIndexer vertex_indexer(vertex_map, 2);

    // Output
    int64_t rows = vertex_indexer.GetShape(0);
    int64_t cols = vertex_indexer.GetShape(1);

    normal_map =
            core::Tensor::Zeros(vertex_map.GetShape(), vertex_map.GetDtype(),
                                vertex_map.GetDevice());
    t::geometry::kernel::NDArrayIndexer normal_indexer(normal_map, 2);

    int64_t n = rows * cols;
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
#else
    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
#endif
                int64_t y = workload_idx / cols;
                int64_t x = workload_idx % cols;

                if (y < rows - 1 && x < cols - 1) {
                    float* v00 =
                            vertex_indexer.GetDataPtrFromCoord<float>(x, y);
                    float* v10 =
                            vertex_indexer.GetDataPtrFromCoord<float>(x + 1, y);
                    float* v01 =
                            vertex_indexer.GetDataPtrFromCoord<float>(x, y + 1);

                    float dx0 = v01[0] - v00[0];
                    float dy0 = v01[1] - v00[1];
                    float dz0 = v01[2] - v00[2];

                    float dx1 = v10[0] - v00[0];
                    float dy1 = v10[1] - v00[1];
                    float dz1 = v10[2] - v00[2];

                    float* normal =
                            normal_indexer.GetDataPtrFromCoord<float>(x, y);
                    normal[0] = dy0 * dz1 - dz0 * dy1;
                    normal[1] = dz0 * dx1 - dx0 * dz1;
                    normal[2] = dx0 * dy1 - dy0 * dx1;

                    float normal_norm =
                            sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                                 normal[2] * normal[2]);
                    if (normal_norm > 1e-5) {
                        normal[0] /= normal_norm;
                        normal[1] /= normal_norm;
                        normal[2] /= normal_norm;
                    }
                }
            });
}

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void ComputePosePointToPlaneCUDA
#else
void ComputePosePointToPlaneCPU
#endif
        (const core::Tensor& source_vertex_map,
         const core::Tensor& target_vertex_map,
         const core::Tensor& source_normal_map,
         const core::Tensor& intrinsics,
         const core::Tensor& init_source_to_target,
         core::Tensor& delta,
         core::Tensor& residual,
         float depth_diff) {

    t::geometry::kernel::NDArrayIndexer source_vertex_indexer(source_vertex_map,
                                                              2);
    t::geometry::kernel::NDArrayIndexer target_vertex_indexer(target_vertex_map,
                                                              2);
    t::geometry::kernel::NDArrayIndexer source_normal_indexer(source_normal_map,
                                                              2);

    t::geometry::kernel::TransformIndexer ti(intrinsics,
                                             init_source_to_target.Inverse());

    // Output
    int64_t rows = source_vertex_indexer.GetShape(0);
    int64_t cols = source_vertex_indexer.GetShape(1);

    core::Device device = source_vertex_map.GetDevice();
    core::Tensor AtA =
            core::Tensor::Zeros({6, 6}, core::Dtype::Float32, device);
    core::Tensor Atb = core::Tensor::Zeros({6}, core::Dtype::Float32, device);

    core::Tensor count = core::Tensor::Zeros({}, core::Dtype::Int32, device);
    residual = core::Tensor::Zeros({}, core::Dtype::Float32, device);

    float* AtA_local_ptr = static_cast<float*>(AtA.GetDataPtr());
    float* Atb_local_ptr = static_cast<float*>(Atb.GetDataPtr());
    float* residual_ptr = static_cast<float*>(residual.GetDataPtr());
    int* count_ptr = static_cast<int*>(count.GetDataPtr());

    int64_t n = rows * cols;
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
#else
    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
#endif
                int64_t y = workload_idx / cols;
                int64_t x = workload_idx % cols;

                float* dst_v =
                        target_vertex_indexer.GetDataPtrFromCoord<float>(x, y);

                float T_dst_v[3], u, v;
                ti.RigidTransform(dst_v[0], dst_v[1], dst_v[2], &T_dst_v[0],
                                  &T_dst_v[1], &T_dst_v[2]);

                ti.Project(T_dst_v[0], T_dst_v[1], T_dst_v[2], &u, &v);
                if (T_dst_v[2] < 0 || !source_vertex_indexer.InBoundary(u, v)) {
                    return;
                }

                float* src_v = source_vertex_indexer.GetDataPtrFromCoord<float>(
                        static_cast<int64_t>(u), static_cast<int64_t>(v));
                float* src_n = source_normal_indexer.GetDataPtrFromCoord<float>(
                        static_cast<int64_t>(u), static_cast<int64_t>(v));

                float r = (T_dst_v[0] - src_v[0]) * src_n[0] +
                          (T_dst_v[1] - src_v[1]) * src_n[1] +
                          (T_dst_v[2] - src_v[2]) * src_n[2];

                if (abs(r) > depth_diff) return;

                float J_ij[6];
                J_ij[0] = -T_dst_v[2] * src_n[1] + T_dst_v[1] * src_n[2];
                J_ij[1] = T_dst_v[2] * src_n[0] - T_dst_v[0] * src_n[2];
                J_ij[2] = -T_dst_v[1] * src_n[0] + T_dst_v[0] * src_n[1];
                J_ij[3] = src_n[0];
                J_ij[4] = src_n[1];
                J_ij[5] = src_n[2];
        // printf("(%ld %ld) -> (%f %f): residual = %f, J = (%f %f %f %f "
        //        "%f %f)\n",
        //        x, y, u, v, r, J_ij[0], J_ij[1], J_ij[2], J_ij[3],
        //        J_ij[4], J_ij[5]);

        // Not optimized; Switch to reduction if necessary.
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
                for (int i_local = 0; i_local < 6; ++i_local) {
                    for (int j_local = 0; j_local < 6; ++j_local) {
                        atomicAdd(&AtA_local_ptr[i_local * 6 + j_local],
                                  J_ij[i_local] * J_ij[j_local]);
                    }
                    atomicAdd(&Atb_local_ptr[i_local], J_ij[i_local] * r);
                }
                atomicAdd(residual_ptr, r * r);
                atomicAdd(count_ptr, 1);
#else
#pragma omp critical
                {
                    for (int i_local = 0; i_local < 6; ++i_local) {
                        for (int j_local = 0; j_local < 6; ++j_local) {
                            AtA_local_ptr[i_local * 6 + j_local] +=
                                    J_ij[i_local] * J_ij[j_local];
                        }
                        Atb_local_ptr[i_local] += J_ij[i_local] * r;
                    }
                    *residual_ptr += r * r;
                    *count_ptr += 1;
                }
#endif
            });

    utility::LogInfo("AtA = {}", AtA.ToString());
    utility::LogInfo("Atb = {}", Atb.ToString());
    utility::LogInfo("residual = {}", residual.ToString());
    utility::LogInfo("count = {}", count.ToString());

    delta = AtA.Solve(Atb.Neg());
}

}  // namespace odometry
}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
