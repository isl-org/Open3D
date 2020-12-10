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

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CPULauncher.h"
#include "open3d/core/kernel/GeneralEW.h"
#include "open3d/core/kernel/GeneralEWMacros.h"
#include "open3d/core/kernel/GeneralIndexer.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {
namespace kernel {

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void CUDAUnprojectKernel
#else
void CPUUnprojectKernel
#endif
        (const std::unordered_map<std::string, Tensor>& srcs,
         std::unordered_map<std::string, Tensor>& dsts) {
    static std::vector<std::string> src_attrs = {
            "depth", "intrinsics", "depth_scale", "depth_max", "stride",
    };
    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[UnprojectKernel] expected Tensor {} in srcs, but "
                    "did not receive",
                    k);
        }
    }

    // Input
    Tensor depth = srcs.at("depth").To(core::Dtype::Float32);
    Tensor intrinsics = srcs.at("intrinsics").To(core::Dtype::Float32);
    float depth_scale = srcs.at("depth_scale").Item<float>();
    float depth_max = srcs.at("depth_max").Item<float>();
    int64_t stride = srcs.at("stride").Item<int64_t>();

    NDArrayIndexer depth_indexer(depth, 2);
    TransformIndexer ti(intrinsics);

    // Output
    int64_t rows_strided = depth_indexer.GetShape(0) / stride;
    int64_t cols_strided = depth_indexer.GetShape(1) / stride;

    Tensor points({rows_strided * cols_strided, 3}, core::Dtype::Float32,
                  depth.GetDevice());
    NDArrayIndexer point_indexer(points, 1);

    // Counter
    Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32,
                 depth.GetDevice());
    int* count_ptr = static_cast<int*>(count.GetDataPtr());

    // Workload
    int64_t n = rows_strided * cols_strided;

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDALauncher::LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                 int64_t workload_idx) {
#else
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
#endif
        int64_t y = (workload_idx / cols_strided) * stride;
        int64_t x = (workload_idx % cols_strided) * stride;

        float d =
                *static_cast<float*>(depth_indexer.GetDataPtrFromCoord(x, y)) /
                depth_scale;
        if (d > 0 && d < depth_max) {

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
            int idx = atomicAdd(count_ptr, 1);
#else
            int idx;
#pragma omp atomic capture
            {
                idx = *count_ptr;
                *count_ptr += 1;
            }
#endif
            float* vertex =
                    static_cast<float*>(point_indexer.GetDataPtrFromCoord(idx));
            ti.Unproject(static_cast<float>(x), static_cast<float>(y), d,
                         vertex + 0, vertex + 1, vertex + 2);
        }
    });

    int total_pts_count = count.Item<int>();
    dsts.emplace("points", points.Slice(0, 0, total_pts_count));
}

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void CUDATSDFIntegrateKernel
#else
void CPUTSDFIntegrateKernel
#endif
        (const std::unordered_map<std::string, Tensor>& srcs,
         std::unordered_map<std::string, Tensor>& dsts) {
    // Decode input tensors
    static std::vector<std::string> src_attrs = {
            "depth",       "indices",    "block_keys", "intrinsics",
            "extrinsics",  "resolution", "voxel_size", "sdf_trunc",
            "depth_scale", "depth_max",
    };
    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[TSDFIntegrateKernel] expected Tensor {} in srcs, but "
                    "did not receive",
                    k);
        }
    }

    Tensor depth = srcs.at("depth").To(core::Dtype::Float32);
    Tensor color = srcs.at("color").To(core::Dtype::Float32);
    Tensor indices = srcs.at("indices");
    Tensor block_keys = srcs.at("block_keys");
    Tensor block_values = dsts.at("block_values");

    // Transforms
    Tensor intrinsics = srcs.at("intrinsics").To(core::Dtype::Float32);
    Tensor extrinsics = srcs.at("extrinsics").To(core::Dtype::Float32);

    // Parameters
    int64_t resolution = srcs.at("resolution").Item<int64_t>();
    int64_t resolution3 = resolution * resolution * resolution;

    float voxel_size = srcs.at("voxel_size").Item<float>();
    float sdf_trunc = srcs.at("sdf_trunc").Item<float>();
    float depth_scale = srcs.at("depth_scale").Item<float>();
    float depth_max = srcs.at("depth_max").Item<float>();

    // Shape / transform indexers, no data involved
    NDArrayIndexer voxel_indexer({resolution, resolution, resolution});
    TransformIndexer transform_indexer(intrinsics, extrinsics, voxel_size);

    // Real data indexer
    NDArrayIndexer depth_indexer(depth, 2);
    NDArrayIndexer color_indexer(color, 2);
    NDArrayIndexer block_keys_indexer(block_keys, 1);
    NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);

    // Plain arrays that does not require indexers
    int64_t* indices_ptr = static_cast<int64_t*>(indices.GetDataPtr());

    int64_t n = indices.GetShape()[0] * resolution3;

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDALauncher::LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                 int64_t workload_idx) {
#else
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
#endif
        // Natural index (0, N) -> (block_idx, voxel_idx)
        int64_t block_idx = indices_ptr[workload_idx / resolution3];
        int64_t voxel_idx = workload_idx % resolution3;

        /// Coordinate transform
        // block_idx -> (x_block, y_block, z_block)
        int* block_key_ptr = static_cast<int*>(
                block_keys_indexer.GetDataPtrFromCoord(block_idx));
        int64_t xb = static_cast<int64_t>(block_key_ptr[0]);
        int64_t yb = static_cast<int64_t>(block_key_ptr[1]);
        int64_t zb = static_cast<int64_t>(block_key_ptr[2]);

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        int64_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        // coordinate in world (in voxel)
        int64_t x = (xb * resolution + xv);
        int64_t y = (yb * resolution + yv);
        int64_t z = (zb * resolution + zv);

        // coordinate in camera (in voxel -> in meter)
        float xc, yc, zc, u, v;
        transform_indexer.RigidTransform(static_cast<float>(x),
                                         static_cast<float>(y),
                                         static_cast<float>(z), &xc, &yc, &zc);

        // coordinate in image (in pixel)
        transform_indexer.Project(xc, yc, zc, &u, &v);
        if (!depth_indexer.InBoundary(u, v)) {
            return;
        }

        /// Associate image workload and compute SDF

        float depth =
                *static_cast<const float*>(depth_indexer.GetDataPtrFromCoord(
                        static_cast<int64_t>(u), static_cast<int64_t>(v))) /
                depth_scale;

        float* color_ptr =
                static_cast<float*>(color_indexer.GetDataPtrFromCoord(
                        static_cast<int64_t>(u), static_cast<int64_t>(v)));

        // Compute multiplier
        float xc_unproj, yc_unproj, zc_unproj;
        transform_indexer.Unproject(static_cast<float>(u),
                                    static_cast<float>(v), 1.0, &xc_unproj,
                                    &yc_unproj, &zc_unproj);
        float multiplier =
                sqrt(xc_unproj * xc_unproj + yc_unproj * yc_unproj + 1.0);
        float sdf = (depth - zc) * multiplier;
        if (depth <= 0 || depth > depth_max || zc <= 0 || sdf < -sdf_trunc) {
            return;
        }
        sdf = sdf < sdf_trunc ? sdf : sdf_trunc;
        sdf /= sdf_trunc;

        /// Associate voxel workload and update TSDF/Weights
        float* voxel_ptr = static_cast<float*>(
                voxel_block_buffer_indexer.GetDataPtrFromCoord(xv, yv, zv,
                                                               block_idx));

        float tsdf_sum = voxel_ptr[0];
        float weight_sum = voxel_ptr[1];
        float r_sum = voxel_ptr[2];
        float g_sum = voxel_ptr[3];
        float b_sum = voxel_ptr[4];

        float new_weight_sum = weight_sum + 1;
        voxel_ptr[0] = (weight_sum * tsdf_sum + sdf) / new_weight_sum;
        voxel_ptr[1] = new_weight_sum;
        voxel_ptr[2] = (weight_sum * r_sum + color_ptr[0]) / new_weight_sum;
        voxel_ptr[3] = (weight_sum * g_sum + color_ptr[1]) / new_weight_sum;
        voxel_ptr[4] = (weight_sum * b_sum + color_ptr[2]) / new_weight_sum;
    });
}

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void CUDASurfaceExtractionKernel
#else
void CPUSurfaceExtractionKernel
#endif
        (const std::unordered_map<std::string, Tensor>& srcs,
         std::unordered_map<std::string, Tensor>& dsts) {
    // Decode input tensors
    static std::vector<std::string> src_attrs = {
            "indices",      "nb_indices", "nb_masks",   "block_keys",
            "block_values", "voxel_size", "resolution",
    };
    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[TSDFSurfaceExtractionKernel] expected Tensor {} in "
                    "srcs, but did not receive",
                    k);
        }
    }
    utility::LogInfo("surface extraction starts");

    Tensor indices = srcs.at("indices");
    Tensor nb_indices = srcs.at("nb_indices");
    Tensor nb_masks = srcs.at("nb_masks");
    Tensor block_keys = srcs.at("block_keys");
    Tensor block_values = srcs.at("block_values");

    // Parameters
    int64_t resolution = srcs.at("resolution").Item<int64_t>();
    int64_t resolution3 = resolution * resolution * resolution;

    float voxel_size = srcs.at("voxel_size").Item<float>();

    // Shape / transform indexers, no data involved
    NDArrayIndexer voxel_indexer({resolution, resolution, resolution});

    // Real data indexer
    NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);
    NDArrayIndexer block_keys_indexer(block_keys, 1);

    // Plain arrays that does not require indexers
    int64_t* nb_indices_ptr = static_cast<int64_t*>(nb_indices.GetDataPtr());
    bool* nb_masks_ptr = static_cast<bool*>(nb_masks.GetDataPtr());
    int64_t* indices_ptr = static_cast<int64_t*>(indices.GetDataPtr());

    int n_blocks = indices.GetShape()[0];
    int64_t n = n_blocks * resolution3;

    // Output
    core::Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32,
                       block_values.GetDevice());
    int* count_ptr = static_cast<int*>(count.GetDataPtr());
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDALauncher::LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                 int64_t workload_idx) {
#else
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
#endif
        auto GetVoxelAt = [&] OPEN3D_DEVICE(int xo, int yo, int zo,
                                            int curr_block_idx) -> float* {
            int xn = (xo + resolution) % resolution;
            int yn = (yo + resolution) % resolution;
            int zn = (zo + resolution) % resolution;

            int64_t dxb = sign(xo - xn);
            int64_t dyb = sign(yo - yn);
            int64_t dzb = sign(zo - zn);

            int64_t nb_idx = (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

            bool block_mask_i =
                    nb_masks_ptr[nb_idx * n_blocks + curr_block_idx];
            if (!block_mask_i) return nullptr;

            int64_t block_idx_i =
                    nb_indices_ptr[nb_idx * n_blocks + curr_block_idx];
            return static_cast<float*>(
                    voxel_block_buffer_indexer.GetDataPtrFromCoord(
                            xn, yn, zn, block_idx_i));
        };

        // Natural index (0, N) -> (block_idx, voxel_idx)
        int64_t workload_block_idx = workload_idx / resolution3;
        int64_t block_idx = indices_ptr[workload_block_idx];
        int64_t voxel_idx = workload_idx % resolution3;

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        int64_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        float* voxel_ptr = static_cast<float*>(
                voxel_block_buffer_indexer.GetDataPtrFromCoord(xv, yv, zv,
                                                               block_idx));
        float tsdf_o = voxel_ptr[0];
        float weight_o = voxel_ptr[1];
        if (weight_o == 0) return;

        for (int i = 0; i < 3; ++i) {
            float* ptr = GetVoxelAt(xv + int64_t(i == 0), yv + int64_t(i == 1),
                                    zv + int64_t(i == 2), workload_block_idx);
            if (ptr == nullptr) continue;

            float tsdf_i = ptr[0];
            float weight_i = ptr[1];

            if (weight_i > 0 && tsdf_i * tsdf_o < 0) {

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
                atomicAdd(count_ptr, 1);
#else
#pragma omp atomic
                *count_ptr += 1;

#endif
            }
        }
    });

    int total_count = count.Item<int>();

    core::Tensor points({total_count, 3}, core::Dtype::Float32,
                        block_values.GetDevice());
    core::Tensor normals({total_count, 3}, core::Dtype::Float32,
                         block_values.GetDevice());
    core::Tensor colors({total_count, 3}, core::Dtype::Float32,
                        block_values.GetDevice());
    NDArrayIndexer point_indexer(points, 1);
    NDArrayIndexer normal_indexer(normals, 1);
    NDArrayIndexer color_indexer(colors, 1);

    // Reset count
    count = core::Tensor(std::vector<int>{0}, {}, core::Dtype::Int32,
                         block_values.GetDevice());
    count_ptr = static_cast<int*>(count.GetDataPtr());

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDALauncher::LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                 int64_t workload_idx) {
#else
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
#endif
        auto GetVoxelAt = [&] OPEN3D_DEVICE(int xo, int yo, int zo,
                                            int curr_block_idx) -> float* {
            int xn = (xo + resolution) % resolution;
            int yn = (yo + resolution) % resolution;
            int zn = (zo + resolution) % resolution;

            int64_t dxb = sign(xo - xn);
            int64_t dyb = sign(yo - yn);
            int64_t dzb = sign(zo - zn);

            int64_t nb_idx = (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

            bool block_mask_i =
                    nb_masks_ptr[nb_idx * n_blocks + curr_block_idx];
            if (!block_mask_i) return nullptr;

            int64_t block_idx_i =
                    nb_indices_ptr[nb_idx * n_blocks + curr_block_idx];
            return static_cast<float*>(
                    voxel_block_buffer_indexer.GetDataPtrFromCoord(
                            xn, yn, zn, block_idx_i));
        };

        auto GetNormalAt = [&] OPEN3D_DEVICE(int xo, int yo, int zo,
                                             int curr_block_idx, float* n) {
            float* vxp = GetVoxelAt(xo + 1, yo, zo, curr_block_idx);
            float* vxn = GetVoxelAt(xo - 1, yo, zo, curr_block_idx);
            float* vyp = GetVoxelAt(xo, yo + 1, zo, curr_block_idx);
            float* vyn = GetVoxelAt(xo, yo - 1, zo, curr_block_idx);
            float* vzp = GetVoxelAt(xo, yo, zo + 1, curr_block_idx);
            float* vzn = GetVoxelAt(xo, yo, zo - 1, curr_block_idx);
            if (vxp && vxn) n[0] = (vxp[0] - vxn[0]) / (2 * voxel_size);
            if (vyp && vyn) n[1] = (vyp[0] - vyn[0]) / (2 * voxel_size);
            if (vzp && vzn) n[2] = (vzp[0] - vzn[0]) / (2 * voxel_size);
        };

        // Natural index (0, N) -> (block_idx, voxel_idx)
        int64_t workload_block_idx = workload_idx / resolution3;
        int64_t block_idx = indices_ptr[workload_block_idx];
        int64_t voxel_idx = workload_idx % resolution3;

        /// Coordinate transform
        // block_idx -> (x_block, y_block, z_block)
        int* block_key_ptr = static_cast<int*>(
                block_keys_indexer.GetDataPtrFromCoord(block_idx));
        int64_t xb = static_cast<int64_t>(block_key_ptr[0]);
        int64_t yb = static_cast<int64_t>(block_key_ptr[1]);
        int64_t zb = static_cast<int64_t>(block_key_ptr[2]);

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        int64_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        float* voxel_ptr = static_cast<float*>(
                voxel_block_buffer_indexer.GetDataPtrFromCoord(xv, yv, zv,
                                                               block_idx));
        float tsdf_o = voxel_ptr[0];
        float weight_o = voxel_ptr[1];
        float r_o = voxel_ptr[2];
        float g_o = voxel_ptr[3];
        float b_o = voxel_ptr[4];
        if (weight_o == 0) return;

        int64_t x = xb * resolution + xv;
        int64_t y = yb * resolution + yv;
        int64_t z = zb * resolution + zv;

        float no[3] = {0}, ni[3] = {0};
        GetNormalAt(xv, yv, zv, workload_block_idx, no);
        for (int i = 0; i < 3; ++i) {
            float* ptr = GetVoxelAt(xv + int64_t(i == 0), yv + int64_t(i == 1),
                                    zv + int64_t(i == 2), workload_block_idx);
            if (ptr == nullptr) continue;

            float tsdf_i = ptr[0];
            float weight_i = ptr[1];

            if (weight_i > 0 && tsdf_i * tsdf_o < 0) {
                float ratio = (0 - tsdf_o) / (tsdf_i - tsdf_o);

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
                int idx = atomicAdd(count_ptr, 1);
#else
                int idx;
#pragma omp atomic capture
                {
                    idx = *count_ptr;
                    *count_ptr += 1;
                }
#endif

                float* point_ptr = static_cast<float*>(
                        point_indexer.GetDataPtrFromCoord(idx));
                point_ptr[0] = voxel_size * (x + ratio * int(i == 0));
                point_ptr[1] = voxel_size * (y + ratio * int(i == 1));
                point_ptr[2] = voxel_size * (z + ratio * int(i == 2));
                GetNormalAt(xv + int64_t(i == 0), yv + int64_t(i == 1),
                            zv + int64_t(i == 2), workload_block_idx, ni);

                float* normal_ptr = static_cast<float*>(
                        normal_indexer.GetDataPtrFromCoord(idx));
                float nx = (1 - ratio) * no[0] + ratio * ni[0];
                float ny = (1 - ratio) * no[1] + ratio * ni[1];
                float nz = (1 - ratio) * no[2] + ratio * ni[2];
                float norm = sqrt(nx * nx + ny * ny + nz * nz) + 1e-5;
                normal_ptr[0] = nx / norm;
                normal_ptr[1] = ny / norm;
                normal_ptr[2] = nz / norm;

                float* color_ptr = static_cast<float*>(
                        color_indexer.GetDataPtrFromCoord(idx));
                float r_i = ptr[2];
                float g_i = ptr[3];
                float b_i = ptr[4];
                color_ptr[0] = ((1 - ratio) * r_o + ratio * r_i) / 255.0f;
                color_ptr[1] = ((1 - ratio) * g_o + ratio * g_i) / 255.0f;
                color_ptr[2] = ((1 - ratio) * b_o + ratio * b_i) / 255.0f;
            }
        }
    });

    dsts.emplace("points", points);
    dsts.emplace("normals", normals);
    dsts.emplace("colors", colors);
}

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void CUDAMarchingCubesKernel
#else
void CPUMarchingCubesKernel
#endif
        (const std::unordered_map<std::string, Tensor>& srcs,
         std::unordered_map<std::string, Tensor>& dsts) {
    // Decode input tensors
    static std::vector<std::string> src_attrs = {
            "indices",    "inv_indices",  "nb_indices", "nb_masks",
            "block_keys", "block_values", "voxel_size", "resolution",
    };
    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[CUDAMarchingCubesKernel] expected Tensor {} in "
                    "srcs, but "
                    "did not receive",
                    k);
        }
    }

    Tensor indices = srcs.at("indices");
    Tensor inv_indices = srcs.at("inv_indices");
    Tensor nb_indices = srcs.at("nb_indices");
    Tensor nb_masks = srcs.at("nb_masks");
    Tensor block_keys = srcs.at("block_keys");
    Tensor block_values = srcs.at("block_values");

    // Parameters
    int64_t resolution = srcs.at("resolution").Item<int64_t>();
    int64_t resolution3 = resolution * resolution * resolution;

    float voxel_size = srcs.at("voxel_size").Item<float>();

    // Shape / transform indexers, no data involved
    NDArrayIndexer voxel_indexer({resolution, resolution, resolution});

    // Output
    int n_blocks = indices.GetShape()[0];
    core::Tensor mesh_structure = core::Tensor::Zeros(
            {n_blocks, resolution, resolution, resolution, 4},
            core::Dtype::Int32, block_keys.GetDevice());

    // Real data indexer
    NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);
    NDArrayIndexer mesh_structure_indexer(mesh_structure, 4);

    // Plain arrays that does not require indexers
    int64_t* nb_indices_ptr = static_cast<int64_t*>(nb_indices.GetDataPtr());
    bool* nb_masks_ptr = static_cast<bool*>(nb_masks.GetDataPtr());
    int64_t* indices_ptr = static_cast<int64_t*>(indices.GetDataPtr());
    int64_t* inv_indices_ptr = static_cast<int64_t*>(inv_indices.GetDataPtr());

    int64_t n = n_blocks * resolution3;

    // Pass 0: analyze mesh structure, set up one-on-one correspondences
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDALauncher::LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                 int64_t workload_idx) {
#else
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
#endif
        // Natural index (0, N) -> (block_idx, voxel_idx)
        int64_t workload_block_idx = workload_idx / resolution3;
        int64_t voxel_idx = workload_idx % resolution3;

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        int64_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        // Check per-vertex sign in the cube to determine cube type
        int table_idx = 0;
        for (int i = 0; i < 8; ++i) {
            int64_t xv_i = xv + vtx_shifts[i][0];
            int64_t yv_i = yv + vtx_shifts[i][1];
            int64_t zv_i = zv + vtx_shifts[i][2];

            int64_t dxb = xv_i / resolution;
            int64_t dyb = yv_i / resolution;
            int64_t dzb = zv_i / resolution;

            int64_t nb_idx = (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

            bool block_mask_i =
                    nb_masks_ptr[nb_idx * n_blocks + workload_block_idx];
            if (!block_mask_i) return;

            int64_t block_idx_i =
                    nb_indices_ptr[nb_idx * n_blocks + workload_block_idx];
            float* voxel_ptr_i = static_cast<float*>(
                    voxel_block_buffer_indexer.GetDataPtrFromCoord(
                            xv_i - dxb * resolution, yv_i - dyb * resolution,
                            zv_i - dzb * resolution, block_idx_i));

            float tsdf_i = voxel_ptr_i[0];
            float weight_i = voxel_ptr_i[1];
            if (weight_i == 0) return;

            table_idx |= ((tsdf_i < 0) ? (1 << i) : 0);
        }

        int* mesh_struct_ptr =
                static_cast<int*>(mesh_structure_indexer.GetDataPtrFromCoord(
                        xv, yv, zv, workload_block_idx));
        mesh_struct_ptr[3] = table_idx;

        if (table_idx == 0 || table_idx == 255) return;

        // Check per-edge sign in the cube to determine cube type
        int edges_with_vertices = edge_table[table_idx];
        for (int i = 0; i < 12; ++i) {
            if (edges_with_vertices & (1 << i)) {
                int64_t xv_i = xv + edge_shifts[i][0];
                int64_t yv_i = yv + edge_shifts[i][1];
                int64_t zv_i = zv + edge_shifts[i][2];
                int edge_i = edge_shifts[i][3];

                int dxb = xv_i / resolution;
                int dyb = yv_i / resolution;
                int dzb = zv_i / resolution;

                int nb_idx = (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

                int64_t block_idx_i =
                        nb_indices_ptr[nb_idx * n_blocks + workload_block_idx];
                int* mesh_struct_ptr_i = static_cast<int*>(
                        mesh_structure_indexer.GetDataPtrFromCoord(
                                xv_i - dxb * resolution,
                                yv_i - dyb * resolution,
                                zv_i - dzb * resolution,
                                inv_indices_ptr[block_idx_i]));

                // Non-atomic write, but we are safe
                mesh_struct_ptr_i[edge_i] = -1;
            }
        }
    });

    // Pass 1: allocate and assign vertices with normals
    core::Tensor vtx_count(std::vector<int>{0}, {}, core::Dtype::Int32,
                           block_values.GetDevice());
    int* vtx_count_ptr = static_cast<int*>(vtx_count.GetDataPtr());
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
#else
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
#endif
                // Natural index (0, N) -> (block_idx, voxel_idx)
                int64_t workload_block_idx = workload_idx / resolution3;
                int64_t voxel_idx = workload_idx % resolution3;

                // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                int64_t xv, yv, zv;
                voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                // Obtain voxel's mesh struct ptr
                int* mesh_struct_ptr = static_cast<int*>(
                        mesh_structure_indexer.GetDataPtrFromCoord(
                                xv, yv, zv, workload_block_idx));

                // Early quit -- no allocated vertex to compute
                if (mesh_struct_ptr[0] != -1 && mesh_struct_ptr[1] != -1 &&
                    mesh_struct_ptr[2] != -1) {
                    return;
                }

                // Enumerate 3 edges in the voxel
                for (int e = 0; e < 3; ++e) {
                    int vertex_idx = mesh_struct_ptr[e];
                    if (vertex_idx != -1) continue;

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
                    atomicAdd(vtx_count_ptr, 1);
#else
#pragma omp atomic
            *vtx_count_ptr += 1;
#endif
                }
            });

    int total_vtx_count = vtx_count.Item<int>();
    utility::LogInfo("Total vertex count = {}", total_vtx_count);

    // Reset counter
    vtx_count = core::Tensor(std::vector<int>{0}, {}, core::Dtype::Int32,
                             block_values.GetDevice());
    vtx_count_ptr = static_cast<int*>(vtx_count.GetDataPtr());

    core::Tensor vertices({total_vtx_count, 3}, core::Dtype::Float32,
                          block_values.GetDevice());
    core::Tensor normals({total_vtx_count, 3}, core::Dtype::Float32,
                         block_values.GetDevice());
    core::Tensor colors({total_vtx_count, 3}, core::Dtype::Float32,
                        block_values.GetDevice());

    NDArrayIndexer block_keys_indexer(block_keys, 1);
    NDArrayIndexer vertex_indexer(vertices, 1);
    NDArrayIndexer normal_indexer(normals, 1);
    NDArrayIndexer color_indexer(colors, 1);

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDALauncher::LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                 int64_t workload_idx) {
#else
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
#endif
        auto GetVoxelAt = [&] OPEN3D_DEVICE(int xo, int yo, int zo,
                                            int curr_block_idx) -> float* {
            int xn = (xo + resolution) % resolution;
            int yn = (yo + resolution) % resolution;
            int zn = (zo + resolution) % resolution;

            int64_t dxb = sign(xo - xn);
            int64_t dyb = sign(yo - yn);
            int64_t dzb = sign(zo - zn);

            int64_t nb_idx = (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

            bool block_mask_i =
                    nb_masks_ptr[nb_idx * n_blocks + curr_block_idx];
            if (!block_mask_i) return nullptr;

            int64_t block_idx_i =
                    nb_indices_ptr[nb_idx * n_blocks + curr_block_idx];
            return static_cast<float*>(
                    voxel_block_buffer_indexer.GetDataPtrFromCoord(
                            xn, yn, zn, block_idx_i));
        };

        auto GetNormalAt = [&] OPEN3D_DEVICE(int xo, int yo, int zo,
                                             int curr_block_idx, float* n) {
            float* vxp = GetVoxelAt(xo + 1, yo, zo, curr_block_idx);
            float* vxn = GetVoxelAt(xo - 1, yo, zo, curr_block_idx);
            float* vyp = GetVoxelAt(xo, yo + 1, zo, curr_block_idx);
            float* vyn = GetVoxelAt(xo, yo - 1, zo, curr_block_idx);
            float* vzp = GetVoxelAt(xo, yo, zo + 1, curr_block_idx);
            float* vzn = GetVoxelAt(xo, yo, zo - 1, curr_block_idx);
            if (vxp && vxn) n[0] = (vxp[0] - vxn[0]) / (2 * voxel_size);
            if (vyp && vyn) n[1] = (vyp[0] - vyn[0]) / (2 * voxel_size);
            if (vzp && vzn) n[2] = (vzp[0] - vzn[0]) / (2 * voxel_size);
        };

        // Natural index (0, N) -> (block_idx, voxel_idx)
        int64_t workload_block_idx = workload_idx / resolution3;
        int64_t block_idx = indices_ptr[workload_block_idx];
        int64_t voxel_idx = workload_idx % resolution3;

        // block_idx -> (x_block, y_block, z_block)
        int* block_key_ptr = static_cast<int*>(
                block_keys_indexer.GetDataPtrFromCoord(block_idx));
        int64_t xb = static_cast<int64_t>(block_key_ptr[0]);
        int64_t yb = static_cast<int64_t>(block_key_ptr[1]);
        int64_t zb = static_cast<int64_t>(block_key_ptr[2]);

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        int64_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        // global coordinate (in voxels)
        int64_t x = xb * resolution + xv;
        int64_t y = yb * resolution + yv;
        int64_t z = zb * resolution + zv;

        // Obtain voxel's mesh struct ptr
        int* mesh_struct_ptr =
                static_cast<int*>(mesh_structure_indexer.GetDataPtrFromCoord(
                        xv, yv, zv, workload_block_idx));

        // Early quit -- no allocated vertex to compute
        if (mesh_struct_ptr[0] != -1 && mesh_struct_ptr[1] != -1 &&
            mesh_struct_ptr[2] != -1) {
            return;
        }

        // Obtain voxel ptr
        float* voxel_ptr = static_cast<float*>(
                voxel_block_buffer_indexer.GetDataPtrFromCoord(xv, yv, zv,
                                                               block_idx));
        float tsdf_o = voxel_ptr[0];

        float r_o = voxel_ptr[2];
        float g_o = voxel_ptr[3];
        float b_o = voxel_ptr[4];
        float no[3] = {0}, ne[3] = {0};
        GetNormalAt(xv, yv, zv, workload_block_idx, no);

        // Enumerate 3 edges in the voxel
        for (int e = 0; e < 3; ++e) {
            int vertex_idx = mesh_struct_ptr[e];
            if (vertex_idx != -1) continue;

            float* voxel_ptr_e =
                    GetVoxelAt(xv + int(e == 0), yv + int(e == 1),
                               zv + int(e == 2), workload_block_idx);
            float tsdf_e = voxel_ptr_e[0];
            float ratio = (0 - tsdf_o) / (tsdf_e - tsdf_o);

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
            int idx = atomicAdd(vtx_count_ptr, 1);
#else
            int idx;
#pragma omp atomic capture
            {
                idx = *vtx_count_ptr;
                *vtx_count_ptr += 1;
            }
#endif
            mesh_struct_ptr[e] = idx;

            float ratio_x = ratio * int(e == 0);
            float ratio_y = ratio * int(e == 1);
            float ratio_z = ratio * int(e == 2);

            float* vertex_ptr = static_cast<float*>(
                    vertex_indexer.GetDataPtrFromCoord(idx));
            vertex_ptr[0] = voxel_size * (x + ratio_x);
            vertex_ptr[1] = voxel_size * (y + ratio_y);
            vertex_ptr[2] = voxel_size * (z + ratio_z);

            float* normal_ptr = static_cast<float*>(
                    normal_indexer.GetDataPtrFromCoord(idx));
            GetNormalAt(xv + int(e == 0), yv + int(e == 1), zv + int(e == 2),
                        workload_block_idx, ne);
            float nx = (1 - ratio) * no[0] + ratio * ne[0];
            float ny = (1 - ratio) * no[1] + ratio * ne[1];
            float nz = (1 - ratio) * no[2] + ratio * ne[2];
            float norm = sqrt(nx * nx + ny * ny + nz * nz) + 1e-5;
            normal_ptr[0] = nx / norm;
            normal_ptr[1] = ny / norm;
            normal_ptr[2] = nz / norm;

            float* color_ptr =
                    static_cast<float*>(color_indexer.GetDataPtrFromCoord(idx));
            float r_e = voxel_ptr_e[2];
            float g_e = voxel_ptr_e[3];
            float b_e = voxel_ptr_e[4];
            color_ptr[0] = ((1 - ratio) * r_o + ratio * r_e) / 255.0;
            color_ptr[1] = ((1 - ratio) * g_o + ratio * g_e) / 255.0;
            color_ptr[2] = ((1 - ratio) * b_o + ratio * b_e) / 255.0;
        }
    });

    dsts.emplace("vertices", vertices);
    dsts.emplace("colors", colors);
    dsts.emplace("normals", normals);

    // Pass 2: connect vertices
    core::Tensor triangle_count(std::vector<int>{0}, {}, core::Dtype::Int32,
                                block_values.GetDevice());
    int* tri_count_ptr = static_cast<int*>(triangle_count.GetDataPtr());

    core::Tensor triangles({total_vtx_count * 3, 3}, core::Dtype::Int64,
                           block_values.GetDevice());
    NDArrayIndexer triangle_indexer(triangles, 1);

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    CUDALauncher::LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                 int64_t workload_idx) {
#else
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
#endif
        // Natural index (0, N) -> (block_idx, voxel_idx)
        int64_t workload_block_idx = workload_idx / resolution3;
        int64_t voxel_idx = workload_idx % resolution3;

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        int64_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        // Obtain voxel's mesh struct ptr
        int* mesh_struct_ptr =
                static_cast<int*>(mesh_structure_indexer.GetDataPtrFromCoord(
                        xv, yv, zv, workload_block_idx));

        int table_idx = mesh_struct_ptr[3];
        if (tri_count[table_idx] == 0) return;

        for (size_t tri = 0; tri < 16; tri += 3) {
            if (tri_table[table_idx][tri] == -1) return;

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
            int tri_idx = atomicAdd(tri_count_ptr, 1);
#else
            int tri_idx;
#pragma omp atomic capture
            {
                tri_idx = *tri_count_ptr;
                *tri_count_ptr += 1;
            }
#endif

            for (size_t vertex = 0; vertex < 3; ++vertex) {
                int edge = tri_table[table_idx][tri + vertex];

                int64_t xv_i = xv + edge_shifts[edge][0];
                int64_t yv_i = yv + edge_shifts[edge][1];
                int64_t zv_i = zv + edge_shifts[edge][2];
                int64_t edge_i = edge_shifts[edge][3];

                int dxb = xv_i / resolution;
                int dyb = yv_i / resolution;
                int dzb = zv_i / resolution;

                int nb_idx = (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

                int64_t block_idx_i =
                        nb_indices_ptr[nb_idx * n_blocks + workload_block_idx];
                int* mesh_struct_ptr_i = static_cast<int*>(
                        mesh_structure_indexer.GetDataPtrFromCoord(
                                xv_i - dxb * resolution,
                                yv_i - dyb * resolution,
                                zv_i - dzb * resolution,
                                inv_indices_ptr[block_idx_i]));

                int64_t* triangle_ptr = static_cast<int64_t*>(
                        triangle_indexer.GetDataPtrFromCoord(tri_idx));
                triangle_ptr[2 - vertex] = mesh_struct_ptr_i[edge_i];
            }
        }
    });

    int total_tri_count = triangle_count.Item<int>();
    utility::LogInfo("Total triangle count = {}", total_tri_count);

    triangles = triangles.Slice(0, 0, total_tri_count);
    dsts.emplace("triangles", triangles);
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
