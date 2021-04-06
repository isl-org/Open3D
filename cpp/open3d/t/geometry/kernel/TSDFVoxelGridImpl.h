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

#include <atomic>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/TSDFVoxel.h"
#include "open3d/t/geometry/kernel/TSDFVoxelGrid.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace tsdf {

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void IntegrateCUDA
#else
void IntegrateCPU
#endif
        (const core::Tensor& depth,
         const core::Tensor& color,
         const core::Tensor& indices,
         const core::Tensor& block_keys,
         core::Tensor& block_values,
         // Transforms
         const core::Tensor& intrinsics,
         const core::Tensor& extrinsics,
         // Parameters
         int64_t resolution,
         float voxel_size,
         float sdf_trunc,
         float depth_scale,
         float depth_max) {
    // Parameters
    int64_t resolution3 = resolution * resolution * resolution;

    // Shape / transform indexers, no data involved
    NDArrayIndexer voxel_indexer({resolution, resolution, resolution});
    TransformIndexer transform_indexer(intrinsics, extrinsics, voxel_size);

    // Real data indexer
    NDArrayIndexer depth_indexer(depth, 2);
    NDArrayIndexer block_keys_indexer(block_keys, 1);
    NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);

    // Optional color integration
    NDArrayIndexer color_indexer;
    bool integrate_color = false;
    if (color.NumElements() != 0) {
        color_indexer = NDArrayIndexer(color, 2);
        integrate_color = true;
    }

    // Plain arrays that does not require indexers
    const int64_t* indices_ptr =
            static_cast<const int64_t*>(indices.GetDataPtr());

    int64_t n = indices.GetLength() * resolution3;

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher launcher;
#else
    core::kernel::CPULauncher launcher;
#endif

    DISPATCH_BYTESIZE_TO_VOXEL(
            voxel_block_buffer_indexer.ElementByteSize(), [&]() {
                launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                        int64_t workload_idx) {
                    // Natural index (0, N) -> (block_idx, voxel_idx)
                    int64_t block_idx = indices_ptr[workload_idx / resolution3];
                    int64_t voxel_idx = workload_idx % resolution3;

                    /// Coordinate transform
                    // block_idx -> (x_block, y_block, z_block)
                    int* block_key_ptr =
                            block_keys_indexer.GetDataPtrFromCoord<int>(
                                    block_idx);
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
                    transform_indexer.RigidTransform(
                            static_cast<float>(x), static_cast<float>(y),
                            static_cast<float>(z), &xc, &yc, &zc);

                    // coordinate in image (in pixel)
                    transform_indexer.Project(xc, yc, zc, &u, &v);
                    if (!depth_indexer.InBoundary(u, v)) {
                        return;
                    }

                    // Associate image workload and compute SDF and TSDF.
                    float depth = *depth_indexer.GetDataPtrFromCoord<float>(
                                          static_cast<int64_t>(u),
                                          static_cast<int64_t>(v)) /
                                  depth_scale;

                    float sdf = (depth - zc);
                    if (depth <= 0 || depth > depth_max || zc <= 0 ||
                        sdf < -sdf_trunc) {
                        return;
                    }
                    sdf = sdf < sdf_trunc ? sdf : sdf_trunc;
                    sdf /= sdf_trunc;

                    // Associate voxel workload and update TSDF/Weights
                    voxel_t* voxel_ptr = voxel_block_buffer_indexer
                                                 .GetDataPtrFromCoord<voxel_t>(
                                                         xv, yv, zv, block_idx);

                    if (integrate_color) {
                        float* color_ptr =
                                color_indexer.GetDataPtrFromCoord<float>(
                                        static_cast<int64_t>(u),
                                        static_cast<int64_t>(v));

                        voxel_ptr->Integrate(sdf, color_ptr[0], color_ptr[1],
                                             color_ptr[2]);
                    } else {
                        voxel_ptr->Integrate(sdf);
                    }
                });
            });
}

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void ExtractSurfacePointsCUDA
#else
void ExtractSurfacePointsCPU
#endif
        (const core::Tensor& indices,
         const core::Tensor& nb_indices,
         const core::Tensor& nb_masks,
         const core::Tensor& block_keys,
         const core::Tensor& block_values,
         core::Tensor& points,
         core::Tensor& normals,
         core::Tensor& colors,
         int64_t resolution,
         float voxel_size,
         float weight_threshold) {
    // Parameters
    int64_t resolution3 = resolution * resolution * resolution;

    // Shape / transform indexers, no data involved
    NDArrayIndexer voxel_indexer({resolution, resolution, resolution});

    // Real data indexer
    NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);
    NDArrayIndexer block_keys_indexer(block_keys, 1);
    NDArrayIndexer nb_block_masks_indexer(nb_masks, 2);
    NDArrayIndexer nb_block_indices_indexer(nb_indices, 2);

    // Plain arrays that does not require indexers
    const int64_t* indices_ptr =
            static_cast<const int64_t*>(indices.GetDataPtr());

    int64_t n_blocks = indices.GetLength();
    int64_t n = n_blocks * resolution3;

    // Output
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32,
                       block_values.GetDevice());
    int* count_ptr = count.GetDataPtr<int>();
#else
    std::atomic<int> count_atomic(0);
    std::atomic<int>* count_ptr = &count_atomic;
#endif

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher launcher;
#else
    core::kernel::CPULauncher launcher;
#endif

    // This pass determines valid number of points.
    DISPATCH_BYTESIZE_TO_VOXEL(
            voxel_block_buffer_indexer.ElementByteSize(), [&]() {
                launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                        int64_t workload_idx) {
                    auto GetVoxelAt = [&] OPEN3D_DEVICE(
                                              int xo, int yo, int zo,
                                              int curr_block_idx) -> voxel_t* {
                        return DeviceGetVoxelAt<voxel_t>(
                                xo, yo, zo, curr_block_idx,
                                static_cast<int>(resolution),
                                nb_block_masks_indexer,
                                nb_block_indices_indexer,
                                voxel_block_buffer_indexer);
                    };

                    // Natural index (0, N) -> (block_idx, voxel_idx)
                    int64_t workload_block_idx = workload_idx / resolution3;
                    int64_t block_idx = indices_ptr[workload_block_idx];
                    int64_t voxel_idx = workload_idx % resolution3;

                    // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                    int64_t xv, yv, zv;
                    voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                    voxel_t* voxel_ptr = voxel_block_buffer_indexer
                                                 .GetDataPtrFromCoord<voxel_t>(
                                                         xv, yv, zv, block_idx);
                    float tsdf_o = voxel_ptr->GetTSDF();
                    float weight_o = voxel_ptr->GetWeight();
                    if (weight_o <= weight_threshold) return;

                    // Enumerate x-y-z directions
                    for (int i = 0; i < 3; ++i) {
                        voxel_t* ptr = GetVoxelAt(
                                static_cast<int>(xv) + (i == 0),
                                static_cast<int>(yv) + (i == 1),
                                static_cast<int>(zv) + (i == 2),
                                static_cast<int>(workload_block_idx));
                        if (ptr == nullptr) continue;

                        float tsdf_i = ptr->GetTSDF();
                        float weight_i = ptr->GetWeight();

                        if (weight_i > weight_threshold &&
                            tsdf_i * tsdf_o < 0) {
                            OPEN3D_ATOMIC_ADD(count_ptr, 1);
                        }
                    }
                });
            });

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    int total_count = count.Item<int>();
#else
    int total_count = (*count_ptr).load();
#endif
    utility::LogInfo("Total point count = {}", total_count);

    points = core::Tensor({total_count, 3}, core::Dtype::Float32,
                          block_values.GetDevice());
    normals = core::Tensor({total_count, 3}, core::Dtype::Float32,
                           block_values.GetDevice());
    NDArrayIndexer point_indexer(points, 1);
    NDArrayIndexer normal_indexer(normals, 1);

    // Reset count
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    count = core::Tensor(std::vector<int>{0}, {}, core::Dtype::Int32,
                         block_values.GetDevice());
    count_ptr = count.GetDataPtr<int>();
#else
    (*count_ptr) = 0;
#endif

    // This pass extracts exact surface points.
    DISPATCH_BYTESIZE_TO_VOXEL(
            voxel_block_buffer_indexer.ElementByteSize(), [&]() {
                bool extract_color = false;
                NDArrayIndexer color_indexer;
                if (voxel_t::HasColor()) {
                    extract_color = true;
                    colors =
                            core::Tensor({total_count, 3}, core::Dtype::Float32,
                                         block_values.GetDevice());
                    color_indexer = NDArrayIndexer(colors, 1);
                }

                launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                        int64_t workload_idx) {
                    auto GetVoxelAt = [&] OPEN3D_DEVICE(
                                              int xo, int yo, int zo,
                                              int curr_block_idx) -> voxel_t* {
                        return DeviceGetVoxelAt<voxel_t>(
                                xo, yo, zo, curr_block_idx,
                                static_cast<int>(resolution),
                                nb_block_masks_indexer,
                                nb_block_indices_indexer,
                                voxel_block_buffer_indexer);
                    };
                    auto GetNormalAt = [&] OPEN3D_DEVICE(int xo, int yo, int zo,
                                                         int curr_block_idx,
                                                         float* n) {
                        return DeviceGetNormalAt<voxel_t>(
                                xo, yo, zo, curr_block_idx, n,
                                static_cast<int>(resolution), voxel_size,
                                nb_block_masks_indexer,
                                nb_block_indices_indexer,
                                voxel_block_buffer_indexer);
                    };

                    // Natural index (0, N) -> (block_idx, voxel_idx)
                    int64_t workload_block_idx = workload_idx / resolution3;
                    int64_t block_idx = indices_ptr[workload_block_idx];
                    int64_t voxel_idx = workload_idx % resolution3;

                    /// Coordinate transform
                    // block_idx -> (x_block, y_block, z_block)
                    int* block_key_ptr =
                            block_keys_indexer.GetDataPtrFromCoord<int>(
                                    block_idx);
                    int64_t xb = static_cast<int64_t>(block_key_ptr[0]);
                    int64_t yb = static_cast<int64_t>(block_key_ptr[1]);
                    int64_t zb = static_cast<int64_t>(block_key_ptr[2]);

                    // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                    int64_t xv, yv, zv;
                    voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                    voxel_t* voxel_ptr = voxel_block_buffer_indexer
                                                 .GetDataPtrFromCoord<voxel_t>(
                                                         xv, yv, zv, block_idx);
                    float tsdf_o = voxel_ptr->GetTSDF();
                    float weight_o = voxel_ptr->GetWeight();

                    if (weight_o <= weight_threshold) return;

                    int64_t x = xb * resolution + xv;
                    int64_t y = yb * resolution + yv;
                    int64_t z = zb * resolution + zv;

                    float no[3] = {0}, ni[3] = {0};
                    GetNormalAt(static_cast<int>(xv), static_cast<int>(yv),
                                static_cast<int>(zv),
                                static_cast<int>(workload_block_idx), no);

                    // Enumerate x-y-z axis
                    for (int i = 0; i < 3; ++i) {
                        voxel_t* ptr = GetVoxelAt(
                                static_cast<int>(xv) + (i == 0),
                                static_cast<int>(yv) + (i == 1),
                                static_cast<int>(zv) + (i == 2),
                                static_cast<int>(workload_block_idx));
                        if (ptr == nullptr) continue;

                        float tsdf_i = ptr->GetTSDF();
                        float weight_i = ptr->GetWeight();

                        if (weight_i > weight_threshold &&
                            tsdf_i * tsdf_o < 0) {
                            float ratio = (0 - tsdf_o) / (tsdf_i - tsdf_o);

                            int idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);

                            float* point_ptr =
                                    point_indexer.GetDataPtrFromCoord<float>(
                                            idx);
                            point_ptr[0] =
                                    voxel_size * (x + ratio * int(i == 0));
                            point_ptr[1] =
                                    voxel_size * (y + ratio * int(i == 1));
                            point_ptr[2] =
                                    voxel_size * (z + ratio * int(i == 2));
                            GetNormalAt(static_cast<int>(xv) + (i == 0),
                                        static_cast<int>(yv) + (i == 1),
                                        static_cast<int>(zv) + (i == 2),
                                        static_cast<int>(workload_block_idx),
                                        ni);

                            float* normal_ptr =
                                    normal_indexer.GetDataPtrFromCoord<float>(
                                            idx);
                            float nx = (1 - ratio) * no[0] + ratio * ni[0];
                            float ny = (1 - ratio) * no[1] + ratio * ni[1];
                            float nz = (1 - ratio) * no[2] + ratio * ni[2];
                            float norm = static_cast<float>(
                                    sqrt(nx * nx + ny * ny + nz * nz) + 1e-5);
                            normal_ptr[0] = nx / norm;
                            normal_ptr[1] = ny / norm;
                            normal_ptr[2] = nz / norm;

                            if (extract_color) {
                                float* color_ptr =
                                        color_indexer
                                                .GetDataPtrFromCoord<float>(
                                                        idx);

                                float r_o = voxel_ptr->GetR();
                                float g_o = voxel_ptr->GetG();
                                float b_o = voxel_ptr->GetB();

                                float r_i = ptr->GetR();
                                float g_i = ptr->GetG();
                                float b_i = ptr->GetB();

                                color_ptr[0] =
                                        ((1 - ratio) * r_o + ratio * r_i) /
                                        255.0f;
                                color_ptr[1] =
                                        ((1 - ratio) * g_o + ratio * g_i) /
                                        255.0f;
                                color_ptr[2] =
                                        ((1 - ratio) * b_o + ratio * b_i) /
                                        255.0f;
                            }
                        }
                    }
                });
            });
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void ExtractSurfaceMeshCUDA
#else
void ExtractSurfaceMeshCPU
#endif
        (const core::Tensor& indices,
         const core::Tensor& inv_indices,
         const core::Tensor& nb_indices,
         const core::Tensor& nb_masks,
         const core::Tensor& block_keys,
         const core::Tensor& block_values,
         core::Tensor& vertices,
         core::Tensor& triangles,
         core::Tensor& normals,
         core::Tensor& colors,
         int64_t resolution,
         float voxel_size,
         float weight_threshold) {

    int64_t resolution3 = resolution * resolution * resolution;

    // Shape / transform indexers, no data involved
    NDArrayIndexer voxel_indexer({resolution, resolution, resolution});

    // Output
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::CUDACachedMemoryManager::ReleaseCache();
#endif

    int n_blocks = static_cast<int>(indices.GetLength());
    // Voxel-wise mesh info. 4 channels correspond to:
    // 3 edges' corresponding vertex index + 1 table index.
    core::Tensor mesh_structure;
    try {
        mesh_structure = core::Tensor::Zeros(
                {n_blocks, resolution, resolution, resolution, 4},
                core::Dtype::Int32, block_keys.GetDevice());
    } catch (const std::runtime_error&) {
        utility::LogError(
                "[MeshExtractionKernel] Unable to allocate assistance mesh "
                "structure for Marching "
                "Cubes with {} active voxel blocks. Please consider using a "
                "larger voxel size (currently {}) for TSDF "
                "integration, or using tsdf_volume.cpu() to perform mesh "
                "extraction on CPU.",
                n_blocks, voxel_size);
    }

    // Real data indexer
    NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);
    NDArrayIndexer mesh_structure_indexer(mesh_structure, 4);
    NDArrayIndexer nb_block_masks_indexer(nb_masks, 2);
    NDArrayIndexer nb_block_indices_indexer(nb_indices, 2);

    // Plain arrays that does not require indexers
    const int64_t* indices_ptr = indices.GetDataPtr<int64_t>();
    const int64_t* inv_indices_ptr = inv_indices.GetDataPtr<int64_t>();

    int64_t n = n_blocks * resolution3;

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher launcher;
#else
    core::kernel::CPULauncher launcher;
#endif

    // Pass 0: analyze mesh structure, set up one-on-one correspondences from
    // edges to vertices.
    DISPATCH_BYTESIZE_TO_VOXEL(
            voxel_block_buffer_indexer.ElementByteSize(), [&]() {
                launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                        int64_t workload_idx) {
                    auto GetVoxelAt = [&] OPEN3D_DEVICE(
                                              int xo, int yo, int zo,
                                              int curr_block_idx) -> voxel_t* {
                        return DeviceGetVoxelAt<voxel_t>(
                                xo, yo, zo, curr_block_idx,
                                static_cast<int>(resolution),
                                nb_block_masks_indexer,
                                nb_block_indices_indexer,
                                voxel_block_buffer_indexer);
                    };

                    // Natural index (0, N) -> (block_idx, voxel_idx)
                    int64_t workload_block_idx = workload_idx / resolution3;
                    int64_t voxel_idx = workload_idx % resolution3;

                    // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                    int64_t xv, yv, zv;
                    voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                    // Check per-vertex sign in the cube to determine cube type
                    int table_idx = 0;
                    for (int i = 0; i < 8; ++i) {
                        voxel_t* voxel_ptr_i = GetVoxelAt(
                                static_cast<int>(xv) + vtx_shifts[i][0],
                                static_cast<int>(yv) + vtx_shifts[i][1],
                                static_cast<int>(zv) + vtx_shifts[i][2],
                                static_cast<int>(workload_block_idx));
                        if (voxel_ptr_i == nullptr) return;

                        float tsdf_i = voxel_ptr_i->GetTSDF();
                        float weight_i = voxel_ptr_i->GetWeight();
                        if (weight_i <= weight_threshold) return;

                        table_idx |= ((tsdf_i < 0) ? (1 << i) : 0);
                    }

                    int* mesh_struct_ptr =
                            mesh_structure_indexer.GetDataPtrFromCoord<int>(
                                    xv, yv, zv, workload_block_idx);
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

                            int dxb = static_cast<int>(xv_i / resolution);
                            int dyb = static_cast<int>(yv_i / resolution);
                            int dzb = static_cast<int>(zv_i / resolution);

                            int nb_idx =
                                    (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

                            int64_t block_idx_i =
                                    *nb_block_indices_indexer
                                             .GetDataPtrFromCoord<int64_t>(
                                                     workload_block_idx,
                                                     nb_idx);
                            int* mesh_ptr_i =
                                    mesh_structure_indexer.GetDataPtrFromCoord<
                                            int>(xv_i - dxb * resolution,
                                                 yv_i - dyb * resolution,
                                                 zv_i - dzb * resolution,
                                                 inv_indices_ptr[block_idx_i]);

                            // Non-atomic write, but we are safe
                            mesh_ptr_i[edge_i] = -1;
                        }
                    }
                });
            });

    // Pass 1: determine valid number of vertices.
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::Tensor vtx_count(std::vector<int>{0}, {}, core::Dtype::Int32,
                           block_values.GetDevice());
    int* vtx_count_ptr = vtx_count.GetDataPtr<int>();
#else
    std::atomic<int> vtx_count_atomic(0);
    std::atomic<int>* vtx_count_ptr = &vtx_count_atomic;
#endif

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
#else
    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
#endif
                // Natural index (0, N) -> (block_idx, voxel_idx)
                int64_t workload_block_idx = workload_idx / resolution3;
                int64_t voxel_idx = workload_idx % resolution3;

                // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                int64_t xv, yv, zv;
                voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                // Obtain voxel's mesh struct ptr
                int* mesh_struct_ptr =
                        mesh_structure_indexer.GetDataPtrFromCoord<int>(
                                xv, yv, zv, workload_block_idx);

                // Early quit -- no allocated vertex to compute
                if (mesh_struct_ptr[0] != -1 && mesh_struct_ptr[1] != -1 &&
                    mesh_struct_ptr[2] != -1) {
                    return;
                }

                // Enumerate 3 edges in the voxel
                for (int e = 0; e < 3; ++e) {
                    int vertex_idx = mesh_struct_ptr[e];
                    if (vertex_idx != -1) continue;

                    OPEN3D_ATOMIC_ADD(vtx_count_ptr, 1);
                }
            });

    // Reset count_ptr
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    int total_vtx_count = vtx_count.Item<int>();
    vtx_count = core::Tensor(std::vector<int>{0}, {}, core::Dtype::Int32,
                             block_values.GetDevice());
    vtx_count_ptr = vtx_count.GetDataPtr<int>();
#else
    int total_vtx_count = (*vtx_count_ptr).load();
    (*vtx_count_ptr) = 0;
#endif

    utility::LogInfo("Total vertex count = {}", total_vtx_count);
    vertices = core::Tensor({total_vtx_count, 3}, core::Dtype::Float32,
                            block_values.GetDevice());
    normals = core::Tensor({total_vtx_count, 3}, core::Dtype::Float32,
                           block_values.GetDevice());

    NDArrayIndexer block_keys_indexer(block_keys, 1);
    NDArrayIndexer vertex_indexer(vertices, 1);
    NDArrayIndexer normal_indexer(normals, 1);

    // Pass 2: extract vertices.
    DISPATCH_BYTESIZE_TO_VOXEL(
            voxel_block_buffer_indexer.ElementByteSize(), [&]() {
                bool extract_color = false;
                NDArrayIndexer color_indexer;
                if (voxel_t::HasColor()) {
                    extract_color = true;
                    colors = core::Tensor({total_vtx_count, 3},
                                          core::Dtype::Float32,
                                          block_values.GetDevice());
                    color_indexer = NDArrayIndexer(colors, 1);
                }
                launcher.LaunchGeneralKernel(n, [=] OPEN3D_DEVICE(
                                                        int64_t workload_idx) {
                    auto GetVoxelAt = [&] OPEN3D_DEVICE(
                                              int xo, int yo, int zo,
                                              int curr_block_idx) -> voxel_t* {
                        return DeviceGetVoxelAt<voxel_t>(
                                xo, yo, zo, curr_block_idx,
                                static_cast<int>(resolution),
                                nb_block_masks_indexer,
                                nb_block_indices_indexer,
                                voxel_block_buffer_indexer);
                    };

                    auto GetNormalAt = [&] OPEN3D_DEVICE(int xo, int yo, int zo,
                                                         int curr_block_idx,
                                                         float* n) {
                        return DeviceGetNormalAt<voxel_t>(
                                xo, yo, zo, curr_block_idx, n,
                                static_cast<int>(resolution), voxel_size,
                                nb_block_masks_indexer,
                                nb_block_indices_indexer,
                                voxel_block_buffer_indexer);
                    };

                    // Natural index (0, N) -> (block_idx, voxel_idx)
                    int64_t workload_block_idx = workload_idx / resolution3;
                    int64_t block_idx = indices_ptr[workload_block_idx];
                    int64_t voxel_idx = workload_idx % resolution3;

                    // block_idx -> (x_block, y_block, z_block)
                    int* block_key_ptr =
                            block_keys_indexer.GetDataPtrFromCoord<int>(
                                    block_idx);
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
                            mesh_structure_indexer.GetDataPtrFromCoord<int>(
                                    xv, yv, zv, workload_block_idx);

                    // Early quit -- no allocated vertex to compute
                    if (mesh_struct_ptr[0] != -1 && mesh_struct_ptr[1] != -1 &&
                        mesh_struct_ptr[2] != -1) {
                        return;
                    }

                    // Obtain voxel ptr
                    voxel_t* voxel_ptr = voxel_block_buffer_indexer
                                                 .GetDataPtrFromCoord<voxel_t>(
                                                         xv, yv, zv, block_idx);
                    float tsdf_o = voxel_ptr->GetTSDF();
                    float no[3] = {0}, ne[3] = {0};
                    GetNormalAt(static_cast<int>(xv), static_cast<int>(yv),
                                static_cast<int>(zv),
                                static_cast<int>(workload_block_idx), no);

                    // Enumerate 3 edges in the voxel
                    for (int e = 0; e < 3; ++e) {
                        int vertex_idx = mesh_struct_ptr[e];
                        if (vertex_idx != -1) continue;

                        voxel_t* voxel_ptr_e = GetVoxelAt(
                                static_cast<int>(xv) + (e == 0),
                                static_cast<int>(yv) + (e == 1),
                                static_cast<int>(zv) + (e == 2),
                                static_cast<int>(workload_block_idx));
                        float tsdf_e = voxel_ptr_e->GetTSDF();
                        float ratio = (0 - tsdf_o) / (tsdf_e - tsdf_o);

                        int idx = OPEN3D_ATOMIC_ADD(vtx_count_ptr, 1);
                        mesh_struct_ptr[e] = idx;

                        float ratio_x = ratio * int(e == 0);
                        float ratio_y = ratio * int(e == 1);
                        float ratio_z = ratio * int(e == 2);

                        float* vertex_ptr =
                                vertex_indexer.GetDataPtrFromCoord<float>(idx);
                        vertex_ptr[0] = voxel_size * (x + ratio_x);
                        vertex_ptr[1] = voxel_size * (y + ratio_y);
                        vertex_ptr[2] = voxel_size * (z + ratio_z);

                        float* normal_ptr =
                                normal_indexer.GetDataPtrFromCoord<float>(idx);
                        GetNormalAt(static_cast<int>(xv) + (e == 0),
                                    static_cast<int>(yv) + (e == 1),
                                    static_cast<int>(zv) + (e == 2),
                                    static_cast<int>(workload_block_idx), ne);
                        float nx = (1 - ratio) * no[0] + ratio * ne[0];
                        float ny = (1 - ratio) * no[1] + ratio * ne[1];
                        float nz = (1 - ratio) * no[2] + ratio * ne[2];
                        float norm = static_cast<float>(
                                sqrt(nx * nx + ny * ny + nz * nz) + 1e-5);
                        normal_ptr[0] = nx / norm;
                        normal_ptr[1] = ny / norm;
                        normal_ptr[2] = nz / norm;

                        if (extract_color) {
                            float* color_ptr =
                                    color_indexer.GetDataPtrFromCoord<float>(
                                            idx);
                            float r_o = voxel_ptr->GetR();
                            float g_o = voxel_ptr->GetG();
                            float b_o = voxel_ptr->GetB();

                            float r_e = voxel_ptr_e->GetR();
                            float g_e = voxel_ptr_e->GetG();
                            float b_e = voxel_ptr_e->GetB();
                            color_ptr[0] =
                                    ((1 - ratio) * r_o + ratio * r_e) / 255.0f;
                            color_ptr[1] =
                                    ((1 - ratio) * g_o + ratio * g_e) / 255.0f;
                            color_ptr[2] =
                                    ((1 - ratio) * b_o + ratio * b_e) / 255.0f;
                        }
                    }
                });
            });

    // Pass 3: connect vertices and form triangles.
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::Tensor triangle_count(std::vector<int>{0}, {}, core::Dtype::Int32,
                                block_values.GetDevice());
    int* tri_count_ptr = triangle_count.GetDataPtr<int>();
#else
    std::atomic<int> tri_count_atomic(0);
    std::atomic<int>* tri_count_ptr = &tri_count_atomic;
#endif

    triangles = core::Tensor({total_vtx_count * 3, 3}, core::Dtype::Int64,
                             block_values.GetDevice());
    NDArrayIndexer triangle_indexer(triangles, 1);

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
#else
    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
#endif
                // Natural index (0, N) -> (block_idx,
                // voxel_idx)
                int64_t workload_block_idx = workload_idx / resolution3;
                int64_t voxel_idx = workload_idx % resolution3;

                // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                int64_t xv, yv, zv;
                voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                // Obtain voxel's mesh struct ptr
                int* mesh_struct_ptr =
                        mesh_structure_indexer.GetDataPtrFromCoord<int>(
                                xv, yv, zv, workload_block_idx);

                int table_idx = mesh_struct_ptr[3];
                if (tri_count[table_idx] == 0) return;

                for (size_t tri = 0; tri < 16; tri += 3) {
                    if (tri_table[table_idx][tri] == -1) return;

                    int tri_idx = OPEN3D_ATOMIC_ADD(tri_count_ptr, 1);

                    for (size_t vertex = 0; vertex < 3; ++vertex) {
                        int edge = tri_table[table_idx][tri + vertex];

                        int64_t xv_i = xv + edge_shifts[edge][0];
                        int64_t yv_i = yv + edge_shifts[edge][1];
                        int64_t zv_i = zv + edge_shifts[edge][2];
                        int64_t edge_i = edge_shifts[edge][3];

                        int dxb = static_cast<int>(xv_i / resolution);
                        int dyb = static_cast<int>(yv_i / resolution);
                        int dzb = static_cast<int>(zv_i / resolution);

                        int nb_idx = (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

                        int64_t block_idx_i =
                                *nb_block_indices_indexer
                                         .GetDataPtrFromCoord<int64_t>(
                                                 workload_block_idx, nb_idx);
                        int* mesh_struct_ptr_i =
                                mesh_structure_indexer.GetDataPtrFromCoord<int>(
                                        xv_i - dxb * resolution,
                                        yv_i - dyb * resolution,
                                        zv_i - dzb * resolution,
                                        inv_indices_ptr[block_idx_i]);

                        int64_t* triangle_ptr =
                                triangle_indexer.GetDataPtrFromCoord<int64_t>(
                                        tri_idx);
                        triangle_ptr[2 - vertex] = mesh_struct_ptr_i[edge_i];
                    }
                }
            });

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    int total_tri_count = triangle_count.Item<int>();
#else
    int total_tri_count = (*tri_count_ptr).load();
#endif
    utility::LogInfo("Total triangle count = {}", total_tri_count);
    triangles = triangles.Slice(0, 0, total_tri_count);
}

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
void RayCastCUDA
#else
void RayCastCPU
#endif
        (std::shared_ptr<core::DeviceHashmap>& hashmap,
         core::Tensor& block_values,
         core::Tensor& vertex_map,
         core::Tensor& color_map,
         core::Tensor& normal_map,
         const core::Tensor& intrinsics,
         const core::Tensor& pose,
         int64_t block_resolution,
         float voxel_size,
         float sdf_trunc,
         int max_steps,
         float depth_min,
         float depth_max,
         float weight_threshold) {
    using Key = core::Block<int, 3>;
    using Hash = core::BlockHash<int, 3>;

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    auto cuda_hashmap =
            std::dynamic_pointer_cast<core::StdGPUHashmap<Key, Hash>>(hashmap);
    if (cuda_hashmap == nullptr) {
        utility::LogError(
                "Unsupported backend: CUDA raycasting only supports STDGPU.");
    }
    auto hashmap_impl = cuda_hashmap->GetImpl();
#else
    auto cpu_hashmap =
            std::dynamic_pointer_cast<core::TBBHashmap<Key, Hash>>(hashmap);
    auto hashmap_impl = *cpu_hashmap->GetImpl();
#endif

    NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);
    NDArrayIndexer vertex_map_indexer(vertex_map, 2);
    NDArrayIndexer color_map_indexer(color_map, 2);
    NDArrayIndexer normal_map_indexer(normal_map, 2);

    TransformIndexer transform_indexer(intrinsics, pose);
    TransformIndexer w2c_transform_indexer(intrinsics, pose.Inverse());

    int64_t rows = vertex_map_indexer.GetShape(0);
    int64_t cols = vertex_map_indexer.GetShape(1);

    float block_size = voxel_size * block_resolution;
#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::kernel::CUDALauncher launcher;
#else
    core::kernel::CPULauncher launcher;
#endif

    DISPATCH_BYTESIZE_TO_VOXEL(
            voxel_block_buffer_indexer.ElementByteSize(), [&]() {
                launcher.LaunchGeneralKernel(
                        rows * cols, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                            auto GetVoxelAtP =
                                    [&] OPEN3D_DEVICE(int x_b, int y_b, int z_b,
                                                      int x_v, int y_v, int z_v,
                                                      core::addr_t block_addr)
                                    -> voxel_t* {
                                int x_vn = (x_v + block_resolution) %
                                           block_resolution;
                                int y_vn = (y_v + block_resolution) %
                                           block_resolution;
                                int z_vn = (z_v + block_resolution) %
                                           block_resolution;

                                int dx_b = sign(x_v - x_vn);
                                int dy_b = sign(y_v - y_vn);
                                int dz_b = sign(z_v - z_vn);

                                if (dx_b == 0 && dy_b == 0 && dz_b == 0) {
                                    return voxel_block_buffer_indexer
                                            .GetDataPtrFromCoord<voxel_t>(
                                                    x_v, y_v, z_v, block_addr);
                                } else {
                                    Key key;
                                    key(0) = x_b + dx_b;
                                    key(1) = y_b + dy_b;
                                    key(2) = z_b + dz_b;

                                    auto iter = hashmap_impl.find(key);
                                    if (iter == hashmap_impl.end())
                                        return nullptr;

                                    return voxel_block_buffer_indexer
                                            .GetDataPtrFromCoord<voxel_t>(
                                                    x_vn, y_vn, z_vn,
                                                    iter->second);
                                }
                            };

                            auto GetVoxelAtT = [&] OPEN3D_DEVICE(
                                                       float x_o, float y_o,
                                                       float z_o, float x_d,
                                                       float y_d, float z_d,
                                                       float t) -> voxel_t* {
                                float x_g = x_o + t * x_d;
                                float y_g = y_o + t * y_d;
                                float z_g = z_o + t * z_d;

                                // Block coordinate and look up
                                int x_b = static_cast<int>(
                                        floor(x_g / block_size));
                                int y_b = static_cast<int>(
                                        floor(y_g / block_size));
                                int z_b = static_cast<int>(
                                        floor(z_g / block_size));

                                Key key;
                                key(0) = x_b;
                                key(1) = y_b;
                                key(2) = z_b;
                                auto iter = hashmap_impl.find(key);
                                if (iter == hashmap_impl.end()) return nullptr;

                                core::addr_t block_addr = iter->second;

                                // Voxel coordinate and look up
                                int x_v = int((x_g - x_b * block_size) /
                                              voxel_size);
                                int y_v = int((y_g - y_b * block_size) /
                                              voxel_size);
                                int z_v = int((z_g - z_b * block_size) /
                                              voxel_size);
                                return voxel_block_buffer_indexer
                                        .GetDataPtrFromCoord<voxel_t>(
                                                x_v, y_v, z_v, block_addr);
                            };

                            int64_t y = workload_idx / cols;
                            int64_t x = workload_idx % cols;

                            float t = depth_min;

                            // Coordinates in camera and global
                            float x_c = 0, y_c = 0, z_c = 0;
                            float x_g = 0, y_g = 0, z_g = 0;
                            float x_o = 0, y_o = 0, z_o = 0;

                            // Iterative ray intersection check
                            float t_prev = t;
                            float tsdf_prev = 1.0f;

                            // Camera origin
                            transform_indexer.RigidTransform(0, 0, 0, &x_o,
                                                             &y_o, &z_o);

                            // Direction
                            transform_indexer.Unproject(static_cast<float>(x),
                                                        static_cast<float>(y),
                                                        1.0f, &x_c, &y_c, &z_c);
                            transform_indexer.RigidTransform(x_c, y_c, z_c,
                                                             &x_g, &y_g, &z_g);
                            float x_d = (x_g - x_o);
                            float y_d = (y_g - y_o);
                            float z_d = (z_g - z_o);

                            for (int step = 0; step < max_steps; ++step) {
                                voxel_t* voxel_ptr = GetVoxelAtT(
                                        x_o, y_o, z_o, x_d, y_d, z_d, t);
                                if (!voxel_ptr) {
                                    t_prev = t;
                                    t += block_size;
                                    continue;
                                }
                                float tsdf = voxel_ptr->GetTSDF();
                                float w = voxel_ptr->GetWeight();

                                if (tsdf_prev > 0 && w >= weight_threshold &&
                                    tsdf <= 0) {
                                    float t_intersect =
                                            (t * tsdf_prev - t_prev * tsdf) /
                                            (tsdf_prev - tsdf);
                                    x_g = x_o + t_intersect * x_d;
                                    y_g = y_o + t_intersect * y_d;
                                    z_g = z_o + t_intersect * z_d;

                                    // Trivial vertex assignment
                                    float* vertex =
                                            vertex_map_indexer
                                                    .GetDataPtrFromCoord<float>(
                                                            x, y);
                                    vertex[0] = x_g;
                                    vertex[1] = y_g;
                                    vertex[2] = z_g;

                                    // Trilinear interpolation
                                    int x_b = static_cast<int>(
                                            floor(x_g / block_size));
                                    int y_b = static_cast<int>(
                                            floor(y_g / block_size));
                                    int z_b = static_cast<int>(
                                            floor(z_g / block_size));
                                    float x_v =
                                            (x_g - float(x_b) * block_size) /
                                            voxel_size;
                                    float y_v =
                                            (y_g - float(y_b) * block_size) /
                                            voxel_size;
                                    float z_v =
                                            (z_g - float(z_b) * block_size) /
                                            voxel_size;

                                    Key key;
                                    key(0) = x_b;
                                    key(1) = y_b;
                                    key(2) = z_b;
                                    auto iter = hashmap_impl.find(key);
                                    if (iter == hashmap_impl.end()) break;

                                    core::addr_t block_addr = iter->second;

                                    int x_v_floor =
                                            static_cast<int>(floor(x_v));
                                    int y_v_floor =
                                            static_cast<int>(floor(y_v));
                                    int z_v_floor =
                                            static_cast<int>(floor(z_v));

                                    float ratio_x = x_v - float(x_v_floor);
                                    float ratio_y = y_v - float(y_v_floor);
                                    float ratio_z = z_v - float(z_v_floor);

                                    // Color inteprolation
                                    float* color =
                                            color_map_indexer
                                                    .GetDataPtrFromCoord<float>(
                                                            x, y);
                                    color[0] = 0;
                                    color[1] = 0;
                                    color[2] = 0;

                                    float* normal =
                                            normal_map_indexer
                                                    .GetDataPtrFromCoord<float>(
                                                            x, y);
                                    normal[0] = 0;
                                    normal[1] = 0;
                                    normal[2] = 0;

                                    float sum_weight_color = 0.0;
                                    float sum_weight_normal = 0.0;
                                    for (int k = 0; k < 8; ++k) {
                                        int dx_v = (k & 1) > 0 ? 1 : 0;
                                        int dy_v = (k & 2) > 0 ? 1 : 0;
                                        int dz_v = (k & 4) > 0 ? 1 : 0;
                                        float ratio =
                                                (dx_v * (ratio_x) +
                                                 (1 - dx_v) * (1 - ratio_x)) *
                                                (dy_v * (ratio_y) +
                                                 (1 - dy_v) * (1 - ratio_y)) *
                                                (dz_v * (ratio_z) +
                                                 (1 - dz_v) * (1 - ratio_z));

                                        voxel_t* voxel_ptr_k = GetVoxelAtP(
                                                x_b, y_b, z_b, x_v_floor + dx_v,
                                                y_v_floor + dy_v,
                                                z_v_floor + dz_v, block_addr);

                                        if (voxel_ptr_k &&
                                            voxel_ptr_k->GetWeight() > 0) {
                                            sum_weight_color += ratio;
                                            color[0] +=
                                                    ratio * voxel_ptr_k->GetR();
                                            color[1] +=
                                                    ratio * voxel_ptr_k->GetG();
                                            color[2] +=
                                                    ratio * voxel_ptr_k->GetB();
                                        }

                                        for (int dim = 0; dim < 3; ++dim) {
                                            voxel_t* voxel_ptr_k_plus =
                                                    GetVoxelAtP(
                                                            x_b, y_b, z_b,
                                                            x_v_floor + dx_v +
                                                                    (dim == 0),
                                                            y_v_floor + dy_v +
                                                                    (dim == 1),
                                                            z_v_floor + dz_v +
                                                                    (dim == 2),
                                                            block_addr);
                                            voxel_t* voxel_ptr_k_minus =
                                                    GetVoxelAtP(
                                                            x_b, y_b, z_b,
                                                            x_v_floor + dx_v -
                                                                    (dim == 0),
                                                            y_v_floor + dy_v -
                                                                    (dim == 1),
                                                            z_v_floor + dz_v -
                                                                    (dim == 2),
                                                            block_addr);

                                            bool valid = false;
                                            if (voxel_ptr_k_plus &&
                                                voxel_ptr_k_plus->GetWeight() >
                                                        0) {
                                                normal[dim] +=
                                                        ratio *
                                                        voxel_ptr_k_plus
                                                                ->GetTSDF() /
                                                        (2 * voxel_size);
                                                valid = true;
                                            }

                                            if (voxel_ptr_k_minus &&
                                                voxel_ptr_k_minus->GetWeight() >
                                                        0) {
                                                normal[dim] -=
                                                        ratio *
                                                        voxel_ptr_k_minus
                                                                ->GetTSDF() /
                                                        (2 * voxel_size);
                                                valid = true;
                                            }
                                            sum_weight_normal +=
                                                    valid ? ratio : 0;
                                        }
                                    }

                                    if (sum_weight_color > 0) {
                                        sum_weight_color *= 255.0;
                                        color[0] /= sum_weight_color;
                                        color[1] /= sum_weight_color;
                                        color[2] /= sum_weight_color;
                                    }
                                    if (sum_weight_normal > 0) {
                                        normal[0] /= sum_weight_normal;
                                        normal[1] /= sum_weight_normal;
                                        normal[2] /= sum_weight_normal;
                                        float norm =
                                                sqrt(normal[0] * normal[0] +
                                                     normal[1] * normal[1] +
                                                     normal[2] * normal[2]);
                                        w2c_transform_indexer.Rotate(
                                                normal[0] / norm,
                                                normal[1] / norm,
                                                normal[2] / norm, normal + 0,
                                                normal + 1, normal + 2);
                                    }
                                    break;
                                }

                                tsdf_prev = tsdf;
                                t_prev = t;
                                float delta = tsdf * sdf_trunc;
                                t += delta < voxel_size ? voxel_size : delta;
                            }
                        });
            });

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

}  // namespace tsdf
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
