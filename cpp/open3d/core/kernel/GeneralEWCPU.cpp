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

#include <tbb/concurrent_unordered_set.h>

#include <unordered_set>

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

void CPUUnprojectKernel(const std::unordered_map<std::string, Tensor>& srcs,
                        std::unordered_map<std::string, Tensor>& dsts) {
    static std::unordered_set<std::string> src_attrs = {
            "depth",
            "intrinsics",
            "depth_scale",
            "depth_max",
    };
    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[CPUUnprojectKernel] expected Tensor {} in srcs, but "
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

    NDArrayIndexer depth_ndi(depth, 2);
    TransformIndexer ti(intrinsics);

    // Output
    int64_t rows_strided = depth_ndi.GetShape(0) / stride;
    int64_t cols_strided = depth_ndi.GetShape(1) / stride;
    Tensor points({rows_strided * cols_strided, 3}, core::Dtype::Float32,
                  depth.GetDevice());
    Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32,
                 depth.GetDevice());
    float* points_ptr = static_cast<float*>(points.GetDataPtr());
    int* count_ptr = static_cast<int*>(count.GetDataPtr());

    // Workload
    int64_t n = rows_strided * cols_strided;

    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
        int64_t y = (workload_idx / cols_strided) * stride;
        int64_t x = (workload_idx % cols_strided) * stride;

        int64_t workload_depth;
        depth_ndi.CoordToWorkload(x, y, &workload_depth);
        float d = *static_cast<float*>(
                          depth_ndi.GetDataPtrFromWorkload(workload_depth)) /
                  depth_scale;
        if (d > 0 && d < depth_max) {
            int idx;
#pragma omp atomic capture
            {
                idx = *count_ptr;
                *count_ptr += 1;
            }

            float* vertex = points_ptr + 3 * idx;
            ti.Unproject(static_cast<float>(x), static_cast<float>(y), d,
                         vertex + 0, vertex + 1, vertex + 2);
        }
    });

    int total_pts_count = count.Item<int>();
    dsts.emplace("points", points.Slice(0, 0, total_pts_count));
}

struct Coord3i {
    Coord3i(int x, int y, int z) : x_(x), y_(y), z_(z) {}
    bool operator==(const Coord3i& other) const {
        return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
    }

    int64_t x_;
    int64_t y_;
    int64_t z_;
};

struct Coord3iHash {
    size_t operator()(const Coord3i& k) const {
        static const size_t p0 = 73856093;
        static const size_t p1 = 19349669;
        static const size_t p2 = 83492791;

        return (static_cast<size_t>(k.x_) * p0) ^
               (static_cast<size_t>(k.y_) * p1) ^
               (static_cast<size_t>(k.z_) * p2);
    }
};

void CPUTSDFTouchKernel(const std::unordered_map<std::string, Tensor>& srcs,
                        std::unordered_map<std::string, Tensor>& dsts) {
    static std::unordered_set<std::string> src_attrs = {
            "points", "voxel_size", "resolution", "sdf_trunc"};

    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[CUDATSDFTouchKernel] expected Tensor {} in srcs, but "
                    "did not receive",
                    k);
        }
    }

    Tensor pcd = srcs.at("points");
    float voxel_size = srcs.at("voxel_size").Item<float>();
    int64_t resolution = srcs.at("resolution").Item<int64_t>();
    float block_size = voxel_size * resolution;

    float sdf_trunc = srcs.at("sdf_trunc").Item<float>();

    int64_t n = pcd.GetShape()[0];
    float* pcd_ptr = static_cast<float*>(pcd.GetDataPtr());

    tbb::concurrent_unordered_set<Coord3i, Coord3iHash> set;
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
        float x = pcd_ptr[3 * workload_idx + 0];
        float y = pcd_ptr[3 * workload_idx + 1];
        float z = pcd_ptr[3 * workload_idx + 2];

        int64_t xb_lo =
                static_cast<int64_t>(std::floor((x - sdf_trunc) / block_size));
        int64_t xb_hi =
                static_cast<int64_t>(std::floor((x + sdf_trunc) / block_size));
        int64_t yb_lo =
                static_cast<int64_t>(std::floor((y - sdf_trunc) / block_size));
        int64_t yb_hi =
                static_cast<int64_t>(std::floor((y + sdf_trunc) / block_size));
        int64_t zb_lo =
                static_cast<int64_t>(std::floor((z - sdf_trunc) / block_size));
        int64_t zb_hi =
                static_cast<int64_t>(std::floor((z + sdf_trunc) / block_size));
        for (int64_t xb = xb_lo; xb <= xb_hi; ++xb) {
            for (int64_t yb = yb_lo; yb <= yb_hi; ++yb) {
                for (int64_t zb = zb_lo; zb <= zb_hi; ++zb) {
                    set.emplace(xb, yb, zb);
                }
            }
        }
    });

    int64_t block_count = set.size();
    core::Tensor block_coords({block_count, 3}, core::Dtype::Int32,
                              pcd.GetDevice());
    int* block_coords_ptr = static_cast<int*>(block_coords.GetDataPtr());
    int count = 0;
    for (auto it = set.begin(); it != set.end(); ++it, ++count) {
        int64_t offset = count * 3;
        block_coords_ptr[offset + 0] = (*it).x_;
        block_coords_ptr[offset + 1] = (*it).y_;
        block_coords_ptr[offset + 2] = (*it).z_;
    }

    dsts.emplace("block_coords", block_coords);
}

void CPUTSDFIntegrateKernel(const std::unordered_map<std::string, Tensor>& srcs,
                            std::unordered_map<std::string, Tensor>& dsts) {
    // Decode input tensors
    static std::unordered_set<std::string> src_attrs = {
            "depth",       "indices",    "block_keys", "intrinsics",
            "extrinsics",  "resolution", "voxel_size", "sdf_trunc",
            "depth_scale", "depth_max",
    };
    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[CPUTSDFIntegrateKernel] expected Tensor {} in srcs, but "
                    "did not receive",
                    k);
        }
    }

    Tensor depth = srcs.at("depth").To(core::Dtype::Float32);
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
    NDArrayIndexer image_indexer(depth, 2);
    NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);

    // Plain arrays that does not require indexers
    int64_t* indices_ptr = static_cast<int64_t*>(indices.GetDataPtr());
    int* block_keys_ptr = static_cast<int*>(block_keys.GetDataPtr());

    int64_t n = indices.GetShape()[0] * resolution3;
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
        // Natural index (0, N) -> (block_idx, voxel_idx)
        int64_t block_idx = indices_ptr[workload_idx / resolution3];
        int64_t voxel_idx = workload_idx % resolution3;

        /// Coordinate transform
        // block_idx -> (x_block, y_block, z_block)
        int64_t xb = static_cast<int64_t>(block_keys_ptr[block_idx * 3 + 0]);
        int64_t yb = static_cast<int64_t>(block_keys_ptr[block_idx * 3 + 1]);
        int64_t zb = static_cast<int64_t>(block_keys_ptr[block_idx * 3 + 2]);

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
        if (!image_indexer.InBoundary(u, v)) {
            return;
        }

        /// Associate image workload and compute SDF
        int64_t workload_image;
        image_indexer.CoordToWorkload(static_cast<int64_t>(u),
                                      static_cast<int64_t>(v), &workload_image);
        float depth =
                *static_cast<const float*>(
                        image_indexer.GetDataPtrFromWorkload(workload_image)) /
                depth_scale;

        // Compute multiplier
        float xc_unproj, yc_unproj, zc_unproj;
        transform_indexer.Unproject(static_cast<float>(u),
                                    static_cast<float>(v), 1.0, &xc_unproj,
                                    &yc_unproj, &zc_unproj);
        float multiplier =
                std::sqrt(xc_unproj * xc_unproj + yc_unproj * yc_unproj + 1.0);
        float sdf = (depth - zc) * multiplier;
        if (depth <= 0 || depth > depth_max || zc <= 0 || sdf < -sdf_trunc) {
            return;
        }
        sdf = sdf < sdf_trunc ? sdf : sdf_trunc;
        sdf /= sdf_trunc;

        /// Associate voxel workload and update TSDF/Weights
        int64_t workload_voxel;
        voxel_block_buffer_indexer.CoordToWorkload(xv, yv, zv, block_idx,
                                                   &workload_voxel);
        float* voxel_ptr = static_cast<float*>(
                voxel_block_buffer_indexer.GetDataPtrFromWorkload(
                        workload_voxel));

        float tsdf_sum = voxel_ptr[0];
        float weight_sum = voxel_ptr[1];
        voxel_ptr[0] = (weight_sum * tsdf_sum + sdf) / (weight_sum + 1);
        voxel_ptr[1] = weight_sum + 1;
        // printf("%ld -> (%ld, %ld %ld %ld): %ld (%f %f)\n", workload_idx,
        //        block_idx, xv, yv, zv, workload_voxel, voxel_ptr[0],
        //        voxel_ptr[1]);
    });
}

void CPUSurfaceExtractionKernel(
        const std::unordered_map<std::string, Tensor>& srcs,
        std::unordered_map<std::string, Tensor>& dsts) {
    // Decode input tensors
    static std::unordered_set<std::string> src_attrs = {
            "indices",      "nb_indices", "nb_masks",   "block_keys",
            "block_values", "voxel_size", "resolution",
    };
    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[CPUTSDFIntegrateKernel] expected Tensor {} in srcs, but "
                    "did not receive",
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

    // Plain arrays that does not require indexers
    int64_t* nb_indices_ptr = static_cast<int64_t*>(nb_indices.GetDataPtr());
    bool* nb_masks_ptr = static_cast<bool*>(nb_masks.GetDataPtr());
    int64_t* indices_ptr = static_cast<int64_t*>(indices.GetDataPtr());
    int* block_keys_ptr = static_cast<int*>(block_keys.GetDataPtr());

    int n_blocks = indices.GetShape()[0];
    int64_t n = n_blocks * resolution3;

    // Output
    core::Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32,
                       block_values.GetDevice());
    core::Tensor points({std::min(n * 3, int64_t(10000000)), 3},
                        core::Dtype::Float32, block_values.GetDevice());
    core::Tensor normals({std::min(n * 3, int64_t(10000000)), 3},
                         core::Dtype::Float32, block_values.GetDevice());
    int* count_ptr = static_cast<int*>(count.GetDataPtr());
    float* points_ptr = static_cast<float*>(points.GetDataPtr());
    float* normals_ptr = static_cast<float*>(normals.GetDataPtr());

    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
        auto GetVoxelAt = [&](int xo, int yo, int zo,
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
            int64_t workload_voxel_i;
            voxel_block_buffer_indexer.CoordToWorkload(xn, yn, zn, block_idx_i,
                                                       &workload_voxel_i);
            return static_cast<float*>(
                    voxel_block_buffer_indexer.GetDataPtrFromWorkload(
                            workload_voxel_i));
        };

        auto GetNormalAt = [&](int xo, int yo, int zo, int curr_block_idx,
                               float* n) {
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
        int64_t xb = static_cast<int64_t>(block_keys_ptr[block_idx * 3 + 0]);
        int64_t yb = static_cast<int64_t>(block_keys_ptr[block_idx * 3 + 1]);
        int64_t zb = static_cast<int64_t>(block_keys_ptr[block_idx * 3 + 2]);

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        int64_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);
        int64_t workload_voxel;
        voxel_block_buffer_indexer.CoordToWorkload(xv, yv, zv, block_idx,
                                                   &workload_voxel);
        float* voxel_ptr = static_cast<float*>(
                voxel_block_buffer_indexer.GetDataPtrFromWorkload(
                        workload_voxel));
        float tsdf_o = voxel_ptr[0];
        float weight_o = voxel_ptr[1];
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

                int idx;
#pragma omp atomic capture
                {
                    idx = *count_ptr;
                    *count_ptr += 1;
                }

                points_ptr[idx * 3 + 0] =
                        voxel_size * (x + ratio * int(i == 0));
                points_ptr[idx * 3 + 1] =
                        voxel_size * (y + ratio * int(i == 1));
                points_ptr[idx * 3 + 2] =
                        voxel_size * (z + ratio * int(i == 2));

                GetNormalAt(xv + int64_t(i == 0), yv + int64_t(i == 1),
                            zv + int64_t(i == 2), workload_block_idx, ni);

                float nx = (1 - ratio) * no[0] + ratio * ni[0];
                float ny = (1 - ratio) * no[1] + ratio * ni[1];
                float nz = (1 - ratio) * no[2] + ratio * ni[2];
                float norm = sqrt(nx * nx + ny * ny + nz * nz) + 1e-5;
                normals_ptr[idx * 3 + 0] = nx / norm;
                normals_ptr[idx * 3 + 1] = ny / norm;
                normals_ptr[idx * 3 + 2] = nz / norm;
            }
        }
    });

    int total_count = count.Item<int>();
    dsts.emplace("points", points.Slice(0, 0, total_count));
    dsts.emplace("normals", normals.Slice(0, 0, total_count));
}

void CPUMarchingCubesKernel(const std::unordered_map<std::string, Tensor>& srcs,
                            std::unordered_map<std::string, Tensor>& dsts) {
    // Decode input tensors
    static std::vector<std::string> src_attrs = {
            "indices",    "inv_indices",  "nb_indices", "nb_masks",
            "block_keys", "block_values", "voxel_size", "resolution",
    };
    for (auto& k : src_attrs) {
        if (srcs.count(k) == 0) {
            utility::LogError(
                    "[CUDAMarchingCubesKernel] expected Tensor {} in srcs, but "
                    "did not receive",
                    k);
        }
    }
    utility::LogInfo("surface extraction starts");

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
    int* block_keys_ptr = static_cast<int*>(block_keys.GetDataPtr());

    int64_t n = n_blocks * resolution3;

    // Pass 0: analyze mesh structure, set up one-on-one correspondences
    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
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
            int64_t workload_voxel_i;
            voxel_block_buffer_indexer.CoordToWorkload(
                    xv_i - dxb * resolution, yv_i - dyb * resolution,
                    zv_i - dzb * resolution, block_idx_i, &workload_voxel_i);
            float* voxel_ptr_i = static_cast<float*>(
                    voxel_block_buffer_indexer.GetDataPtrFromWorkload(
                            workload_voxel_i));

            float tsdf_i = voxel_ptr_i[0];
            float weight_i = voxel_ptr_i[1];
            if (weight_i == 0) return;

            table_idx |= ((tsdf_i < 0) ? (1 << i) : 0);
        }

        int64_t workload_mesh_struct_idx;
        mesh_structure_indexer.CoordToWorkload(xv, yv, zv, workload_block_idx,
                                               &workload_mesh_struct_idx);
        int* mesh_struct_ptr =
                static_cast<int*>(mesh_structure_indexer.GetDataPtrFromWorkload(
                        workload_mesh_struct_idx));
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
                int64_t workload_mesh_struct_i;
                mesh_structure_indexer.CoordToWorkload(
                        xv_i - dxb * resolution, yv_i - dyb * resolution,
                        zv_i - dzb * resolution, inv_indices_ptr[block_idx_i],
                        &workload_mesh_struct_i);
                if (indices_ptr[inv_indices_ptr[block_idx_i]] != block_idx_i) {
                    printf("inv indices error!\n");
                }
                int* mesh_struct_ptr_i = static_cast<int*>(
                        mesh_structure_indexer.GetDataPtrFromWorkload(
                                workload_mesh_struct_i));

                // Non-atomic write, but we are safe
                mesh_struct_ptr_i[edge_i] = -1;
            }
        }
    });

    // Pass 1: allocate and assign vertices with normals
    core::Tensor vtx_count(std::vector<int>{0}, {}, core::Dtype::Int32,
                           block_values.GetDevice());
    core::Tensor vertices({std::min(n * 3, int64_t(10000000)), 3},
                          core::Dtype::Float32, block_values.GetDevice());
    core::Tensor normals({std::min(n * 3, int64_t(10000000)), 3},
                         core::Dtype::Float32, block_values.GetDevice());
    int* vtx_count_ptr = static_cast<int*>(vtx_count.GetDataPtr());
    float* vertices_ptr = static_cast<float*>(vertices.GetDataPtr());
    float* normals_ptr = static_cast<float*>(normals.GetDataPtr());

    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
        auto GetVoxelAt = [&](int xo, int yo, int zo,
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
            int64_t workload_voxel_i;
            voxel_block_buffer_indexer.CoordToWorkload(xn, yn, zn, block_idx_i,
                                                       &workload_voxel_i);
            return static_cast<float*>(
                    voxel_block_buffer_indexer.GetDataPtrFromWorkload(
                            workload_voxel_i));
        };

        auto GetNormalAt = [&](int xo, int yo, int zo, int curr_block_idx,
                               float* n) {
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
        int64_t xb = static_cast<int64_t>(block_keys_ptr[block_idx * 3 + 0]);
        int64_t yb = static_cast<int64_t>(block_keys_ptr[block_idx * 3 + 1]);
        int64_t zb = static_cast<int64_t>(block_keys_ptr[block_idx * 3 + 2]);

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        int64_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        // global coordinate (in voxels)
        int64_t x = xb * resolution + xv;
        int64_t y = yb * resolution + yv;
        int64_t z = zb * resolution + zv;

        // Obtain voxel's mesh struct ptr
        int64_t workload_mesh_struct_idx;
        mesh_structure_indexer.CoordToWorkload(xv, yv, zv, workload_block_idx,
                                               &workload_mesh_struct_idx);
        int* mesh_struct_ptr =
                static_cast<int*>(mesh_structure_indexer.GetDataPtrFromWorkload(
                        workload_mesh_struct_idx));

        // Early quit -- no allocated vertex to compute
        if (mesh_struct_ptr[0] != -1 && mesh_struct_ptr[1] != -1 &&
            mesh_struct_ptr[2] != -1) {
            return;
        }

        // Obtain voxel ptr
        int64_t workload_voxel_idx;
        voxel_block_buffer_indexer.CoordToWorkload(xv, yv, zv, block_idx,
                                                   &workload_voxel_idx);
        float* voxel_ptr = static_cast<float*>(
                voxel_block_buffer_indexer.GetDataPtrFromWorkload(
                        workload_voxel_idx));
        float tsdf_o = voxel_ptr[0];
        if (voxel_ptr[1] == 0) {
            printf("voxel weight error!\n");
        }

        float no[3] = {0}, ne[3] = {0};
        GetNormalAt(xv, yv, zv, workload_block_idx, no);

        // Enumerate 3 edges in the voxel
        for (int e = 0; e < 3; ++e) {
            int vertex_idx = mesh_struct_ptr[e];
            if (vertex_idx != -1) continue;

            float* voxel_ptr_e =
                    GetVoxelAt(xv + int(e == 0), yv + int(e == 1),
                               zv + int(e == 2), workload_block_idx);
            if (voxel_ptr_e == nullptr) {
                printf("ptr_e error!\n");
            }
            float tsdf_e = voxel_ptr_e[0];
            if (tsdf_e * tsdf_o > 0) {
                printf("tsdf error: %f %f\n", tsdf_e, tsdf_o);
                return;
            }

            float ratio = (0 - tsdf_o) / (tsdf_e - tsdf_o);

            int idx;
#pragma omp atomic capture
            {
                idx = *vtx_count_ptr;
                *vtx_count_ptr += 1;
            }
            mesh_struct_ptr[e] = idx;
            /// printf("%d\n", idx);

            float ratio_x = ratio * int(e == 0);
            float ratio_y = ratio * int(e == 1);
            float ratio_z = ratio * int(e == 2);

            vertices_ptr[3 * idx + 0] = voxel_size * (x + ratio_x);
            vertices_ptr[3 * idx + 1] = voxel_size * (y + ratio_y);
            vertices_ptr[3 * idx + 2] = voxel_size * (z + ratio_z);

            GetNormalAt(xv + int(e == 0), yv + int(e == 1), zv + int(e == 2),
                        workload_block_idx, ne);
            float nx = (1 - ratio) * no[0] + ratio * ne[0];
            float ny = (1 - ratio) * no[1] + ratio * ne[1];
            float nz = (1 - ratio) * no[2] + ratio * ne[2];
            float norm = sqrt(nx * nx + ny * ny + nz * nz) + 1e-5;
            normals_ptr[idx * 3 + 0] = nx / norm;
            normals_ptr[idx * 3 + 1] = ny / norm;
            normals_ptr[idx * 3 + 2] = nz / norm;
        }
    });

    int total_vtx_count = vtx_count.Item<int>();
    utility::LogInfo("Total vertex count = {}", total_vtx_count);
    vertices = vertices.Slice(0, 0, total_vtx_count);
    normals = normals.Slice(0, 0, total_vtx_count);
    dsts.emplace("vertices", vertices);
    dsts.emplace("normals", normals);

    // Pass 2: connect vertices
    core::Tensor triangle_count(std::vector<int>{0}, {}, core::Dtype::Int32,
                                block_values.GetDevice());
    core::Tensor triangles({total_vtx_count * 3, 3}, core::Dtype::Int64,
                           block_values.GetDevice());
    int* tri_count_ptr = static_cast<int*>(triangle_count.GetDataPtr());
    int64_t* triangles_ptr = static_cast<int64_t*>(triangles.GetDataPtr());

    CPULauncher::LaunchGeneralKernel(n, [&](int64_t workload_idx) {
        // Natural index (0, N) -> (block_idx, voxel_idx)
        int64_t workload_block_idx = workload_idx / resolution3;
        int64_t voxel_idx = workload_idx % resolution3;

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        int64_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        // Obtain voxel's mesh struct ptr
        int64_t workload_mesh_struct_idx;
        mesh_structure_indexer.CoordToWorkload(xv, yv, zv, workload_block_idx,
                                               &workload_mesh_struct_idx);
        int* mesh_struct_ptr =
                static_cast<int*>(mesh_structure_indexer.GetDataPtrFromWorkload(
                        workload_mesh_struct_idx));

        int table_idx = mesh_struct_ptr[3];
        if (tri_count[table_idx] == 0) return;

        for (size_t tri = 0; tri < 16; tri += 3) {
            if (tri_table[table_idx][tri] == -1) return;

            int tri_idx;
#pragma omp atomic capture
            {
                tri_idx = *tri_count_ptr;
                *tri_count_ptr += 1;
            }

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
                int64_t workload_mesh_struct_i;
                mesh_structure_indexer.CoordToWorkload(
                        xv_i - dxb * resolution, yv_i - dyb * resolution,
                        zv_i - dzb * resolution, inv_indices_ptr[block_idx_i],
                        &workload_mesh_struct_i);
                if (indices_ptr[inv_indices_ptr[block_idx_i]] != block_idx_i) {
                    printf("inv indices error!\n");
                }
                int* mesh_struct_ptr_i = static_cast<int*>(
                        mesh_structure_indexer.GetDataPtrFromWorkload(
                                workload_mesh_struct_i));

                if (mesh_struct_ptr_i[edge_i] < 0) {
                    printf("triangle: mesh struct error");
                }
                triangles_ptr[3 * tri_idx + 2 - vertex] =
                        mesh_struct_ptr_i[edge_i];
            }
        }
    });

    int total_tri_count = triangle_count.Item<int>();
    utility::LogInfo("Total triangle count = {}", total_tri_count);
    triangles = triangles.Slice(0, 0, total_tri_count);
    dsts.emplace("triangles", triangles);
}

void GeneralEWCPU(const std::unordered_map<std::string, Tensor>& srcs,
                  std::unordered_map<std::string, Tensor>& dsts,
                  GeneralEWOpCode op_code) {
    switch (op_code) {
        case GeneralEWOpCode::Unproject:
            CPUUnprojectKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::TSDFTouch:
            CPUTSDFTouchKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::TSDFIntegrate:
            CPUTSDFIntegrateKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::TSDFSurfaceExtraction:
            CPUSurfaceExtractionKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::MarchingCubes:
            CPUMarchingCubesKernel(srcs, dsts);
            break;
        case GeneralEWOpCode::RayCasting:
            break;
        case GeneralEWOpCode::Debug: {
            int64_t n = 10;
            CPULauncher::LaunchGeneralKernel(n, [=](int64_t workload_idx) {});
            break;
        }
        default:
            break;
    }
}

}  // namespace kernel
}  // namespace core
}  // namespace open3d
