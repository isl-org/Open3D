// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
#include <cmath>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/Dispatch.h"
#include "open3d/t/geometry/Utility.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/VoxelBlockGrid.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Timer.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace voxel_grid {

using index_t = int;
using ArrayIndexer = TArrayIndexer<index_t>;

#if defined(__CUDACC__)
void GetVoxelCoordinatesAndFlattenedIndicesCUDA
#else
void GetVoxelCoordinatesAndFlattenedIndicesCPU
#endif
        (const core::Tensor& buf_indices,
         const core::Tensor& block_keys,
         core::Tensor& voxel_coords,
         core::Tensor& flattened_indices,
         index_t resolution,
         float voxel_size) {
    core::Device device = buf_indices.GetDevice();

    const index_t* buf_indices_ptr = buf_indices.GetDataPtr<index_t>();
    const index_t* block_key_ptr = block_keys.GetDataPtr<index_t>();

    float* voxel_coords_ptr = voxel_coords.GetDataPtr<float>();
    int64_t* flattened_indices_ptr = flattened_indices.GetDataPtr<int64_t>();

    index_t n = flattened_indices.GetLength();
    ArrayIndexer voxel_indexer({resolution, resolution, resolution});
    index_t resolution3 = resolution * resolution * resolution;

    core::ParallelFor(device, n, [=] OPEN3D_DEVICE(index_t workload_idx) {
        index_t block_idx = buf_indices_ptr[workload_idx / resolution3];
        index_t voxel_idx = workload_idx % resolution3;

        index_t block_key_offset = block_idx * 3;
        index_t xb = block_key_ptr[block_key_offset + 0];
        index_t yb = block_key_ptr[block_key_offset + 1];
        index_t zb = block_key_ptr[block_key_offset + 2];

        index_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        float x = (xb * resolution + xv) * voxel_size;
        float y = (yb * resolution + yv) * voxel_size;
        float z = (zb * resolution + zv) * voxel_size;

        flattened_indices_ptr[workload_idx] =
                block_idx * resolution3 + voxel_idx;

        index_t voxel_coords_offset = workload_idx * 3;
        voxel_coords_ptr[voxel_coords_offset + 0] = x;
        voxel_coords_ptr[voxel_coords_offset + 1] = y;
        voxel_coords_ptr[voxel_coords_offset + 2] = z;
    });
}

inline OPEN3D_DEVICE index_t
DeviceGetLinearIdx(index_t xo,
                   index_t yo,
                   index_t zo,
                   index_t curr_block_idx,
                   index_t resolution,
                   const ArrayIndexer& nb_block_masks_indexer,
                   const ArrayIndexer& nb_block_indices_indexer) {
    index_t xn = (xo + resolution) % resolution;
    index_t yn = (yo + resolution) % resolution;
    index_t zn = (zo + resolution) % resolution;

    index_t dxb = Sign(xo - xn);
    index_t dyb = Sign(yo - yn);
    index_t dzb = Sign(zo - zn);

    index_t nb_idx = (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

    bool block_mask_i =
            *nb_block_masks_indexer.GetDataPtr<bool>(curr_block_idx, nb_idx);
    if (!block_mask_i) return -1;

    index_t block_idx_i = *nb_block_indices_indexer.GetDataPtr<index_t>(
            curr_block_idx, nb_idx);

    return (((block_idx_i * resolution) + zn) * resolution + yn) * resolution +
           xn;
}

template <typename tsdf_t>
inline OPEN3D_DEVICE void DeviceGetNormal(
        const tsdf_t* tsdf_base_ptr,
        index_t xo,
        index_t yo,
        index_t zo,
        index_t curr_block_idx,
        float* n,
        index_t resolution,
        const ArrayIndexer& nb_block_masks_indexer,
        const ArrayIndexer& nb_block_indices_indexer) {
    auto GetLinearIdx = [&] OPEN3D_DEVICE(index_t xo, index_t yo,
                                          index_t zo) -> index_t {
        return DeviceGetLinearIdx(xo, yo, zo, curr_block_idx, resolution,
                                  nb_block_masks_indexer,
                                  nb_block_indices_indexer);
    };
    index_t vxp = GetLinearIdx(xo + 1, yo, zo);
    index_t vxn = GetLinearIdx(xo - 1, yo, zo);
    index_t vyp = GetLinearIdx(xo, yo + 1, zo);
    index_t vyn = GetLinearIdx(xo, yo - 1, zo);
    index_t vzp = GetLinearIdx(xo, yo, zo + 1);
    index_t vzn = GetLinearIdx(xo, yo, zo - 1);
    if (vxp >= 0 && vxn >= 0) n[0] = tsdf_base_ptr[vxp] - tsdf_base_ptr[vxn];
    if (vyp >= 0 && vyn >= 0) n[1] = tsdf_base_ptr[vyp] - tsdf_base_ptr[vyn];
    if (vzp >= 0 && vzn >= 0) n[2] = tsdf_base_ptr[vzp] - tsdf_base_ptr[vzn];
};

template <typename input_depth_t,
          typename input_color_t,
          typename tsdf_t,
          typename weight_t,
          typename color_t>
#if defined(__CUDACC__)
void IntegrateCUDA
#else
void IntegrateCPU
#endif
        (const core::Tensor& depth,
         const core::Tensor& color,
         const core::Tensor& indices,
         const core::Tensor& block_keys,
         TensorMap& block_value_map,
         const core::Tensor& depth_intrinsic,
         const core::Tensor& color_intrinsic,
         const core::Tensor& extrinsics,
         index_t resolution,
         float voxel_size,
         float sdf_trunc,
         float depth_scale,
         float depth_max) {
    // Parameters
    index_t resolution2 = resolution * resolution;
    index_t resolution3 = resolution2 * resolution;

    TransformIndexer transform_indexer(depth_intrinsic, extrinsics, voxel_size);
    TransformIndexer colormap_indexer(
            color_intrinsic,
            core::Tensor::Eye(4, core::Dtype::Float64, core::Device("CPU:0")));

    ArrayIndexer voxel_indexer({resolution, resolution, resolution});

    ArrayIndexer block_keys_indexer(block_keys, 1);
    ArrayIndexer depth_indexer(depth, 2);
    core::Device device = block_keys.GetDevice();

    const index_t* indices_ptr = indices.GetDataPtr<index_t>();

    if (!block_value_map.Contains("tsdf") ||
        !block_value_map.Contains("weight")) {
        utility::LogError(
                "TSDF and/or weight not allocated in blocks, please implement "
                "customized integration.");
    }
    tsdf_t* tsdf_base_ptr = block_value_map.at("tsdf").GetDataPtr<tsdf_t>();
    weight_t* weight_base_ptr =
            block_value_map.at("weight").GetDataPtr<weight_t>();

    bool integrate_color =
            block_value_map.Contains("color") && color.NumElements() > 0;
    color_t* color_base_ptr = nullptr;
    ArrayIndexer color_indexer;

    float color_multiplier = 1.0;
    if (integrate_color) {
        color_base_ptr = block_value_map.at("color").GetDataPtr<color_t>();
        color_indexer = ArrayIndexer(color, 2);

        // Float32: [0, 1] -> [0, 255]
        if (color.GetDtype() == core::Float32) {
            color_multiplier = 255.0;
        }
    }

    index_t n = indices.GetLength() * resolution3;
    core::ParallelFor(device, n, [=] OPEN3D_DEVICE(index_t workload_idx) {
        // Natural index (0, N) -> (block_idx, voxel_idx)
        index_t block_idx = indices_ptr[workload_idx / resolution3];
        index_t voxel_idx = workload_idx % resolution3;

        /// Coordinate transform
        // block_idx -> (x_block, y_block, z_block)
        index_t* block_key_ptr =
                block_keys_indexer.GetDataPtr<index_t>(block_idx);
        index_t xb = block_key_ptr[0];
        index_t yb = block_key_ptr[1];
        index_t zb = block_key_ptr[2];

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        index_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        // coordinate in world (in voxel)
        index_t x = xb * resolution + xv;
        index_t y = yb * resolution + yv;
        index_t z = zb * resolution + zv;

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

        index_t ui = static_cast<index_t>(u);
        index_t vi = static_cast<index_t>(v);

        // Associate image workload and compute SDF and
        // TSDF.
        float depth =
                *depth_indexer.GetDataPtr<input_depth_t>(ui, vi) / depth_scale;

        float sdf = depth - zc;
        if (depth <= 0 || depth > depth_max || zc <= 0 || sdf < -sdf_trunc) {
            return;
        }
        sdf = sdf < sdf_trunc ? sdf : sdf_trunc;
        sdf /= sdf_trunc;

        index_t linear_idx = block_idx * resolution3 + voxel_idx;

        tsdf_t* tsdf_ptr = tsdf_base_ptr + linear_idx;
        weight_t* weight_ptr = weight_base_ptr + linear_idx;

        float inv_wsum = 1.0f / (*weight_ptr + 1);
        float weight = *weight_ptr;
        *tsdf_ptr = (weight * (*tsdf_ptr) + sdf) * inv_wsum;

        if (integrate_color) {
            color_t* color_ptr = color_base_ptr + 3 * linear_idx;

            // Unproject ui, vi with depth_intrinsic, then project back with
            // color_intrinsic
            float x, y, z;
            transform_indexer.Unproject(ui, vi, 1.0, &x, &y, &z);

            float uf, vf;
            colormap_indexer.Project(x, y, z, &uf, &vf);
            if (color_indexer.InBoundary(uf, vf)) {
                ui = round(uf);
                vi = round(vf);

                input_color_t* input_color_ptr =
                        color_indexer.GetDataPtr<input_color_t>(ui, vi);

                for (index_t i = 0; i < 3; ++i) {
                    color_ptr[i] = (weight * color_ptr[i] +
                                    input_color_ptr[i] * color_multiplier) *
                                   inv_wsum;
                }
            }
        }
        *weight_ptr = weight + 1;
    });

#if defined(__CUDACC__)
    core::cuda::Synchronize();
#endif
}

#if defined(__CUDACC__)
void EstimateRangeCUDA
#else
void EstimateRangeCPU
#endif
        (const core::Tensor& block_keys,
         core::Tensor& range_minmax_map,
         const core::Tensor& intrinsics,
         const core::Tensor& extrinsics,
         int h,
         int w,
         int down_factor,
         int64_t block_resolution,
         float voxel_size,
         float depth_min,
         float depth_max) {

    // TODO(wei): reserve it in a reusable buffer

    // Every 2 channels: (min, max)
    int h_down = h / down_factor;
    int w_down = w / down_factor;
    range_minmax_map = core::Tensor({h_down, w_down, 2}, core::Float32,
                                    block_keys.GetDevice());
    NDArrayIndexer range_map_indexer(range_minmax_map, 2);

    // Every 6 channels: (v_min, u_min, v_max, u_max, z_min, z_max)
    const int fragment_size = 16;
    const int frag_buffer_size = 65535;

    // TODO(wei): explicit buffer
    core::Tensor fragment_buffer = core::Tensor(
            {frag_buffer_size, 6}, core::Float32, block_keys.GetDevice());

    NDArrayIndexer frag_buffer_indexer(fragment_buffer, 1);
    NDArrayIndexer block_keys_indexer(block_keys, 1);
    TransformIndexer w2c_transform_indexer(intrinsics, extrinsics);
#if defined(__CUDACC__)
    core::Tensor count(std::vector<int>{0}, {1}, core::Int32,
                       block_keys.GetDevice());
    int* count_ptr = count.GetDataPtr<int>();
#else
    std::atomic<int> count_atomic(0);
    std::atomic<int>* count_ptr = &count_atomic;
#endif

#ifndef __CUDACC__
    using std::max;
    using std::min;
#endif

    // Pass 0: iterate over blocks, fill-in an rendering fragment array
    core::ParallelFor(
            block_keys.GetDevice(), block_keys.GetLength(),
            [=] OPEN3D_DEVICE(int64_t workload_idx) {
                int* key = block_keys_indexer.GetDataPtr<int>(workload_idx);

                int u_min = w_down - 1, v_min = h_down - 1, u_max = 0,
                    v_max = 0;
                float z_min = depth_max, z_max = depth_min;

                float xc, yc, zc, u, v;

                // Project 8 corners to low-res image and form a rectangle
                for (int i = 0; i < 8; ++i) {
                    float xw = (key[0] + ((i & 1) > 0)) * block_resolution *
                               voxel_size;
                    float yw = (key[1] + ((i & 2) > 0)) * block_resolution *
                               voxel_size;
                    float zw = (key[2] + ((i & 4) > 0)) * block_resolution *
                               voxel_size;

                    w2c_transform_indexer.RigidTransform(xw, yw, zw, &xc, &yc,
                                                         &zc);
                    if (zc <= 0) continue;

                    // Project to the down sampled image buffer
                    w2c_transform_indexer.Project(xc, yc, zc, &u, &v);
                    u /= down_factor;
                    v /= down_factor;

                    v_min = min(static_cast<int>(floorf(v)), v_min);
                    v_max = max(static_cast<int>(ceilf(v)), v_max);

                    u_min = min(static_cast<int>(floorf(u)), u_min);
                    u_max = max(static_cast<int>(ceilf(u)), u_max);

                    z_min = min(z_min, zc);
                    z_max = max(z_max, zc);
                }

                v_min = max(0, v_min);
                v_max = min(h_down - 1, v_max);

                u_min = max(0, u_min);
                u_max = min(w_down - 1, u_max);

                if (v_min >= v_max || u_min >= u_max || z_min >= z_max) return;

                // Divide the rectangle into small 16x16 fragments
                int frag_v_count =
                        ceilf(float(v_max - v_min + 1) / float(fragment_size));
                int frag_u_count =
                        ceilf(float(u_max - u_min + 1) / float(fragment_size));

                int frag_count = frag_v_count * frag_u_count;
                int frag_count_start = OPEN3D_ATOMIC_ADD(count_ptr, 1);
                int frag_count_end = frag_count_start + frag_count;
                if (frag_count_end >= frag_buffer_size) {
                    printf("Fragment count exceeding buffer size, abort!\n");
                }

                int offset = 0;
                for (int frag_v = 0; frag_v < frag_v_count; ++frag_v) {
                    for (int frag_u = 0; frag_u < frag_u_count;
                         ++frag_u, ++offset) {
                        float* frag_ptr = frag_buffer_indexer.GetDataPtr<float>(
                                frag_count_start + offset);
                        // zmin, zmax
                        frag_ptr[0] = z_min;
                        frag_ptr[1] = z_max;

                        // vmin, umin
                        frag_ptr[2] = v_min + frag_v * fragment_size;
                        frag_ptr[3] = u_min + frag_u * fragment_size;

                        // vmax, umax
                        frag_ptr[4] = min(frag_ptr[2] + fragment_size - 1,
                                          static_cast<float>(v_max));
                        frag_ptr[5] = min(frag_ptr[3] + fragment_size - 1,
                                          static_cast<float>(u_max));
                    }
                }
            });
#if defined(__CUDACC__)
    int frag_count = count[0].Item<int>();
#else
    int frag_count = (*count_ptr).load();
#endif

    // Pass 0.5: Fill in range map to prepare for atomic min/max
    core::ParallelFor(block_keys.GetDevice(), h_down * w_down,
                      [=] OPEN3D_DEVICE(int64_t workload_idx) {
                          int v = workload_idx / w_down;
                          int u = workload_idx % w_down;
                          float* range_ptr =
                                  range_map_indexer.GetDataPtr<float>(u, v);
                          range_ptr[0] = depth_max;
                          range_ptr[1] = depth_min;
                      });

    // Pass 1: iterate over rendering fragment array, fill-in range
    core::ParallelFor(
            block_keys.GetDevice(), frag_count * fragment_size * fragment_size,
            [=] OPEN3D_DEVICE(int64_t workload_idx) {
                int frag_idx = workload_idx / (fragment_size * fragment_size);
                int local_idx = workload_idx % (fragment_size * fragment_size);
                int dv = local_idx / fragment_size;
                int du = local_idx % fragment_size;

                float* frag_ptr =
                        frag_buffer_indexer.GetDataPtr<float>(frag_idx);
                int v_min = static_cast<int>(frag_ptr[2]);
                int u_min = static_cast<int>(frag_ptr[3]);
                int v_max = static_cast<int>(frag_ptr[4]);
                int u_max = static_cast<int>(frag_ptr[5]);

                int v = v_min + dv;
                int u = u_min + du;
                if (v > v_max || u > u_max) return;

                float z_min = frag_ptr[0];
                float z_max = frag_ptr[1];
                float* range_ptr = range_map_indexer.GetDataPtr<float>(u, v);
#ifdef __CUDACC__
                atomicMinf(&(range_ptr[0]), z_min);
                atomicMaxf(&(range_ptr[1]), z_max);
#else
#pragma omp critical(EstimateRangeCPU)
                {
                    range_ptr[0] = min(z_min, range_ptr[0]);
                    range_ptr[1] = max(z_max, range_ptr[1]);
                }
#endif
            });
#if defined(__CUDACC__)
    core::cuda::Synchronize();
#endif
}

struct MiniVecCache {
    index_t x;
    index_t y;
    index_t z;
    index_t block_idx;

    inline index_t OPEN3D_DEVICE Check(index_t xin, index_t yin, index_t zin) {
        return (xin == x && yin == y && zin == z) ? block_idx : -1;
    }

    inline void OPEN3D_DEVICE Update(index_t xin,
                                     index_t yin,
                                     index_t zin,
                                     index_t block_idx_in) {
        x = xin;
        y = yin;
        z = zin;
        block_idx = block_idx_in;
    }
};

template <typename tsdf_t, typename weight_t, typename color_t>
#if defined(__CUDACC__)
void RayCastCUDA
#else
void RayCastCPU
#endif
        (std::shared_ptr<core::HashMap>& hashmap,
         const TensorMap& block_value_map,
         const core::Tensor& range,
         TensorMap& renderings_map,
         const core::Tensor& intrinsic,
         const core::Tensor& extrinsics,
         index_t h,
         index_t w,
         index_t block_resolution,
         float voxel_size,
         float depth_scale,
         float depth_min,
         float depth_max,
         float weight_threshold,
         float trunc_voxel_multiplier,
         int range_map_down_factor) {
    using Key = utility::MiniVec<index_t, 3>;
    using Hash = utility::MiniVecHash<index_t, 3>;
    using Eq = utility::MiniVecEq<index_t, 3>;

    auto device_hashmap = hashmap->GetDeviceHashBackend();
#if defined(__CUDACC__)
    auto cuda_hashmap =
            std::dynamic_pointer_cast<core::StdGPUHashBackend<Key, Hash, Eq>>(
                    device_hashmap);
    if (cuda_hashmap == nullptr) {
        utility::LogError(
                "Unsupported backend: CUDA raycasting only supports STDGPU.");
    }
    auto hashmap_impl = cuda_hashmap->GetImpl();
#else
    auto cpu_hashmap =
            std::dynamic_pointer_cast<core::TBBHashBackend<Key, Hash, Eq>>(
                    device_hashmap);
    if (cpu_hashmap == nullptr) {
        utility::LogError(
                "Unsupported backend: CPU raycasting only supports TBB.");
    }
    auto hashmap_impl = *cpu_hashmap->GetImpl();
#endif

    core::Device device = hashmap->GetDevice();

    ArrayIndexer range_indexer(range, 2);

    // Geometry
    ArrayIndexer depth_indexer;
    ArrayIndexer vertex_indexer;
    ArrayIndexer normal_indexer;

    // Diff rendering
    ArrayIndexer index_indexer;
    ArrayIndexer mask_indexer;
    ArrayIndexer interp_ratio_indexer;
    ArrayIndexer interp_ratio_dx_indexer;
    ArrayIndexer interp_ratio_dy_indexer;
    ArrayIndexer interp_ratio_dz_indexer;

    // Color
    ArrayIndexer color_indexer;

    if (!block_value_map.Contains("tsdf") ||
        !block_value_map.Contains("weight")) {
        utility::LogError(
                "TSDF and/or weight not allocated in blocks, please implement "
                "customized integration.");
    }
    const tsdf_t* tsdf_base_ptr =
            block_value_map.at("tsdf").GetDataPtr<tsdf_t>();
    const weight_t* weight_base_ptr =
            block_value_map.at("weight").GetDataPtr<weight_t>();

    // Geometry
    if (renderings_map.Contains("depth")) {
        depth_indexer = ArrayIndexer(renderings_map.at("depth"), 2);
    }
    if (renderings_map.Contains("vertex")) {
        vertex_indexer = ArrayIndexer(renderings_map.at("vertex"), 2);
    }
    if (renderings_map.Contains("normal")) {
        normal_indexer = ArrayIndexer(renderings_map.at("normal"), 2);
    }

    // Diff rendering
    if (renderings_map.Contains("index")) {
        index_indexer = ArrayIndexer(renderings_map.at("index"), 2);
    }
    if (renderings_map.Contains("mask")) {
        mask_indexer = ArrayIndexer(renderings_map.at("mask"), 2);
    }
    if (renderings_map.Contains("interp_ratio")) {
        interp_ratio_indexer =
                ArrayIndexer(renderings_map.at("interp_ratio"), 2);
    }
    if (renderings_map.Contains("interp_ratio_dx")) {
        interp_ratio_dx_indexer =
                ArrayIndexer(renderings_map.at("interp_ratio_dx"), 2);
    }
    if (renderings_map.Contains("interp_ratio_dy")) {
        interp_ratio_dy_indexer =
                ArrayIndexer(renderings_map.at("interp_ratio_dy"), 2);
    }
    if (renderings_map.Contains("interp_ratio_dz")) {
        interp_ratio_dz_indexer =
                ArrayIndexer(renderings_map.at("interp_ratio_dz"), 2);
    }

    // Color
    bool render_color = false;
    if (block_value_map.Contains("color") && renderings_map.Contains("color")) {
        render_color = true;
        color_indexer = ArrayIndexer(renderings_map.at("color"), 2);
    }
    const color_t* color_base_ptr =
            render_color ? block_value_map.at("color").GetDataPtr<color_t>()
                         : nullptr;

    bool visit_neighbors = render_color || normal_indexer.GetDataPtr() ||
                           mask_indexer.GetDataPtr() ||
                           index_indexer.GetDataPtr() ||
                           interp_ratio_indexer.GetDataPtr() ||
                           interp_ratio_dx_indexer.GetDataPtr() ||
                           interp_ratio_dy_indexer.GetDataPtr() ||
                           interp_ratio_dz_indexer.GetDataPtr();

    TransformIndexer c2w_transform_indexer(
            intrinsic, t::geometry::InverseTransformation(extrinsics));
    TransformIndexer w2c_transform_indexer(intrinsic, extrinsics);

    index_t rows = h;
    index_t cols = w;
    index_t n = rows * cols;

    float block_size = voxel_size * block_resolution;
    index_t resolution2 = block_resolution * block_resolution;
    index_t resolution3 = resolution2 * block_resolution;

#ifndef __CUDACC__
    using std::max;
    using std::sqrt;
#endif

    core::ParallelFor(device, n, [=] OPEN3D_DEVICE(index_t workload_idx) {
        auto GetLinearIdxAtP = [&] OPEN3D_DEVICE(
                                       index_t x_b, index_t y_b, index_t z_b,
                                       index_t x_v, index_t y_v, index_t z_v,
                                       core::buf_index_t block_buf_idx,
                                       MiniVecCache & cache) -> index_t {
            index_t x_vn = (x_v + block_resolution) % block_resolution;
            index_t y_vn = (y_v + block_resolution) % block_resolution;
            index_t z_vn = (z_v + block_resolution) % block_resolution;

            index_t dx_b = Sign(x_v - x_vn);
            index_t dy_b = Sign(y_v - y_vn);
            index_t dz_b = Sign(z_v - z_vn);

            if (dx_b == 0 && dy_b == 0 && dz_b == 0) {
                return block_buf_idx * resolution3 + z_v * resolution2 +
                       y_v * block_resolution + x_v;
            } else {
                Key key(x_b + dx_b, y_b + dy_b, z_b + dz_b);

                index_t block_buf_idx = cache.Check(key[0], key[1], key[2]);
                if (block_buf_idx < 0) {
                    auto iter = hashmap_impl.find(key);
                    if (iter == hashmap_impl.end()) return -1;
                    block_buf_idx = iter->second;
                    cache.Update(key[0], key[1], key[2], block_buf_idx);
                }

                return block_buf_idx * resolution3 + z_vn * resolution2 +
                       y_vn * block_resolution + x_vn;
            }
        };

        auto GetLinearIdxAtT = [&] OPEN3D_DEVICE(
                                       float x_o, float y_o, float z_o,
                                       float x_d, float y_d, float z_d, float t,
                                       MiniVecCache& cache) -> index_t {
            float x_g = x_o + t * x_d;
            float y_g = y_o + t * y_d;
            float z_g = z_o + t * z_d;

            // MiniVec coordinate and look up
            index_t x_b = static_cast<index_t>(floorf(x_g / block_size));
            index_t y_b = static_cast<index_t>(floorf(y_g / block_size));
            index_t z_b = static_cast<index_t>(floorf(z_g / block_size));

            Key key(x_b, y_b, z_b);
            index_t block_buf_idx = cache.Check(x_b, y_b, z_b);
            if (block_buf_idx < 0) {
                auto iter = hashmap_impl.find(key);
                if (iter == hashmap_impl.end()) return -1;
                block_buf_idx = iter->second;
                cache.Update(x_b, y_b, z_b, block_buf_idx);
            }

            // Voxel coordinate and look up
            index_t x_v = index_t((x_g - x_b * block_size) / voxel_size);
            index_t y_v = index_t((y_g - y_b * block_size) / voxel_size);
            index_t z_v = index_t((z_g - z_b * block_size) / voxel_size);

            return block_buf_idx * resolution3 + z_v * resolution2 +
                   y_v * block_resolution + x_v;
        };

        index_t y = workload_idx / cols;
        index_t x = workload_idx % cols;

        const float* range = range_indexer.GetDataPtr<float>(
                x / range_map_down_factor, y / range_map_down_factor);

        float* depth_ptr = nullptr;
        float* vertex_ptr = nullptr;
        float* color_ptr = nullptr;
        float* normal_ptr = nullptr;

        int64_t* index_ptr = nullptr;
        bool* mask_ptr = nullptr;
        float* interp_ratio_ptr = nullptr;
        float* interp_ratio_dx_ptr = nullptr;
        float* interp_ratio_dy_ptr = nullptr;
        float* interp_ratio_dz_ptr = nullptr;

        if (vertex_indexer.GetDataPtr()) {
            vertex_ptr = vertex_indexer.GetDataPtr<float>(x, y);
            vertex_ptr[0] = 0;
            vertex_ptr[1] = 0;
            vertex_ptr[2] = 0;
        }
        if (depth_indexer.GetDataPtr()) {
            depth_ptr = depth_indexer.GetDataPtr<float>(x, y);
            depth_ptr[0] = 0;
        }
        if (normal_indexer.GetDataPtr()) {
            normal_ptr = normal_indexer.GetDataPtr<float>(x, y);
            normal_ptr[0] = 0;
            normal_ptr[1] = 0;
            normal_ptr[2] = 0;
        }

        if (mask_indexer.GetDataPtr()) {
            mask_ptr = mask_indexer.GetDataPtr<bool>(x, y);
#ifdef __CUDACC__
#pragma unroll
#endif
            for (int i = 0; i < 8; ++i) {
                mask_ptr[i] = false;
            }
        }
        if (index_indexer.GetDataPtr()) {
            index_ptr = index_indexer.GetDataPtr<int64_t>(x, y);
#ifdef __CUDACC__
#pragma unroll
#endif
            for (int i = 0; i < 8; ++i) {
                index_ptr[i] = 0;
            }
        }
        if (interp_ratio_indexer.GetDataPtr()) {
            interp_ratio_ptr = interp_ratio_indexer.GetDataPtr<float>(x, y);
#ifdef __CUDACC__
#pragma unroll
#endif
            for (int i = 0; i < 8; ++i) {
                interp_ratio_ptr[i] = 0;
            }
        }
        if (interp_ratio_dx_indexer.GetDataPtr()) {
            interp_ratio_dx_ptr =
                    interp_ratio_dx_indexer.GetDataPtr<float>(x, y);
#ifdef __CUDACC__
#pragma unroll
#endif
            for (int i = 0; i < 8; ++i) {
                interp_ratio_dx_ptr[i] = 0;
            }
        }
        if (interp_ratio_dy_indexer.GetDataPtr()) {
            interp_ratio_dy_ptr =
                    interp_ratio_dy_indexer.GetDataPtr<float>(x, y);
#ifdef __CUDACC__
#pragma unroll
#endif
            for (int i = 0; i < 8; ++i) {
                interp_ratio_dy_ptr[i] = 0;
            }
        }
        if (interp_ratio_dz_indexer.GetDataPtr()) {
            interp_ratio_dz_ptr =
                    interp_ratio_dz_indexer.GetDataPtr<float>(x, y);
#ifdef __CUDACC__
#pragma unroll
#endif
            for (int i = 0; i < 8; ++i) {
                interp_ratio_dz_ptr[i] = 0;
            }
        }

        if (color_indexer.GetDataPtr()) {
            color_ptr = color_indexer.GetDataPtr<float>(x, y);
            color_ptr[0] = 0;
            color_ptr[1] = 0;
            color_ptr[2] = 0;
        }

        float t = range[0];
        const float t_max = range[1];
        if (t >= t_max) return;

        // Coordinates in camera and global
        float x_c = 0, y_c = 0, z_c = 0;
        float x_g = 0, y_g = 0, z_g = 0;
        float x_o = 0, y_o = 0, z_o = 0;

        // Iterative ray intersection check
        float t_prev = t;

        float tsdf_prev = -1.0f;
        float tsdf = 1.0;
        float sdf_trunc = voxel_size * trunc_voxel_multiplier;
        float w = 0.0;

        // Camera origin
        c2w_transform_indexer.RigidTransform(0, 0, 0, &x_o, &y_o, &z_o);

        // Direction
        c2w_transform_indexer.Unproject(static_cast<float>(x),
                                        static_cast<float>(y), 1.0f, &x_c, &y_c,
                                        &z_c);
        c2w_transform_indexer.RigidTransform(x_c, y_c, z_c, &x_g, &y_g, &z_g);
        float x_d = (x_g - x_o);
        float y_d = (y_g - y_o);
        float z_d = (z_g - z_o);

        MiniVecCache cache{0, 0, 0, -1};
        bool surface_found = false;
        while (t < t_max) {
            index_t linear_idx =
                    GetLinearIdxAtT(x_o, y_o, z_o, x_d, y_d, z_d, t, cache);

            if (linear_idx < 0) {
                t_prev = t;
                t += block_size;
            } else {
                tsdf_prev = tsdf;
                tsdf = tsdf_base_ptr[linear_idx];
                w = weight_base_ptr[linear_idx];
                if (tsdf_prev > 0 && w >= weight_threshold && tsdf <= 0) {
                    surface_found = true;
                    break;
                }
                t_prev = t;
                float delta = tsdf * sdf_trunc;
                t += delta < voxel_size ? voxel_size : delta;
            }
        }

        if (surface_found) {
            float t_intersect =
                    (t * tsdf_prev - t_prev * tsdf) / (tsdf_prev - tsdf);
            x_g = x_o + t_intersect * x_d;
            y_g = y_o + t_intersect * y_d;
            z_g = z_o + t_intersect * z_d;

            // Trivial vertex assignment
            if (depth_ptr) {
                *depth_ptr = t_intersect * depth_scale;
            }
            if (vertex_ptr) {
                w2c_transform_indexer.RigidTransform(
                        x_g, y_g, z_g, vertex_ptr + 0, vertex_ptr + 1,
                        vertex_ptr + 2);
            }
            if (!visit_neighbors) return;

            // Trilinear interpolation
            // TODO(wei): simplify the flow by splitting the
            // functions given what is enabled
            index_t x_b = static_cast<index_t>(floorf(x_g / block_size));
            index_t y_b = static_cast<index_t>(floorf(y_g / block_size));
            index_t z_b = static_cast<index_t>(floorf(z_g / block_size));
            float x_v = (x_g - float(x_b) * block_size) / voxel_size;
            float y_v = (y_g - float(y_b) * block_size) / voxel_size;
            float z_v = (z_g - float(z_b) * block_size) / voxel_size;

            Key key(x_b, y_b, z_b);

            index_t block_buf_idx = cache.Check(x_b, y_b, z_b);
            if (block_buf_idx < 0) {
                auto iter = hashmap_impl.find(key);
                if (iter == hashmap_impl.end()) return;
                block_buf_idx = iter->second;
                cache.Update(x_b, y_b, z_b, block_buf_idx);
            }

            index_t x_v_floor = static_cast<index_t>(floorf(x_v));
            index_t y_v_floor = static_cast<index_t>(floorf(y_v));
            index_t z_v_floor = static_cast<index_t>(floorf(z_v));

            float ratio_x = x_v - float(x_v_floor);
            float ratio_y = y_v - float(y_v_floor);
            float ratio_z = z_v - float(z_v_floor);

            float sum_r = 0.0;
            for (index_t k = 0; k < 8; ++k) {
                index_t dx_v = (k & 1) > 0 ? 1 : 0;
                index_t dy_v = (k & 2) > 0 ? 1 : 0;
                index_t dz_v = (k & 4) > 0 ? 1 : 0;

                index_t linear_idx_k = GetLinearIdxAtP(
                        x_b, y_b, z_b, x_v_floor + dx_v, y_v_floor + dy_v,
                        z_v_floor + dz_v, block_buf_idx, cache);

                if (linear_idx_k >= 0 && weight_base_ptr[linear_idx_k] > 0) {
                    float rx = dx_v * (ratio_x) + (1 - dx_v) * (1 - ratio_x);
                    float ry = dy_v * (ratio_y) + (1 - dy_v) * (1 - ratio_y);
                    float rz = dz_v * (ratio_z) + (1 - dz_v) * (1 - ratio_z);
                    float r = rx * ry * rz;

                    if (interp_ratio_ptr) {
                        interp_ratio_ptr[k] = r;
                    }
                    if (mask_ptr) {
                        mask_ptr[k] = true;
                    }
                    if (index_ptr) {
                        index_ptr[k] = linear_idx_k;
                    }

                    float tsdf_k = tsdf_base_ptr[linear_idx_k];
                    float interp_ratio_dx = ry * rz * (2 * dx_v - 1);
                    float interp_ratio_dy = rx * rz * (2 * dy_v - 1);
                    float interp_ratio_dz = rx * ry * (2 * dz_v - 1);

                    if (interp_ratio_dx_ptr) {
                        interp_ratio_dx_ptr[k] = interp_ratio_dx;
                    }
                    if (interp_ratio_dy_ptr) {
                        interp_ratio_dy_ptr[k] = interp_ratio_dy;
                    }
                    if (interp_ratio_dz_ptr) {
                        interp_ratio_dz_ptr[k] = interp_ratio_dz;
                    }

                    if (normal_ptr) {
                        normal_ptr[0] += interp_ratio_dx * tsdf_k;
                        normal_ptr[1] += interp_ratio_dy * tsdf_k;
                        normal_ptr[2] += interp_ratio_dz * tsdf_k;
                    }

                    if (color_ptr) {
                        index_t color_linear_idx = linear_idx_k * 3;
                        color_ptr[0] +=
                                r * color_base_ptr[color_linear_idx + 0];
                        color_ptr[1] +=
                                r * color_base_ptr[color_linear_idx + 1];
                        color_ptr[2] +=
                                r * color_base_ptr[color_linear_idx + 2];
                    }

                    sum_r += r;
                }
            }  // loop over 8 neighbors

            if (sum_r > 0) {
                sum_r *= 255.0;
                if (color_ptr) {
                    color_ptr[0] /= sum_r;
                    color_ptr[1] /= sum_r;
                    color_ptr[2] /= sum_r;
                }

                if (normal_ptr) {
                    constexpr float EPSILON = 1e-5f;
                    float norm = sqrt(normal_ptr[0] * normal_ptr[0] +
                                      normal_ptr[1] * normal_ptr[1] +
                                      normal_ptr[2] * normal_ptr[2]);
                    norm = std::max(norm, EPSILON);
                    w2c_transform_indexer.Rotate(
                            -normal_ptr[0] / norm, -normal_ptr[1] / norm,
                            -normal_ptr[2] / norm, normal_ptr + 0,
                            normal_ptr + 1, normal_ptr + 2);
                }
            }
        }  // surface-found
    });

#if defined(__CUDACC__)
    core::cuda::Synchronize();
#endif
}

template <typename tsdf_t, typename weight_t, typename color_t>
#if defined(__CUDACC__)
void ExtractPointCloudCUDA
#else
void ExtractPointCloudCPU
#endif
        (const core::Tensor& indices,
         const core::Tensor& nb_indices,
         const core::Tensor& nb_masks,
         const core::Tensor& block_keys,
         const TensorMap& block_value_map,
         core::Tensor& points,
         core::Tensor& normals,
         core::Tensor& colors,
         index_t resolution,
         float voxel_size,
         float weight_threshold,
         int& valid_size) {
    core::Device device = block_keys.GetDevice();

    // Parameters
    index_t resolution2 = resolution * resolution;
    index_t resolution3 = resolution2 * resolution;

    // Shape / transform indexers, no data involved
    ArrayIndexer voxel_indexer({resolution, resolution, resolution});

    // Real data indexer
    ArrayIndexer block_keys_indexer(block_keys, 1);
    ArrayIndexer nb_block_masks_indexer(nb_masks, 2);
    ArrayIndexer nb_block_indices_indexer(nb_indices, 2);

    // Plain arrays that does not require indexers
    const index_t* indices_ptr = indices.GetDataPtr<index_t>();

    if (!block_value_map.Contains("tsdf") ||
        !block_value_map.Contains("weight")) {
        utility::LogError(
                "TSDF and/or weight not allocated in blocks, please implement "
                "customized integration.");
    }
    const tsdf_t* tsdf_base_ptr =
            block_value_map.at("tsdf").GetDataPtr<tsdf_t>();
    const weight_t* weight_base_ptr =
            block_value_map.at("weight").GetDataPtr<weight_t>();
    const color_t* color_base_ptr = nullptr;
    if (block_value_map.Contains("color")) {
        color_base_ptr = block_value_map.at("color").GetDataPtr<color_t>();
    }

    index_t n_blocks = indices.GetLength();
    index_t n = n_blocks * resolution3;

    // Output
#if defined(__CUDACC__)
    core::Tensor count(std::vector<index_t>{0}, {1}, core::Int32,
                       block_keys.GetDevice());
    index_t* count_ptr = count.GetDataPtr<index_t>();
#else
    std::atomic<index_t> count_atomic(0);
    std::atomic<index_t>* count_ptr = &count_atomic;
#endif

    if (valid_size < 0) {
        utility::LogDebug(
                "No estimated max point cloud size provided, using a 2-pass "
                "estimation. Surface extraction could be slow.");
        // This pass determines valid number of points.

        core::ParallelFor(device, n, [=] OPEN3D_DEVICE(index_t workload_idx) {
            auto GetLinearIdx = [&] OPEN3D_DEVICE(
                                        index_t xo, index_t yo, index_t zo,
                                        index_t curr_block_idx) -> index_t {
                return DeviceGetLinearIdx(xo, yo, zo, curr_block_idx,
                                          resolution, nb_block_masks_indexer,
                                          nb_block_indices_indexer);
            };

            // Natural index (0, N) -> (block_idx,
            // voxel_idx)
            index_t workload_block_idx = workload_idx / resolution3;
            index_t block_idx = indices_ptr[workload_block_idx];
            index_t voxel_idx = workload_idx % resolution3;

            // voxel_idx -> (x_voxel, y_voxel, z_voxel)
            index_t xv, yv, zv;
            voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

            index_t linear_idx = block_idx * resolution3 + voxel_idx;
            float tsdf_o = tsdf_base_ptr[linear_idx];
            float weight_o = weight_base_ptr[linear_idx];
            if (weight_o <= weight_threshold) return;

            // Enumerate x-y-z directions
            for (index_t i = 0; i < 3; ++i) {
                index_t linear_idx_i =
                        GetLinearIdx(xv + (i == 0), yv + (i == 1),
                                     zv + (i == 2), workload_block_idx);
                if (linear_idx_i < 0) continue;

                float tsdf_i = tsdf_base_ptr[linear_idx_i];
                float weight_i = weight_base_ptr[linear_idx_i];
                if (weight_i > weight_threshold && tsdf_i * tsdf_o < 0) {
                    OPEN3D_ATOMIC_ADD(count_ptr, 1);
                }
            }
        });

#if defined(__CUDACC__)
        valid_size = count[0].Item<index_t>();
        count[0] = 0;
#else
        valid_size = (*count_ptr).load();
        (*count_ptr) = 0;
#endif
    }

    if (points.GetLength() == 0) {
        points = core::Tensor({valid_size, 3}, core::Float32, device);
    }
    ArrayIndexer point_indexer(points, 1);

    // Normals
    ArrayIndexer normal_indexer;
    normals = core::Tensor({valid_size, 3}, core::Float32, device);
    normal_indexer = ArrayIndexer(normals, 1);

    // This pass extracts exact surface points.

    // Colors
    ArrayIndexer color_indexer;
    if (color_base_ptr) {
        colors = core::Tensor({valid_size, 3}, core::Float32, device);
        color_indexer = ArrayIndexer(colors, 1);
    }

    core::ParallelFor(device, n, [=] OPEN3D_DEVICE(index_t workload_idx) {
        auto GetLinearIdx = [&] OPEN3D_DEVICE(
                                    index_t xo, index_t yo, index_t zo,
                                    index_t curr_block_idx) -> index_t {
            return DeviceGetLinearIdx(xo, yo, zo, curr_block_idx, resolution,
                                      nb_block_masks_indexer,
                                      nb_block_indices_indexer);
        };

        auto GetNormal = [&] OPEN3D_DEVICE(index_t xo, index_t yo, index_t zo,
                                           index_t curr_block_idx, float* n) {
            return DeviceGetNormal<tsdf_t>(
                    tsdf_base_ptr, xo, yo, zo, curr_block_idx, n, resolution,
                    nb_block_masks_indexer, nb_block_indices_indexer);
        };

        // Natural index (0, N) -> (block_idx, voxel_idx)
        index_t workload_block_idx = workload_idx / resolution3;
        index_t block_idx = indices_ptr[workload_block_idx];
        index_t voxel_idx = workload_idx % resolution3;

        /// Coordinate transform
        // block_idx -> (x_block, y_block, z_block)
        index_t* block_key_ptr =
                block_keys_indexer.GetDataPtr<index_t>(block_idx);
        index_t xb = block_key_ptr[0];
        index_t yb = block_key_ptr[1];
        index_t zb = block_key_ptr[2];

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        index_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        index_t linear_idx = block_idx * resolution3 + voxel_idx;
        float tsdf_o = tsdf_base_ptr[linear_idx];
        float weight_o = weight_base_ptr[linear_idx];
        if (weight_o <= weight_threshold) return;

        float no[3] = {0}, ne[3] = {0};

        // Get normal at origin
        GetNormal(xv, yv, zv, workload_block_idx, no);

        index_t x = xb * resolution + xv;
        index_t y = yb * resolution + yv;
        index_t z = zb * resolution + zv;

        // Enumerate x-y-z axis
        for (index_t i = 0; i < 3; ++i) {
            index_t linear_idx_i =
                    GetLinearIdx(xv + (i == 0), yv + (i == 1), zv + (i == 2),
                                 workload_block_idx);
            if (linear_idx_i < 0) continue;

            float tsdf_i = tsdf_base_ptr[linear_idx_i];
            float weight_i = weight_base_ptr[linear_idx_i];
            if (weight_i > weight_threshold && tsdf_i * tsdf_o < 0) {
                float ratio = (0 - tsdf_o) / (tsdf_i - tsdf_o);

                index_t idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);
                if (idx >= valid_size) {
                    printf("Point cloud size larger than "
                           "estimated, please increase the "
                           "estimation!\n");
                    return;
                }

                float* point_ptr = point_indexer.GetDataPtr<float>(idx);
                point_ptr[0] = voxel_size * (x + ratio * int(i == 0));
                point_ptr[1] = voxel_size * (y + ratio * int(i == 1));
                point_ptr[2] = voxel_size * (z + ratio * int(i == 2));

                // Get normal at edge and interpolate
                float* normal_ptr = normal_indexer.GetDataPtr<float>(idx);
                GetNormal(xv + (i == 0), yv + (i == 1), zv + (i == 2),
                          workload_block_idx, ne);
                float nx = (1 - ratio) * no[0] + ratio * ne[0];
                float ny = (1 - ratio) * no[1] + ratio * ne[1];
                float nz = (1 - ratio) * no[2] + ratio * ne[2];
                float norm = static_cast<float>(
                        sqrt(nx * nx + ny * ny + nz * nz) + 1e-5);
                normal_ptr[0] = nx / norm;
                normal_ptr[1] = ny / norm;
                normal_ptr[2] = nz / norm;

                if (color_base_ptr) {
                    float* color_ptr = color_indexer.GetDataPtr<float>(idx);
                    const color_t* color_o_ptr =
                            color_base_ptr + 3 * linear_idx;
                    float r_o = color_o_ptr[0];
                    float g_o = color_o_ptr[1];
                    float b_o = color_o_ptr[2];

                    const color_t* color_i_ptr =
                            color_base_ptr + 3 * linear_idx_i;
                    float r_i = color_i_ptr[0];
                    float g_i = color_i_ptr[1];
                    float b_i = color_i_ptr[2];

                    color_ptr[0] = ((1 - ratio) * r_o + ratio * r_i) / 255.0f;
                    color_ptr[1] = ((1 - ratio) * g_o + ratio * g_i) / 255.0f;
                    color_ptr[2] = ((1 - ratio) * b_o + ratio * b_i) / 255.0f;
                }
            }
        }
    });

#if defined(__CUDACC__)
    index_t total_count = count.Item<index_t>();
#else
    index_t total_count = (*count_ptr).load();
#endif

    utility::LogDebug("{} vertices extracted", total_count);
    valid_size = total_count;

#if defined(BUILD_CUDA_MODULE) && defined(__CUDACC__)
    core::cuda::Synchronize();
#endif
}

template <typename tsdf_t, typename weight_t, typename color_t>
#if defined(__CUDACC__)
void ExtractTriangleMeshCUDA
#else
void ExtractTriangleMeshCPU
#endif
        (const core::Tensor& block_indices,
         const core::Tensor& inv_block_indices,
         const core::Tensor& nb_block_indices,
         const core::Tensor& nb_block_masks,
         const core::Tensor& block_keys,
         const TensorMap& block_value_map,
         core::Tensor& vertices,
         core::Tensor& triangles,
         core::Tensor& vertex_normals,
         core::Tensor& vertex_colors,
         index_t block_resolution,
         float voxel_size,
         float weight_threshold,
         index_t& vertex_count) {
    core::Device device = block_indices.GetDevice();

    index_t resolution = block_resolution;
    index_t resolution3 = resolution * resolution * resolution;

    // Shape / transform indexers, no data involved
    ArrayIndexer voxel_indexer({resolution, resolution, resolution});
    index_t n_blocks = static_cast<index_t>(block_indices.GetLength());

    // TODO(wei): profile performance by replacing the table to a hashmap.
    // Voxel-wise mesh info. 4 channels correspond to:
    // 3 edges' corresponding vertex index + 1 table index.
    core::Tensor mesh_structure;
    try {
        mesh_structure = core::Tensor::Zeros(
                {n_blocks, resolution, resolution, resolution, 4}, core::Int32,
                device);
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
    ArrayIndexer mesh_structure_indexer(mesh_structure, 4);
    ArrayIndexer nb_block_masks_indexer(nb_block_masks, 2);
    ArrayIndexer nb_block_indices_indexer(nb_block_indices, 2);

    // Plain arrays that does not require indexers
    const index_t* indices_ptr = block_indices.GetDataPtr<index_t>();
    const index_t* inv_indices_ptr = inv_block_indices.GetDataPtr<index_t>();

    if (!block_value_map.Contains("tsdf") ||
        !block_value_map.Contains("weight")) {
        utility::LogError(
                "TSDF and/or weight not allocated in blocks, please implement "
                "customized integration.");
    }
    const tsdf_t* tsdf_base_ptr =
            block_value_map.at("tsdf").GetDataPtr<tsdf_t>();
    const weight_t* weight_base_ptr =
            block_value_map.at("weight").GetDataPtr<weight_t>();
    const color_t* color_base_ptr = nullptr;
    if (block_value_map.Contains("color")) {
        color_base_ptr = block_value_map.at("color").GetDataPtr<color_t>();
    }

    index_t n = n_blocks * resolution3;
    // Pass 0: analyze mesh structure, set up one-on-one correspondences
    // from edges to vertices.

    core::ParallelFor(device, n, [=] OPEN3D_DEVICE(index_t widx) {
        auto GetLinearIdx = [&] OPEN3D_DEVICE(
                                    index_t xo, index_t yo, index_t zo,
                                    index_t curr_block_idx) -> index_t {
            return DeviceGetLinearIdx(xo, yo, zo, curr_block_idx,
                                      static_cast<index_t>(resolution),
                                      nb_block_masks_indexer,
                                      nb_block_indices_indexer);
        };

        // Natural index (0, N) -> (block_idx, voxel_idx)
        index_t workload_block_idx = widx / resolution3;
        index_t voxel_idx = widx % resolution3;

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        index_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        // Check per-vertex sign in the cube to determine cube
        // type
        index_t table_idx = 0;
        for (index_t i = 0; i < 8; ++i) {
            index_t linear_idx_i =
                    GetLinearIdx(xv + vtx_shifts[i][0], yv + vtx_shifts[i][1],
                                 zv + vtx_shifts[i][2], workload_block_idx);
            if (linear_idx_i < 0) return;

            float tsdf_i = tsdf_base_ptr[linear_idx_i];
            float weight_i = weight_base_ptr[linear_idx_i];
            if (weight_i <= weight_threshold) return;

            table_idx |= ((tsdf_i < 0) ? (1 << i) : 0);
        }

        index_t* mesh_struct_ptr = mesh_structure_indexer.GetDataPtr<index_t>(
                xv, yv, zv, workload_block_idx);
        mesh_struct_ptr[3] = table_idx;

        if (table_idx == 0 || table_idx == 255) return;

        // Check per-edge sign determine the cube type
        index_t edges_with_vertices = edge_table[table_idx];
        for (index_t i = 0; i < 12; ++i) {
            if (edges_with_vertices & (1 << i)) {
                index_t xv_i = xv + edge_shifts[i][0];
                index_t yv_i = yv + edge_shifts[i][1];
                index_t zv_i = zv + edge_shifts[i][2];
                index_t edge_i = edge_shifts[i][3];

                index_t dxb = xv_i / resolution;
                index_t dyb = yv_i / resolution;
                index_t dzb = zv_i / resolution;

                index_t nb_idx = (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

                index_t block_idx_i =
                        *nb_block_indices_indexer.GetDataPtr<index_t>(
                                workload_block_idx, nb_idx);
                index_t* mesh_ptr_i =
                        mesh_structure_indexer.GetDataPtr<index_t>(
                                xv_i - dxb * resolution,
                                yv_i - dyb * resolution,
                                zv_i - dzb * resolution,
                                inv_indices_ptr[block_idx_i]);

                // Non-atomic write, but we are safe
                mesh_ptr_i[edge_i] = -1;
            }
        }
    });

    // Pass 1: determine valid number of vertices (if not preset)
#if defined(__CUDACC__)
    core::Tensor count(std::vector<index_t>{0}, {}, core::Int32, device);

    index_t* count_ptr = count.GetDataPtr<index_t>();
#else
    std::atomic<index_t> count_atomic(0);
    std::atomic<index_t>* count_ptr = &count_atomic;
#endif

    if (vertex_count < 0) {
        core::ParallelFor(device, n, [=] OPEN3D_DEVICE(index_t widx) {
            // Natural index (0, N) -> (block_idx, voxel_idx)
            index_t workload_block_idx = widx / resolution3;
            index_t voxel_idx = widx % resolution3;

            // voxel_idx -> (x_voxel, y_voxel, z_voxel)
            index_t xv, yv, zv;
            voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

            // Obtain voxel's mesh struct ptr
            index_t* mesh_struct_ptr =
                    mesh_structure_indexer.GetDataPtr<index_t>(
                            xv, yv, zv, workload_block_idx);

            // Early quit -- no allocated vertex to compute
            if (mesh_struct_ptr[0] != -1 && mesh_struct_ptr[1] != -1 &&
                mesh_struct_ptr[2] != -1) {
                return;
            }

            // Enumerate 3 edges in the voxel
            for (index_t e = 0; e < 3; ++e) {
                index_t vertex_idx = mesh_struct_ptr[e];
                if (vertex_idx != -1) continue;

                OPEN3D_ATOMIC_ADD(count_ptr, 1);
            }
        });

#if defined(__CUDACC__)
        vertex_count = count.Item<index_t>();
#else
        vertex_count = (*count_ptr).load();
#endif
    }

    utility::LogDebug("Total vertex count = {}", vertex_count);
    vertices = core::Tensor({vertex_count, 3}, core::Float32, device);

    vertex_normals = core::Tensor({vertex_count, 3}, core::Float32, device);
    ArrayIndexer normal_indexer = ArrayIndexer(vertex_normals, 1);

    ArrayIndexer color_indexer;
    if (color_base_ptr) {
        vertex_colors = core::Tensor({vertex_count, 3}, core::Float32, device);
        color_indexer = ArrayIndexer(vertex_colors, 1);
    }

    ArrayIndexer block_keys_indexer(block_keys, 1);
    ArrayIndexer vertex_indexer(vertices, 1);

#if defined(__CUDACC__)
    count = core::Tensor(std::vector<index_t>{0}, {}, core::Int32, device);
    count_ptr = count.GetDataPtr<index_t>();
#else
    (*count_ptr) = 0;
#endif

    // Pass 2: extract vertices.

    core::ParallelFor(device, n, [=] OPEN3D_DEVICE(index_t widx) {
        auto GetLinearIdx = [&] OPEN3D_DEVICE(
                                    index_t xo, index_t yo, index_t zo,
                                    index_t curr_block_idx) -> index_t {
            return DeviceGetLinearIdx(xo, yo, zo, curr_block_idx, resolution,
                                      nb_block_masks_indexer,
                                      nb_block_indices_indexer);
        };

        auto GetNormal = [&] OPEN3D_DEVICE(index_t xo, index_t yo, index_t zo,
                                           index_t curr_block_idx, float* n) {
            return DeviceGetNormal<tsdf_t>(
                    tsdf_base_ptr, xo, yo, zo, curr_block_idx, n, resolution,
                    nb_block_masks_indexer, nb_block_indices_indexer);
        };

        // Natural index (0, N) -> (block_idx, voxel_idx)
        index_t workload_block_idx = widx / resolution3;
        index_t block_idx = indices_ptr[workload_block_idx];
        index_t voxel_idx = widx % resolution3;

        // block_idx -> (x_block, y_block, z_block)
        index_t* block_key_ptr =
                block_keys_indexer.GetDataPtr<index_t>(block_idx);
        index_t xb = block_key_ptr[0];
        index_t yb = block_key_ptr[1];
        index_t zb = block_key_ptr[2];

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        index_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        // global coordinate (in voxels)
        index_t x = xb * resolution + xv;
        index_t y = yb * resolution + yv;
        index_t z = zb * resolution + zv;

        // Obtain voxel's mesh struct ptr
        index_t* mesh_struct_ptr = mesh_structure_indexer.GetDataPtr<index_t>(
                xv, yv, zv, workload_block_idx);

        // Early quit -- no allocated vertex to compute
        if (mesh_struct_ptr[0] != -1 && mesh_struct_ptr[1] != -1 &&
            mesh_struct_ptr[2] != -1) {
            return;
        }

        // Obtain voxel ptr
        index_t linear_idx = resolution3 * block_idx + voxel_idx;
        float tsdf_o = tsdf_base_ptr[linear_idx];

        float no[3] = {0}, ne[3] = {0};

        // Get normal at origin
        GetNormal(xv, yv, zv, workload_block_idx, no);

        // Enumerate 3 edges in the voxel
        for (index_t e = 0; e < 3; ++e) {
            index_t vertex_idx = mesh_struct_ptr[e];
            if (vertex_idx != -1) continue;

            index_t linear_idx_e =
                    GetLinearIdx(xv + (e == 0), yv + (e == 1), zv + (e == 2),
                                 workload_block_idx);
            OPEN3D_ASSERT(linear_idx_e > 0 &&
                          "Internal error: GetVoxelAt returns nullptr.");
            float tsdf_e = tsdf_base_ptr[linear_idx_e];
            float ratio = (0 - tsdf_o) / (tsdf_e - tsdf_o);

            index_t idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);
            mesh_struct_ptr[e] = idx;

            float ratio_x = ratio * index_t(e == 0);
            float ratio_y = ratio * index_t(e == 1);
            float ratio_z = ratio * index_t(e == 2);

            float* vertex_ptr = vertex_indexer.GetDataPtr<float>(idx);
            vertex_ptr[0] = voxel_size * (x + ratio_x);
            vertex_ptr[1] = voxel_size * (y + ratio_y);
            vertex_ptr[2] = voxel_size * (z + ratio_z);

            // Get normal at edge and interpolate
            float* normal_ptr = normal_indexer.GetDataPtr<float>(idx);
            GetNormal(xv + (e == 0), yv + (e == 1), zv + (e == 2),
                      workload_block_idx, ne);
            float nx = (1 - ratio) * no[0] + ratio * ne[0];
            float ny = (1 - ratio) * no[1] + ratio * ne[1];
            float nz = (1 - ratio) * no[2] + ratio * ne[2];
            float norm = static_cast<float>(sqrt(nx * nx + ny * ny + nz * nz) +
                                            1e-5);
            normal_ptr[0] = nx / norm;
            normal_ptr[1] = ny / norm;
            normal_ptr[2] = nz / norm;

            if (color_base_ptr) {
                float* color_ptr = color_indexer.GetDataPtr<float>(idx);
                float r_o = color_base_ptr[linear_idx * 3 + 0];
                float g_o = color_base_ptr[linear_idx * 3 + 1];
                float b_o = color_base_ptr[linear_idx * 3 + 2];

                float r_e = color_base_ptr[linear_idx_e * 3 + 0];
                float g_e = color_base_ptr[linear_idx_e * 3 + 1];
                float b_e = color_base_ptr[linear_idx_e * 3 + 2];

                color_ptr[0] = ((1 - ratio) * r_o + ratio * r_e) / 255.0f;
                color_ptr[1] = ((1 - ratio) * g_o + ratio * g_e) / 255.0f;
                color_ptr[2] = ((1 - ratio) * b_o + ratio * b_e) / 255.0f;
            }
        }
    });

    // Pass 3: connect vertices and form triangles.
    index_t triangle_count = vertex_count * 3;
    triangles = core::Tensor({triangle_count, 3}, core::Int32, device);
    ArrayIndexer triangle_indexer(triangles, 1);

#if defined(__CUDACC__)
    count = core::Tensor(std::vector<index_t>{0}, {}, core::Int32, device);
    count_ptr = count.GetDataPtr<index_t>();
#else
    (*count_ptr) = 0;
#endif
    core::ParallelFor(device, n, [=] OPEN3D_DEVICE(index_t widx) {
        // Natural index (0, N) -> (block_idx, voxel_idx)
        index_t workload_block_idx = widx / resolution3;
        index_t voxel_idx = widx % resolution3;

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        index_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        // Obtain voxel's mesh struct ptr
        index_t* mesh_struct_ptr = mesh_structure_indexer.GetDataPtr<index_t>(
                xv, yv, zv, workload_block_idx);

        index_t table_idx = mesh_struct_ptr[3];
        if (tri_count[table_idx] == 0) return;

        for (index_t tri = 0; tri < 16; tri += 3) {
            if (tri_table[table_idx][tri] == -1) return;

            index_t tri_idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);

            for (index_t vertex = 0; vertex < 3; ++vertex) {
                index_t edge = tri_table[table_idx][tri + vertex];

                index_t xv_i = xv + edge_shifts[edge][0];
                index_t yv_i = yv + edge_shifts[edge][1];
                index_t zv_i = zv + edge_shifts[edge][2];
                index_t edge_i = edge_shifts[edge][3];

                index_t dxb = xv_i / resolution;
                index_t dyb = yv_i / resolution;
                index_t dzb = zv_i / resolution;

                index_t nb_idx = (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

                index_t block_idx_i =
                        *nb_block_indices_indexer.GetDataPtr<index_t>(
                                workload_block_idx, nb_idx);
                index_t* mesh_struct_ptr_i =
                        mesh_structure_indexer.GetDataPtr<index_t>(
                                xv_i - dxb * resolution,
                                yv_i - dyb * resolution,
                                zv_i - dzb * resolution,
                                inv_indices_ptr[block_idx_i]);

                index_t* triangle_ptr =
                        triangle_indexer.GetDataPtr<index_t>(tri_idx);
                triangle_ptr[2 - vertex] = mesh_struct_ptr_i[edge_i];
            }
        }
    });

#if defined(__CUDACC__)
    triangle_count = count.Item<index_t>();
#else
    triangle_count = (*count_ptr).load();
#endif
    utility::LogDebug("Total triangle count = {}", triangle_count);
    triangles = triangles.Slice(0, 0, triangle_count);
}

}  // namespace voxel_grid
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
