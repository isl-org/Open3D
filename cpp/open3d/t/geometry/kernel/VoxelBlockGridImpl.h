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

inline OPEN3D_DEVICE int64_t
DeviceGetLinearIdx(int xo,
                   int yo,
                   int zo,
                   int curr_block_idx,
                   int resolution,
                   const NDArrayIndexer& nb_block_masks_indexer,
                   const NDArrayIndexer& nb_block_indices_indexer) {
    int xn = (xo + resolution) % resolution;
    int yn = (yo + resolution) % resolution;
    int zn = (zo + resolution) % resolution;

    int64_t dxb = Sign(xo - xn);
    int64_t dyb = Sign(yo - yn);
    int64_t dzb = Sign(zo - zn);

    int64_t nb_idx = (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

    bool block_mask_i =
            *nb_block_masks_indexer.GetDataPtr<bool>(curr_block_idx, nb_idx);
    if (!block_mask_i) return -1;

    int block_idx_i =
            *nb_block_indices_indexer.GetDataPtr<int>(curr_block_idx, nb_idx);

    return (((block_idx_i * resolution) + zn) * resolution + yn) * resolution +
           xn;
}

template <typename tsdf_t>
inline OPEN3D_DEVICE void DeviceGetNormal(
        const tsdf_t* tsdf_base_ptr,
        int xo,
        int yo,
        int zo,
        int curr_block_idx,
        float* n,
        int resolution,
        const NDArrayIndexer& nb_block_masks_indexer,
        const NDArrayIndexer& nb_block_indices_indexer) {
    auto GetLinearIdx = [&] OPEN3D_DEVICE(int xo, int yo, int zo) -> int64_t {
        return DeviceGetLinearIdx(
                xo, yo, zo, curr_block_idx, static_cast<int>(resolution),
                nb_block_masks_indexer, nb_block_indices_indexer);
    };
    int64_t vxp = GetLinearIdx(xo + 1, yo, zo);
    int64_t vxn = GetLinearIdx(xo - 1, yo, zo);
    int64_t vyp = GetLinearIdx(xo, yo + 1, zo);
    int64_t vyn = GetLinearIdx(xo, yo - 1, zo);
    int64_t vzp = GetLinearIdx(xo, yo, zo + 1);
    int64_t vzn = GetLinearIdx(xo, yo, zo - 1);
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
         std::vector<core::Tensor>& block_values,
         const core::Tensor& intrinsics,
         const core::Tensor& extrinsics,
         int64_t resolution,
         float voxel_size,
         float sdf_trunc,
         float depth_scale,
         float depth_max) {
    // Parameters
    int64_t resolution2 = resolution * resolution;
    int64_t resolution3 = resolution2 * resolution;

    TransformIndexer transform_indexer(intrinsics, extrinsics, voxel_size);

    NDArrayIndexer voxel_indexer({resolution, resolution, resolution});

    NDArrayIndexer block_keys_indexer(block_keys, 1);
    NDArrayIndexer depth_indexer(depth, 2);
    NDArrayIndexer color_indexer;
    bool integrate_color = false;
    if (color.NumElements() != 0) {
        color_indexer = NDArrayIndexer(color, 2);
        integrate_color = true;
    }

    const int* indices_ptr = indices.GetDataPtr<int>();
    tsdf_t* tsdf_base_ptr = block_values[0].GetDataPtr<tsdf_t>();
    weight_t* weight_base_ptr = block_values[1].GetDataPtr<weight_t>();
    color_t* color_base_ptr = block_values[2].GetDataPtr<color_t>();

    int64_t n = indices.GetLength() * resolution3;
    core::ParallelFor(
            depth.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                // Natural index (0, N) -> (block_idx, voxel_idx)
                int64_t block_idx = indices_ptr[workload_idx / resolution3];
                int64_t voxel_idx = workload_idx % resolution3;

                /// Coordinate transform
                // block_idx -> (x_block, y_block, z_block)
                int* block_key_ptr =
                        block_keys_indexer.GetDataPtr<int>(block_idx);
                int64_t xb = static_cast<int64_t>(block_key_ptr[0]);
                int64_t yb = static_cast<int64_t>(block_key_ptr[1]);
                int64_t zb = static_cast<int64_t>(block_key_ptr[2]);

                // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                int64_t xv, yv, zv;
                voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                // coordinate in world (in voxel)
                int64_t x = xb * resolution + xv;
                int64_t y = yb * resolution + yv;
                int64_t z = zb * resolution + zv;

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

                int64_t ui = static_cast<int64_t>(u);
                int64_t vi = static_cast<int64_t>(v);

                // Associate image workload and compute SDF and
                // TSDF.
                float depth = *depth_indexer.GetDataPtr<input_depth_t>(ui, vi) /
                              depth_scale;

                float sdf = depth - zc;
                if (depth <= 0 || depth > depth_max || zc <= 0 ||
                    sdf < -sdf_trunc) {
                    return;
                }
                sdf = sdf < sdf_trunc ? sdf : sdf_trunc;
                sdf /= sdf_trunc;

                int64_t linear_idx = block_idx * resolution3 + voxel_idx;

                tsdf_t* tsdf_ptr = tsdf_base_ptr + linear_idx;
                weight_t* weight_ptr = weight_base_ptr + linear_idx;
                color_t* color_ptr = color_base_ptr + 3 * linear_idx;

                float inv_wsum = 1.0f / (*weight_ptr + 1);
                float weight = *weight_ptr;
                *tsdf_ptr = (weight * (*tsdf_ptr) + sdf) * inv_wsum;
                if (integrate_color) {
                    input_color_t* input_color_ptr =
                            color_indexer.GetDataPtr<input_color_t>(ui, vi);
                    for (int i = 0; i < 3; ++i) {
                        color_ptr[i] =
                                (weight * color_ptr[i] + input_color_ptr[i]) *
                                inv_wsum;
                    }
                }
                *weight_ptr = weight + 1;
            });

#if defined(__CUDACC__)
    core::cuda::Synchronize();
#endif
}

struct MiniVecCache {
    int x;
    int y;
    int z;
    int block_idx;

    inline int OPEN3D_DEVICE Check(int xin, int yin, int zin) {
        return (xin == x && yin == y && zin == z) ? block_idx : -1;
    }

    inline void OPEN3D_DEVICE Update(int xin,
                                     int yin,
                                     int zin,
                                     int block_idx_in) {
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
         const std::vector<core::Tensor>& block_values,
         const core::Tensor& range_map,
         std::unordered_map<std::string, core::Tensor>& renderings_map,
         const core::Tensor& intrinsics,
         const core::Tensor& extrinsics,
         int h,
         int w,
         int64_t block_resolution,
         float voxel_size,
         float sdf_trunc,
         float depth_scale,
         float depth_min,
         float depth_max,
         float weight_threshold) {
    using Key = utility::MiniVec<int, 3>;
    using Hash = utility::MiniVecHash<int, 3>;
    using Eq = utility::MiniVecEq<int, 3>;

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

    NDArrayIndexer range_map_indexer(range_map, 2);

    NDArrayIndexer vertex_map_indexer;
    NDArrayIndexer depth_map_indexer;
    NDArrayIndexer color_map_indexer;
    NDArrayIndexer normal_map_indexer;
    vertex_map_indexer = NDArrayIndexer(renderings_map.at("vertex"), 2);
    depth_map_indexer = NDArrayIndexer(renderings_map.at("depth"), 2);
    color_map_indexer = NDArrayIndexer(renderings_map.at("color"), 2);
    normal_map_indexer = NDArrayIndexer(renderings_map.at("normal"), 2);

    NDArrayIndexer index_map_indexer;
    NDArrayIndexer mask_map_indexer;
    NDArrayIndexer ratio_map_indexer;
    index_map_indexer = NDArrayIndexer(renderings_map.at("index"), 2);
    mask_map_indexer = NDArrayIndexer(renderings_map.at("mask"), 2);
    ratio_map_indexer = NDArrayIndexer(renderings_map.at("ratio"), 2);

    const tsdf_t* tsdf_base_ptr = block_values[0].GetDataPtr<tsdf_t>();
    const weight_t* weight_base_ptr = block_values[1].GetDataPtr<weight_t>();
    const color_t* color_base_ptr = block_values[2].GetDataPtr<color_t>();
    TransformIndexer c2w_transform_indexer(
            intrinsics, t::geometry::InverseTransformation(extrinsics));
    TransformIndexer w2c_transform_indexer(intrinsics, extrinsics);

    int64_t rows = h;
    int64_t cols = w;

    float block_size = voxel_size * block_resolution;
    int64_t resolution2 = block_resolution * block_resolution;
    int64_t resolution3 = resolution2 * block_resolution;

#ifndef __CUDACC__
    using std::max;
    using std::sqrt;
#endif

    core::ParallelFor(
            hashmap->GetDevice(), rows * cols,
            [=] OPEN3D_DEVICE(int64_t workload_idx) {
                auto GetLinearIdxAtP = [&] OPEN3D_DEVICE(
                                               int x_b, int y_b, int z_b,
                                               int x_v, int y_v, int z_v,
                                               core::buf_index_t block_buf_idx,
                                               MiniVecCache& cache) -> int64_t {
                    int x_vn = (x_v + block_resolution) % block_resolution;
                    int y_vn = (y_v + block_resolution) % block_resolution;
                    int z_vn = (z_v + block_resolution) % block_resolution;

                    int dx_b = Sign(x_v - x_vn);
                    int dy_b = Sign(y_v - y_vn);
                    int dz_b = Sign(z_v - z_vn);

                    if (dx_b == 0 && dy_b == 0 && dz_b == 0) {
                        return block_buf_idx * resolution3 + z_v * resolution2 +
                               y_v * block_resolution + x_v;
                    } else {
                        Key key(x_b + dx_b, y_b + dy_b, z_b + dz_b);

                        int block_buf_idx = cache.Check(key[0], key[1], key[2]);
                        if (block_buf_idx < 0) {
                            auto iter = hashmap_impl.find(key);
                            if (iter == hashmap_impl.end()) return -1;
                            block_buf_idx = iter->second;
                            cache.Update(key[0], key[1], key[2], block_buf_idx);
                        }

                        return block_buf_idx * resolution3 +
                               z_vn * resolution2 + y_vn * block_resolution +
                               x_vn;
                    }
                };

                auto GetLinearIdxAtT = [&] OPEN3D_DEVICE(
                                               float x_o, float y_o, float z_o,
                                               float x_d, float y_d, float z_d,
                                               float t,
                                               MiniVecCache& cache) -> int64_t {
                    float x_g = x_o + t * x_d;
                    float y_g = y_o + t * y_d;
                    float z_g = z_o + t * z_d;

                    // MiniVec coordinate and look up
                    int x_b = static_cast<int>(floorf(x_g / block_size));
                    int y_b = static_cast<int>(floorf(y_g / block_size));
                    int z_b = static_cast<int>(floorf(z_g / block_size));

                    Key key(x_b, y_b, z_b);
                    int block_buf_idx = cache.Check(x_b, y_b, z_b);
                    if (block_buf_idx < 0) {
                        auto iter = hashmap_impl.find(key);
                        if (iter == hashmap_impl.end()) return -1;
                        block_buf_idx = iter->second;
                        cache.Update(x_b, y_b, z_b, block_buf_idx);
                    }

                    // Voxel coordinate and look up
                    int x_v = int((x_g - x_b * block_size) / voxel_size);
                    int y_v = int((y_g - y_b * block_size) / voxel_size);
                    int z_v = int((z_g - z_b * block_size) / voxel_size);

                    return block_buf_idx * resolution3 + z_v * resolution2 +
                           y_v * block_resolution + x_v;
                };

                int64_t y = workload_idx / cols;
                int64_t x = workload_idx % cols;

                float *depth_ptr = nullptr, *vertex_ptr = nullptr,
                      *color_ptr = nullptr, *normal_ptr = nullptr;

                depth_ptr = depth_map_indexer.GetDataPtr<float>(x, y);
                depth_ptr[0] = 0;

                vertex_ptr = vertex_map_indexer.GetDataPtr<float>(x, y);
                vertex_ptr[0] = 0;
                vertex_ptr[1] = 0;
                vertex_ptr[2] = 0;

                color_ptr = color_map_indexer.GetDataPtr<float>(x, y);
                color_ptr[0] = 0;
                color_ptr[1] = 0;
                color_ptr[2] = 0;

                normal_ptr = normal_map_indexer.GetDataPtr<float>(x, y);
                normal_ptr[0] = 0;
                normal_ptr[1] = 0;
                normal_ptr[2] = 0;

                bool* mask_ptr = mask_map_indexer.GetDataPtr<bool>(x, y);
                float* ratio_ptr = ratio_map_indexer.GetDataPtr<float>(x, y);
                int64_t* index_ptr =
                        index_map_indexer.GetDataPtr<int64_t>(x, y);

                const float* range =
                        range_map_indexer.GetDataPtr<float>(x / 8, y / 8);
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
                float w = 0.0;

                // Camera origin
                c2w_transform_indexer.RigidTransform(0, 0, 0, &x_o, &y_o, &z_o);

                // Direction
                c2w_transform_indexer.Unproject(static_cast<float>(x),
                                                static_cast<float>(y), 1.0f,
                                                &x_c, &y_c, &z_c);
                c2w_transform_indexer.RigidTransform(x_c, y_c, z_c, &x_g, &y_g,
                                                     &z_g);
                float x_d = (x_g - x_o);
                float y_d = (y_g - y_o);
                float z_d = (z_g - z_o);

                MiniVecCache cache{0, 0, 0, -1};
                bool surface_found = false;
                while (t < t_max) {
                    int64_t linear_idx = GetLinearIdxAtT(x_o, y_o, z_o, x_d,
                                                         y_d, z_d, t, cache);

                    if (linear_idx < 0) {
                        t_prev = t;
                        t += block_size;
                    } else {
                        tsdf_prev = tsdf;
                        tsdf = tsdf_base_ptr[linear_idx];
                        w = weight_base_ptr[linear_idx];
                        if (tsdf_prev > 0 && w >= weight_threshold &&
                            tsdf <= 0) {
                            surface_found = true;
                            break;
                        }
                        t_prev = t;
                        float delta = tsdf * sdf_trunc;
                        t += delta < voxel_size ? voxel_size : delta;
                    }
                }

                if (surface_found) {
                    float t_intersect = (t * tsdf_prev - t_prev * tsdf) /
                                        (tsdf_prev - tsdf);
                    x_g = x_o + t_intersect * x_d;
                    y_g = y_o + t_intersect * y_d;
                    z_g = z_o + t_intersect * z_d;

                    // Trivial vertex assignment
                    *depth_ptr = t_intersect * depth_scale;
                    w2c_transform_indexer.RigidTransform(
                            x_g, y_g, z_g, vertex_ptr + 0, vertex_ptr + 1,
                            vertex_ptr + 2);

                    // Trilinear interpolation
                    // TODO(wei): simplify the flow by splitting the
                    // functions given what is enabled
                    int x_b = static_cast<int>(floorf(x_g / block_size));
                    int y_b = static_cast<int>(floorf(y_g / block_size));
                    int z_b = static_cast<int>(floorf(z_g / block_size));
                    float x_v = (x_g - float(x_b) * block_size) / voxel_size;
                    float y_v = (y_g - float(y_b) * block_size) / voxel_size;
                    float z_v = (z_g - float(z_b) * block_size) / voxel_size;

                    Key key(x_b, y_b, z_b);

                    int block_buf_idx = cache.Check(x_b, y_b, z_b);
                    if (block_buf_idx < 0) {
                        auto iter = hashmap_impl.find(key);
                        if (iter == hashmap_impl.end()) return;
                        block_buf_idx = iter->second;
                        cache.Update(x_b, y_b, z_b, block_buf_idx);
                    }

                    int x_v_floor = static_cast<int>(floorf(x_v));
                    int y_v_floor = static_cast<int>(floorf(y_v));
                    int z_v_floor = static_cast<int>(floorf(z_v));

                    float ratio_x = x_v - float(x_v_floor);
                    float ratio_y = y_v - float(y_v_floor);
                    float ratio_z = z_v - float(z_v_floor);

                    float sum_r = 0.0;
                    for (int k = 0; k < 8; ++k) {
                        int dx_v = (k & 1) > 0 ? 1 : 0;
                        int dy_v = (k & 2) > 0 ? 1 : 0;
                        int dz_v = (k & 4) > 0 ? 1 : 0;

                        int64_t linear_idx_k = GetLinearIdxAtP(
                                x_b, y_b, z_b, x_v_floor + dx_v,
                                y_v_floor + dy_v, z_v_floor + dz_v,
                                block_buf_idx, cache);

                        if (linear_idx_k >= 0 &&
                            weight_base_ptr[linear_idx_k] > 0) {
                            mask_ptr[k] = true;
                            index_ptr[k] = linear_idx_k;

                            float rx = dx_v * (ratio_x) +
                                       (1 - dx_v) * (1 - ratio_x);
                            float ry = dy_v * (ratio_y) +
                                       (1 - dy_v) * (1 - ratio_y);
                            float rz = dz_v * (ratio_z) +
                                       (1 - dz_v) * (1 - ratio_z);

                            float r = rx * ry * rz;
                            ratio_ptr[k] = r;

                            int64_t color_linear_idx = linear_idx_k * 3;
                            color_ptr[0] +=
                                    r * color_base_ptr[color_linear_idx + 0];
                            color_ptr[1] +=
                                    r * color_base_ptr[color_linear_idx + 1];
                            color_ptr[2] +=
                                    r * color_base_ptr[color_linear_idx + 2];

                            float tsdf_k = tsdf_base_ptr[linear_idx_k];
                            float grad_x = ry * rz * tsdf_k * (2 * dx_v - 1);
                            float grad_y = rx * rz * tsdf_k * (2 * dy_v - 1);
                            float grad_z = rx * ry * tsdf_k * (2 * dz_v - 1);

                            normal_ptr[0] += grad_x;
                            normal_ptr[1] += grad_y;
                            normal_ptr[2] += grad_z;

                            sum_r += r;
                        }
                    }  // loop over 8 neighbors

                    if (sum_r > 0) {
                        sum_r *= 255.0;
                        color_ptr[0] /= sum_r;
                        color_ptr[1] /= sum_r;
                        color_ptr[2] /= sum_r;

                        float norm = sqrt(normal_ptr[0] * normal_ptr[0] +
                                          normal_ptr[1] * normal_ptr[1] +
                                          normal_ptr[2] * normal_ptr[2]);
                        w2c_transform_indexer.Rotate(
                                -normal_ptr[0] / norm, -normal_ptr[1] / norm,
                                -normal_ptr[2] / norm, normal_ptr + 0,
                                normal_ptr + 1, normal_ptr + 2);
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
         const std::vector<core::Tensor>& block_values,
         core::Tensor& points,
         core::Tensor& normals,
         core::Tensor& colors,
         int64_t resolution,
         float voxel_size,
         float weight_threshold,
         int& valid_size) {
    // Parameters
    int64_t resolution2 = resolution * resolution;
    int64_t resolution3 = resolution2 * resolution;

    // Shape / transform indexers, no data involved
    NDArrayIndexer voxel_indexer({resolution, resolution, resolution});

    // Real data indexer
    NDArrayIndexer block_keys_indexer(block_keys, 1);
    NDArrayIndexer nb_block_masks_indexer(nb_masks, 2);
    NDArrayIndexer nb_block_indices_indexer(nb_indices, 2);

    // Plain arrays that does not require indexers
    const int* indices_ptr = indices.GetDataPtr<int>();

    const tsdf_t* tsdf_base_ptr = block_values[0].GetDataPtr<tsdf_t>();
    const weight_t* weight_base_ptr = block_values[1].GetDataPtr<weight_t>();
    const color_t* color_base_ptr = block_values[2].GetDataPtr<color_t>();

    int64_t n_blocks = indices.GetLength();
    int64_t n = n_blocks * resolution3;

    // Output
#if defined(__CUDACC__)
    core::Tensor count(std::vector<int>{0}, {1}, core::Int32,
                       block_values[0].GetDevice());
    int* count_ptr = count.GetDataPtr<int>();
#else
    std::atomic<int> count_atomic(0);
    std::atomic<int>* count_ptr = &count_atomic;
#endif

    if (valid_size < 0) {
        utility::LogWarning(
                "No estimated max point cloud size provided, using a 2-pass "
                "estimation. Surface extraction could be slow.");
        // This pass determines valid number of points.

        core::ParallelFor(
                indices.GetDevice(), n,
                [=] OPEN3D_DEVICE(int64_t workload_idx) {
                    auto GetLinearIdx = [&] OPEN3D_DEVICE(
                                                int xo, int yo, int zo,
                                                int curr_block_idx) -> int64_t {
                        return DeviceGetLinearIdx(xo, yo, zo, curr_block_idx,
                                                  static_cast<int>(resolution),
                                                  nb_block_masks_indexer,
                                                  nb_block_indices_indexer);
                    };

                    // Natural index (0, N) -> (block_idx,
                    // voxel_idx)
                    int64_t workload_block_idx = workload_idx / resolution3;
                    int64_t block_idx = indices_ptr[workload_block_idx];
                    int64_t voxel_idx = workload_idx % resolution3;

                    // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                    int64_t xv, yv, zv;
                    voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                    int64_t linear_idx = block_idx * resolution3 + voxel_idx;
                    float tsdf_o = tsdf_base_ptr[linear_idx];
                    float weight_o = weight_base_ptr[linear_idx];
                    if (weight_o <= weight_threshold) return;

                    // Enumerate x-y-z directions
                    for (int i = 0; i < 3; ++i) {
                        int64_t linear_idx_i = GetLinearIdx(
                                static_cast<int>(xv) + (i == 0),
                                static_cast<int>(yv) + (i == 1),
                                static_cast<int>(zv) + (i == 2),
                                static_cast<int>(workload_block_idx));
                        if (linear_idx_i < 0) continue;

                        float tsdf_i = tsdf_base_ptr[linear_idx_i];
                        float weight_i = weight_base_ptr[linear_idx_i];
                        if (weight_i > weight_threshold &&
                            tsdf_i * tsdf_o < 0) {
                            OPEN3D_ATOMIC_ADD(count_ptr, 1);
                        }
                    }
                });

#if defined(__CUDACC__)
        valid_size = count[0].Item<int>();
        count[0] = 0;
#else
        valid_size = (*count_ptr).load();
        (*count_ptr) = 0;
#endif
    }

    if (points.GetLength() == 0) {
        points = core::Tensor({valid_size, 3}, core::Float32,
                              block_values[0].GetDevice());
    }
    NDArrayIndexer point_indexer(points, 1);

    // Normals
    NDArrayIndexer normal_indexer;
    normals = core::Tensor({valid_size, 3}, core::Float32,
                           block_values[0].GetDevice());
    normal_indexer = NDArrayIndexer(normals, 1);

    // This pass extracts exact surface points.

    // Colors
    NDArrayIndexer color_indexer;
    colors = core::Tensor({valid_size, 3}, core::Float32,
                          block_values[0].GetDevice());
    color_indexer = NDArrayIndexer(colors, 1);

    core::ParallelFor(
            indices.GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                auto GetLinearIdx = [&] OPEN3D_DEVICE(
                                            int xo, int yo, int zo,
                                            int curr_block_idx) -> int64_t {
                    return DeviceGetLinearIdx(xo, yo, zo, curr_block_idx,
                                              static_cast<int>(resolution),
                                              nb_block_masks_indexer,
                                              nb_block_indices_indexer);
                };

                auto GetNormal = [&] OPEN3D_DEVICE(int xo, int yo, int zo,
                                                   int curr_block_idx,
                                                   float* n) {
                    return DeviceGetNormal<tsdf_t>(
                            tsdf_base_ptr, xo, yo, zo, curr_block_idx, n,
                            static_cast<int>(resolution),
                            nb_block_masks_indexer, nb_block_indices_indexer);
                };

                // Natural index (0, N) -> (block_idx, voxel_idx)
                int64_t workload_block_idx = workload_idx / resolution3;
                int64_t block_idx = indices_ptr[workload_block_idx];
                int64_t voxel_idx = workload_idx % resolution3;

                /// Coordinate transform
                // block_idx -> (x_block, y_block, z_block)
                int* block_key_ptr =
                        block_keys_indexer.GetDataPtr<int>(block_idx);
                int64_t xb = static_cast<int64_t>(block_key_ptr[0]);
                int64_t yb = static_cast<int64_t>(block_key_ptr[1]);
                int64_t zb = static_cast<int64_t>(block_key_ptr[2]);

                // voxel_idx -> (x_voxel, y_voxel, z_voxel)
                int64_t xv, yv, zv;
                voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

                int64_t linear_idx = block_idx * resolution3 + voxel_idx;
                float tsdf_o = tsdf_base_ptr[linear_idx];
                float weight_o = weight_base_ptr[linear_idx];
                if (weight_o <= weight_threshold) return;

                float no[3] = {0}, ne[3] = {0};

                // Get normal at origin
                GetNormal(static_cast<int>(xv), static_cast<int>(yv),
                          static_cast<int>(zv),
                          static_cast<int>(workload_block_idx), no);

                int64_t x = xb * resolution + xv;
                int64_t y = yb * resolution + yv;
                int64_t z = zb * resolution + zv;

                // Enumerate x-y-z axis
                for (int i = 0; i < 3; ++i) {
                    int64_t linear_idx_i =
                            GetLinearIdx(static_cast<int>(xv) + (i == 0),
                                         static_cast<int>(yv) + (i == 1),
                                         static_cast<int>(zv) + (i == 2),
                                         static_cast<int>(workload_block_idx));
                    if (linear_idx_i < 0) continue;

                    float tsdf_i = tsdf_base_ptr[linear_idx_i];
                    float weight_i = weight_base_ptr[linear_idx_i];
                    if (weight_i > weight_threshold && tsdf_i * tsdf_o < 0) {
                        float ratio = (0 - tsdf_o) / (tsdf_i - tsdf_o);

                        int idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);
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
                        float* normal_ptr =
                                normal_indexer.GetDataPtr<float>(idx);
                        GetNormal(static_cast<int>(xv) + (i == 0),
                                  static_cast<int>(yv) + (i == 1),
                                  static_cast<int>(zv) + (i == 2),
                                  static_cast<int>(workload_block_idx), ne);
                        float nx = (1 - ratio) * no[0] + ratio * ne[0];
                        float ny = (1 - ratio) * no[1] + ratio * ne[1];
                        float nz = (1 - ratio) * no[2] + ratio * ne[2];
                        float norm = static_cast<float>(
                                sqrt(nx * nx + ny * ny + nz * nz) + 1e-5);
                        normal_ptr[0] = nx / norm;
                        normal_ptr[1] = ny / norm;
                        normal_ptr[2] = nz / norm;

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

                        color_ptr[0] =
                                ((1 - ratio) * r_o + ratio * r_i) / 255.0f;
                        color_ptr[1] =
                                ((1 - ratio) * g_o + ratio * g_i) / 255.0f;
                        color_ptr[2] =
                                ((1 - ratio) * b_o + ratio * b_i) / 255.0f;
                    }
                }
            });

#if defined(__CUDACC__)
    int total_count = count.Item<int>();
#else
    int total_count = (*count_ptr).load();
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
         const std::vector<core::Tensor>& block_values,
         core::Tensor& vertices,
         core::Tensor& triangles,
         core::Tensor& vertex_normals,
         core::Tensor& vertex_colors,
         int64_t block_resolution,
         float voxel_size,
         float weight_threshold,
         int& vertex_count) {
    core::Device device = block_indices.GetDevice();

    int64_t resolution = block_resolution;
    int64_t resolution3 = resolution * resolution * resolution;

    // Shape / transform indexers, no data involved
    NDArrayIndexer voxel_indexer({resolution, resolution, resolution});
    int n_blocks = static_cast<int>(block_indices.GetLength());

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
    NDArrayIndexer mesh_structure_indexer(mesh_structure, 4);
    NDArrayIndexer nb_block_masks_indexer(nb_block_masks, 2);
    NDArrayIndexer nb_block_indices_indexer(nb_block_indices, 2);

    // Plain arrays that does not require indexers
    const int* indices_ptr = block_indices.GetDataPtr<int>();
    const int* inv_indices_ptr = inv_block_indices.GetDataPtr<int>();

    const tsdf_t* tsdf_base_ptr = block_values[0].GetDataPtr<tsdf_t>();
    const weight_t* weight_base_ptr = block_values[1].GetDataPtr<weight_t>();
    const color_t* color_base_ptr = block_values[2].GetDataPtr<color_t>();

    int64_t n = n_blocks * resolution3;

    // Pass 0: analyze mesh structure, set up one-on-one correspondences
    // from edges to vertices.

    printf("pass 0\n");
    core::ParallelFor(device, n, [=] OPEN3D_DEVICE(int64_t widx) {
        auto GetLinearIdx = [&] OPEN3D_DEVICE(int xo, int yo, int zo,
                                              int curr_block_idx) -> int64_t {
            return DeviceGetLinearIdx(
                    xo, yo, zo, curr_block_idx, static_cast<int>(resolution),
                    nb_block_masks_indexer, nb_block_indices_indexer);
        };

        // Natural index (0, N) -> (block_idx, voxel_idx)
        int64_t workload_block_idx = widx / resolution3;
        int64_t voxel_idx = widx % resolution3;

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        int64_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        // Check per-vertex sign in the cube to determine cube
        // type
        int table_idx = 0;
        for (int i = 0; i < 8; ++i) {
            int64_t linear_idx_i =
                    GetLinearIdx(static_cast<int>(xv) + vtx_shifts[i][0],
                                 static_cast<int>(yv) + vtx_shifts[i][1],
                                 static_cast<int>(zv) + vtx_shifts[i][2],
                                 static_cast<int>(workload_block_idx));
            if (linear_idx_i < 0) return;

            float tsdf_i = tsdf_base_ptr[linear_idx_i];
            float weight_i = weight_base_ptr[linear_idx_i];
            if (weight_i <= weight_threshold) return;

            table_idx |= ((tsdf_i < 0) ? (1 << i) : 0);
        }

        int* mesh_struct_ptr = mesh_structure_indexer.GetDataPtr<int>(
                xv, yv, zv, workload_block_idx);
        mesh_struct_ptr[3] = table_idx;

        if (table_idx == 0 || table_idx == 255) return;

        // Check per-edge sign determine the cube type
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

                int nb_idx = (dxb + 1) + (dyb + 1) * 3 + (dzb + 1) * 9;

                int block_idx_i = *nb_block_indices_indexer.GetDataPtr<int>(
                        workload_block_idx, nb_idx);
                int* mesh_ptr_i = mesh_structure_indexer.GetDataPtr<int>(
                        xv_i - dxb * resolution, yv_i - dyb * resolution,
                        zv_i - dzb * resolution, inv_indices_ptr[block_idx_i]);

                // Non-atomic write, but we are safe
                mesh_ptr_i[edge_i] = -1;
            }
        }
    });

    // Pass 1: determine valid number of vertices (if not preset)
#if defined(__CUDACC__)
    core::Tensor count(std::vector<int>{0}, {}, core::Int32, device);

    int* count_ptr = count.GetDataPtr<int>();
#else
    std::atomic<int> count_atomic(0);
    std::atomic<int>* count_ptr = &count_atomic;
#endif

    printf("pass 1\n");
    if (vertex_count < 0) {
        core::ParallelFor(device, n, [=] OPEN3D_DEVICE(int64_t widx) {
            // Natural index (0, N) -> (block_idx, voxel_idx)
            int64_t workload_block_idx = widx / resolution3;
            int64_t voxel_idx = widx % resolution3;

            // voxel_idx -> (x_voxel, y_voxel, z_voxel)
            int64_t xv, yv, zv;
            voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

            // Obtain voxel's mesh struct ptr
            int* mesh_struct_ptr = mesh_structure_indexer.GetDataPtr<int>(
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

                OPEN3D_ATOMIC_ADD(count_ptr, 1);
            }
        });

#if defined(__CUDACC__)
        vertex_count = count.Item<int>();
#else
        vertex_count = (*count_ptr).load();
#endif
    }

    utility::LogDebug("Total vertex count = {}", vertex_count);
    vertices = core::Tensor({vertex_count, 3}, core::Float32, device);

    vertex_normals = core::Tensor({vertex_count, 3}, core::Float32, device);
    NDArrayIndexer normal_indexer = NDArrayIndexer(vertex_normals, 1);

    vertex_colors = core::Tensor({vertex_count, 3}, core::Float32, device);
    NDArrayIndexer color_indexer = NDArrayIndexer(vertex_colors, 1);

    NDArrayIndexer block_keys_indexer(block_keys, 1);
    NDArrayIndexer vertex_indexer(vertices, 1);

#if defined(__CUDACC__)
    count = core::Tensor(std::vector<int>{0}, {}, core::Int32, device);
    count_ptr = count.GetDataPtr<int>();
#else
    (*count_ptr) = 0;
#endif

    // Pass 2: extract vertices.

    printf("pass 2\n");
    core::ParallelFor(device, n, [=] OPEN3D_DEVICE(int64_t widx) {
        auto GetLinearIdx = [&] OPEN3D_DEVICE(int xo, int yo, int zo,
                                              int curr_block_idx) -> int64_t {
            return DeviceGetLinearIdx(
                    xo, yo, zo, curr_block_idx, static_cast<int>(resolution),
                    nb_block_masks_indexer, nb_block_indices_indexer);
        };

        auto GetNormal = [&] OPEN3D_DEVICE(int xo, int yo, int zo,
                                           int curr_block_idx, float* n) {
            return DeviceGetNormal<tsdf_t>(
                    tsdf_base_ptr, xo, yo, zo, curr_block_idx, n,
                    static_cast<int>(resolution), nb_block_masks_indexer,
                    nb_block_indices_indexer);
        };

        // Natural index (0, N) -> (block_idx, voxel_idx)
        int64_t workload_block_idx = widx / resolution3;
        int64_t block_idx = indices_ptr[workload_block_idx];
        int64_t voxel_idx = widx % resolution3;

        // block_idx -> (x_block, y_block, z_block)
        int* block_key_ptr = block_keys_indexer.GetDataPtr<int>(block_idx);
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
        int* mesh_struct_ptr = mesh_structure_indexer.GetDataPtr<int>(
                xv, yv, zv, workload_block_idx);

        // Early quit -- no allocated vertex to compute
        if (mesh_struct_ptr[0] != -1 && mesh_struct_ptr[1] != -1 &&
            mesh_struct_ptr[2] != -1) {
            return;
        }

        // Obtain voxel ptr
        int64_t linear_idx = resolution3 * block_idx + voxel_idx;
        float tsdf_o = tsdf_base_ptr[linear_idx];

        float no[3] = {0}, ne[3] = {0};

        // Get normal at origin
        GetNormal(static_cast<int>(xv), static_cast<int>(yv),
                  static_cast<int>(zv), static_cast<int>(workload_block_idx),
                  no);

        // Enumerate 3 edges in the voxel
        for (int e = 0; e < 3; ++e) {
            int vertex_idx = mesh_struct_ptr[e];
            if (vertex_idx != -1) continue;

            int64_t linear_idx_e =
                    GetLinearIdx(static_cast<int>(xv) + (e == 0),
                                 static_cast<int>(yv) + (e == 1),
                                 static_cast<int>(zv) + (e == 2),
                                 static_cast<int>(workload_block_idx));
            OPEN3D_ASSERT(linear_idx_e > 0 &&
                          "Internal error: GetVoxelAt returns nullptr.");
            float tsdf_e = tsdf_base_ptr[linear_idx_e];
            float ratio = (0 - tsdf_o) / (tsdf_e - tsdf_o);

            int idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);
            mesh_struct_ptr[e] = idx;

            float ratio_x = ratio * int(e == 0);
            float ratio_y = ratio * int(e == 1);
            float ratio_z = ratio * int(e == 2);

            float* vertex_ptr = vertex_indexer.GetDataPtr<float>(idx);
            vertex_ptr[0] = voxel_size * (x + ratio_x);
            vertex_ptr[1] = voxel_size * (y + ratio_y);
            vertex_ptr[2] = voxel_size * (z + ratio_z);

            // Get normal at edge and interpolate
            float* normal_ptr = normal_indexer.GetDataPtr<float>(idx);
            GetNormal(static_cast<int>(xv) + (e == 0),
                      static_cast<int>(yv) + (e == 1),
                      static_cast<int>(zv) + (e == 2),
                      static_cast<int>(workload_block_idx), ne);
            float nx = (1 - ratio) * no[0] + ratio * ne[0];
            float ny = (1 - ratio) * no[1] + ratio * ne[1];
            float nz = (1 - ratio) * no[2] + ratio * ne[2];
            float norm = static_cast<float>(sqrt(nx * nx + ny * ny + nz * nz) +
                                            1e-5);
            normal_ptr[0] = nx / norm;
            normal_ptr[1] = ny / norm;
            normal_ptr[2] = nz / norm;

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
    });

    // Pass 3: connect vertices and form triangles.
    int triangle_count = vertex_count * 3;
    triangles = core::Tensor({triangle_count, 3}, core::Int64, device);
    NDArrayIndexer triangle_indexer(triangles, 1);

#if defined(__CUDACC__)
    count = core::Tensor(std::vector<int>{0}, {}, core::Int32, device);
    count_ptr = count.GetDataPtr<int>();
#else
    (*count_ptr) = 0;
#endif
    core::ParallelFor(device, n, [=] OPEN3D_DEVICE(int64_t widx) {
        // Natural index (0, N) -> (block_idx, voxel_idx)
        int64_t workload_block_idx = widx / resolution3;
        int64_t voxel_idx = widx % resolution3;

        // voxel_idx -> (x_voxel, y_voxel, z_voxel)
        int64_t xv, yv, zv;
        voxel_indexer.WorkloadToCoord(voxel_idx, &xv, &yv, &zv);

        // Obtain voxel's mesh struct ptr
        int* mesh_struct_ptr = mesh_structure_indexer.GetDataPtr<int>(
                xv, yv, zv, workload_block_idx);

        int table_idx = mesh_struct_ptr[3];
        if (tri_count[table_idx] == 0) return;

        for (size_t tri = 0; tri < 16; tri += 3) {
            if (tri_table[table_idx][tri] == -1) return;

            int tri_idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);

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

                int block_idx_i = *nb_block_indices_indexer.GetDataPtr<int>(
                        workload_block_idx, nb_idx);
                int* mesh_struct_ptr_i = mesh_structure_indexer.GetDataPtr<int>(
                        xv_i - dxb * resolution, yv_i - dyb * resolution,
                        zv_i - dzb * resolution, inv_indices_ptr[block_idx_i]);

                int64_t* triangle_ptr =
                        triangle_indexer.GetDataPtr<int64_t>(tri_idx);
                triangle_ptr[2 - vertex] = mesh_struct_ptr_i[edge_i];
            }
        }
    });

#if defined(__CUDACC__)
    triangle_count = count.Item<int>();
#else
    triangle_count = (*count_ptr).load();
#endif
    utility::LogInfo("Total triangle count = {}", triangle_count);
    triangles = triangles.Slice(0, 0, triangle_count);
}

}  // namespace voxel_grid
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
