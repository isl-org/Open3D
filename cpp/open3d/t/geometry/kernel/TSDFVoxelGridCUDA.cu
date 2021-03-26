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
#include "open3d/core/hashmap/CUDA/STDGPUHashmap.h"
#include "open3d/core/hashmap/DeviceHashmap.h"
#include "open3d/core/hashmap/Dispatch.h"
#include "open3d/core/hashmap/Hashmap.h"
#include "open3d/core/kernel/CUDALauncher.cuh"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/TSDFVoxelGrid.h"
#include "open3d/t/geometry/kernel/TSDFVoxelGridImpl.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace tsdf {
struct Coord3i {
    OPEN3D_HOST_DEVICE Coord3i(int x, int y, int z) : x_(x), y_(y), z_(z) {}
    bool OPEN3D_HOST_DEVICE operator==(const Coord3i& other) const {
        return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
    }

    int64_t x_;
    int64_t y_;
    int64_t z_;
};

void TouchCUDA(const core::Tensor& points,
               core::Tensor& voxel_block_coords,
               int64_t voxel_grid_resolution,
               float voxel_size,
               float sdf_trunc) {
    int64_t resolution = voxel_grid_resolution;
    float block_size = voxel_size * resolution;

    int64_t n = points.GetLength();
    const float* pcd_ptr = static_cast<const float*>(points.GetDataPtr());

    core::Device device = points.GetDevice();
    core::Tensor block_coordi({8 * n, 3}, core::Dtype::Int32, device);
    int* block_coordi_ptr = static_cast<int*>(block_coordi.GetDataPtr());
    core::Tensor count(std::vector<int>{0}, {}, core::Dtype::Int32, device);
    int* count_ptr = static_cast<int*>(count.GetDataPtr());

    core::kernel::CUDALauncher::LaunchGeneralKernel(
            n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                float x = pcd_ptr[3 * workload_idx + 0];
                float y = pcd_ptr[3 * workload_idx + 1];
                float z = pcd_ptr[3 * workload_idx + 2];

                int xb_lo =
                        static_cast<int>(floor((x - sdf_trunc) / block_size));
                int xb_hi =
                        static_cast<int>(floor((x + sdf_trunc) / block_size));
                int yb_lo =
                        static_cast<int>(floor((y - sdf_trunc) / block_size));
                int yb_hi =
                        static_cast<int>(floor((y + sdf_trunc) / block_size));
                int zb_lo =
                        static_cast<int>(floor((z - sdf_trunc) / block_size));
                int zb_hi =
                        static_cast<int>(floor((z + sdf_trunc) / block_size));

                for (int xb = xb_lo; xb <= xb_hi; ++xb) {
                    for (int yb = yb_lo; yb <= yb_hi; ++yb) {
                        for (int zb = zb_lo; zb <= zb_hi; ++zb) {
                            int idx = atomicAdd(count_ptr, 1);
                            block_coordi_ptr[3 * idx + 0] = xb;
                            block_coordi_ptr[3 * idx + 1] = yb;
                            block_coordi_ptr[3 * idx + 2] = zb;
                        }
                    }
                }
            });

    int total_block_count = count.Item<int>();
    if (total_block_count == 0) {
        utility::LogError(
                "[CUDATSDFTouchKernel] No block is touched in TSDF volume, "
                "abort integration. Please check specified parameters, "
                "especially depth_scale and voxel_size");
    }
    block_coordi = block_coordi.Slice(0, 0, total_block_count);
    core::Hashmap pcd_block_hashmap(total_block_count, core::Dtype::Int32,
                                    core::Dtype::Int32, {3}, {1}, device);
    core::Tensor block_addrs, block_masks;
    pcd_block_hashmap.Activate(block_coordi.Slice(0, 0, count.Item<int>()),
                               block_addrs, block_masks);
    voxel_block_coords = block_coordi.IndexGet({block_masks});
}

void RayCastCUDA(std::shared_ptr<core::DeviceHashmap>& hashmap,
                 core::Tensor& block_values,
                 core::Tensor& vertex_map,
                 core::Tensor& color_map,
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

    auto cuda_hashmap =
            std::dynamic_pointer_cast<core::STDGPUHashmap<Key, Hash>>(hashmap);
    if (cuda_hashmap == nullptr) {
        utility::LogError(
                "Unsupported backend: CUDA raycasting only supports STDGPU.");
    }
    auto hashmap_ctx = cuda_hashmap->GetContext();

    NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);
    NDArrayIndexer vertex_map_indexer(vertex_map, 2);
    NDArrayIndexer color_map_indexer(color_map, 2);

    TransformIndexer transform_indexer(intrinsics, pose, 1);

    int64_t rows = vertex_map_indexer.GetShape(0);
    int64_t cols = vertex_map_indexer.GetShape(1);

    float block_size = voxel_size * block_resolution;
    DISPATCH_BYTESIZE_TO_VOXEL(
            voxel_block_buffer_indexer.ElementByteSize(), [&]() {
                core::kernel::CUDALauncher::LaunchGeneralKernel(
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
                                    // printf("%d %d %d-> %d %d %d->%d %d %d\n",
                                    //        x_v, y_v, z_v, dx_b, dy_b, dz_b,
                                    //        x_vn, y_vn, z_vn);
                                    Key key;
                                    key(0) = x_b + dx_b;
                                    key(1) = y_b + dy_b;
                                    key(2) = z_b + dz_b;

                                    auto iter = hashmap_ctx.find(key);
                                    if (iter == hashmap_ctx.end())
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
                                auto iter = hashmap_ctx.find(key);
                                if (iter == hashmap_ctx.end()) return nullptr;

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

                                    // Color map assignment:
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
                                    auto iter = hashmap_ctx.find(key);
                                    if (iter == hashmap_ctx.end()) break;

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

                                    float sum_weight = 0.0;
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

                                        // if (voxel_ptr_k &&
                                        //     voxel_ptr_k->GetWeight() > 0) {
                                        //     sum_weight += ratio;
                                        //     color[0] +=
                                        //             ratio *
                                        //             voxel_ptr_k->GetR();
                                        //     color[1] +=
                                        //             ratio *
                                        //             voxel_ptr_k->GetG();
                                        //     color[2] +=
                                        //             ratio *
                                        //             voxel_ptr_k->GetB();
                                        // }

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
                                                color[dim] +=
                                                        ratio *
                                                        voxel_ptr_k_plus
                                                                ->GetTSDF() /
                                                        (2 * voxel_size);
                                                valid = true;
                                            }

                                            if (voxel_ptr_k_minus &&
                                                voxel_ptr_k_minus->GetWeight() >
                                                        0) {
                                                color[dim] -=
                                                        ratio *
                                                        voxel_ptr_k_minus
                                                                ->GetTSDF() /
                                                        (2 * voxel_size);
                                                valid = true;
                                            }
                                            sum_weight += valid ? ratio : 0;
                                        }
                                    }

                                    // sum_weight *= 255.0;
                                    if (sum_weight > 0) {
                                        color[0] = (color[0]) / sum_weight;
                                        color[1] = (color[1]) / sum_weight;
                                        color[2] = (color[2]) / sum_weight;
                                        // color[0] /= sum_weight;
                                        // color[1] /= sum_weight;
                                        // color[2] /= sum_weight;
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

    // For profiling
    cudaDeviceSynchronize();
}

}  // namespace tsdf
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
