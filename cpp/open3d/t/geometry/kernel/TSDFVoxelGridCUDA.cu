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
#include "open3d/core/hashmap/CUDA/HashmapCUDA.h"
#include "open3d/core/hashmap/DeviceHashmap.h"
#include "open3d/core/hashmap/Hashmap.h"
#include "open3d/core/kernel/CUDALauncher.cuh"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/TSDFVoxelGrid.h"
#include "open3d/t/geometry/kernel/TSDFVoxelGridShared.h"
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

void RayCastCUDA(std::shared_ptr<core::DefaultDeviceHashmap>& hashmap,
                 core::Tensor& block_values,
                 core::Tensor& vertex_map,
                 core::Tensor& color_map,
                 const core::Tensor& intrinsics,
                 const core::Tensor& extrinsics,
                 int64_t block_resolution,
                 float voxel_size,
                 float sdf_trunc,
                 int max_steps,
                 float depth_min,
                 float depth_max,
                 float weight_threshold) {
    auto cuda_hashmap = std::dynamic_pointer_cast<
            core::CUDAHashmap<core::DefaultHash, core::DefaultKeyEq>>(hashmap);
    auto hashmap_ctx = cuda_hashmap->GetContext();

    NDArrayIndexer voxel_block_buffer_indexer(block_values, 4);
    NDArrayIndexer vertex_map_indexer(vertex_map, 2);
    NDArrayIndexer color_map_indexer(color_map, 2);

    TransformIndexer transform_indexer(intrinsics, extrinsics, 1);

    int64_t rows = vertex_map_indexer.GetShape(0);
    int64_t cols = vertex_map_indexer.GetShape(1);

    float block_size = voxel_size * block_resolution;
    DISPATCH_BYTESIZE_TO_VOXEL(
            voxel_block_buffer_indexer.ElementByteSize(), [&]() {
                core::kernel::CUDALauncher::LaunchGeneralKernel(
                        rows * cols, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                            auto hashmap_ctx_instance = hashmap_ctx;

                            int64_t y = workload_idx / cols;
                            int64_t x = workload_idx % cols;
                            uint32_t lane_id = workload_idx & 0x1F;

                            float t = depth_min;

                            // Coordinates in camera and global
                            float x_c = 0, y_c = 0, z_c = 0;
                            float x_g = 0, y_g = 0, z_g = 0;
                            float x_o = 0, y_o = 0, z_o = 0;

                            // Coordinates in voxel blocks and voxels
                            int key[3] = {0};
                            int x_b = 0, y_b = 0, z_b = 0;
                            int x_v = 0, y_v = 0, z_v = 0;

                            // Iterative ray intersection check
                            float t_prev = t;
                            float tsdf_prev = 1.0f;
                            bool active = true;

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
                                // Release a warp if all of the threads are
                                // inactive.
                                int all_inactive = __all_sync(
                                        core::kSyncLanesMask, !active);
                                if (all_inactive) {
                                    break;
                                }

                                x_g = x_o + t * x_d;
                                y_g = y_o + t * y_d;
                                z_g = z_o + t * z_d;

                                x_b = static_cast<int>(floor(x_g / block_size));
                                y_b = static_cast<int>(floor(y_g / block_size));
                                z_b = static_cast<int>(floor(z_g / block_size));

                                key[0] = x_b;
                                key[1] = y_b;
                                key[2] = z_b;

                                uint32_t bucket_id =
                                        hashmap_ctx.ComputeBucket(key);
                                core::Pair<core::addr_t, bool> result =
                                        hashmap_ctx_instance.Find(
                                                active, lane_id, bucket_id,
                                                static_cast<const void*>(key));

                                bool flag = active && result.second;
                                if (!flag) {
                                    t_prev = t;
                                    t += block_size;
                                    continue;
                                }

                                core::addr_t block_addr = result.first;
                                x_v = int((x_g - x_b * block_size) /
                                          voxel_size);
                                y_v = int((y_g - y_b * block_size) /
                                          voxel_size);
                                z_v = int((z_g - z_b * block_size) /
                                          voxel_size);

                                voxel_t* voxel_ptr =
                                        voxel_block_buffer_indexer
                                                .GetDataPtrFromCoord<voxel_t>(
                                                        x_v, y_v, z_v,
                                                        block_addr);
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

                                    float* vertex =
                                            vertex_map_indexer
                                                    .GetDataPtrFromCoord<float>(
                                                            x, y);
                                    vertex[0] = x_g;
                                    vertex[1] = y_g;
                                    vertex[2] = z_g;

                                    float* color =
                                            color_map_indexer
                                                    .GetDataPtrFromCoord<float>(
                                                            x, y);
                                    color[0] = voxel_ptr->GetR() / 255.0f;
                                    color[1] = voxel_ptr->GetG() / 255.0f;
                                    color[2] = voxel_ptr->GetB() / 255.0f;

                                    active = false;
                                    continue;
                                }

                                tsdf_prev = tsdf;
                                t_prev = t;
                                float delta = tsdf * sdf_trunc;
                                t += delta < voxel_size ? voxel_size : delta;
                                continue;
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
