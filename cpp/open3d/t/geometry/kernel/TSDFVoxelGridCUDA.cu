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
                 float depth_max) {
    auto cuda_hashmap = std::dynamic_pointer_cast<
            core::CUDAHashmap<core::DefaultHash, core::DefaultKeyEq>>(hashmap);
    auto hashmap_ctx = cuda_hashmap->GetContext();

    TransformIndexer transform_indexer(intrinsics, extrinsics, 1);
    NDArrayIndexer vertex_map_indexer(vertex_map, 2);
    core::SizeVector shape = vertex_map.GetShape();

    int64_t rows = vertex_map_indexer.GetShape(0);
    int64_t cols = vertex_map_indexer.GetShape(1);

    float block_size = voxel_size * block_resolution;
    core::kernel::CUDALauncher::LaunchGeneralKernel(
            rows * cols, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                // Copy the constant reference
                auto hashmap_ctx_instance = hashmap_ctx;
                int64_t y = workload_idx / cols;
                int64_t x = workload_idx % cols;

                float d = 1.0f;
                float x_c = 0, y_c = 0, z_c = 0;
                transform_indexer.Unproject(static_cast<float>(x),
                                            static_cast<float>(y), d, &x_c,
                                            &y_c, &z_c);

                int key[3];
                key[0] = static_cast<int>(floor(x_c / block_size));
                key[1] = static_cast<int>(floor(y_c / block_size));
                key[2] = static_cast<int>(floor(z_c / block_size));

                uint32_t bucket_id = hashmap_ctx.ComputeBucket(key);
                uint32_t lane_id = workload_idx & 0x1F;
                core::Pair<core::addr_t, bool> result =
                        hashmap_ctx_instance.Find(
                                true, lane_id, bucket_id,
                                static_cast<const void*>(key));
                if (result.second) {
                    printf("%d\n", result.first);
                }
            });
}

}  // namespace tsdf
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
