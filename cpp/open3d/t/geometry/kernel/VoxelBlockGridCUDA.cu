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

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/CUDA/StdGPUHashBackend.h"
#include "open3d/core/hashmap/DeviceHashBackend.h"
#include "open3d/core/hashmap/Dispatch.h"
#include "open3d/core/hashmap/HashMap.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/geometry/kernel/VoxelBlockGrid.h"
#include "open3d/t/geometry/kernel/VoxelBlockGridImpl.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {
namespace kernel {
namespace voxel_grid {

struct Coord3i {
    OPEN3D_HOST_DEVICE Coord3i(int x, int y, int z) : x_(x), y_(y), z_(z) {}
    OPEN3D_HOST_DEVICE bool operator==(const Coord3i& other) const {
        return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
    }

    int64_t x_;
    int64_t y_;
    int64_t z_;
};

void TouchCUDA(std::shared_ptr<core::HashMap>& hashmap,
               const core::Tensor& points,
               core::Tensor& voxel_block_coords,
               int64_t voxel_grid_resolution,
               float voxel_size,
               float sdf_trunc) {
    int64_t resolution = voxel_grid_resolution;
    float block_size = voxel_size * resolution;

    int64_t n = points.GetLength();
    const float* pcd_ptr = static_cast<const float*>(points.GetDataPtr());

    core::Device device = points.GetDevice();
    core::Tensor block_coordi({8 * n, 3}, core::Int32, device);
    int* block_coordi_ptr = static_cast<int*>(block_coordi.GetDataPtr());
    core::Tensor count(std::vector<int>{0}, {}, core::Int32, device);
    int* count_ptr = static_cast<int*>(count.GetDataPtr());

    core::ParallelFor(
            hashmap->GetDevice(), n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
                float x = pcd_ptr[3 * workload_idx + 0];
                float y = pcd_ptr[3 * workload_idx + 1];
                float z = pcd_ptr[3 * workload_idx + 2];

                int xb_lo =
                        static_cast<int>(floorf((x - sdf_trunc) / block_size));
                int xb_hi =
                        static_cast<int>(floorf((x + sdf_trunc) / block_size));
                int yb_lo =
                        static_cast<int>(floorf((y - sdf_trunc) / block_size));
                int yb_hi =
                        static_cast<int>(floorf((y + sdf_trunc) / block_size));
                int zb_lo =
                        static_cast<int>(floorf((z - sdf_trunc) / block_size));
                int zb_hi =
                        static_cast<int>(floorf((z + sdf_trunc) / block_size));

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
    core::Tensor block_buf_indices, block_masks;
    hashmap->Activate(block_coordi.Slice(0, 0, count.Item<int>()),
                      block_buf_indices, block_masks);
    voxel_block_coords = block_coordi.IndexGet({block_masks});
}

void DepthTouchCUDA(std::shared_ptr<core::HashMap>& hashmap,
                    const core::Tensor& depth,
                    const core::Tensor& intrinsics,
                    const core::Tensor& extrinsics,
                    core::Tensor& voxel_block_coords,
                    int64_t voxel_grid_resolution,
                    float voxel_size,
                    float sdf_trunc,
                    float depth_scale,
                    float depth_max,
                    int stride) {
    core::Device device = depth.GetDevice();
    NDArrayIndexer depth_indexer(depth, 2);
    core::Tensor pose = t::geometry::InverseTransformation(extrinsics);
    TransformIndexer ti(intrinsics, pose, 1.0f);

    // Output
    int64_t rows_strided = depth_indexer.GetShape(0) / stride;
    int64_t cols_strided = depth_indexer.GetShape(1) / stride;

    // Counter
    core::Tensor count(std::vector<int>{0}, {1}, core::Dtype::Int32, device);
    int* count_ptr = count.GetDataPtr<int>();

    int64_t n = rows_strided * cols_strided;
    static core::Tensor block_coordi;
    if (block_coordi.GetLength() != 4 * n) {
        block_coordi = core::Tensor({4 * n, 3}, core::Dtype::Int32, device);
    }

    int* block_coordi_ptr = block_coordi.GetDataPtr<int>();

    int64_t resolution = voxel_grid_resolution;
    float block_size = voxel_size * resolution;

    DISPATCH_DTYPE_TO_TEMPLATE(depth.GetDtype(), [&]() {
        core::ParallelFor(device, n, [=] OPEN3D_DEVICE(int64_t workload_idx) {
            int64_t y = (workload_idx / cols_strided) * stride;
            int64_t x = (workload_idx % cols_strided) * stride;

            float d = *depth_indexer.GetDataPtr<scalar_t>(x, y) / depth_scale;
            if (d > 0 && d < depth_max) {
                float x_c = 0, y_c = 0, z_c = 0;
                ti.Unproject(static_cast<float>(x), static_cast<float>(y), 1.0,
                             &x_c, &y_c, &z_c);
                float x_g = 0, y_g = 0, z_g = 0;
                ti.RigidTransform(x_c, y_c, z_c, &x_g, &y_g, &z_g);

                // Origin
                float x_o = 0, y_o = 0, z_o = 0;
                ti.GetCameraPosition(&x_o, &y_o, &z_o);

                // Direction
                float x_d = x_g - x_o;
                float y_d = y_g - y_o;
                float z_d = z_g - z_o;

                const int step_size = 3;
                const float t_min = max(d - sdf_trunc, 0.0);
                const float t_max = min(d + sdf_trunc, depth_max);
                const float t_step = (t_max - t_min) / step_size;

                float t = t_min;
                int idx = atomicAdd(count_ptr, step_size);
                for (int step = 0; step <= step_size; ++step) {
                    int offset = (step + idx) * 3;

                    int xb = static_cast<int>(
                            floorf((x_o + t * x_d) / block_size));
                    int yb = static_cast<int>(
                            floorf((y_o + t * y_d) / block_size));
                    int zb = static_cast<int>(
                            floorf((z_o + t * z_d) / block_size));

                    block_coordi_ptr[offset + 0] = xb;
                    block_coordi_ptr[offset + 1] = yb;
                    block_coordi_ptr[offset + 2] = zb;

                    t += t_step;
                }
            }
        });
    });

    int total_block_count = count[0].Item<int>();
    if (total_block_count == 0) {
        utility::LogError(
                "No block is touched in TSDF volume, "
                "abort integration. Please check specified parameters, "
                "especially depth_scale and voxel_size");
    }
    block_coordi = block_coordi.Slice(0, 0, total_block_count);
    core::Tensor block_addrs, block_masks;
    hashmap->Activate(block_coordi, block_addrs, block_masks);

    // Customized IndexGet (generic version too slow)
    voxel_block_coords =
            core::Tensor({hashmap->Size(), 3}, core::Int32, device);
    int* voxel_block_coord_ptr = voxel_block_coords.GetDataPtr<int>();
    bool* block_masks_ptr = block_masks.GetDataPtr<bool>();
    count[0] = 0;
    core::ParallelFor(device, total_block_count,
                      [=] OPEN3D_DEVICE(int64_t workload_idx) {
                          if (block_masks_ptr[workload_idx]) {
                              int idx = OPEN3D_ATOMIC_ADD(count_ptr, 1);
                              int offset_lhs = 3 * idx;
                              int offset_rhs = 3 * workload_idx;
                              voxel_block_coord_ptr[offset_lhs + 0] =
                                      block_coordi_ptr[offset_rhs + 0];
                              voxel_block_coord_ptr[offset_lhs + 1] =
                                      block_coordi_ptr[offset_rhs + 1];
                              voxel_block_coord_ptr[offset_lhs + 2] =
                                      block_coordi_ptr[offset_rhs + 2];
                          }
                      });
    OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace voxel_grid
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
