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

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/CPULauncher.h"
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

void TouchCPU(const core::Tensor& points,
              core::Tensor& voxel_block_coords,
              int64_t voxel_grid_resolution,
              float voxel_size,
              float sdf_trunc) {
    int64_t resolution = voxel_grid_resolution;
    float block_size = voxel_size * resolution;

    int64_t n = points.GetLength();
    const float* pcd_ptr = static_cast<const float*>(points.GetDataPtr());

    tbb::concurrent_unordered_set<Coord3i, Coord3iHash> set;
    core::kernel::CPULauncher::LaunchGeneralKernel(
            n, [&](int64_t workload_idx) {
                float x = pcd_ptr[3 * workload_idx + 0];
                float y = pcd_ptr[3 * workload_idx + 1];
                float z = pcd_ptr[3 * workload_idx + 2];

                int xb_lo = static_cast<int>(
                        std::floor((x - sdf_trunc) / block_size));
                int xb_hi = static_cast<int>(
                        std::floor((x + sdf_trunc) / block_size));
                int yb_lo = static_cast<int>(
                        std::floor((y - sdf_trunc) / block_size));
                int yb_hi = static_cast<int>(
                        std::floor((y + sdf_trunc) / block_size));
                int zb_lo = static_cast<int>(
                        std::floor((z - sdf_trunc) / block_size));
                int zb_hi = static_cast<int>(
                        std::floor((z + sdf_trunc) / block_size));
                for (int xb = xb_lo; xb <= xb_hi; ++xb) {
                    for (int yb = yb_lo; yb <= yb_hi; ++yb) {
                        for (int zb = zb_lo; zb <= zb_hi; ++zb) {
                            set.emplace(xb, yb, zb);
                        }
                    }
                }
            });

    int64_t block_count = set.size();
    if (block_count == 0) {
        utility::LogError(
                "No block is touched in TSDF volume, abort integration. Please "
                "check specified parameters, "
                "especially depth_scale and voxel_size");
    }

    voxel_block_coords = core::Tensor({block_count, 3}, core::Dtype::Int32,
                                      points.GetDevice());
    int* block_coords_ptr = static_cast<int*>(voxel_block_coords.GetDataPtr());
    int count = 0;
    for (auto it = set.begin(); it != set.end(); ++it, ++count) {
        int64_t offset = count * 3;
        block_coords_ptr[offset + 0] = static_cast<int>(it->x_);
        block_coords_ptr[offset + 1] = static_cast<int>(it->y_);
        block_coords_ptr[offset + 2] = static_cast<int>(it->z_);
    }
}
}  // namespace tsdf
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
