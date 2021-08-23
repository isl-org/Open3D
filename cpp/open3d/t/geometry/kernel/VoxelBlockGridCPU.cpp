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

#include <tbb/concurrent_unordered_set.h>

#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/CPU/TBBHashBackend.h"
#include "open3d/core/hashmap/Dispatch.h"
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
    Coord3i(int x, int y, int z) : x_(x), y_(y), z_(z) {}
    bool operator==(const Coord3i &other) const {
        return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
    }

    int64_t x_;
    int64_t y_;
    int64_t z_;
};

struct Coord3iHash {
    size_t operator()(const Coord3i &k) const {
        static const size_t p0 = 73856093;
        static const size_t p1 = 19349669;
        static const size_t p2 = 83492791;

        return (static_cast<size_t>(k.x_) * p0) ^
               (static_cast<size_t>(k.y_) * p1) ^
               (static_cast<size_t>(k.z_) * p2);
    }
};

void PointCloudTouchCPU(
        std::shared_ptr<core::HashMap>
                &hashmap,  // dummy for now, one pass insertion is faster
        const core::Tensor &points,
        core::Tensor &voxel_block_coords,
        int64_t voxel_grid_resolution,
        float voxel_size,
        float sdf_trunc) {
    int64_t resolution = voxel_grid_resolution;
    float block_size = voxel_size * resolution;

    int64_t n = points.GetLength();
    const float *pcd_ptr = static_cast<const float *>(points.GetDataPtr());

    tbb::concurrent_unordered_set<Coord3i, Coord3iHash> set;
    core::ParallelFor(core::Device("CPU:0"), n, [&](int64_t workload_idx) {
        float x = pcd_ptr[3 * workload_idx + 0];
        float y = pcd_ptr[3 * workload_idx + 1];
        float z = pcd_ptr[3 * workload_idx + 2];

        int xb_lo = static_cast<int>(std::floor((x - sdf_trunc) / block_size));
        int xb_hi = static_cast<int>(std::floor((x + sdf_trunc) / block_size));
        int yb_lo = static_cast<int>(std::floor((y - sdf_trunc) / block_size));
        int yb_hi = static_cast<int>(std::floor((y + sdf_trunc) / block_size));
        int zb_lo = static_cast<int>(std::floor((z - sdf_trunc) / block_size));
        int zb_hi = static_cast<int>(std::floor((z + sdf_trunc) / block_size));
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

    voxel_block_coords =
            core::Tensor({block_count, 3}, core::Int32, points.GetDevice());
    int *block_coords_ptr = static_cast<int *>(voxel_block_coords.GetDataPtr());
    int count = 0;
    for (auto it = set.begin(); it != set.end(); ++it, ++count) {
        int64_t offset = count * 3;
        block_coords_ptr[offset + 0] = static_cast<int>(it->x_);
        block_coords_ptr[offset + 1] = static_cast<int>(it->y_);
        block_coords_ptr[offset + 2] = static_cast<int>(it->z_);
    }
}

void DepthTouchCPU(std::shared_ptr<core::HashMap> &hashmap,
                   const core::Tensor &depth,
                   const core::Tensor &intrinsics,
                   const core::Tensor &extrinsics,
                   core::Tensor &voxel_block_coords,
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
    int64_t n = rows_strided * cols_strided;

    int64_t resolution = voxel_grid_resolution;
    float block_size = voxel_size * resolution;

    tbb::concurrent_unordered_set<Coord3i, Coord3iHash> set;
    DISPATCH_DTYPE_TO_TEMPLATE(depth.GetDtype(), [&]() {
        core::ParallelFor(device, n, [&](int64_t workload_idx) {
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
                const float t_min = std::max(d - sdf_trunc, 0.0f);
                const float t_max = std::min(d + sdf_trunc, depth_max);
                const float t_step = (t_max - t_min) / step_size;

                float t = t_min;
                for (int step = 0; step <= step_size; ++step) {
                    int xb = static_cast<int>(
                            std::floor((x_o + t * x_d) / block_size));
                    int yb = static_cast<int>(
                            std::floor((y_o + t * y_d) / block_size));
                    int zb = static_cast<int>(
                            std::floor((z_o + t * z_d) / block_size));
                    set.emplace(xb, yb, zb);
                    t += t_step;
                }
            }
        });
    });

    int64_t block_count = set.size();
    if (block_count == 0) {
        utility::LogError(
                "No block is touched in TSDF volume, abort integration. Please "
                "check specified parameters, "
                "especially depth_scale and voxel_size");
    }

    voxel_block_coords = core::Tensor({block_count, 3}, core::Int32, device);
    int *block_coords_ptr = voxel_block_coords.GetDataPtr<int>();
    int count = 0;
    for (auto it = set.begin(); it != set.end(); ++it, ++count) {
        int64_t offset = count * 3;
        block_coords_ptr[offset + 0] = static_cast<int>(it->x_);
        block_coords_ptr[offset + 1] = static_cast<int>(it->y_);
        block_coords_ptr[offset + 2] = static_cast<int>(it->z_);
    }
}

#define FN_ARGUMENTS                                                        \
    const core::Tensor &depth, const core::Tensor &color,                   \
            const core::Tensor &indices, const core::Tensor &block_keys,    \
            std::vector<core::Tensor> &block_values,                        \
            const core::Tensor &intrinsics, const core::Tensor &extrinsics, \
            int64_t resolution, float voxel_size, float sdf_trunc,          \
            float depth_scale, float depth_max

template void IntegrateCPU<uint16_t, uint8_t, float, uint16_t, uint16_t>(
        FN_ARGUMENTS);
template void IntegrateCPU<uint16_t, uint8_t, float, float, float>(
        FN_ARGUMENTS);
template void IntegrateCPU<float, float, float, uint16_t, uint16_t>(
        FN_ARGUMENTS);
template void IntegrateCPU<float, float, float, float, float>(FN_ARGUMENTS);

#undef FN_ARGUMENTS

#define FN_ARGUMENTS                                                     \
    std::shared_ptr<core::HashMap> &hashmap,                             \
            const std::vector<core::Tensor> &block_values,               \
            const core::Tensor &range_map, core::Tensor &vertex_map,     \
            core::Tensor &depth_map, core::Tensor &color_map,            \
            core::Tensor &normal_map, const core::Tensor &intrinsics,    \
            const core::Tensor &extrinsics, int h, int w,                \
            int64_t block_resolution, float voxel_size, float sdf_trunc, \
            float depth_scale, float depth_min, float depth_max,         \
            float weight_threshold

template void RayCastCPU<float, uint16_t, uint16_t>(FN_ARGUMENTS);
template void RayCastCPU<float, float, float>(FN_ARGUMENTS);

#undef FN_ARGUMENTS

#define FN_ARGUMENTS                                                           \
    const core::Tensor &block_indices, const core::Tensor &nb_block_indices,   \
            const core::Tensor &nb_block_masks,                                \
            const core::Tensor &block_keys,                                    \
            const std::vector<core::Tensor> &block_values,                     \
            core::Tensor &points, core::Tensor &normals, core::Tensor &colors, \
            int64_t block_resolution, float voxel_size,                        \
            float weight_threshold, int &valid_size

template void ExtractSurfacePointsCPU<float, uint16_t, uint16_t>(FN_ARGUMENTS);
template void ExtractSurfacePointsCPU<float, float, float>(FN_ARGUMENTS);

#undef FN_ARGUMENTS

}  // namespace voxel_grid
}  // namespace kernel
}  // namespace geometry
}  // namespace t
}  // namespace open3d
