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

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/Hashmap.h"
#include "open3d/t/geometry/Geometry.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/PointCloud.h"

/// ControlGrid is a spatially hashed voxel grid used for non-rigid point cloud
/// registration and TSDF integration.
/// Each grid stores a R^3 coordinate after non-rigid transformation.
/// You can imagine a control grid as a jelly that is warped upon perturbation
/// with its overall shape preserved.
/// Reference:
/// https://github.com/qianyizh/ElasticReconstruction/blob/master/FragmentOptimizer/OptApp.cpp
/// http://vladlen.info/papers/elastic-fragments.pdf

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {
class ControlGrid {
public:
    static const std::string kAttrNbGridIdx;
    static const std::string kAttrNbGridPointInterp;
    static const std::string kAttrNbGridNormalInterp;

    /// Default constructor.
    ControlGrid() {}

    /// Constructor with initial grid size and grid count estimation.
    ControlGrid(float grid_size,
                int64_t grid_count = 1000,
                const core::Device& device = core::Device("CPU:0"));

    /// Constructor with known keys (Int32 x 3 positions) and values (Float32 x
    /// 3 shifted positions).
    ControlGrid(float grid_size,
                const core::Tensor& keys,
                const core::Tensor& values,
                const core::Device& device = core::Device("CPU:0"));
    ~ControlGrid() {}

    /// Allocate control grids in the shared camera space.
    void Touch(const geometry::PointCloud& pcd);

    /// Force rehashing, so that all entries are remapped to [0, size) and form
    /// a contiguous index map.
    void Compactify();

    /// \return A 6-way neighbor grid map for all the active entries.
    /// - addrs (N, ) Active indices in the buffer
    /// - addrs_nb (N, 6) Neighbor indices (including non-allocated entries) for
    /// the active entries.
    /// - masks_nb (N, 6) corresponding neighbor masks.
    std::tuple<core::Tensor, core::Tensor, core::Tensor> GetNeighborGridMap();

    /// Parameterize an input point cloud with the control grids via indexing
    /// and interpolation.
    /// \return A PointCloud with parameterization attributes:
    /// - neighbors: (8, ) Int64, index of 8 neighbor control grid points.
    /// - ratios: (8, ) Float32, interpolation ratios of 8 neighbor control grid
    /// points.
    geometry::PointCloud Parameterize(const geometry::PointCloud& pcd);

    /// Non-rigidly warp a point cloud using the control grid.
    geometry::PointCloud Warp(const geometry::PointCloud& pcd);

    /// Non-rigidly warp a depth image by
    /// - unprojecting the image to a point cloud;
    /// - warp the point cloud;
    /// - project the warped point cloud back to the image.
    geometry::Image Warp(const geometry::Image& depth,
                         const core::Tensor& intrinsics,
                         const core::Tensor& extrinsics,
                         float depth_scale,
                         float depth_max);

    std::pair<geometry::Image, geometry::Image> Warp(
            const geometry::Image& depth,
            const geometry::Image& color,
            const core::Tensor& intrinsics,
            const core::Tensor& extrinsics,
            float depth_scale,
            float depth_max);

    /// Get control grid original positions directly from tensor keys.
    core::Tensor GetInitPositions() {
        return ctr_hashmap_->GetKeyTensor().To(core::Dtype::Float32) *
               grid_size_;
    }

    /// Get control grid shifted positions from tensor values (optimized
    /// in-place).
    core::Tensor GetCurrPositions() { return ctr_hashmap_->GetValueTensor(); }

    std::shared_ptr<core::Hashmap> GetHashmap() { return ctr_hashmap_; }
    int64_t Size() { return ctr_hashmap_->Size(); }

    int64_t anchor_idx_ = 0;

private:
    float grid_size_;

    core::Device device_ = core::Device("CPU:0");
    std::shared_ptr<core::Hashmap> ctr_hashmap_;
};

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
