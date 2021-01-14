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

    ControlGrid(float grid_size,
                int64_t grid_count = 1000,
                const core::Device& device = core::Device("CPU:0"));
    ~ControlGrid(){};

    int64_t Size() { return ctr_hashmap_->Size(); }

    /// Allocate control grids in the shared camera space.
    void Touch(const geometry::PointCloud& pcd);

    /// Parameterize an input point cloud with the control grids via indexing
    /// and interpolation.
    /// \return A PointCloud with parameterization attributes:
    /// - neighbors: (8, ) Int64, index of 8 neighbor control grid points.
    /// - ratios: (8, ) Float32, interpolation ratios of 8 neighbor control grid
    /// points.
    geometry::PointCloud Parameterize(const geometry::PointCloud& pcd);

    /// Get control grid positions directly by buffer tensor indices.
    core::Tensor GetPositions() { return ctr_hashmap_->GetValueTensor(); }

    /// Warp a point cloud using control grid.
    geometry::PointCloud Warp(const geometry::PointCloud& pcd);

    /// Warp a depth image by
    /// - unprojecting the image to a point cloud;
    /// - warp the point cloud;
    /// - project the warped point cloud back to the image.
    geometry::Image Warp(const geometry::Image& depth,
                         const core::Tensor& extrinsics,
                         const core::Tensor& intrinsics);

    /// Warp an RGB-D image with the geometry provided by the depth.
    std::pair<geometry::Image, geometry::Image> Warp(
            const geometry::Image& depth,
            const geometry::Image& color,
            const core::Tensor& extrinsics,
            const core::Tensor& intrinsics);

private:
    float grid_size_;

    core::Device device_ = core::Device("CPU:0");
    std::shared_ptr<core::Hashmap> ctr_hashmap_;
};

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
