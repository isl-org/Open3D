// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/HashMap.h"
#include "open3d/t/geometry/Geometry.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/PointCloud.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace slac {

/// ControlGrid is a spatially hashed voxel grid used for non-rigid point cloud
/// registration and TSDF integration.
/// Each grid stores a map from the initial grid location to the deformed
/// location.
/// You can imagine a control grid as a jelly that is warped upon
/// perturbation with its overall shape preserved.
/// Reference:
/// https://github.com/qianyizh/ElasticReconstruction/blob/master/FragmentOptimizer/OptApp.cpp
/// http://vladlen.info/papers/elastic-fragments.pdf
class ControlGrid {
public:
    /// Attributes used to extend the point cloud to support neighbor lookup.
    /// 8 neighbor grid index per point.
    static const std::string kGrid8NbIndices;
    /// 8 neighbor grid interpolation ratio for vertex per point.
    static const std::string kGrid8NbVertexInterpRatios;
    /// 8 neighbor grid interpolation ratio for normal per point.
    static const std::string kGrid8NbNormalInterpRatios;

    /// Default constructor.
    ControlGrid() = default;

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

    /// Allocate control grids in the shared camera space.
    void Touch(const geometry::PointCloud& pcd);

    /// Force rehashing, so that all entries are remapped to [0, size) and form
    /// a contiguous index map.
    void Compactify();

    /// Get the neighbor indices per grid to construct the regularizer.
    /// \return A 6-way neighbor grid map for all the active entries of shape
    /// (N, ).
    /// - buf_indices Active indices in the buffer of shape (N, )
    /// - buf_indices_nb Neighbor indices (including non-allocated entries) for
    /// the active entries of shape (N, 6).
    /// - masks_nb Corresponding neighbor masks of shape (N, 6).
    std::tuple<core::Tensor, core::Tensor, core::Tensor> GetNeighborGridMap();

    /// Parameterize an input point cloud by embedding each point in the grid
    /// with 8 corners via indexing and interpolation.
    /// \return A PointCloud with parameterization attributes:
    /// - neighbors: Index of 8 neighbor control grid points of shape (8, ) in
    /// Int64.
    /// - ratios: Interpolation ratios of 8 neighbor control grid
    /// points of shape (8, ) in Float32.
    geometry::PointCloud Parameterize(const geometry::PointCloud& pcd);

    /// Non-rigidly deform a point cloud using the control grid.
    geometry::PointCloud Deform(const geometry::PointCloud& pcd);

    /// Non-rigidly deform a depth image by
    /// - unprojecting the depth image to a point cloud;
    /// - deform the point cloud;
    /// - project the deformed point cloud back to the image.
    geometry::Image Deform(const geometry::Image& depth,
                           const core::Tensor& intrinsics,
                           const core::Tensor& extrinsics,
                           float depth_scale,
                           float depth_max);

    /// Non-rigidly deform an RGBD image by
    /// - unprojecting the depth image to a point cloud;
    /// - deform the point cloud;
    /// - project the deformed point cloud back to the image.
    geometry::RGBDImage Deform(const geometry::RGBDImage& rgbd,
                               const core::Tensor& intrinsics,
                               const core::Tensor& extrinsics,
                               float depth_scale,
                               float depth_max);

    /// Get control grid original positions directly from tensor keys.
    core::Tensor GetInitPositions() {
        return ctr_hashmap_->GetKeyTensor().To(core::Float32) * grid_size_;
    }

    /// Get control grid shifted positions from tensor values (optimized
    /// in-place).
    core::Tensor GetCurrPositions() { return ctr_hashmap_->GetValueTensor(); }

    std::shared_ptr<core::HashMap> GetHashMap() { return ctr_hashmap_; }
    int64_t Size() { return ctr_hashmap_->Size(); }

    core::Device GetDevice() { return device_; }
    int64_t GetAnchorIdx() { return anchor_idx_; }

private:
    /// Anchor grid point index in the regularizer.
    int64_t anchor_idx_ = 0;

    /// Grid size, typically much larger than the voxel size (e.g. 0.375m).
    float grid_size_;

    core::Device device_ = core::Device("CPU:0");
    std::shared_ptr<core::HashMap> ctr_hashmap_;
};

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
