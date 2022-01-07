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

#pragma once

#include <Eigen/Core>
#include <memory>
#include <unordered_map>
#include <vector>

#include "open3d/geometry/Geometry3D.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {

namespace camera {
class PinholeCameraParameters;
}

namespace geometry {

class PointCloud;
class TriangleMesh;
class Octree;
class Image;

/// \class Voxel
///
/// \brief Base Voxel class, containing grid id and color.
class Voxel {
public:
    /// \brief Default Constructor.
    Voxel() {}
    /// \brief Parameterized Constructor.
    ///
    /// \param grid_index Grid coordinate index of the voxel.
    Voxel(const Eigen::Vector3i &grid_index) : grid_index_(grid_index) {}
    /// \brief Parameterized Constructor.
    ///
    /// \param grid_index Grid coordinate index of the voxel.
    /// \param color Color of the voxel.
    Voxel(const Eigen::Vector3i &grid_index, const Eigen::Vector3d &color)
        : grid_index_(grid_index), color_(color) {}
    ~Voxel() {}

public:
    /// Grid coordinate index of the voxel.
    Eigen::Vector3i grid_index_ = Eigen::Vector3i(0, 0, 0);
    /// Color of the voxel.
    Eigen::Vector3d color_ = Eigen::Vector3d(0, 0, 0);
};

/// \class VoxelGrid
///
/// \brief VoxelGrid is a collection of voxels which are aligned in grid.
class VoxelGrid : public Geometry3D {
public:
    /// \brief Default Constructor.
    VoxelGrid() : Geometry3D(Geometry::GeometryType::VoxelGrid) {}
    /// \brief Copy Constructor.
    VoxelGrid(const VoxelGrid &src_voxel_grid);
    ~VoxelGrid() override {}

    VoxelGrid &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetMaxBound() const override;
    Eigen::Vector3d GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    OrientedBoundingBox GetOrientedBoundingBox(
            bool robust = false) const override;
    VoxelGrid &Transform(const Eigen::Matrix4d &transformation) override;
    VoxelGrid &Translate(const Eigen::Vector3d &translation,
                         bool relative = true) override;
    VoxelGrid &Scale(const double scale,
                     const Eigen::Vector3d &center) override;
    VoxelGrid &Rotate(const Eigen::Matrix3d &R,
                      const Eigen::Vector3d &center) override;

    VoxelGrid &operator+=(const VoxelGrid &voxelgrid);
    VoxelGrid operator+(const VoxelGrid &voxelgrid) const;

    /// Returns `true` if the voxel grid contains voxels.
    bool HasVoxels() const { return voxels_.size() > 0; }
    /// Returns `true` if the voxel grid contains voxel colors.
    bool HasColors() const {
        return true;  // By default, the colors are (0, 0, 0)
    }
    /// Returns voxel index given query point.
    Eigen::Vector3i GetVoxel(const Eigen::Vector3d &point) const;

    /// Function that returns the 3d coordinates of the queried voxel center.
    Eigen::Vector3d GetVoxelCenterCoordinate(const Eigen::Vector3i &idx) const {
        auto it = voxels_.find(idx);
        if (it != voxels_.end()) {
            auto voxel = it->second;
            return ((voxel.grid_index_.cast<double>() +
                     Eigen::Vector3d(0.5, 0.5, 0.5)) *
                    voxel_size_) +
                   origin_;
        } else {
            return Eigen::Vector3d::Zero();
        }
    }

    /// Add a voxel with specified grid index and color.
    void AddVoxel(const Voxel &voxel);

    /// Return a vector of 3D coordinates that define the indexed voxel cube.
    std::vector<Eigen::Vector3d> GetVoxelBoundingPoints(
            const Eigen::Vector3i &index) const;

    /// Element-wise check if a query in the list is included in the VoxelGrid
    /// Queries are double precision and are mapped to the closest voxel.
    std::vector<bool> CheckIfIncluded(
            const std::vector<Eigen::Vector3d> &queries);

    /// Remove all voxels from the VoxelGrid where none of the boundary points
    /// of the voxel projects to depth value that is smaller, or equal than the
    /// projected depth of the boundary point. If keep_voxels_outside_image is
    /// true then voxels are only carved if all boundary points project to a
    /// valid image location.
    ///
    /// \param depth_map Depth map (Image) used for VoxelGrid carving.
    /// \param camera_parameter Input Camera Parameters.
    /// \param keep_voxels_outside_image Project all voxels to a valid location.
    VoxelGrid &CarveDepthMap(
            const Image &depth_map,
            const camera::PinholeCameraParameters &camera_parameter,
            bool keep_voxels_outside_image);

    /// Remove all voxels from the VoxelGrid where none of the boundary points
    /// of the voxel projects to a valid mask pixel (pixel value > 0). If
    /// keep_voxels_outside_image is true then voxels are only carved if
    /// all boundary points project to a valid image location.
    ///
    /// \param silhouette_mask Silhouette mask (Image) used for VoxelGrid
    /// carving.
    /// \param camera_parameter Input Camera Parameters.
    /// \param keep_voxels_outside_image Project all voxels to a valid location.
    VoxelGrid &CarveSilhouette(
            const Image &silhouette_mask,
            const camera::PinholeCameraParameters &camera_parameter,
            bool keep_voxels_outside_image);

    /// Create VoxelGrid from Octree
    ///
    /// \param octree The input Octree.
    void CreateFromOctree(const Octree &octree);

    /// Convert to Octree.
    ///
    /// \param max_depth Maximum depth of the octree.
    std::shared_ptr<geometry::Octree> ToOctree(const size_t &max_depth) const;

    /// Creates a voxel grid where every voxel is set (hence dense). This is a
    /// useful starting point for voxel carving.
    ///
    /// \param origin Coordinate center of the VoxelGrid
    /// \param color Voxel color for all voxels of the VoxelGrid.
    /// \param voxel_size Voxel size of of the VoxelGrid construction.
    /// \param width Spatial width extend of the VoxelGrid.
    /// \param height Spatial height extend of the VoxelGrid.
    /// \param depth Spatial depth extend of the VoxelGrid.
    static std::shared_ptr<VoxelGrid> CreateDense(const Eigen::Vector3d &origin,
                                                  const Eigen::Vector3d &color,
                                                  double voxel_size,
                                                  double width,
                                                  double height,
                                                  double depth);

    /// Creates a VoxelGrid from a given PointCloud. The color value of a given
    /// voxel is the average color value of the points that fall into it (if the
    /// PointCloud has colors).
    /// The bounds of the created VoxelGrid are computed from the PointCloud.
    ///
    /// \param input The input PointCloud.
    /// \param voxel_size Voxel size of of the VoxelGrid construction.
    static std::shared_ptr<VoxelGrid> CreateFromPointCloud(
            const PointCloud &input, double voxel_size);

    /// Creates a VoxelGrid from a given PointCloud. The color value of a given
    /// voxel is the average color value of the points that fall into it (if the
    /// PointCloud has colors).
    /// The bounds of the created VoxelGrid are defined by the given parameters.
    ///
    /// \param input The input PointCloud.
    /// \param voxel_size Voxel size of of the VoxelGrid construction.
    /// \param min_bound Minimum boundary point for the VoxelGrid to create.
    /// \param max_bound Maximum boundary point for the VoxelGrid to create.
    static std::shared_ptr<VoxelGrid> CreateFromPointCloudWithinBounds(
            const PointCloud &input,
            double voxel_size,
            const Eigen::Vector3d &min_bound,
            const Eigen::Vector3d &max_bound);

    /// Creates a VoxelGrid from a given TriangleMesh. No color information is
    /// converted. The bounds of the created VoxelGrid are computed from the
    /// TriangleMesh.
    ///
    /// \param input The input TriangleMesh.
    /// \param voxel_size Voxel size of of the VoxelGrid construction.
    static std::shared_ptr<VoxelGrid> CreateFromTriangleMesh(
            const TriangleMesh &input, double voxel_size);

    /// Creates a VoxelGrid from a given TriangleMesh. No color information is
    /// converted. The bounds of the created VoxelGrid are defined by the given
    /// parameters..
    ///
    /// \param input The input TriangleMesh.
    /// \param voxel_size Voxel size of of the VoxelGrid construction.
    /// \param min_bound Minimum boundary point for the VoxelGrid to create.
    /// \param max_bound Maximum boundary point for the VoxelGrid to create.
    static std::shared_ptr<VoxelGrid> CreateFromTriangleMeshWithinBounds(
            const TriangleMesh &input,
            double voxel_size,
            const Eigen::Vector3d &min_bound,
            const Eigen::Vector3d &max_bound);

    /// Returns List of ``Voxel``: Voxels contained in voxel grid.
    /// Changes to the voxels returned from this method are not reflected in
    /// the voxel grid.
    std::vector<Voxel> GetVoxels() const;

public:
    /// Size of the voxel.
    double voxel_size_ = 0.0;
    /// Coorindate of the origin point.
    Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
    /// Voxels contained in voxel grid
    std::unordered_map<Eigen::Vector3i,
                       Voxel,
                       utility::hash_eigen<Eigen::Vector3i>>
            voxels_;
};

/// \class AvgColorVoxel
///
/// \brief Class to aggregate color values from different votes in one voxel
/// Computes the average color value in the voxel.
class AvgColorVoxel {
public:
    AvgColorVoxel() : num_of_points_(0), color_(0.0, 0.0, 0.0) {}

public:
    void Add(const Eigen::Vector3i &voxel_index) {
        if (num_of_points_ > 0 && voxel_index != voxel_index_) {
            utility::LogWarning(
                    "Tried to aggregate ColorVoxel with different "
                    "voxel_index");
        }
        voxel_index_ = voxel_index;
    }

    void Add(const Eigen::Vector3i &voxel_index, const Eigen::Vector3d &color) {
        Add(voxel_index);
        color_ += color;
        num_of_points_++;
    }

    Eigen::Vector3i GetVoxelIndex() const { return voxel_index_; }

    Eigen::Vector3d GetAverageColor() const {
        if (num_of_points_ > 0) {
            return color_ / double(num_of_points_);
        } else {
            return color_;
        }
    }

public:
    int num_of_points_;
    Eigen::Vector3i voxel_index_;
    Eigen::Vector3d color_;
};

}  // namespace geometry
}  // namespace open3d
