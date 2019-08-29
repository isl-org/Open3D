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

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "Open3D/Geometry/Geometry3D.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

namespace camera {
class PinholeCameraParameters;
}

namespace geometry {

class PointCloud;
class TriangleMesh;
class Octree;
class Image;

class Voxel {
public:
    Voxel() {}
    Voxel(const Eigen::Vector3i &grid_index) : grid_index_(grid_index) {}
    Voxel(const Eigen::Vector3i &grid_index, const Eigen::Vector3d &color)
        : grid_index_(grid_index), color_(color) {}
    ~Voxel() {}

public:
    Eigen::Vector3i grid_index_ = Eigen::Vector3i(0, 0, 0);
    Eigen::Vector3d color_ = Eigen::Vector3d(0, 0, 0);
};

class VoxelGrid : public Geometry3D {
public:
    VoxelGrid() : Geometry3D(Geometry::GeometryType::VoxelGrid) {}
    VoxelGrid(const VoxelGrid &src_voxel_grid);
    ~VoxelGrid() override {}

    VoxelGrid &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetMaxBound() const override;
    Eigen::Vector3d GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    OrientedBoundingBox GetOrientedBoundingBox() const override;
    VoxelGrid &Transform(const Eigen::Matrix4d &transformation) override;
    VoxelGrid &Translate(const Eigen::Vector3d &translation,
                         bool relative = true) override;
    VoxelGrid &Scale(const double scale, bool center = true) override;
    VoxelGrid &Rotate(const Eigen::Vector3d &rotation,
                      bool center = true,
                      RotationType type = RotationType::XYZ) override;

    VoxelGrid &operator+=(const VoxelGrid &voxelgrid);
    VoxelGrid operator+(const VoxelGrid &voxelgrid) const;

    bool HasVoxels() const { return voxels_.size() > 0; }
    bool HasColors() const {
        return true;  // By default, the colors are (0, 0, 0)
    }
    Eigen::Vector3i GetVoxel(const Eigen::Vector3d &point) const;

    // Function that returns the 3d coordinates of the queried voxel center
    Eigen::Vector3d GetVoxelCenterCoordinate(int idx) const {
        const Eigen::Vector3i &grid_index = voxels_[idx].grid_index_;
        return ((grid_index.cast<double>() + Eigen::Vector3d(0.5, 0.5, 0.5)) *
                voxel_size_) +
               origin_;
    }

    /// Return a vector of 3D coordinates that define the indexed voxel cube.
    std::vector<Eigen::Vector3d> GetVoxelBoundingPoints(int index) const;

    /// Remove all voxels from the VoxelGrid where none of the boundary points
    /// of the voxel projects to depth value that is smaller, or equal than the
    /// projected depth of the boundary point. The point is not carved if none
    /// of the boundary points of the voxel projects to a valid image location.
    VoxelGrid &CarveDepthMap(
            const Image &depth_map,
            const camera::PinholeCameraParameters &camera_parameter);

    /// Remove all voxels from the VoxelGrid where none of the boundary points
    /// of the voxel projects to a valid mask pixel (pixel value > 0). The point
    /// is not carved if none of the boundary points of the voxel projects to a
    /// valid image location.
    VoxelGrid &CarveSilhouette(
            const Image &silhouette_mask,
            const camera::PinholeCameraParameters &camera_parameter);

    void CreateFromOctree(const Octree &octree);

    std::shared_ptr<geometry::Octree> ToOctree(const size_t &max_depth) const;

    // Creates a voxel grid where every voxel is set (hence dense). This is a
    // useful starting point for voxel carving.
    static std::shared_ptr<VoxelGrid> CreateDense(const Eigen::Vector3d &origin,
                                                  double voxel_size,
                                                  double width,
                                                  double height,
                                                  double depth);

    // Creates a VoxelGrid from a given PointCloud. The color value of a given
    // voxel is the average color value of the points that fall into it (if the
    // PointCloud has colors).
    // The bounds of the created VoxelGrid are computed from the PointCloud.
    static std::shared_ptr<VoxelGrid> CreateFromPointCloud(
            const PointCloud &input, double voxel_size);

    // Creates a VoxelGrid from a given PointCloud. The color value of a given
    // voxel is the average color value of the points that fall into it (if the
    // PointCloud has colors).
    // The bounds of the created VoxelGrid are defined by the given parameters.
    static std::shared_ptr<VoxelGrid> CreateFromPointCloudWithinBounds(
            const PointCloud &input,
            double voxel_size,
            const Eigen::Vector3d &min_bound,
            const Eigen::Vector3d &max_bound);

    // Creates a VoxelGrid from a given TriangleMesh. No color information is
    // converted. The bounds of the created VoxelGrid are computed from the
    // TriangleMesh..
    static std::shared_ptr<VoxelGrid> CreateFromTriangleMesh(
            const TriangleMesh &input, double voxel_size);

    // Creates a VoxelGrid from a given TriangleMesh. No color information is
    // converted. The bounds of the created VoxelGrid are defined by the given
    // parameters..
    static std::shared_ptr<VoxelGrid> CreateFromTriangleMeshWithinBounds(
            const TriangleMesh &input,
            double voxel_size,
            const Eigen::Vector3d &min_bound,
            const Eigen::Vector3d &max_bound);

public:
    double voxel_size_;
    Eigen::Vector3d origin_;
    std::vector<Voxel> voxels_;
};

/// Class to aggregate color values from different votes in one voxel
/// Computes the average color value in the voxel.
class AvgColorVoxel {
public:
    AvgColorVoxel() : num_of_points_(0), color_(0.0, 0.0, 0.0) {}

public:
    void Add(const Eigen::Vector3i &voxel_index) {
        if (num_of_points_ > 0 && voxel_index != voxel_index_) {
            utility::LogWarning(
                    "Tried to aggregate ColorVoxel with different "
                    "voxel_index\n");
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
