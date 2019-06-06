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

namespace open3d {
namespace geometry {

class PointCloud;
class TriangleMesh;
class Octree;

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
    VoxelGrid &Transform(const Eigen::Matrix4d &transformation) override;
    VoxelGrid &Translate(const Eigen::Vector3d &translation) override;
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

    void FromOctree(const Octree &octree);

    std::shared_ptr<geometry::Octree> ToOctree(const size_t &max_depth) const;

    static std::shared_ptr<VoxelGrid> CreateFromPointCloud(
            const PointCloud &input, double voxel_size);

public:
    double voxel_size_;
    Eigen::Vector3d origin_;
    std::vector<Voxel> voxels_;
};

}  // namespace geometry
}  // namespace open3d
