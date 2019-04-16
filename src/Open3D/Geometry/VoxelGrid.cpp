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

#include "Open3D/Geometry/VoxelGrid.h"

#include <unordered_map>

#include "Open3D/Geometry/Octree.h"

namespace open3d {
namespace geometry {

VoxelGrid::VoxelGrid(const VoxelGrid &src_voxel_grid)
    : Geometry3D(Geometry::GeometryType::VoxelGrid),
      voxel_size_(src_voxel_grid.voxel_size_),
      origin_(src_voxel_grid.origin_),
      voxels_(src_voxel_grid.voxels_),
      colors_(src_voxel_grid.colors_) {}

void VoxelGrid::Clear() {
    voxel_size_ = 0.0;
    origin_ = Eigen::Vector3d::Zero();
    voxels_.clear();
    colors_.clear();
}

bool VoxelGrid::IsEmpty() const { return !HasVoxels(); }

Eigen::Vector3d VoxelGrid::GetMinBound() const {
    if (!HasVoxels()) {
        return origin_;
    } else {
        Eigen::Array3i min_voxel = voxels_[0];
        for (const Eigen::Vector3i &voxel : voxels_) {
            min_voxel = min_voxel.min(voxel.array());
        }
        return min_voxel.cast<double>() * voxel_size_ + origin_.array();
    }
}

Eigen::Vector3d VoxelGrid::GetMaxBound() const {
    if (!HasVoxels()) {
        return origin_;
    } else {
        Eigen::Array3i max_voxel = voxels_[0];
        for (const Eigen::Vector3i &voxel : voxels_) {
            max_voxel = max_voxel.max(voxel.array());
        }
        return (max_voxel.cast<double>() + 1) * voxel_size_ + origin_.array();
    }
}

void VoxelGrid::Transform(const Eigen::Matrix4d &transformation) {
    throw std::runtime_error("VoxelGrid::Transform is not supported");
}

VoxelGrid &VoxelGrid::operator+=(const VoxelGrid &voxelgrid) {
    throw std::runtime_error("VoxelGrid::operator+= is not supported");
}

VoxelGrid VoxelGrid::operator+(const VoxelGrid &voxelgrid) const {
    throw std::runtime_error("VoxelGrid::operator+ is not supported");
}

Eigen::Vector3i VoxelGrid::GetVoxel(const Eigen::Vector3d &point) const {
    Eigen::Vector3d voxel_f = (point - origin_) / voxel_size_;
    return Eigen::floor(voxel_f.array()).cast<int>();
}

void VoxelGrid::FromOctree(const Octree &octree) {
    // Get leaf nodes and their node_info
    std::unordered_map<std::shared_ptr<OctreeLeafNode>,
                       std::shared_ptr<OctreeNodeInfo>>
            map_node_to_node_info;
    auto f_collect_nodes =
            [&map_node_to_node_info](
                    const std::shared_ptr<OctreeNode> &node,
                    const std::shared_ptr<OctreeNodeInfo> &node_info) -> void {
        if (auto leaf_node = std::dynamic_pointer_cast<OctreeLeafNode>(node)) {
            map_node_to_node_info[leaf_node] = node_info;
        }
    };
    octree.Traverse(f_collect_nodes);

    // Prepare dimensions for voxel
    origin_ = octree.origin_;
    voxel_size_ = octree.size_;  // Maximum possible voxel size
    voxels_.clear();
    colors_.clear();
    for (const auto &it : map_node_to_node_info) {
        voxel_size_ = std::min(voxel_size_, it.second->size_);
    }

    // Convert nodes to voxels
    for (const auto &it : map_node_to_node_info) {
        const std::shared_ptr<OctreeLeafNode> &node = it.first;
        const std::shared_ptr<OctreeNodeInfo> &node_info = it.second;
        Eigen::Array3d node_center =
                Eigen::Array3d(node_info->origin_) + node_info->size_ / 2.0;
        Eigen::Vector3i voxel =
                Eigen::floor((node_center - Eigen::Array3d(origin_)) /
                             voxel_size_)
                        .cast<int>();
        voxels_.push_back(voxel);
        colors_.push_back(node->color_);
    }
}

}  // namespace geometry
}  // namespace open3d
