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

#include "Open3D/Camera/PinholeCameraParameters.h"
#include "Open3D/Geometry/Image.h"
#include "Open3D/Geometry/Octree.h"
#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Utility/Helper.h"

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

VoxelGrid &VoxelGrid::Transform(const Eigen::Matrix4d &transformation) {
    throw std::runtime_error("VoxelGrid::Transform is not supported");
    return *this;
}

VoxelGrid &VoxelGrid::Translate(const Eigen::Vector3d &translation) {
    throw std::runtime_error("Not implemented");
    return *this;
}

VoxelGrid &VoxelGrid::Scale(const double scale) {
    throw std::runtime_error("Not implemented");
    return *this;
}

VoxelGrid &VoxelGrid::Rotate(const Eigen::Vector3d &rotation,
                             RotationType type) {
    throw std::runtime_error("Not implemented");
    return *this;
}

Eigen::Vector3i VoxelGrid::GetVoxel(const Eigen::Vector3d &point) const {
    Eigen::Vector3d voxel_f = (point - origin_) / voxel_size_;
    return (Eigen::floor(voxel_f.array())).cast<int>();
}

void VoxelGrid::FromOctree(const Octree &octree) {
    // TODO: currently only handles color leaf nodes
    // Get leaf nodes and their node_info
    std::unordered_map<std::shared_ptr<OctreeColorLeafNode>,
                       std::shared_ptr<OctreeNodeInfo>>
            map_node_to_node_info;
    auto f_collect_nodes =
            [&map_node_to_node_info](
                    const std::shared_ptr<OctreeNode> &node,
                    const std::shared_ptr<OctreeNodeInfo> &node_info) -> void {
        if (auto color_leaf_node =
                    std::dynamic_pointer_cast<OctreeColorLeafNode>(node)) {
            map_node_to_node_info[color_leaf_node] = node_info;
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
        const std::shared_ptr<OctreeColorLeafNode> &node = it.first;
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

class PointCloudVoxel {
public:
    PointCloudVoxel() : num_of_points_(0), color_(0.0, 0.0, 0.0) {}

public:
    void AddVoxel(const Eigen::Vector3i &voxel_index,
                  const VoxelGrid &voxelgrid,
                  int index) {
        coordinate_ = voxel_index;
        if (voxelgrid.HasColors()) {
            color_ += voxelgrid.colors_[index];
        }
        num_of_points_++;
    }

    Eigen::Vector3i GetVoxelCoordinate() const { return coordinate_; }

    Eigen::Vector3d GetAverageColor() const {
        return color_ / double(num_of_points_);
    }

public:
    int num_of_points_;
    Eigen::Vector3i coordinate_;
    Eigen::Vector3d color_;
};

VoxelGrid &VoxelGrid::operator+=(const VoxelGrid &voxelgrid) {
    assert(voxel_size_ == voxelgrid.voxel_size_);
    assert(origin_ == voxelgrid.origin_);
    assert(*this.HasColors() == voxelgrid.HasColors());
    std::unordered_map<Eigen::Vector3i, PointCloudVoxel,
                       utility::hash_eigen::hash<Eigen::Vector3i>>
            voxelindex_to_accpoint;
    Eigen::Vector3d ref_coord;
    Eigen::Vector3i voxel_index;
    int i = 0;
    for (auto &voxel : voxelgrid.voxels_) {
        voxel_index << voxel(0), voxel(1), voxel(2);
        voxelindex_to_accpoint[voxel_index].AddVoxel(voxel_index, voxelgrid,
                                                     i++);
    }
    i = 0;
    for (auto &voxel : voxels_) {
        voxel_index << voxel(0), voxel(1), voxel(2);
        voxelindex_to_accpoint[voxel_index].AddVoxel(voxel_index, *this, i++);
    }
    this->voxels_.clear();
    this->colors_.clear();
    bool has_colors = voxelgrid.HasColors();
    for (auto accpoint : voxelindex_to_accpoint) {
        this->voxels_.push_back(accpoint.second.GetVoxelCoordinate());
        if (has_colors) {
            this->colors_.push_back(accpoint.second.GetAverageColor());
        }
    }
    return *this;
}

VoxelGrid VoxelGrid::operator+(const VoxelGrid &voxelgrid) const {
    return (VoxelGrid(*this) += voxelgrid);
}

std::vector<Eigen::Vector3d> VoxelGrid::GetBoundingPointsOfVoxel(int id) {
    double r = voxel_size_ / 2.0;
    auto x = GetOriginalCoordinate(id);
    std::vector<Eigen::Vector3d> points;
    points.push_back(x + Eigen::Vector3d(-r, -r, -r));
    points.push_back(x + Eigen::Vector3d(-r, -r, r));
    points.push_back(x + Eigen::Vector3d(r, -r, -r));
    points.push_back(x + Eigen::Vector3d(r, -r, r));
    points.push_back(x + Eigen::Vector3d(-r, r, -r));
    points.push_back(x + Eigen::Vector3d(-r, r, r));
    points.push_back(x + Eigen::Vector3d(r, r, -r));
    points.push_back(x + Eigen::Vector3d(r, r, r));
    return std::move(points);
}

std::shared_ptr<VoxelGrid> CarveVoxelGridUsingDepthMap(
        VoxelGrid &voxel_grid,
        const Image &depth_map,
        const camera::PinholeCameraParameters &camera_parameter) {
    auto rot = camera_parameter.extrinsic_.block<3, 3>(0, 0);
    auto trans = camera_parameter.extrinsic_.block<3, 1>(0, 3);
    auto intrinsic = camera_parameter.intrinsic_.intrinsic_matrix_;

    if (voxel_grid.HasColors())
        assert(voxel_grid.voxels_.size() == voxel_grid.colors_.size());
    assert(depth_map.height_ == camera_parameter.intrinsic_.height_);
    assert(depth_map.width_ == camera_parameter.intrinsic_.width_);
    size_t n_voxels = voxel_grid.voxels_.size();
    std::vector<bool> valid(n_voxels, true);
    for (size_t i = 0; i < n_voxels; i++) {
        auto pts = voxel_grid.GetBoundingPointsOfVoxel(i);
        std::vector<bool> valid_i(8, true);
        int cnt = 0;
        for (auto &x : pts) {
            auto x_trans = rot * x + trans;
            auto uvz = intrinsic * x_trans;
            double z = uvz(2);
            double u = uvz(0) / z;
            double v = uvz(1) / z;
            double d;
            bool boundary;
            std::tie(boundary, d) = depth_map.FloatValueAt(u, v);
            if (boundary) {
                if ((d == 0.0) || (z > 0.0 && z < d)) {
                    valid_i[cnt] = false;
                }
            }
            cnt++;
        }
        if (std::all_of(valid_i.begin(), valid_i.end(),
                        [](bool v) { return !v; })) {
            valid[i] = false;
        }
    }
    auto voxel_grid_output = std::make_shared<VoxelGrid>();
    voxel_grid_output->voxel_size_ = voxel_grid.voxel_size_;
    voxel_grid_output->origin_ = voxel_grid.origin_;
    for (size_t i = 0; i < n_voxels; i++) {
        if (valid[i]) {
            voxel_grid_output->voxels_.push_back(voxel_grid.voxels_[i]);
            if (voxel_grid.HasColors())
                voxel_grid_output->colors_.push_back(voxel_grid.colors_[i]);
        }
    }
    return voxel_grid_output;
}

void CarveVoxelGridUsingSilhouette(
        VoxelGrid &voxel_grid,
        const Image &silhouette_mask,
        const camera::PinholeCameraParameters &camera_parameter) {
    auto voxel_pcd = CreatePointCloudFromVoxelGrid(voxel_grid);
    voxel_pcd->Transform(camera_parameter.extrinsic_);
    if (voxel_grid.HasColors())
        assert(voxel_grid.voxels_.size() == voxel_grid.colors_.size());
    assert(silhouette_mask.height_ == camera_parameter.intrinsic_.height_);
    assert(silhouette_mask.width_ == camera_parameter.intrinsic_.width_);
    for (size_t j = 0; j < voxel_pcd->points_.size(); j++) {
        Eigen::Vector3d &x = voxel_pcd->points_[j];
        auto uvz = camera_parameter.intrinsic_.intrinsic_matrix_ * x;
        if (silhouette_mask.TestImageBoundary(uvz(0), uvz(1), 0)) {
            int u = std::round(uvz(0));
            int v = std::round(uvz(1));
            double z = uvz(2);
            unsigned char mask =
                    *PointerAt<unsigned char>(silhouette_mask, u, v);
            if (mask == 0) {
                voxel_grid.voxels_.erase(voxel_grid.voxels_.begin() + j);
                if (voxel_grid.HasColors()) {
                    voxel_grid.colors_.erase(voxel_grid.colors_.begin() + j);
                }
            }
        }
    }
}

}  // namespace geometry
}  // namespace open3d
