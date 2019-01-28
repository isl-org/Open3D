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

#include "VoxelGrid.h"
#include <Eigen/Dense>
#include <Core/Geometry/Image.h>
#include <Core/Geometry/PointCloud.h>
#include <Core/Camera/PinholeCameraParameters.h>

namespace open3d {

void VoxelGrid::Clear()
{
    voxel_size_ = 0.0;
    origin_ = Eigen::Vector3d::Zero();
    voxels_.clear();
    colors_.clear();
}

bool VoxelGrid::IsEmpty() const
{
    return !HasVoxels();
}

Eigen::Vector3d VoxelGrid::GetMinBound() const
{
    if (!HasVoxels()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    auto itr_x = std::min_element(voxels_.begin(), voxels_.end(),
        [](const Eigen::Vector3i &a, const Eigen::Vector3i &b) { return a(0) < b(0); });
    auto itr_y = std::min_element(voxels_.begin(), voxels_.end(),
        [](const Eigen::Vector3i &a, const Eigen::Vector3i &b) { return a(1) < b(1); });
    auto itr_z = std::min_element(voxels_.begin(), voxels_.end(),
        [](const Eigen::Vector3i &a, const Eigen::Vector3i &b) { return a(2) < b(2); });
    return Eigen::Vector3d((*itr_x)(0) * voxel_size_ + origin_(0),
                           (*itr_y)(1) * voxel_size_ + origin_(1),
                           (*itr_z)(2) * voxel_size_ + origin_(2));
}

Eigen::Vector3d VoxelGrid::GetMaxBound() const
{
    if (!HasVoxels()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    auto itr_x = std::max_element(voxels_.begin(), voxels_.end(),
        [](const Eigen::Vector3i &a, const Eigen::Vector3i &b) { return a(0) < b(0); });
    auto itr_y = std::max_element(voxels_.begin(), voxels_.end(),
        [](const Eigen::Vector3i &a, const Eigen::Vector3i &b) { return a(1) < b(1); });
    auto itr_z = std::max_element(voxels_.begin(), voxels_.end(),
        [](const Eigen::Vector3i &a, const Eigen::Vector3i &b) { return a(2) < b(2); });
    return Eigen::Vector3d((*itr_x)(0) * voxel_size_ + origin_(0),
                           (*itr_y)(1) * voxel_size_ + origin_(1),
                           (*itr_z)(2) * voxel_size_ + origin_(2));
}

void VoxelGrid::Transform(const Eigen::Matrix4d &transformation)
{
    // not implemented.
}

VoxelGrid &VoxelGrid::operator+=(const VoxelGrid &voxelgrid)
{
    // not implemented.
    return (*this);
}

VoxelGrid VoxelGrid::operator+(const VoxelGrid &voxelgrid) const
{
    return (VoxelGrid(*this) += voxelgrid);
}

std::vector<Eigen::Vector3d> VoxelGrid::GetBoundingPointsOfVoxel(int id)
{
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
    VoxelGrid& voxel_grid,
    const Image& depth_map,
    const PinholeCameraParameters& camera_parameter)
{
    auto rot = camera_parameter.extrinsic_.block<3, 3>(0, 0);
    auto trans = camera_parameter.extrinsic_.block<3, 1>(0, 3);
    auto intrinsic = camera_parameter.intrinsic_.intrinsic_matrix_;
    
    if (voxel_grid.HasColors())
        assert(voxel_grid.voxels_.size() == voxel_grid.colors_.size());
    assert(depth_map.height_ == 
            camera_parameter.intrinsic_.height_);
    assert(depth_map.width_ ==
            camera_parameter.intrinsic_.width_);
    size_t n_voxels = voxel_grid.voxels_.size();
    std::vector<bool> valid(n_voxels, true);
    for (size_t i = 0; i < n_voxels; i++) {
        auto pts = voxel_grid.GetBoundingPointsOfVoxel(i);
        std::vector<bool> valid_i(8, true);
        int cnt = 0;
        for (auto& x : pts) {
            auto x_trans = rot * x + trans;
            auto uvz = intrinsic * x_trans;
            double z = uvz(2);
            int u = std::round(uvz(0) / z);
            int v = std::round(uvz(1) / z);
            if (depth_map.TestImageBoundary(u, v, 0.0)) {
                double d = *PointerAt<float>(depth_map, u, v);
                if ((d == 0.0f) || (z > 0.0 && z < d)) { 
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
    for (size_t i=0; i<n_voxels; i++) {
        if (valid[i]) {
            voxel_grid_output->voxels_.push_back(
                    voxel_grid.voxels_[i]);
            if (voxel_grid.HasColors())
                voxel_grid_output->colors_.push_back(
                        voxel_grid.colors_[i]);
        }
    }
    return voxel_grid_output;
}

void CarveVoxelGridUsingSilhouette(
    VoxelGrid& voxel_grid,
    const Image& silhouette_mask,
    const PinholeCameraParameters& camera_parameter)
{
    auto voxel_pcd = CreatePointCloudFromVoxelGrid(voxel_grid);
    voxel_pcd->Transform(camera_parameter.extrinsic_);
    if (voxel_grid.HasColors())
        assert(voxel_grid.voxels_.size() == voxel_grid.colors_.size());
    assert(silhouette_mask.height_ == 
            camera_parameter.intrinsic_.height_);
    assert(silhouette_mask.width_ ==
            camera_parameter.intrinsic_.width_);
    for (size_t j=0; j<voxel_pcd->points_.size(); j++) {
        Eigen::Vector3d &x = voxel_pcd->points_[j];
        auto uvz = camera_parameter.
                intrinsic_.intrinsic_matrix_ * x;
        if (silhouette_mask.TestImageBoundary(uvz(0), uvz(1), 0)) {
            int u = std::round(uvz(0));
            int v = std::round(uvz(1));
            double z = uvz(2);
            unsigned char mask = 
                    *PointerAt<unsigned char>(silhouette_mask, u, v);
            if (mask == 0) {
                voxel_grid.voxels_.erase(
                    voxel_grid.voxels_.begin() + j);
                if (voxel_grid.HasColors()) {
                    voxel_grid.colors_.erase(
                            voxel_grid.colors_.begin() + j);
                }
            }
        }
    }
}

}   // namespace open3d