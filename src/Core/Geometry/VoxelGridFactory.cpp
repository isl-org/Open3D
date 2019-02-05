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

#include <numeric>
#include <unordered_map>

#include <Core/Geometry/PointCloud.h>
#include <Core/Utility/Console.h>
#include <Core/Utility/Helper.h>

namespace open3d {

namespace {

class PointCloudVoxel {
public:
    PointCloudVoxel() : num_of_points_(0), color_(0.0, 0.0, 0.0) {}

public:
    void AddPoint(const Eigen::Vector3i &voxel_index,
                  const PointCloud &cloud,
                  int index) {
        coordinate_ = voxel_index;
        if (cloud.HasColors()) {
            color_ += cloud.colors_[index];
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

}  // namespace

std::shared_ptr<VoxelGrid> CreateSurfaceVoxelGridFromPointCloud(
        const PointCloud &input, double voxel_size) {
    auto output = std::make_shared<VoxelGrid>();
    if (voxel_size <= 0.0) {
        PrintDebug("[VoxelGridFromPointCloud] voxel_size <= 0.\n");
        return output;
    }
    Eigen::Vector3d voxel_size3 =
            Eigen::Vector3d(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d voxel_min_bound = input.GetMinBound() - voxel_size3 * 0.5;
    Eigen::Vector3d voxel_max_bound = input.GetMaxBound() + voxel_size3 * 0.5;
    if (voxel_size * std::numeric_limits<int>::max() <
        (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        PrintDebug("[VoxelGridFromPointCloud] voxel_size is too small.\n");
        return output;
    }
    output->voxel_size_ = voxel_size;
    output->origin_ = voxel_min_bound;
    std::unordered_map<Eigen::Vector3i, PointCloudVoxel,
                       hash_eigen::hash<Eigen::Vector3i>>
            voxelindex_to_accpoint;
    Eigen::Vector3d ref_coord;
    Eigen::Vector3i voxel_index;
    for (int i = 0; i < (int)input.points_.size(); i++) {
        ref_coord = (input.points_[i] - voxel_min_bound) / voxel_size;
        voxel_index << int(floor(ref_coord(0))), int(floor(ref_coord(1))),
                int(floor(ref_coord(2)));
        voxelindex_to_accpoint[voxel_index].AddPoint(voxel_index, input, i);
    }
    bool has_colors = input.HasColors();
    for (auto accpoint : voxelindex_to_accpoint) {
        output->voxels_.push_back(accpoint.second.GetVoxelCoordinate());
        if (has_colors) {
            output->colors_.push_back(accpoint.second.GetAverageColor());
        }
    }
    PrintDebug("Pointcloud is voxelized from %d points to %d voxels.\n",
               (int)input.points_.size(), (int)output->voxels_.size());
    return output;
}

}  // namespace open3d
