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

#include <numeric>
#include <unordered_map>

#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/VoxelGrid.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/Helper.h"

namespace open3d {

namespace {
using namespace geometry;

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

namespace geometry {

std::shared_ptr<VoxelGrid> CreateSurfaceVoxelGridFromPointCloud(
    const PointCloud &input, double voxel_size,
    const Eigen::Vector3d voxel_min_bound,
    const Eigen::Vector3d voxel_max_bound) {
    auto output = std::make_shared<VoxelGrid>();
    if (voxel_size <= 0.0) {
        utility::PrintDebug("[VoxelGridFromPointCloud] voxel_size <= 0.\n");
        return output;
    }
    if (voxel_size * std::numeric_limits<int>::max() <
        (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        utility::PrintDebug("[VoxelGridFromPointCloud] voxel_size is too small.\n");
        return output;
    }
    output->voxel_size_ = voxel_size;
    output->origin_ = voxel_min_bound;
    std::unordered_map<Eigen::Vector3i, PointCloudVoxel,
                       utility::hash_eigen::hash<Eigen::Vector3i>>
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
    utility::PrintDebug("Pointcloud is voxelized from %d points to %d voxels.\n",
               (int)input.points_.size(), (int)output->voxels_.size());
    return output;
}

std::shared_ptr<VoxelGrid> CreateVoxelGrid(double w, double h, double d,
                                           double voxel_size,
                                           const Eigen::Vector3d origin) {
    auto output = std::make_shared<VoxelGrid>();
    int num_w = std::round(w / voxel_size);
    int num_h = std::round(h / voxel_size);
    int num_d = std::round(d / voxel_size);
    if (num_w * num_h * num_d > 512 * 512 * 512) {
        utility::PrintWarning("Too many voxels were requested");
        return output;
    }
    output->origin_ = origin;
    output->voxel_size_ = voxel_size;
    output->voxels_.resize(num_w * num_h * num_d);
    int cnt = 0;
    for (int i = 0; i < num_w; i++) {
        for (int j = 0; j < num_h; j++) {
            for (int k = 0; k < num_d; k++) {
                output->voxels_[cnt](0) = i;
                output->voxels_[cnt](1) = j;
                output->voxels_[cnt](2) = k;
                cnt++;
            }
        }
    }
    return output;
}

}  // namespace geometry
}  // namespace open3d
