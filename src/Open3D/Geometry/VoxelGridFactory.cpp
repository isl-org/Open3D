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
namespace geometry {

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromPointCloud(
        const PointCloud &input, double voxel_size) {
    auto output = std::make_shared<VoxelGrid>();
    if (voxel_size <= 0.0) {
        utility::PrintError("[VoxelGridFromPointCloud] voxel_size <= 0.\n");
        return output;
    }
    Eigen::Vector3d voxel_size3 =
            Eigen::Vector3d(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d voxel_min_bound = input.GetMinBound() - voxel_size3 * 0.5;
    Eigen::Vector3d voxel_max_bound = input.GetMaxBound() + voxel_size3 * 0.5;
    if (voxel_size * std::numeric_limits<int>::max() <
        (voxel_max_bound - voxel_min_bound).maxCoeff()) {
        utility::PrintError(
                "[VoxelGridFromPointCloud] voxel_size is too small.\n");
        return output;
    }
    output->voxel_size_ = voxel_size;
    output->origin_ = voxel_min_bound;
    std::unordered_map<Eigen::Vector3i, AvgColorVoxel,
                       utility::hash_eigen::hash<Eigen::Vector3i>>
            voxelindex_to_accpoint;
    Eigen::Vector3d ref_coord;
    Eigen::Vector3i voxel_index;
    bool has_colors = input.HasColors();
    for (int i = 0; i < (int)input.points_.size(); i++) {
        ref_coord = (input.points_[i] - voxel_min_bound) / voxel_size;
        voxel_index << int(floor(ref_coord(0))), int(floor(ref_coord(1))),
                int(floor(ref_coord(2)));
        if (has_colors) {
            voxelindex_to_accpoint[voxel_index].Add(voxel_index,
                                                    input.colors_[i]);
        } else {
            voxelindex_to_accpoint[voxel_index].Add(voxel_index);
        }
    }
    for (auto accpoint : voxelindex_to_accpoint) {
        const Eigen::Vector3i &grid_index =
                accpoint.second.GetVoxelCoordinate();
        const Eigen::Vector3d &color =
                has_colors ? accpoint.second.GetAverageColor()
                           : Eigen::Vector3d(0, 0, 0);
        output->voxels_.emplace_back(grid_index, color);
    }
    utility::PrintDebug(
            "Pointcloud is voxelized from %d points to %d voxels.\n",
            (int)input.points_.size(), (int)output->voxels_.size());
    return output;
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateDense(const Eigen::Vector3d &origin,
                                                  double voxel_size,
                                                  double width,
                                                  double height,
                                                  double depth) {
    auto output = std::make_shared<VoxelGrid>();
    int num_w = std::round(width / voxel_size);
    int num_h = std::round(height / voxel_size);
    int num_d = std::round(depth / voxel_size);
    output->origin_ = origin;
    output->voxel_size_ = voxel_size;
    output->voxels_.resize(num_w * num_h * num_d);
    int cnt = 0;
    for (int widx = 0; widx < num_w; widx++) {
        for (int hidx = 0; hidx < num_h; hidx++) {
            for (int didx = 0; didx < num_d; didx++) {
                output->voxels_[cnt].grid_index_ << widx, hidx, didx;
                cnt++;
            }
        }
    }
    return output;
}

}  // namespace geometry
}  // namespace open3d
