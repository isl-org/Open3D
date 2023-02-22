// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <numeric>
#include <unordered_map>

#include "open3d/geometry/IntersectionTest.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/geometry/VoxelGrid.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace geometry {

std::shared_ptr<VoxelGrid> VoxelGrid::CreateDense(const Eigen::Vector3d &origin,
                                                  const Eigen::Vector3d &color,
                                                  double voxel_size,
                                                  double width,
                                                  double height,
                                                  double depth) {
    auto output = std::make_shared<VoxelGrid>();
    int num_w = int(std::round(width / voxel_size));
    int num_h = int(std::round(height / voxel_size));
    int num_d = int(std::round(depth / voxel_size));
    output->origin_ = origin;
    output->voxel_size_ = voxel_size;
    for (int widx = 0; widx < num_w; widx++) {
        for (int hidx = 0; hidx < num_h; hidx++) {
            for (int didx = 0; didx < num_d; didx++) {
                Eigen::Vector3i grid_index(widx, hidx, didx);
                output->AddVoxel(geometry::Voxel(grid_index, color));
            }
        }
    }
    return output;
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromPointCloudWithinBounds(
        const PointCloud &input,
        double voxel_size,
        const Eigen::Vector3d &min_bound,
        const Eigen::Vector3d &max_bound) {
    auto output = std::make_shared<VoxelGrid>();
    if (voxel_size <= 0.0) {
        utility::LogError("voxel_size <= 0.");
    }

    if (voxel_size * std::numeric_limits<int>::max() <
        (max_bound - min_bound).maxCoeff()) {
        utility::LogError("voxel_size is too small.");
    }
    output->voxel_size_ = voxel_size;
    output->origin_ = min_bound;
    std::unordered_map<Eigen::Vector3i, AvgColorVoxel,
                       utility::hash_eigen<Eigen::Vector3i>>
            voxelindex_to_accpoint;
    Eigen::Vector3d ref_coord;
    Eigen::Vector3i voxel_index;
    bool has_colors = input.HasColors();
    for (int i = 0; i < (int)input.points_.size(); i++) {
        ref_coord = (input.points_[i] - min_bound) / voxel_size;
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
        const Eigen::Vector3i &grid_index = accpoint.second.GetVoxelIndex();
        const Eigen::Vector3d &color =
                has_colors ? accpoint.second.GetAverageColor()
                           : Eigen::Vector3d(0, 0, 0);
        output->AddVoxel(geometry::Voxel(grid_index, color));
    }
    utility::LogDebug(
            "Pointcloud is voxelized from {:d} points to {:d} voxels.",
            (int)input.points_.size(), (int)output->voxels_.size());
    return output;
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromPointCloud(
        const PointCloud &input, double voxel_size) {
    Eigen::Vector3d voxel_size3(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d min_bound = input.GetMinBound() - voxel_size3 * 0.5;
    Eigen::Vector3d max_bound = input.GetMaxBound() + voxel_size3 * 0.5;
    return CreateFromPointCloudWithinBounds(input, voxel_size, min_bound,
                                            max_bound);
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromTriangleMeshWithinBounds(
        const TriangleMesh &input,
        double voxel_size,
        const Eigen::Vector3d &min_bound,
        const Eigen::Vector3d &max_bound) {
    auto output = std::make_shared<VoxelGrid>();
    if (voxel_size <= 0.0) {
        utility::LogError("voxel_size <= 0.");
    }

    if (voxel_size * std::numeric_limits<int>::max() <
        (max_bound - min_bound).maxCoeff()) {
        utility::LogError("voxel_size is too small.");
    }
    output->voxel_size_ = voxel_size;
    output->origin_ = min_bound;

    const Eigen::Vector3d box_half_size(voxel_size / 2, voxel_size / 2,
                                        voxel_size / 2);
    for (const Eigen::Vector3i &tria : input.triangles_) {
        const Eigen::Vector3d &v0 = input.vertices_[tria(0)];
        const Eigen::Vector3d &v1 = input.vertices_[tria(1)];
        const Eigen::Vector3d &v2 = input.vertices_[tria(2)];
        double minx, miny, minz, maxx, maxy, maxz;
        int num_w, num_h, num_d, inix, iniy, iniz;
        minx = std::min(v0[0], std::min(v1[0], v2[0]));
        miny = std::min(v0[1], std::min(v1[1], v2[1]));
        minz = std::min(v0[2], std::min(v1[2], v2[2]));
        maxx = std::max(v0[0], std::max(v1[0], v2[0]));
        maxy = std::max(v0[1], std::max(v1[1], v2[1]));
        maxz = std::max(v0[2], std::max(v1[2], v2[2]));
        inix = static_cast<int>(std::floor((minx - min_bound[0]) / voxel_size));
        iniy = static_cast<int>(std::floor((miny - min_bound[1]) / voxel_size));
        iniz = static_cast<int>(std::floor((minz - min_bound[2]) / voxel_size));
        num_w = static_cast<int>((std::round((maxx - minx) / voxel_size))) + 2;
        num_h = static_cast<int>((std::round((maxy - miny) / voxel_size))) + 2;
        num_d = static_cast<int>((std::round((maxz - minz) / voxel_size))) + 2;
        for (int widx = inix; widx < inix + num_w; widx++) {
            for (int hidx = iniy; hidx < iniy + num_h; hidx++) {
                for (int didx = iniz; didx < iniz + num_d; didx++) {
                    const Eigen::Vector3d box_center =
                            min_bound + box_half_size +
                            Eigen::Vector3d(widx, hidx, didx) * voxel_size;
                    if (IntersectionTest::TriangleAABB(
                                box_center, box_half_size, v0, v1, v2)) {
                        Eigen::Vector3i grid_index(widx, hidx, didx);
                        output->AddVoxel(geometry::Voxel(grid_index));
                        break;
                    }
                }
            }
        }
    }

    return output;
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromTriangleMesh(
        const TriangleMesh &input, double voxel_size) {
    Eigen::Vector3d voxel_size3(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3d min_bound = input.GetMinBound() - voxel_size3 * 0.5;
    Eigen::Vector3d max_bound = input.GetMaxBound() + voxel_size3 * 0.5;
    return CreateFromTriangleMeshWithinBounds(input, voxel_size, min_bound,
                                              max_bound);
}

}  // namespace geometry
}  // namespace open3d
