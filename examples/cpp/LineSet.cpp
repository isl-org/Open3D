// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Eigen/Dense>
#include <cstdio>
#include <vector>

#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > LineSet [filename]");
    utility::LogInfo("    The program will :");
    utility::LogInfo("    1. load the pointcloud in [filename].");
    utility::LogInfo("    2. use KDTreeFlann to compute 50 nearest neighbors of point0.");
    utility::LogInfo("    3. convert the correspondences to LineSet and render it.");
    utility::LogInfo("    4. rotate the point cloud slightly to get another point cloud.");
    utility::LogInfo("    5. find closest point of the original point cloud on the new point cloud, mark as correspondences.");
    utility::LogInfo("    6. convert to LineSet and render it.");
    utility::LogInfo("    7. distance below 0.05 are rendered as red, others as black.");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc != 2 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    auto cloud_ptr = io::CreatePointCloudFromFile(argv[1]);
    std::vector<std::pair<int, int>> correspondences;

    const int nn = 50;
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(*cloud_ptr);
    std::vector<int> indices_vec(nn);
    std::vector<double> dists_vec(nn);
    kdtree.SearchKNN(cloud_ptr->points_[0], nn, indices_vec, dists_vec);
    for (int i = 0; i < nn; i++) {
        correspondences.push_back(std::make_pair(0, indices_vec[i]));
    }
    auto lineset_ptr = geometry::LineSet::CreateFromPointCloudCorrespondences(
            *cloud_ptr, *cloud_ptr, correspondences);
    visualization::DrawGeometries({cloud_ptr, lineset_ptr});

    auto new_cloud_ptr = std::make_shared<geometry::PointCloud>();
    *new_cloud_ptr = *cloud_ptr;
    auto bounding_box = new_cloud_ptr->GetAxisAlignedBoundingBox();
    Eigen::Matrix4d trans_to_origin = Eigen::Matrix4d::Identity();
    trans_to_origin.block<3, 1>(0, 3) = bounding_box.GetCenter() * -1.0;
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 3>(0, 0) = static_cast<Eigen::Matrix3d>(
            Eigen::AngleAxisd(M_PI / 6.0, Eigen::Vector3d::UnitX()));
    new_cloud_ptr->Transform(trans_to_origin.inverse() * transformation *
                             trans_to_origin);
    correspondences.clear();
    for (size_t i = 0; i < new_cloud_ptr->points_.size(); i++) {
        kdtree.SearchKNN(new_cloud_ptr->points_[i], 1, indices_vec, dists_vec);
        correspondences.push_back(std::make_pair(indices_vec[0], (int)i));
    }
    auto new_lineset_ptr =
            geometry::LineSet::CreateFromPointCloudCorrespondences(
                    *cloud_ptr, *new_cloud_ptr, correspondences);
    new_lineset_ptr->colors_.resize(new_lineset_ptr->lines_.size());
    for (size_t i = 0; i < new_lineset_ptr->lines_.size(); i++) {
        auto point_pair = new_lineset_ptr->GetLineCoordinate(i);
        if ((point_pair.first - point_pair.second).norm() <
            0.05 * bounding_box.GetMaxExtent()) {
            new_lineset_ptr->colors_[i] = Eigen::Vector3d(1.0, 0.0, 0.0);
        } else {
            new_lineset_ptr->colors_[i] = Eigen::Vector3d(0.0, 0.0, 0.0);
        }
    }
    visualization::DrawGeometries({cloud_ptr, new_cloud_ptr, new_lineset_ptr});

    return 0;
}
