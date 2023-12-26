// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <vector>

#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > Flann [filename]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace open3d;
    if (argc != 2 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    auto new_cloud_ptr = std::make_shared<geometry::PointCloud>();
    if (io::ReadPointCloud(argv[1], *new_cloud_ptr)) {
        utility::LogInfo("Successfully read {}", argv[1]);
    } else {
        utility::LogWarning("Failed to read {}", argv[1]);
        return 1;
    }

    if ((int)new_cloud_ptr->points_.size() < 100) {
        utility::LogWarning("Boring point cloud.");
        return 1;
    }

    if (!new_cloud_ptr->HasColors()) {
        new_cloud_ptr->colors_.resize(new_cloud_ptr->points_.size());
        for (size_t i = 0; i < new_cloud_ptr->points_.size(); i++) {
            new_cloud_ptr->colors_[i].setZero();
        }
    }

    int nn = std::min(20, (int)new_cloud_ptr->points_.size() - 1);
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(*new_cloud_ptr);
    std::vector<int> new_indices_vec(nn);
    std::vector<double> new_dists_vec(nn);
    kdtree.SearchKNN(new_cloud_ptr->points_[0], nn, new_indices_vec,
                     new_dists_vec);

    for (size_t i = 0; i < new_indices_vec.size(); i++) {
        utility::LogInfo("{:d}, {:f}", (int)new_indices_vec[i],
                         sqrt(new_dists_vec[i]));
        new_cloud_ptr->colors_[new_indices_vec[i]] =
                Eigen::Vector3d(1.0, 0.0, 0.0);
    }

    new_cloud_ptr->colors_[0] = Eigen::Vector3d(0.0, 1.0, 0.0);

    float r = float(sqrt(new_dists_vec[nn - 1]) * 2.0);
    int k = kdtree.SearchRadius(new_cloud_ptr->points_[99], r, new_indices_vec,
                                new_dists_vec);

    utility::LogInfo("======== {:d}, {:f} ========", k, r);
    for (int i = 0; i < k; i++) {
        utility::LogInfo("{:d}, {:f}", (int)new_indices_vec[i],
                         sqrt(new_dists_vec[i]));
        new_cloud_ptr->colors_[new_indices_vec[i]] =
                Eigen::Vector3d(0.0, 0.0, 1.0);
    }
    new_cloud_ptr->colors_[99] = Eigen::Vector3d(0.0, 1.0, 1.0);

    k = kdtree.Search(new_cloud_ptr->points_[199],
                      geometry::KDTreeSearchParamRadius(r), new_indices_vec,
                      new_dists_vec);

    utility::LogInfo("======== {:d}, {:f} ========", k, r);
    for (int i = 0; i < k; i++) {
        utility::LogInfo("{:d}, {:f}", (int)new_indices_vec[i],
                         sqrt(new_dists_vec[i]));
        new_cloud_ptr->colors_[new_indices_vec[i]] =
                Eigen::Vector3d(0.0, 0.0, 1.0);
    }
    new_cloud_ptr->colors_[199] = Eigen::Vector3d(0.0, 1.0, 1.0);

    visualization::DrawGeometries({new_cloud_ptr}, "KDTreeFlann", 1600, 900);
    return 0;
}
