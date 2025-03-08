// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdlib>

#include "open3d/Open3D.h"

using namespace open3d;

void Visualize_GaussianSplat() {
    // Please use you own file path.
    std::string file = "./point_cloud_29999_only_table.npz";
    std::shared_ptr<t::geometry::PointCloud> pointcloud = std::make_shared<t::geometry::PointCloud>();
    io::ReadPointCloudOption tmp = io::ReadPointCloudOption();
    t::io::ReadPointCloudFromNPZ(file, *pointcloud, tmp);
 
    visualization::Draw({pointcloud});
}

int main(int argc, char **argv) {
    Visualize_GaussianSplat();
}