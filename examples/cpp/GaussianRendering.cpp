// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdlib>

#include "open3d/Open3D.h"

using namespace open3d;

double GetRandom() { return double(std::rand()) / double(RAND_MAX); }

void SingleObject() {
    std::string file = "/home/zy/ws/Open3D/pointcloud/converted_point_cloud_29999_only_table.npz";
    std::shared_ptr<t::geometry::PointCloud> pointcloud = std::make_shared<t::geometry::PointCloud>();

    io::ReadPointCloudOption tmp = io::ReadPointCloudOption();
    t::io::ReadPointCloudFromNPZ(file, *pointcloud, tmp);

    visualization::Draw({pointcloud});
}

int main(int argc, char **argv) {
    SingleObject();
}
