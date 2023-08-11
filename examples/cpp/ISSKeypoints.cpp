// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#include <Eigen/Core>
#include <cstdlib>
#include <memory>
#include <string>

#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > ISSKeypoints [mesh|pointcloud] [filename]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc != 3 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    const std::string option(argv[1]);
    const std::string filename(argv[2]);
    auto cloud = std::make_shared<geometry::PointCloud>();
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    if (option == "mesh") {
        if (!io::ReadTriangleMesh(filename, *mesh)) {
            utility::LogWarning("Failed to read {}", filename);
            return 1;
        }
        cloud->points_ = mesh->vertices_;
    } else if (option == "pointcloud") {
        if (!io::ReadPointCloud(filename, *cloud)) {
            utility::LogWarning("Failed to read {}\n\n", filename);
            return 1;
        }
    } else {
        utility::LogError("option {} not supported\n", option);
    }

    // Compute the ISS Keypoints
    auto iss_keypoints = std::make_shared<geometry::PointCloud>();
    {
        utility::ScopeTimer timer("ISS Keypoints estimation");
        iss_keypoints = geometry::keypoint::ComputeISSKeypoints(*cloud);
        utility::LogInfo("Detected {} keypoints",
                         iss_keypoints->points_.size());
    }

    // Visualize the results
    cloud->PaintUniformColor(Eigen::Vector3d(0.5, 0.5, 0.5));
    iss_keypoints->PaintUniformColor(Eigen::Vector3d(1.0, 0.75, 0.0));
    if (option == "mesh") {
        mesh->PaintUniformColor(Eigen::Vector3d(0.5, 0.5, 0.5));
        mesh->ComputeVertexNormals();
        mesh->ComputeTriangleNormals();
        visualization::DrawGeometries({mesh, iss_keypoints}, "ISS", 1600, 900);
    } else {
        visualization::DrawGeometries({iss_keypoints}, "ISS", 1600, 900);
    }

    return 0;
}
