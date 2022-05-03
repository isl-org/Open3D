// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include <chrono>
#include <iostream>

#include "open3d/Open3D.h"

int main(int argc, char **argv) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Info);

    if (argc < 2) {
        PrintOpen3DVersion();
        // clang-format off
        utility::LogInfo("Usage:");
        utility::LogInfo("    > PlanarPatchDetection [filename]");
        utility::LogInfo("    The program will :");
        utility::LogInfo("    1. load the point cloud in [filename].");
        utility::LogInfo("    2. estimate planar patches within point cloud");
        utility::LogInfo("    3. visualize those planar patches");
        // clang-format on
        return 1;
    }

    auto cloud_ptr = std::make_shared<geometry::PointCloud>();
    if (io::ReadPointCloud(argv[1], *cloud_ptr)) {
        utility::LogInfo("Successfully read {}\n", argv[1]);
    } else {
        utility::LogWarning("Failed to read {}\n\n", argv[1]);
        return 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    static constexpr int nrNeighbors = 75;
    const geometry::KDTreeSearchParam &search_param =
            geometry::KDTreeSearchParamKNN(nrNeighbors);
    cloud_ptr->EstimateNormals(search_param);
    std::cout << "EstimateNormals: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                         std::chrono::high_resolution_clock::now() - t1)
                         .count()
              << " seconds" << std::endl;

    // set parameters
    constexpr double normal_similarity = 0.5;
    constexpr double coplanarity = 0.25;
    constexpr double outlier_ratio = 0.75;
    constexpr double min_plane_edge_length = 0.0;
    constexpr size_t min_num_points = 30;

    t1 = std::chrono::high_resolution_clock::now();
    const std::vector<Eigen::Matrix<double, 12, 1>> patches =
            cloud_ptr->DetectPlanarPatches(normal_similarity, coplanarity,
                                           outlier_ratio, min_plane_edge_length,
                                           min_num_points, search_param);
    std::cout << "DetectPlanarPatches: " << patches.size() << " in "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                         std::chrono::high_resolution_clock::now() - t1)
                         .count()
              << " seconds" << std::endl;

    // Colors (default MATLAB colors)
    static constexpr int NUM_COLORS = 6;
    static std::array<Eigen::Vector3d, NUM_COLORS> colors = {
            Eigen::Vector3d(0.8500, 0.3250, 0.0980),
            Eigen::Vector3d(0.9290, 0.6940, 0.1250),
            Eigen::Vector3d(0.4940, 0.1840, 0.5560),
            Eigen::Vector3d(0.4660, 0.6740, 0.1880),
            Eigen::Vector3d(0.3010, 0.7450, 0.9330),
            Eigen::Vector3d(0.6350, 0.0780, 0.1840)};

    // for const-correctness
    std::vector<std::shared_ptr<const geometry::Geometry>> geometries;
    geometries.reserve(patches.size());
    for (size_t i = 0; i < patches.size(); ++i) {
        const auto &patch = patches[i];

        const Eigen::Vector3d normal = patch.head<3>().normalized();
        // const double d = patch.head<3>().norm();
        const Eigen::Vector3d center = patch.segment<3>(3);
        const Eigen::Vector3d basis_x = patch.segment<3>(6);
        const Eigen::Vector3d basis_y = patch.segment<3>(9);

        auto mesh = geometry::TriangleMesh::CreatePlanarPatch(
                center, basis_x, basis_y, normal, true);
        mesh->PaintUniformColor(colors[i % NUM_COLORS]);
        geometries.push_back(mesh);
    }

    // visualize point cloud, too
    geometries.push_back(cloud_ptr);

    visualization::DrawGeometries(geometries, "Visualize", 1600, 900);

    return 0;
}
