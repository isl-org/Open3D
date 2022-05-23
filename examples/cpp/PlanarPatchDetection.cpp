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
    static constexpr int nrNeighbors = 100;
    const geometry::KDTreeSearchParam &normals_search_param =
            geometry::KDTreeSearchParamKNN(nrNeighbors);
    cloud_ptr->EstimateNormals(normals_search_param);
    cloud_ptr->OrientNormalsTowardsCameraLocation();
    std::cout << "EstimateNormals: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                         std::chrono::high_resolution_clock::now() - t1)
                         .count()
              << " seconds" << std::endl;

    static constexpr double radius = 1;
    static constexpr int max_nn = 100;
    const geometry::KDTreeSearchParam &search_param =
            geometry::KDTreeSearchParamHybrid(radius, max_nn);

    // set parameters
    constexpr double normal_variance_threshold_deg = 60;
    constexpr double coplanarity_deg = 75;
    constexpr double outlier_ratio = 0.75;
    constexpr double min_plane_edge_length = 0.0;
    constexpr size_t min_num_points = 30;

    t1 = std::chrono::high_resolution_clock::now();
    const std::vector<std::shared_ptr<geometry::OrientedBoundingBox>> patches =
            cloud_ptr->DetectPlanarPatches(normal_variance_threshold_deg,
                                           coplanarity_deg, outlier_ratio,
                                           min_plane_edge_length,
                                           min_num_points, search_param);
    std::cout << "DetectPlanarPatches: " << patches.size() << " in "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                         std::chrono::high_resolution_clock::now() - t1)
                         .count()
              << " seconds" << std::endl;

    cloud_ptr->PaintUniformColor(Eigen::Vector3d(0.8, 0.8, 0.8));

    // for const-correctness
    std::vector<std::shared_ptr<const geometry::Geometry>> geometries;
    geometries.reserve(patches.size());
    for (const auto &obox : patches) {
        std::vector<size_t> idxs =
                obox->GetPointIndicesWithinBoundingBox(cloud_ptr->points_);
        for (size_t j = 0; j < idxs.size(); ++j) {
            cloud_ptr->colors_[idxs[j]] = obox->color_;
        }
        geometries.push_back(obox);

        // turn bounding box into a plane
        auto mesh = geometry::TriangleMesh::CreateFromOrientedBoundingBox(
                *obox, Eigen::Vector3d(1, 1, 0.0001));
        mesh->PaintUniformColor(obox->color_);
        geometries.push_back(mesh);
    }

    // visualize point cloud, too
    geometries.push_back(cloud_ptr);

    visualization::DrawGeometries(geometries, "Visualize", 1600, 900);

    return 0;
}
