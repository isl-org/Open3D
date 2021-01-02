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

#include <chrono>
#include <iostream>

#include "open3d/Open3D.h"
#include "open3d/geometry/Qhull.h"

int main(int argc, char **argv) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 2) {
        PrintOpen3DVersion();
        // clang-format off
        utility::LogInfo("Usage:");
        utility::LogInfo("    > PlanarPatchSegmentation [filename]");
        utility::LogInfo("    The program will :");
        utility::LogInfo("    1. load the pointcloud in [filename].");
        utility::LogInfo("    2. estimate planar patches within pointcloud");
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
    cloud_ptr->EstimateNormals();
    std::cout << "EstimateNormals: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                         std::chrono::high_resolution_clock::now() - t1)
                         .count()
              << " seconds" << std::endl;


    t1 = std::chrono::high_resolution_clock::now();
    const auto patches = cloud_ptr->DetectPlanarPatches();
    std::cout << "DetectPlanarPatches: " << patches.size() << " in "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                         std::chrono::high_resolution_clock::now() - t1)
                         .count()
              << " seconds" << std::endl;

    std::cout << "==============================" << std::endl;
    for (const auto& p : patches) {
        std::cout << p->normal_.transpose() << " ";
        std::cout << p->dist_from_origin_ << " |\t";
        std::cout << p->center_.transpose() << " |\t";
        std::cout << p->basis_x_.transpose() << " |\t";
        std::cout << p->basis_y_.transpose() << std::endl;
    }
    std::cout << "==============================" << std::endl;

    // for const-correctness
    std::vector<std::shared_ptr<const geometry::Geometry>> geometries;
    geometries.reserve(patches.size());
    for (const auto& patch : patches) {
        geometries.push_back(patch);
    }

    // visualize point cloud, too
    geometries.push_back(cloud_ptr);

    // const auto ret = geometry::Qhull::ComputeConvexHull(cloud_ptr->points_);
    // geometries.push_back(std::get<0>(ret));

    // auto cloud2 = cloud_ptr->SelectByIndex(std::get<1>(ret));
    // cloud2->PaintUniformColor(Eigen::Vector3d(255,0,0));
    // geometries.push_back(cloud2);

    visualization::DrawGeometries(geometries, "Visualize", 1600, 900);

    // const auto patch = geometry::PlanarPatch::CreateFromPointCloud(*cloud_ptr);
    // utility::LogInfo("Patch with center {} and normal {}", patch->center_, patch->normal_);


    // auto cloud_ptr_without_normals = std::make_shared<geometry::PointCloud>();
    // io::ReadPointCloud(argv[1], *cloud_ptr_without_normals);

    // visualization::DrawGeometries({/*cloud_ptr_without_normals,*/ patch}, "Visualize", 1600, 900);

    return 0;
}
