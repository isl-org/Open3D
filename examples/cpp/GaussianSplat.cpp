// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// To run this example, download some sample Gaussian Splats. You can use these
// samples to get started:
// clang-format off
// curl -O https://github.com/isl-org/open3d_downloads/releases/download/3dgs-1/mipnerf360_garden_crop_table.ply
// curl -O https://github.com/isl-org/open3d_downloads/releases/download/3dgs-1/mipnerf360_garden_crop_table.splat
// clang-format on

#include <cstdlib>

#include "open3d/Open3D.h"

using namespace open3d;
using visualization::gui::Application;

void PrintUsage() {
    utility::LogInfo("Visualize Gaussian Splat from PLY or SPLAT file.");
    utility::LogInfo("Usage: GaussianSplat <filename.[ply|splat]>");
}

int main(int argc, char** argv) {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Info);
    if (argc != 2 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintUsage();
        return 1;
    }
    std::shared_ptr<t::geometry::PointCloud> gsplat =
            std::make_shared<t::geometry::PointCloud>();
    if (!t::io::ReadPointCloud(argv[1], *gsplat)) {
        utility::LogWarning("Failed to read file {}", argv[1]);
        return 1;
    }

    // Create a test sphere mesh for depth compositing testing.
    auto sphere = t::geometry::TriangleMesh::CreateSphere(0.2, 20);
    sphere.ComputeVertexNormals();
    auto colors = core::Tensor::Init<float>({{1.0f, 0.2f, 0.2f}})
                          .Expand(sphere.GetVertexPositions().GetShape());
    sphere.SetVertexColors(colors);  // Red: shape {1,3}
    // Position the sphere at the 3DGS scene center.
    auto bbox = gsplat->GetAxisAlignedBoundingBox();
    auto center = bbox.GetCenter();
    sphere.Translate(center, /*relative=*/false);
    auto p_sphere = std::make_shared<t::geometry::TriangleMesh>(sphere);

    // Create a material that limits SH degree to 0 and filters low-alpha
    // splats (only splats with opacity >= 0.2 will be cached/rendered).
    visualization::rendering::MaterialRecord mat;
    mat.shader = "gaussianSplat";
    mat.gaussian_splat_sh_degree = 0;
    mat.gaussian_splat_min_alpha = 0.2f;

    // Launch visualizer directly so we can pass the material to AddGeometry.
    auto& app = Application::GetInstance();
    app.Initialize();
    auto vis = std::make_shared<visualization::visualizer::O3DVisualizer>(
            "Gaussian Splat + Mesh Depth Test", 1024, 768);

    vis->AddGeometry(argv[1], gsplat, &mat);
    vis->AddGeometry("test_sphere", p_sphere);
    vis->ShowGeometry(argv[1], true);
    vis->ShowGeometry("test_sphere", true);

    vis->ResetCameraToDefault();
    app.AddWindow(vis);
    vis.reset();
    app.Run();
}