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
    utility::LogInfo(
            "Usage: GaussianSplat <filename.[ply|splat]> [sh_degree] "
            "[min_alpha] [antialias]");
    utility::LogInfo("  sh_degree:  integer 0..2 (default 2)");
    utility::LogInfo("  min_alpha:  float 0..1 (default 0)");
    utility::LogInfo(
            "  antialias:  0 or 1 (default 0); enables density compensation "
            "to correct small-splat over-brightening");
}

int main(int argc, char** argv) {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Info);
    if (argc < 2 ||
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

    // Create a test cube mesh for depth compositing testing.
    auto cube = t::geometry::TriangleMesh::CreateBox(0.25, 0.25, 0.25);
    cube.ComputeVertexNormals();
    auto colors = core::Tensor::Init<float>({{1.0f, 0.2f, 0.2f}})
                          .Expand(cube.GetVertexPositions().GetShape());
    cube.SetVertexColors(colors);  // Red: shape {1,3}
    // Position the cube at the 3DGS scene center.
    auto bbox = gsplat->GetAxisAlignedBoundingBox();
    auto center = bbox.GetCenter();
    cube.Translate(center, /*relative=*/false);
    auto p_cube = std::make_shared<t::geometry::TriangleMesh>(cube);

    // Parse optional args: sh_degree, min_alpha, antialias
    int sh_degree = 2;
    float min_alpha = 0.0f;
    bool antialias = false;
    if (argc >= 3) {
        try {
            sh_degree = std::stoi(argv[2]);
        } catch (...) {
            utility::LogWarning("Invalid sh_degree '{}', using default {}",
                                argv[2], sh_degree);
        }
    }
    if (argc >= 4) {
        try {
            min_alpha = std::stof(argv[3]);
        } catch (...) {
            utility::LogWarning("Invalid min_alpha '{}', using default {}",
                                argv[3], min_alpha);
        }
    }
    if (argc >= 5) {
        try {
            antialias = std::stoi(argv[4]) != 0;
        } catch (...) {
            utility::LogWarning("Invalid antialias '{}', using default 0",
                                argv[4]);
        }
    }

    // Clamp values to sane ranges
    sh_degree = std::max(0, std::min(2, sh_degree));
    if (min_alpha < 0.0f) min_alpha = 0.0f;
    if (min_alpha > 1.0f) min_alpha = 1.0f;

    utility::LogInfo("Using sh_degree={} min_alpha={} antialias={}", sh_degree,
                     min_alpha, antialias);

    // Create a material that sets SH degree, alpha filter, and anti-aliasing.
    visualization::rendering::MaterialRecord mat;
    mat.shader = "gaussianSplat";
    mat.gaussian_splat_sh_degree = sh_degree;
    mat.gaussian_splat_min_alpha = min_alpha;
    mat.gaussian_splat_antialias = antialias;

    // Launch visualizer directly so we can pass the material to AddGeometry.
    auto& app = Application::GetInstance();
    app.Initialize();
    auto vis = std::make_shared<visualization::visualizer::O3DVisualizer>(
            "Gaussian Splat + Mesh Depth Test", 1024, 768);

    vis->AddGeometry(argv[1], gsplat, &mat);
    vis->AddGeometry("test_cube", p_cube);
    vis->ResetCameraToDefault();
    app.AddWindow(vis);
    vis.reset();
    app.Run();
}