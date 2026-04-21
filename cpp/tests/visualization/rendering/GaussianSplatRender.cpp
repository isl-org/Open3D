// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Integration test for the 3D Gaussian Splatting offscreen render pipeline.
//
// Prerequisites:
//   - Requires a GPU / display server (e.g. Xvfb on CI headless Linux).
//   - Reference PNG: testdata/GaussianSplatRender_RenderToImage.png.
//     Test is skipped when the reference is absent.
//     Run with OPEN3D_TEST_GENERATE_REFERENCE=1 to regenerate it.

#include <cstdlib>
#include <string>

#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/io/ImageIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/MaterialRecord.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

namespace {

std::string RenderToImageReferencePngPath(bool depth = false) {
    std::string here(__FILE__);
    const auto pos = here.find_last_of("/\\");
    const std::string dir =
            (pos == std::string::npos) ? std::string() : here.substr(0, pos);
    if (depth) {
        return dir + "/testdata/GaussianSplatRender_RenderToDepth.png";
    } else {
        return dir + "/testdata/GaussianSplatRender_RenderToImage.png";
    }
}

t::geometry::PointCloud MakeTwoSplatCloud() {
    auto positions = core::Tensor::Init<float>(
            {{0.0f, 0.0f, -2.0f}, {1.0f, 0.0f, -2.0f}});
    auto scale =
            core::Tensor::Init<float>({{0.1f, 0.1f, 0.1f}, {0.1f, 0.1f, 0.1f}});
    auto opacity = core::Tensor::Init<float>({{2.197f}, {2.197f}});
    auto rot = core::Tensor::Init<float>(
            {{1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}});
    auto f_dc =
            core::Tensor::Init<float>({{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}});

    t::geometry::PointCloud pcd(positions);
    pcd.SetPointAttr("scale", scale);
    pcd.SetPointAttr("opacity", opacity);
    pcd.SetPointAttr("rot", rot);
    pcd.SetPointAttr("f_dc", f_dc);
    return pcd;
}

struct OffscreenCtx {
    visualization::rendering::FilamentRenderer* renderer = nullptr;
    visualization::rendering::Open3DScene* scene = nullptr;

    OffscreenCtx(int w, int h) {
        using namespace visualization::rendering;
        renderer = std::make_unique<FilamentRenderer>(
                EngineInstance::GetInstance(), w, h,
                EngineInstance::GetResourceManager());
        scene = std::make_unique<Open3DScene>(*renderer);
    }

    ~OffscreenCtx() {
        visualization::rendering::EngineInstance::DestroyInstance();
    }

    OffscreenCtx(const OffscreenCtx&) = delete;
    OffscreenCtx& operator=(const OffscreenCtx&) = delete;
};

// Render the canonical two-splat scene using the given sort algorithm and
// return the resulting image.  The camera is placed so both splats are visible.
std::shared_ptr<geometry::Image> RenderTwoSplats(
        visualization::gui::Application& app,
        const visualization::rendering::MaterialRecord& mat,
        int w,
        int h,
        bool use_onesweep) {
    OffscreenCtx ctx(w, h);
    auto* gs = ctx.renderer->GetGaussianSplatRenderer();
    if (gs) {
        auto cfg = gs->GetRenderConfig();
        cfg.use_onesweep_sort = use_onesweep;
        gs->SetRenderConfig(cfg);
    }
    auto pcd = MakeTwoSplatCloud();
    ctx.scene->AddGeometry("splats", &pcd, mat, /*add_downsampled_copy=*/false);
    auto* cam = ctx.scene->GetCamera();
    cam->LookAt(Eigen::Vector3f(0.5f, 0.0f, -2.0f),
                Eigen::Vector3f(0.5f, 0.0f, 0.0f),
                Eigen::Vector3f(0.0f, 1.0f, 0.0f));
    cam->SetProjection(60.0, static_cast<double>(w) / h, 0.01, 100.0,
                       visualization::rendering::Camera::FovType::Vertical);
    return app.RenderToImage(*ctx.renderer, ctx.scene->GetView(),
                             ctx.scene->GetScene(), w, h);
}

}  // namespace

TEST(GaussianSplatRender, RenderToImage) {
    constexpr int kW = 36;
    constexpr int kH = 20;

    const std::string ref_path = RenderToImageReferencePngPath();
    const std::string ref_depth_path = RenderToImageReferencePngPath(true);

    // If OPEN3D_TEST_GENERATE_REFERENCE=1, write the rendered image as the new
    // golden reference and skip comparison. Use this to update the reference
    // after verifying the visual output is correct.
    const bool generate_ref = []() {
        const char* e = std::getenv("OPEN3D_TEST_GENERATE_REFERENCE");
        return e && std::string(e) == "1";
    }();

    t::geometry::Image ref_img;
    t::geometry::Image ref_depth_img;
    if (!generate_ref) {
        if (!utility::filesystem::FileExists(ref_path)) {
            GTEST_SKIP() << "Missing golden PNG: " << ref_path
                         << "\n  Re-run with OPEN3D_TEST_GENERATE_REFERENCE=1 "
                            "to create it.";
        }
        ASSERT_TRUE(t::io::ReadImageFromPNG(ref_path, ref_img))
                << "Failed to read reference PNG: " << ref_path;

        if (!utility::filesystem::FileExists(ref_depth_path)) {
            GTEST_SKIP() << "Missing golden depth PNG: " << ref_depth_path
                         << "\n  Re-run with OPEN3D_TEST_GENERATE_REFERENCE=1 "
                            "to create it.";
        }
        ASSERT_TRUE(t::io::ReadImageFromPNG(ref_depth_path, ref_depth_img))
                << "Failed to read reference depth PNG: " << ref_depth_path;
    }

    auto& app = visualization::gui::Application::GetInstance();
    app.Initialize();

    OffscreenCtx ctx(kW, kH);

    visualization::rendering::MaterialRecord mat;
    mat.shader = "gaussianSplat";
    mat.gaussian_splat_sh_degree = 0;
    mat.gaussian_splat_min_alpha = 0.0f;
    mat.gaussian_splat_antialias = false;

    auto pcd = MakeTwoSplatCloud();
    ctx.scene->AddGeometry("splats", &pcd, mat,
                           /*add_downsampled_copy=*/false);

    auto* cam = ctx.scene->GetCamera();
    cam->LookAt(/*center=*/Eigen::Vector3f(0.5f, 0.0f, -2.0f),
                /*eye=*/Eigen::Vector3f(0.5f, 0.0f, 0.0f),
                /*up=*/Eigen::Vector3f(0.0f, 1.0f, 0.0f));
    cam->SetProjection(60.0, static_cast<double>(kW) / kH, 0.01, 100.0,
                       visualization::rendering::Camera::FovType::Vertical);

    auto img = app.RenderToImage(*ctx.renderer, ctx.scene->GetView(),
                                 ctx.scene->GetScene(), kW, kH);

    ASSERT_NE(img, nullptr) << "RenderToImage returned null";
    EXPECT_EQ(img->width_, kW);
    EXPECT_EQ(img->height_, kH);
    EXPECT_EQ(img->num_of_channels_, 3);

    t::geometry::Image rendered = t::geometry::Image::FromLegacy(*img);

    if (generate_ref) {
        ASSERT_TRUE(t::io::WriteImageToPNG(ref_path, rendered))
                << "Failed to write reference PNG: " << ref_path;
        utility::LogInfo("Reference PNG written to {}", ref_path);
    }

    if (!generate_ref) {
        ASSERT_TRUE(ref_img.AsTensor().GetShape() ==
                    rendered.AsTensor().GetShape())
                << "Reference shape "
                << ref_img.AsTensor().GetShape().ToString() << " vs rendered "
                << rendered.AsTensor().GetShape().ToString();
        AllCloseOrShow(ref_img.AsTensor(), rendered.AsTensor(), 0.0, 5.0);
    }

    auto depth = app.RenderToDepthImage(*ctx.renderer, ctx.scene->GetView(),
                                        ctx.scene->GetScene(), kW, kH);
    ASSERT_NE(depth, nullptr) << "RenderToDepthImage returned null";
    EXPECT_EQ(depth->width_, kW);
    EXPECT_EQ(depth->height_, kH);
    EXPECT_EQ(depth->num_of_channels_, 1);
    EXPECT_EQ(depth->bytes_per_channel_, 4);

    t::geometry::Image rendered_depth = t::geometry::Image::FromLegacy(*depth);
    utility::LogInfo("Rendered depth image {}: min {}, max {}",
                     rendered_depth.ToString(),
                     rendered_depth.AsTensor().Min({0, 1, 2}).ToString(),
                     rendered_depth.AsTensor().Max({0, 1, 2}).ToString());

    // Filament convention: default depth is inverse (near=1, far=0).
    const float depth_min =
            rendered_depth.AsTensor().Min({0, 1, 2}).Item<float>();
    const float depth_max =
            rendered_depth.AsTensor().Max({0, 1, 2}).Item<float>();
    EXPECT_GE(depth_min, 0.0);
    EXPECT_LE(depth_max, 1.0);

    // Depth references are stored as PNG, so compare as UInt16.
    t::geometry::Image rendered_depth_u16 =
            rendered_depth.To(core::UInt16, /*copy=*/false, /*scale=*/65535.0f);

    if (generate_ref) {
        ASSERT_TRUE(t::io::WriteImageToPNG(ref_depth_path, rendered_depth_u16))
                << "Failed to write reference depth PNG: " << ref_depth_path;
        utility::LogInfo("Reference depth PNG written to {}", ref_depth_path);
    }

    if (!generate_ref) {
        ASSERT_TRUE(ref_depth_img.AsTensor().GetShape() ==
                    rendered_depth_u16.AsTensor().GetShape())
                << "Reference depth shape "
                << ref_depth_img.AsTensor().GetShape().ToString()
                << " vs rendered depth "
                << rendered_depth_u16.AsTensor().GetShape().ToString();
        AllCloseOrShow(ref_depth_img.AsTensor(), rendered_depth_u16.AsTensor(),
                       0.0, 5.0);
    } else {
        GTEST_SKIP() << "Reference PNGs generated. Remove "
                        "OPEN3D_TEST_GENERATE_REFERENCE=1 "
                        "to run comparison.";
    }
}

// Verify that the optimised OneSweep sort produces pixel-identical output to
// the classical 4-pass radix sort on the same two-splat scene.  The test is
// effectively skipped when OneSweep programs are unavailable (e.g. Windows
// without subgroup arithmetic): the runner falls back to classical radix, so
// both renders are identical and the comparison passes regardless.
TEST(GaussianSplatRender, OneSweepSortMatchesRadixSort) {
    constexpr int kW = 36;
    constexpr int kH = 20;

    auto& app = visualization::gui::Application::GetInstance();
    app.Initialize();

    visualization::rendering::MaterialRecord mat;
    mat.shader = "gaussianSplat";
    mat.gaussian_splat_sh_degree = 0;
    mat.gaussian_splat_min_alpha = 0.0f;
    mat.gaussian_splat_antialias = false;

    auto classical_img =
            RenderTwoSplats(app, mat, kW, kH, /*use_onesweep=*/false);
    ASSERT_NE(classical_img, nullptr) << "Classical sort RenderToImage failed";

    auto onesweep_img =
            RenderTwoSplats(app, mat, kW, kH, /*use_onesweep=*/true);
    ASSERT_NE(onesweep_img, nullptr) << "OneSweep sort RenderToImage failed";

    ASSERT_EQ(onesweep_img->width_, classical_img->width_);
    ASSERT_EQ(onesweep_img->height_, classical_img->height_);

    // Allow ε=5 to tolerate fp16 rounding in center_xy.
    t::geometry::Image classical_t =
            t::geometry::Image::FromLegacy(*classical_img);
    t::geometry::Image onesweep_t =
            t::geometry::Image::FromLegacy(*onesweep_img);
    AllCloseOrShow(classical_t.AsTensor(), onesweep_t.AsTensor(), 0.0, 5.0);
}

}  // namespace tests
}  // namespace open3d
