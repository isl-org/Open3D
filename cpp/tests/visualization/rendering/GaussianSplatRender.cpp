// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Offscreen Gaussian splat rendering tests.
//
// Creates 2 test splats (red + green, high opacity, identity rotation),
// renders to 64x64 colour + depth images via Application::RenderToImage
// / RenderToDepthImage, and verifies non-black output.
//
// Set OPEN3D_TEST_SAVE_OUTPUTS=1 to write 64x64 PNGs to /tmp for human
// review.  Once approved, the 64x64 images are downsampled to 8x8
// grayscale and stored as golden byte arrays in the test for automated
// regression detection.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "open3d/Open3D.h"
#include "open3d/core/Tensor.h"
#include "open3d/io/ImageIO.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"

using namespace open3d;

namespace {

// ---------------------------------------------------------------------------
// Golden 8x8 grayscale references -- TODO: populate after human review of
// 64x64 outputs saved via OPEN3D_TEST_SAVE_OUTPUTS=1.
// ---------------------------------------------------------------------------

const std::vector<uint8_t> kRefColorGray = {
        0, 0, 0, 0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
};
const std::vector<uint8_t> kRefDepthGray = {
        0, 0, 0, 0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Two-unit Gaussian splat PointCloud.  SH degree 0 (DC only).
/// Splat 0: red at (0,0,4), scale (0.3,0.3,0.3)
/// Splat 1: green at (1,0.5,3.5), scale (0.4,0.2,0.2)
t::geometry::PointCloud MakeTwoTestSplats() {
    using namespace t::geometry;
    PointCloud pcd;
    const int N = 2;

    pcd.SetPointPositions(core::Tensor(std::vector<float>{
            0.0f, 0.0f, 4.0f, 1.0f, 0.5f, 3.5f},
            {N, 3}, core::Dtype::Float32));
    pcd.SetPointAttr("opacity", core::Tensor(std::vector<float>{1.0f, 1.0f},
            {N, 1}, core::Dtype::Float32));
    pcd.SetPointAttr("rot", core::Tensor(std::vector<float>{
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
            {N, 4}, core::Dtype::Float32));
    pcd.SetPointAttr("scale", core::Tensor(std::vector<float>{
            0.3f, 0.3f, 0.3f, 0.4f, 0.2f, 0.2f},
            {N, 3}, core::Dtype::Float32));
    pcd.SetPointAttr("f_dc", core::Tensor(std::vector<float>{
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
            {N, 3}, core::Dtype::Float32));
    return pcd;
}

/// Convert a color or depth image to an 8x8 grayscale uint8 vector.
/// Uses t::geometry::Image::Resize for downsampling and
/// t::geometry::Image::RGBToGray for color-to-gray conversion.
std::vector<uint8_t> ImageToGray8x8(const geometry::Image& img,
                                     bool is_depth) {
    // Convert legacy -> tensor image.
    auto t_img = t::geometry::Image::FromLegacy(
            img, core::Device("CPU:0"));

    // Resize to 8x8 using bilinear filtering.
    const float rate = 8.0f / static_cast<float>(img.width_);
    t_img = t_img.Resize(rate,
                          t::geometry::Image::InterpType::Linear);

    if (!is_depth) {
        // RGB(A) -> grayscale using BT.601 luma.
        t_img = t_img.RGBToGray();
    }

    // Convert back to legacy for data extraction.
    auto legacy = t_img.ToLegacy();
    std::vector<uint8_t> result(legacy.data_.begin(), legacy.data_.end());
    return result;
}

// =========================================================================
// GaussianSplatRenderTest fixture
// =========================================================================

class GaussianSplatRenderTest : public testing::Test {
protected:
    void SetUp() override {
        if (!initialized_) {
            // Filament resources are at <build>/bin/resources/ relative
            // to the test executable.  Set the path explicitly so the
            // test works when run from any working directory.
            const char* res_env =
                    std::getenv("OPEN3D_RESOURCE_PATH");
            if (res_env && res_env[0] != '\0') {
                visualization::rendering::EngineInstance::SetResourcePath(
                        res_env);
            }
            visualization::gui::Application::GetInstance().Initialize();
            initialized_ = true;
        }
    }

    static bool initialized_;
};

bool GaussianSplatRenderTest::initialized_ = false;

}  // namespace

// ---------------------------------------------------------------------------
// Render 2 splats -> 64x64 -> downsample to 8x8 -> verify non-empty
// ---------------------------------------------------------------------------

TEST_F(GaussianSplatRenderTest, RenderToImageTwoSplats) {
    constexpr int kW = 64;
    constexpr int kH = 64;

    auto& engine = visualization::rendering::EngineInstance::GetInstance();
    auto& resource_mgr =
            visualization::rendering::EngineInstance::GetResourceManager();

    auto renderer =
            std::make_unique<visualization::rendering::FilamentRenderer>(
                    engine, kW, kH, resource_mgr);

    auto scene = std::make_unique<visualization::rendering::Open3DScene>(
            *renderer);

    visualization::rendering::MaterialRecord mat;
    mat.shader = "gaussianSplat";
    mat.point_size = 0.01f;
    mat.gaussian_splat_sh_degree = 0;

    auto pcd = MakeTwoTestSplats();
    scene->AddGeometry("test_splats", &pcd, mat);

    auto* cam = scene->GetCamera();
    cam->SetProjection(60.0f, static_cast<float>(kW) / kH, 0.1f, 50.0f,
                       visualization::rendering::Camera::FovType::Vertical);
    cam->LookAt({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 5.0f},
                {0.0f, 1.0f, 0.0f});

    auto& app = visualization::gui::Application::GetInstance();

    // ---- colour ----
    auto color_img =
            app.RenderToImage(*renderer, scene->GetView(), scene->GetScene(),
                              kW, kH);
    ASSERT_TRUE(color_img);
    EXPECT_EQ(color_img->width_, kW);
    EXPECT_EQ(color_img->height_, kH);

    // ---- depth ----
    auto depth_img = app.RenderToDepthImage(
            *renderer, scene->GetView(), scene->GetScene(), kW, kH);
    ASSERT_TRUE(depth_img);
    EXPECT_EQ(depth_img->width_, kW);
    EXPECT_EQ(depth_img->height_, kH);

    // ---- save 64x64 PNGs for human review ----
    const char* save_env = std::getenv("OPEN3D_TEST_SAVE_OUTPUTS");
    if (save_env && save_env[0] != '\0' && std::strcmp(save_env, "0") != 0) {
        io::WriteImage("/tmp/gs_test_color_64x64.png", *color_img);
        io::WriteImage("/tmp/gs_test_depth_64x64.png", *depth_img);
        utility::LogInfo(
                "Saved /tmp/gs_test_color_64x64.png and "
                "/tmp/gs_test_depth_64x64.png");
    }

    // ---- downsample to 8x8 grayscale ----
    auto gray_color = ImageToGray8x8(*color_img, false);
    auto gray_depth = ImageToGray8x8(*depth_img, true);
    ASSERT_EQ(gray_color.size(), 64u);
    ASSERT_EQ(gray_depth.size(), 64u);

    // Sanity: output must contain non-zero pixels.
    int nc = 0, nd = 0;
    for (int i = 0; i < 64; ++i) {
        if (gray_color[i] > 0) ++nc;
        if (gray_depth[i] > 0) ++nd;
    }
    EXPECT_GE(nc, 1) << "Color output is all black -- GS render failed.";
    EXPECT_GE(nd, 1) << "Depth output is all black -- GS render failed.";

    // TODO: once golden references are captured:
    //   EXPECT_EQ(gray_color, kRefColorGray);
    //   EXPECT_EQ(gray_depth,  kRefDepthGray);
}