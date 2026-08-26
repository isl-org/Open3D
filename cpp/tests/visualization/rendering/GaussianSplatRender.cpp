// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Offscreen Gaussian splat rendering tests.
//
// Renders two test splats into an 8x8 offscreen color/depth grid and compares
// the grayscale values against deterministic golden references.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "open3d/Open3D.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"

using namespace open3d;

namespace {

// ---------------------------------------------------------------------------
// Golden 8x8 grayscale references.
// ---------------------------------------------------------------------------

const std::vector<uint8_t> kRefColorGray = {
        27, 24, 13, 5, 0, 0, 0, 0,  //
        58, 54, 31, 11, 2, 0, 0, 0,  //
        73, 70, 42, 16, 4, 0, 0, 0,  //
        54, 54, 34, 13, 3, 0, 3, 0,  //
        23, 24, 16, 6, 1, 26, 76, 17,  //
        6, 7, 4, 2, 0, 44, 133, 31,  //
        0, 0, 0, 0, 0, 5, 17, 4,  //
        0, 0, 0, 0, 0, 0, 0, 0,  //
};
const std::vector<uint8_t> kRefDepthGray = {
        255, 255, 255, 255, 255, 255, 255, 255,  //
        199, 199, 255, 255, 255, 255, 255, 255,  //
        199, 199, 199, 255, 255, 255, 255, 255,  //
        199, 199, 255, 255, 255, 255, 255, 255,  //
        255, 255, 255, 255, 255, 255, 243, 255,  //
        255, 255, 255, 255, 255, 255, 243, 255,  //
        255, 255, 255, 255, 255, 255, 255, 255,  //
        255, 255, 255, 255, 255, 255, 255, 255,  //
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Two separated Gaussian splats. SH degree 0 stores DC coefficients, not
/// direct RGB: color = 0.5 + 0.2820947918 * f_dc.
t::geometry::PointCloud MakeTwoTestSplats() {
    using namespace t::geometry;
    PointCloud pcd;
    const int N = 2;

    pcd.SetPointPositions(core::Tensor(std::vector<float>{
            -0.20f, -0.10f, 4.55f, 0.70f, 0.35f, 3.00f},
            {N, 3}, core::Dtype::Float32));
    pcd.SetPointAttr("opacity", core::Tensor(std::vector<float>{8.0f, 8.0f},
            {N, 1}, core::Dtype::Float32));
    pcd.SetPointAttr("rot", core::Tensor(std::vector<float>{
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
            {N, 4}, core::Dtype::Float32));
    pcd.SetPointAttr("scale", core::Tensor(std::vector<float>{
            0.08f, 0.08f, 0.08f, 0.08f, 0.08f, 0.08f},
            {N, 3}, core::Dtype::Float32));
    pcd.SetPointAttr("f_dc", core::Tensor(std::vector<float>{
            1.7724539f, -1.7724539f, -1.7724539f,
            -1.7724539f, 1.7724539f, -1.7724539f},
            {N, 3}, core::Dtype::Float32));
    return pcd;
}

std::vector<uint8_t> ImageToGray(const geometry::Image& image,
                                 bool is_color) {
    auto tensor_image = t::geometry::Image::FromLegacy(
            image, core::Device("CPU:0"));
    if (is_color) {
        return tensor_image.RGBToGray()
                .AsTensor()
                .ToFlatVector<uint8_t>();
    }
    return tensor_image.To(core::Dtype::UInt8, false, 255.0)
            .AsTensor()
            .ToFlatVector<uint8_t>();
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

        static void TearDownTestSuite() {
                if (initialized_) {
                        visualization::gui::Application::GetInstance().OnTerminate();
                        initialized_ = false;
                }
        }

    static bool initialized_;
};

bool GaussianSplatRenderTest::initialized_ = false;

}  // namespace

// ---------------------------------------------------------------------------
// Render two splats and compare the direct 8x8 output grid.
// ---------------------------------------------------------------------------

TEST_F(GaussianSplatRenderTest, RenderToImageTwoSplats) {
        constexpr int kW = 8;
        constexpr int kH = 8;

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

        // Visual debugging:
        // io::WriteImage("/tmp/gs_test_color_8x8.png", *color_img);
        // io::WriteImage("/tmp/gs_test_depth_8x8.png", *depth_img);

        auto gray_color = ImageToGray(*color_img, true);
        auto gray_depth = ImageToGray(*depth_img, false);
    ASSERT_EQ(gray_color.size(), 64u);
    ASSERT_EQ(gray_depth.size(), 64u);

        EXPECT_EQ(gray_color, kRefColorGray);
        EXPECT_EQ(gray_depth, kRefDepthGray);
}