// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Offscreen Gaussian splat rendering tests.
//
// Renders test splats (plus, in the mixed-geometry test, an opaque mesh cube)
// into an 8x8 offscreen color/depth grid and compares the grayscale values
// against deterministic golden references.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>

#include "open3d/Open3D.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/SYCLUtils.h"
#include "open3d/core/Tensor.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatRenderer.h"

using namespace open3d;

namespace {

// ---------------------------------------------------------------------------
// Golden 8x8 grayscale references.
// ---------------------------------------------------------------------------

const std::vector<uint8_t> kRefColorGray = {
        255, 255, 255, 255, 255, 255, 255, 255,  //
        247, 245, 247, 251, 255, 246, 223, 245,  //
        223, 221, 229, 241, 249, 222, 150, 222,  //
        175, 172, 193, 221, 242, 241, 222, 245,  //
        114, 113, 152, 200, 234, 249, 255, 255,  //
        81,  85,  135, 193, 232, 249, 255, 255,  //
        106, 114, 158, 207, 237, 250, 255, 255,  //
        167, 173, 200, 228, 246, 255, 255, 255,  //
};
const std::vector<uint8_t> kRefDepthGray = {
        0,  0,  0,  0, 0, 0, 0,   0,  //
        0,  0,  0,  0, 0, 0, 0,   0,  //
        0,  0,  0,  0, 0, 0, 128, 0,  //
        0,  0,  0,  0, 0, 0, 0,   0,  //
        29, 29, 29, 0, 0, 0, 0,   0,  //
        29, 29, 29, 0, 0, 0, 0,   0,  //
        29, 29, 29, 0, 0, 0, 0,   0,  //
        0,  0,  0,  0, 0, 0, 0,   0,  //
};

// Mixed-geometry (splats + mesh) scene: an opaque blue cube occludes the lower
// green splat. The splat is at (0.72, 0.43, 3.0); the cube is at z=2.8 and its
// XY coordinates scale by 2.8 / 3.0 to project to the same image position.
const std::vector<uint8_t> kRefMixedColorGray = {
        255, 255, 255, 255, 255, 255, 255, 255,  //
        247, 245, 247, 251, 255, 246, 223, 245,  //
        223, 221, 229, 241, 249, 57,  150, 222,  //
        175, 172, 193, 221, 242, 241, 222, 245,  //
        114, 113, 152, 200, 234, 249, 255, 255,  //
        81,  85,  135, 193, 232, 249, 255, 255,  //
        106, 114, 158, 207, 237, 250, 255, 255,  //
        167, 173, 200, 228, 246, 255, 255, 255,  //
};
const std::vector<uint8_t> kRefMixedDepthGray = {
        0,  0,  0,  0, 0, 0, 0,   0,  //
        0,  0,  0,  0, 0, 0, 0,   0,  //
        0,  0,  0,  0, 0, 0, 128, 0,  //
        0,  0,  0,  0, 0, 0, 0,   0,  //
        29, 29, 29, 0, 0, 0, 0,   0,  //
        29, 29, 29, 0, 0, 0, 0,   0,  //
        29, 29, 29, 0, 0, 0, 0,   0,  //
        0,  0,  0,  0, 0, 0, 0,   0,  //
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

    pcd.SetPointPositions(core::Tensor(
            std::vector<float>{-0.20f, -0.10f, 4.55f, 0.72f, 0.43f, 3.00f},
            {N, 3}, core::Dtype::Float32));
    pcd.SetPointAttr("opacity", core::Tensor(std::vector<float>{8.0f, 8.0f},
                                             {N, 1}, core::Dtype::Float32));
    pcd.SetPointAttr(
            "rot", core::Tensor(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                                   0.0f, 0.0f, 0.0f},
                                {N, 4}, core::Dtype::Float32));
    pcd.SetPointAttr("scale",
                     core::Tensor(std::vector<float>{0.10f, 0.10f, 0.10f, 0.10f,
                                                     0.10f, 0.10f},
                                  {N, 3}, core::Dtype::Float32));
    pcd.SetPointAttr("f_dc",
                     core::Tensor(std::vector<float>{1.7724539f, -1.7724539f,
                                                     -1.7724539f, -1.7724539f,
                                                     1.7724539f, -1.7724539f},
                                  {N, 3}, core::Dtype::Float32));
    return pcd;
}

/// Opaque blue occluder cube placed between the two test splats (see comment
/// on kRefMixedColorGray for placement rationale).
std::shared_ptr<geometry::TriangleMesh> MakeOcclusionTestCube() {
    constexpr double kSize = 0.4;
    auto cube = geometry::TriangleMesh::CreateBox(kSize, kSize, kSize);
    cube->Translate(Eigen::Vector3d(0.672, 0.4013333333, 2.8) -
                    Eigen::Vector3d(kSize / 2, kSize / 2, kSize / 2));
    cube->ComputeVertexNormals();
    return cube;
}

std::vector<uint8_t> ImageToGray(const geometry::Image& image, bool is_color) {
    auto tensor_image =
            t::geometry::Image::FromLegacy(image, core::Device("CPU:0"));
    if (is_color) {
        return tensor_image.RGBToGray().AsTensor().ToFlatVector<uint8_t>();
    }

    // RenderToDepthImage(..., true) returns linear view-space distance, with
    // infinity for no-hit pixels. Encode no-hit as 0 and metres as 64 levels.
    const auto depth = tensor_image.AsTensor().ToFlatVector<float>();
    std::vector<uint8_t> gray(depth.size());
    for (size_t index = 0; index < depth.size(); ++index) {
        gray[index] =
                std::isfinite(depth[index])
                        ? static_cast<uint8_t>(depth[index] * 64.0f + 0.5f)
                        : 0;
    }
    return gray;
}

geometry::Image MakeDepthPreview(const geometry::Image& depth_image) {
    const auto depth =
            t::geometry::Image::FromLegacy(depth_image, core::Device("CPU:0"))
                    .AsTensor()
                    .ToFlatVector<float>();
    geometry::Image preview;
    preview.Prepare(depth_image.width_, depth_image.height_, 1, 1);
    float min_depth = std::numeric_limits<float>::infinity();
    float max_depth = 0.0f;
    for (const float value : depth) {
        if (std::isfinite(value)) {
            min_depth = std::min(min_depth, value);
            max_depth = std::max(max_depth, value);
        }
    }
    if (!std::isfinite(min_depth)) {
        return preview;
    }
    const float range = std::max(max_depth - min_depth, 1e-6f);
    for (size_t index = 0; index < depth.size(); ++index) {
        float intensity = 0.0f;
        if (std::isfinite(depth[index])) {
            intensity = 48.0f + 207.0f * (max_depth - depth[index]) / range;
        }
        *preview.PointerAt<uint8_t>(index % depth_image.width_,
                                    index / depth_image.width_) =
                static_cast<uint8_t>(std::clamp(intensity, 0.0f, 255.0f));
    }
    return preview;
}

constexpr int kW = 8;
constexpr int kH = 8;

void SetUpTestCamera(visualization::rendering::Open3DScene& scene) {
    auto* cam = scene.GetCamera();
    cam->SetProjection(60.0f, static_cast<float>(kW) / kH, 0.1f, 50.0f,
                       visualization::rendering::Camera::FovType::Vertical);
    cam->LookAt({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f});
}

/// Renders `scene` to an 8x8 color/depth pair and checks it against the given
/// golden grayscale references. `dump_prefix`, when non-null, writes the
/// rendered images to /tmp/<dump_prefix>_{color,depth}_8x8.png for visual
/// inspection while calibrating new golden references.
void RenderAndCheckGolden(visualization::rendering::FilamentRenderer& renderer,
                          visualization::rendering::Open3DScene& scene,
                          const std::vector<uint8_t>& ref_color,
                          const std::vector<uint8_t>& ref_depth,
                          const char* dump_prefix = nullptr) {
    auto& app = visualization::gui::Application::GetInstance();

    auto color_img = app.RenderToImage(renderer, scene.GetView(),
                                       scene.GetScene(), kW, kH);
    ASSERT_TRUE(color_img);
    EXPECT_EQ(color_img->width_, kW);
    EXPECT_EQ(color_img->height_, kH);

    auto depth_img =
            app.RenderToDepthImage(renderer, scene.GetView(), scene.GetScene(),
                                   kW, kH, true /*z_in_view_space*/);
    ASSERT_TRUE(depth_img);
    EXPECT_EQ(depth_img->width_, kW);
    EXPECT_EQ(depth_img->height_, kH);

    if (dump_prefix) {
        io::WriteImage(std::string("/tmp/") + dump_prefix + "_color_8x8.png",
                       *color_img);
        auto depth_preview = MakeDepthPreview(*depth_img);
        io::WriteImage(std::string("/tmp/") + dump_prefix + "_depth_8x8.png",
                       depth_preview);
    }

    auto gray_color = ImageToGray(*color_img, true);
    auto gray_depth = ImageToGray(*depth_img, false);
    ASSERT_EQ(gray_color.size(), 64u);
    ASSERT_EQ(gray_depth.size(), 64u);

    const auto max_difference = [](const std::vector<uint8_t>& actual,
                                   const std::vector<uint8_t>& expected) {
        uint8_t maximum = 0;
        for (size_t index = 0; index < actual.size(); ++index) {
            maximum = std::max(maximum,
                               static_cast<uint8_t>(std::abs(
                                       static_cast<int>(actual[index]) -
                                       static_cast<int>(expected[index]))));
        }
        return maximum;
    };
    utility::LogInfo("Gaussian splat color max difference: {}",
                     max_difference(gray_color, ref_color));
    utility::LogInfo("Gaussian splat depth max difference: {}",
                     max_difference(gray_depth, ref_depth));
    EXPECT_EQ(gray_color, ref_color);
    EXPECT_EQ(gray_depth, ref_depth);
}

// =========================================================================
// GaussianSplatRenderTest fixture
// =========================================================================

class GaussianSplatRenderTest : public testing::Test {
protected:
    void SetUp() override {
        const char* ci = std::getenv("CI");
        // Very rough way to tell if a CI machine has a GPU
        if (ci && !core::cuda::IsAvailable() &&
            core::sy::GetDeviceCount() < 2) {
            GTEST_SKIP() << "Gaussian splat rendering requires GPU in CI";
        }
        if (!initialized_) {
            // Filament resources are at <build>/bin/resources/ relative
            // to the test executable.  Set the path explicitly so the
            // test works when run from any working directory.
            const char* res_env = std::getenv("OPEN3D_RESOURCE_PATH");
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
    auto& engine = visualization::rendering::EngineInstance::GetInstance();
    auto& resource_mgr =
            visualization::rendering::EngineInstance::GetResourceManager();

    auto renderer =
            std::make_unique<visualization::rendering::FilamentRenderer>(
                    engine, kW, kH, resource_mgr);

    auto scene =
            std::make_unique<visualization::rendering::Open3DScene>(*renderer);

    visualization::rendering::MaterialRecord mat;
    mat.shader = "gaussianSplat";
    mat.point_size = 0.01f;
    mat.gaussian_splat_sh_degree = 0;

    auto pcd = MakeTwoTestSplats();
    scene->AddGeometry("test_splats", &pcd, mat);

    SetUpTestCamera(*scene);
    RenderAndCheckGolden(*renderer, *scene, kRefColorGray, kRefDepthGray,
                         "twosplats_");
}

// ---------------------------------------------------------------------------
// Same two splats plus an opaque mesh cube occluder; see kRefMixedColorGray
// for the placement rationale. Exercises the composite shader's per-splat
// scene-depth occlusion test in both directions (mesh occludes splat, splat
// occludes mesh), which the splat-only test above cannot reach.
// ---------------------------------------------------------------------------

TEST_F(GaussianSplatRenderTest, RenderToImageSplatsAndMeshOcclusion) {
    auto& engine = visualization::rendering::EngineInstance::GetInstance();
    auto& resource_mgr =
            visualization::rendering::EngineInstance::GetResourceManager();

    auto renderer =
            std::make_unique<visualization::rendering::FilamentRenderer>(
                    engine, kW, kH, resource_mgr);

    auto scene =
            std::make_unique<visualization::rendering::Open3DScene>(*renderer);

    visualization::rendering::MaterialRecord gs_mat;
    gs_mat.shader = "gaussianSplat";
    gs_mat.point_size = 0.01f;
    gs_mat.gaussian_splat_sh_degree = 0;

    auto pcd = MakeTwoTestSplats();
    scene->AddGeometry("test_splats", &pcd, gs_mat);

    visualization::rendering::MaterialRecord cube_mat;
    cube_mat.shader = "defaultUnlit";
    cube_mat.base_color = {0.0f, 0.0f, 1.0f, 1.0f};

    auto cube = MakeOcclusionTestCube();
    scene->AddGeometry("occluder_cube", cube.get(), cube_mat);

    SetUpTestCamera(*scene);
    RenderAndCheckGolden(*renderer, *scene, kRefMixedColorGray,
                         kRefMixedDepthGray, "mixed_");
}
