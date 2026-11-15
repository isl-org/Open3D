// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Regression test for Scene::RenderToDepthImage()'s z_in_view_space default.
//
// Scene::RenderToDepthImage() (bound to Python as
// open3d.visualization.rendering.Scene.render_to_depth_image, used e.g. by
// SceneWidget mouse-picking code) is documented to return depth normalized to
// [0.0, 1.0] (near to far plane) unless the caller explicitly opts into linear
// view-space depth via z_in_view_space=true. FilamentScene::RenderToDepthImage
// previously ignored the parameter and always rendered linear depth, breaking
// every caller written against the documented [0, 1] contract (including
// Open3D's own mouse_and_point_coord.py example, which feeds the depth value
// straight into Camera::Unproject(), which still expects normalized depth).

#include <cmath>
#include <limits>
#include <memory>

#include "open3d/Open3D.h"
#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

namespace {

constexpr int kW = 32;
constexpr int kH = 32;

// A 1x1x1 box a few meters in front of the camera -- simple, unambiguous
// depth structure. Large enough to guarantee the box covers the center pixel
// and the far background (empty space) covers the corners, regardless of
// the exact FOV/near/far planes used by SetUpTestCamera().
std::shared_ptr<geometry::TriangleMesh> MakeTestBox() {
    auto box = geometry::TriangleMesh::CreateBox(1, 1, 1);
    box->Translate(Eigen::Vector3d(-0.5, -0.5, 4.5));
    box->ComputeVertexNormals();
    return box;
}

void SetUpTestCamera(visualization::rendering::Open3DScene& scene) {
    auto* cam = scene.GetCamera();
    cam->SetProjection(60.0f, static_cast<float>(kW) / kH, 0.1f, 50.0f,
                       visualization::rendering::Camera::FovType::Vertical);
    cam->LookAt({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
}

// Renders `scene`'s depth image via the Scene-level API (the one bound to
// Python's Scene.render_to_depth_image, as opposed to the already-correct
// Application::RenderToDepthImage/OffscreenRenderer path), mirroring how
// Application::RenderToDepthImage pumps the callback-based render itself.
std::shared_ptr<geometry::Image> RenderSceneDepthImage(
        visualization::rendering::FilamentRenderer& renderer,
        visualization::rendering::Open3DScene& scene,
        bool z_in_view_space) {
    scene.GetView()->SetViewport(0, 0, kW, kH);

    std::shared_ptr<geometry::Image> depth_img;
    auto callback = [&depth_img](std::shared_ptr<geometry::Image> img) {
        depth_img = img;
    };
    scene.GetScene()->RenderToDepthImage(callback, z_in_view_space);
    renderer.BeginFrame();
    renderer.EndFrame();
    return depth_img;
}

}  // namespace

class SceneRenderToDepthImageTest : public testing::Test {
protected:
    void SetUp() override {
#if defined(__APPLE__)
        if (std::getenv("CI") != nullptr) {
            GTEST_SKIP() << "Offscreen rendering is not supported on "
                            "virtualized macOS CI runners.";
        }
#elif defined(_WIN32)
        if (std::getenv("CI") != nullptr) {
            GTEST_SKIP() << "Offscreen Vulkan rendering is not set up on "
                            "Windows CI runners.";
        }
#elif defined(__aarch64__)
        if (std::getenv("CI") != nullptr) {
            GTEST_SKIP() << "Offscreen Vulkan rendering is not supported on "
                            "ARM64 Linux CI runners.";
        }
#endif
        if (!initialized_) {
            const char* res_env = std::getenv("OPEN3D_RESOURCE_PATH");
            if (res_env && res_env[0] != '\0') {
                visualization::rendering::EngineInstance::SetResourcePath(
                        res_env);
            }
            visualization::gui::Application::GetInstance().Initialize();
            initialized_ = true;
        }

        auto& engine = visualization::rendering::EngineInstance::GetInstance();
        auto& resource_mgr =
                visualization::rendering::EngineInstance::GetResourceManager();
        renderer_ = std::make_unique<visualization::rendering::FilamentRenderer>(
                engine, kW, kH, resource_mgr);
        scene_ = std::make_unique<visualization::rendering::Open3DScene>(
                *renderer_);

        visualization::rendering::MaterialRecord mat;
        mat.shader = "defaultLit";
        box_ = MakeTestBox();
        scene_->AddGeometry("box", box_.get(), mat);
        SetUpTestCamera(*scene_);
    }

    static void TearDownTestSuite() {
        if (initialized_) {
            visualization::gui::Application::GetInstance().OnTerminate();
            initialized_ = false;
        }
    }

    std::unique_ptr<visualization::rendering::FilamentRenderer> renderer_;
    std::unique_ptr<visualization::rendering::Open3DScene> scene_;
    std::shared_ptr<geometry::TriangleMesh> box_;

    static bool initialized_;
};

bool SceneRenderToDepthImageTest::initialized_ = false;

// Default behavior (z_in_view_space omitted) must match the documented
// contract: every pixel in [0.0, 1.0], with the far/no-hit corner at exactly
// 1.0 and the box (center pixel) strictly less than 1.0.
TEST_F(SceneRenderToDepthImageTest, DefaultsToNormalizedDepth) {
    auto depth_img = RenderSceneDepthImage(*renderer_, *scene_,
                                           /*z_in_view_space=*/false);
    ASSERT_TRUE(depth_img);
    ASSERT_EQ(depth_img->width_, kW);
    ASSERT_EQ(depth_img->height_, kH);

    const float* data = depth_img->PointerAt<float>(0, 0);
    const int num_pixels = kW * kH;

    float min_depth = std::numeric_limits<float>::infinity();
    float max_depth = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < num_pixels; ++i) {
        EXPECT_GE(data[i], 0.0f) << "pixel " << i;
        EXPECT_LE(data[i], 1.0f) << "pixel " << i;
        min_depth = std::min(min_depth, data[i]);
        max_depth = std::max(max_depth, data[i]);
    }

    // The far corner (no geometry) must be exactly the far-plane sentinel.
    EXPECT_FLOAT_EQ(depth_img->PointerAt<float>(0, 0)[0], 1.0f);
    // The box, centered in view, must be strictly closer than the far plane.
    const int cx = kW / 2;
    const int cy = kH / 2;
    const float center_depth = *depth_img->PointerAt<float>(cx, cy);
    EXPECT_LT(center_depth, 1.0f);
    EXPECT_GT(center_depth, 0.0f);
}

// Explicitly requesting z_in_view_space=true must still return linear,
// metric depth (not bounded to [0, 1]), with the no-hit sentinel at
// infinity -- this is the pre-existing behavior relied on elsewhere (e.g.
// Gaussian splat depth compositing) and must keep working unchanged.
TEST_F(SceneRenderToDepthImageTest, ZInViewSpaceReturnsLinearDepth) {
    auto depth_img = RenderSceneDepthImage(*renderer_, *scene_,
                                           /*z_in_view_space=*/true);
    ASSERT_TRUE(depth_img);

    const float far_corner_depth = *depth_img->PointerAt<float>(0, 0);
    EXPECT_TRUE(std::isinf(far_corner_depth));

    const int cx = kW / 2;
    const int cy = kH / 2;
    const float center_depth = *depth_img->PointerAt<float>(cx, cy);
    // The box's near face is at z=4.5; the far face at z=5.5. Linear
    // view-space depth at the center pixel should fall within that range.
    EXPECT_GT(center_depth, 4.0f);
    EXPECT_LT(center_depth, 6.0f);
}

}  // namespace tests
}  // namespace open3d
