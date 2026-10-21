// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/Camera.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>

#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/rendering/filament/FilamentCamera.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

namespace {

class CameraTest : public testing::Test {
protected:
    void SetUp() override {
#if defined(__APPLE__)
        if (std::getenv("CI") != nullptr) {
            GTEST_SKIP() << "Filament Metal rendering is not supported on "
                            "virtualized macOS CI runners.";
        }
#elif defined(_WIN32)
        if (std::getenv("CI") != nullptr) {
            GTEST_SKIP() << "Filament Vulkan rendering is not supported on "
                            "Windows CI runners.";
        }
#elif defined(__aarch64__)
        if (std::getenv("CI") != nullptr) {
            GTEST_SKIP() << "Filament Vulkan rendering is not supported on "
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
    }

    static void TearDownTestSuite() {
        if (initialized_) {
            visualization::gui::Application::GetInstance().OnTerminate();
            initialized_ = false;
        }
    }

    static bool initialized_;
};

bool CameraTest::initialized_ = false;

}  // namespace

TEST_F(CameraTest, SetProjectionIntrinsicsOpenCVOffset) {
    auto& engine = visualization::rendering::EngineInstance::GetInstance();
    visualization::rendering::FilamentCamera camera(engine);

    // Standard optical-center intrinsics for a 640x480 frame
    const double width = 640.0;
    const double height = 480.0;
    const double cx = (width - 1.0) / 2.0;   // 319.5
    const double cy = (height - 1.0) / 2.0;  // 239.5
    const double fx = 525.0;
    const double fy = 525.0;
    const double near = 0.1;
    const double far = 100.0;

    Eigen::Matrix3d intrinsics;
    intrinsics << fx, 0.0, cx,  //
            0.0, fy, cy,        //
            0.0, 0.0, 1.0;

    camera.SetProjection(intrinsics, near, far, width, height);
    auto proj = camera.GetProjectionMatrix();

    // In OpenCV convention where integer pixel coordinates (u, v) represent
    // pixel centers, a principal point at ((width - 1)/2, (height - 1)/2)
    // maps exactly to the optical center (NDC (0, 0)), so the Z-shear terms
    // in the projection matrix must be zero.
    EXPECT_NEAR(proj(0, 2), 0.0f, 1e-6f);
    EXPECT_NEAR(proj(1, 2), 0.0f, 1e-6f);

    // Point along optical axis should project exactly to (0, 0) in NDC
    Eigen::Vector3f pt_axis(0.0f, 0.0f, -5.0f);
    auto ndc = camera.GetNDC(pt_axis);
    EXPECT_NEAR(ndc.x(), 0.0f, 1e-6f);
    EXPECT_NEAR(ndc.y(), 0.0f, 1e-6f);

    // Unprojecting the center of the continuous window [0, W] x [0, H]
    // should yield a point on the optical axis
    auto unproj = camera.Unproject(
            static_cast<float>(width / 2.0), static_cast<float>(height / 2.0),
            0.5f, static_cast<float>(width), static_cast<float>(height));
    EXPECT_NEAR(unproj.x(), 0.0f, 1e-5f);
    EXPECT_NEAR(unproj.y(), 0.0f, 1e-5f);
}

TEST_F(CameraTest, SetProjectionIntrinsicsArbitraryPrincipalPoint) {
    auto& engine = visualization::rendering::EngineInstance::GetInstance();
    visualization::rendering::FilamentCamera camera(engine);

    const double width = 640.0;
    const double height = 480.0;
    const double cx = 310.0;
    const double cy = 250.0;
    const double fx = 500.0;
    const double fy = 500.0;
    const double near = 0.1;
    const double far = 100.0;

    Eigen::Matrix3d intrinsics;
    intrinsics << fx, 0.0, cx,  //
            0.0, fy, cy,        //
            0.0, 0.0, 1.0;

    camera.SetProjection(intrinsics, near, far, width, height);
    auto proj = camera.GetProjectionMatrix();

    // In OpenCV convention, x_win = u + 0.5, so X_ndc = 2 * (u + 0.5)/W - 1
    // proj(0, 2) = 1 - 2 * (cx + 0.5) / width
    // proj(1, 2) = -1 + 2 * (cy + 0.5) / height
    const float expected_m02 =
            static_cast<float>(1.0 - 2.0 * (cx + 0.5) / width);
    const float expected_m12 =
            static_cast<float>(-1.0 + 2.0 * (cy + 0.5) / height);
    EXPECT_NEAR(proj(0, 2), expected_m02, 1e-6f);
    EXPECT_NEAR(proj(1, 2), expected_m12, 1e-6f);

    // A 3D point on the optical axis (0, 0, -5) has NDC:
    // X_ndc = -proj(0, 2) = -expected_m02
    // Y_ndc = -proj(1, 2) = -expected_m12
    Eigen::Vector3f pt_axis(0.0f, 0.0f, -5.0f);
    auto ndc = camera.GetNDC(pt_axis);
    EXPECT_NEAR(ndc.x(), -expected_m02, 1e-6f);
    EXPECT_NEAR(ndc.y(), -expected_m12, 1e-6f);
}

}  // namespace tests
}  // namespace open3d
