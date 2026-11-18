// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/LineSet.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/MaterialRecord.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/rendering/View.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"

using namespace open3d;

namespace {

constexpr int kWidth = 64;
constexpr int kHeight = 64;

t::geometry::LineSet MakeSquareLines(bool with_colors = true) {
    auto points = core::Tensor::Init<float>({{-1.f, -1.f, 0.f},
                                             {1.f, -1.f, 0.f},
                                             {1.f, 1.f, 0.f},
                                             {-1.f, 1.f, 0.f}});
    auto lines = core::Tensor::Init<int64_t>({{0, 1}, {1, 2}, {2, 3}, {3, 0}});
    t::geometry::LineSet line_set(points, lines);
    if (with_colors) {
        line_set.SetLineColors(core::Tensor::Init<float>({{1.f, 0.f, 0.f},
                                                          {0.f, 1.f, 0.f},
                                                          {0.f, 0.f, 1.f},
                                                          {1.f, 1.f, 0.f}}));
    }
    return line_set;
}

class TensorGeometryUpdateTest : public testing::Test {
protected:
    void SetUp() override {
#if defined(__APPLE__) || defined(_WIN32) || defined(__aarch64__)
        if (std::getenv("CI") != nullptr) {
            GTEST_SKIP() << "Filament rendering is not configured for this CI "
                            "runner.";
        }
#endif
        if (!initialized_) {
            const char* resource_path = std::getenv("OPEN3D_RESOURCE_PATH");
            if (resource_path && resource_path[0] != '\0') {
                visualization::rendering::EngineInstance::SetResourcePath(
                        resource_path);
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

bool TensorGeometryUpdateTest::initialized_ = false;

TEST_F(TensorGeometryUpdateTest, ThinLineSetUpdatesExpandedBuffersAndBounds) {
    auto& engine = visualization::rendering::EngineInstance::GetInstance();
    auto& resources =
            visualization::rendering::EngineInstance::GetResourceManager();
    auto renderer =
            std::make_unique<visualization::rendering::FilamentRenderer>(
                    engine, kWidth, kHeight, resources);
    auto scene =
            std::make_unique<visualization::rendering::Open3DScene>(*renderer);

    auto line_set = MakeSquareLines();
    visualization::rendering::MaterialRecord material;
    material.shader = "defaultUnlit";
    scene->AddGeometry("thin-lines", &line_set, material, false);

    line_set.SetPointPositions(line_set.GetPointPositions() +
                               core::Tensor::Init<float>({{0.5f, 0.f, 0.f},
                                                          {0.5f, 0.f, 0.f},
                                                          {0.5f, 0.f, 0.f},
                                                          {0.5f, 0.f, 0.f}}));
    line_set.SetLineColors(core::Tensor::Ones({4, 3}, core::Float32));
    const uint32_t flags = visualization::rendering::Scene::kUpdatePointsFlag |
                           visualization::rendering::Scene::kUpdateColorsFlag;
    for (int i = 0; i < 8; ++i) {
        scene->UpdateGeometry("thin-lines", line_set, flags);
    }

    const auto& bounds = scene->GetBoundingBox();
    EXPECT_NEAR(bounds.min_bound_.x(), -0.5, 1e-6);
    EXPECT_NEAR(bounds.max_bound_.x(), 1.5, 1e-6);

    auto changed_topology = line_set;
    changed_topology.SetLineIndices(
            core::Tensor::Init<int64_t>({{0, 2}, {2, 1}, {1, 3}, {3, 0}}));
    changed_topology.SetPointPositions(
            line_set.GetPointPositions() +
            core::Tensor::Ones({4, 3}, core::Float32));
    scene->UpdateGeometry("thin-lines", changed_topology,
                          visualization::rendering::Scene::kUpdatePointsFlag);
    EXPECT_NEAR(scene->GetBoundingBox().min_bound_.x(), -0.5, 1e-6);
    EXPECT_NEAR(scene->GetBoundingBox().max_bound_.x(), 1.5, 1e-6);

    auto changed_layout = line_set;
    changed_layout.RemoveLineAttr("colors");
    changed_layout.SetPointPositions(line_set.GetPointPositions() +
                                     core::Tensor::Ones({4, 3}, core::Float32));
    scene->UpdateGeometry("thin-lines", changed_layout,
                          visualization::rendering::Scene::kUpdatePointsFlag);
    EXPECT_NEAR(scene->GetBoundingBox().min_bound_.x(), -0.5, 1e-6);
    EXPECT_NEAR(scene->GetBoundingBox().max_bound_.x(), 1.5, 1e-6);
}

TEST_F(TensorGeometryUpdateTest, ThinUncoloredLineSetUpdatesIndexedVertices) {
    auto& engine = visualization::rendering::EngineInstance::GetInstance();
    auto& resources =
            visualization::rendering::EngineInstance::GetResourceManager();
    auto renderer =
            std::make_unique<visualization::rendering::FilamentRenderer>(
                    engine, kWidth, kHeight, resources);
    auto scene =
            std::make_unique<visualization::rendering::Open3DScene>(*renderer);

    auto line_set = MakeSquareLines(false);
    visualization::rendering::MaterialRecord material;
    material.shader = "defaultUnlit";
    scene->AddGeometry("indexed-lines", &line_set, material, false);

    line_set.SetPointPositions(line_set.GetPointPositions() * 0.25f);
    scene->UpdateGeometry("indexed-lines", line_set,
                          visualization::rendering::Scene::kUpdatePointsFlag);

    const auto bounds =
            scene->GetScene()->GetGeometryBoundingBox("indexed-lines");
    EXPECT_NEAR(bounds.min_bound_.x(), -0.25, 1e-6);
    EXPECT_NEAR(bounds.max_bound_.x(), 0.25, 1e-6);
}

TEST_F(TensorGeometryUpdateTest, WideLineSetUpdatesPairedEndpoints) {
    auto& engine = visualization::rendering::EngineInstance::GetInstance();
    auto& resources =
            visualization::rendering::EngineInstance::GetResourceManager();
    auto renderer =
            std::make_unique<visualization::rendering::FilamentRenderer>(
                    engine, kWidth, kHeight, resources);
    auto scene =
            std::make_unique<visualization::rendering::Open3DScene>(*renderer);

    auto line_set = MakeSquareLines();
    visualization::rendering::MaterialRecord material;
    material.shader = "unlitLine";
    material.line_width = 4.f;
    scene->AddGeometry("wide-lines", &line_set, material, false);

    auto* camera = scene->GetCamera();
    camera->SetProjection(60.f, static_cast<float>(kWidth) / kHeight, 0.1f,
                          50.f,
                          visualization::rendering::Camera::FovType::Vertical);
    camera->LookAt({0.f, 0.f, 0.f}, {0.f, 0.f, 5.f}, {0.f, 1.f, 0.f});
    scene->GetView()->SetPostProcessing(false);

    line_set.SetPointPositions(core::Tensor::Init<float>({{-0.25f, -0.4f, 0.f},
                                                          {0.5f, -0.5f, 0.f},
                                                          {0.5f, 0.5f, 0.f},
                                                          {-0.5f, 0.5f, 0.f}}));
    line_set.SetLineColors(core::Tensor::Ones({4, 3}, core::Float32));
    scene->UpdateGeometry(
            "wide-lines", line_set,
            visualization::rendering::Scene::kUpdatePointsFlag |
                    visualization::rendering::Scene::kUpdateColorsFlag);

    const auto bounds = scene->GetScene()->GetGeometryBoundingBox("wide-lines");
    EXPECT_NEAR(bounds.min_bound_.x(), -0.5, 1e-6);
    EXPECT_NEAR(bounds.max_bound_.x(), 0.5, 1e-6);

    auto& app = visualization::gui::Application::GetInstance();
    auto updated = app.RenderToImage(*renderer, scene->GetView(),
                                     scene->GetScene(), kWidth, kHeight);
    ASSERT_TRUE(updated);
    scene->RemoveGeometry("wide-lines");
    scene->AddGeometry("wide-lines", &line_set, material, false);
    auto rebuilt = app.RenderToImage(*renderer, scene->GetView(),
                                     scene->GetScene(), kWidth, kHeight);
    ASSERT_TRUE(rebuilt);
    EXPECT_EQ(updated->data_, rebuilt->data_);
}

}  // namespace
