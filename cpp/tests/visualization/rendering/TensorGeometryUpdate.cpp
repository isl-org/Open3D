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
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/rendering/Camera.h"
#include "open3d/visualization/rendering/MaterialRecord.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"

using namespace open3d;

namespace {

constexpr int kWidth = 64;
constexpr int kHeight = 64;

t::geometry::TriangleMesh MakeQuad() {
    auto vertices = core::Tensor::Init<float>({{-1.f, -1.f, 0.f},
                                               {1.f, -1.f, 0.f},
                                               {1.f, 1.f, 0.f},
                                               {-1.f, 1.f, 0.f}});
    auto triangles = core::Tensor::Init<int64_t>({{0, 1, 2}, {0, 2, 3}});
    return t::geometry::TriangleMesh(vertices, triangles);
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

TEST_F(TensorGeometryUpdateTest, TriangleMeshUpdatesBuffersAndBoundsInPlace) {
    auto& engine = visualization::rendering::EngineInstance::GetInstance();
    auto& resources =
            visualization::rendering::EngineInstance::GetResourceManager();
    auto renderer =
            std::make_unique<visualization::rendering::FilamentRenderer>(
                    engine, kWidth, kHeight, resources);
    auto scene =
            std::make_unique<visualization::rendering::Open3DScene>(*renderer);

    auto mesh = MakeQuad();
    mesh.SetVertexNormals(core::Tensor::Init<float>({{0.f, 0.f, 1.f},
                                                     {0.f, 0.f, 1.f},
                                                     {0.f, 0.f, 1.f},
                                                     {0.f, 0.f, 1.f}}));
    mesh.SetVertexColors(core::Tensor::Init<float>({{1.f, 0.f, 0.f},
                                                    {1.f, 0.f, 0.f},
                                                    {1.f, 0.f, 0.f},
                                                    {1.f, 0.f, 0.f}}));
    mesh.SetVertexAttr(
            "texture_uvs",
            core::Tensor::Init<float>(
                    {{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}}));
    visualization::rendering::MaterialRecord material;
    material.shader = "defaultUnlit";
    material.base_color = {1.f, 1.f, 1.f, 1.f};
    scene->AddGeometry("mesh", &mesh, material, false);

    auto* camera = scene->GetCamera();
    camera->SetProjection(60.f, static_cast<float>(kWidth) / kHeight, 0.1f,
                          50.f,
                          visualization::rendering::Camera::FovType::Vertical);
    camera->LookAt({0.f, 0.f, 0.f}, {0.f, 0.f, 5.f}, {0.f, 1.f, 0.f});
    auto& app = visualization::gui::Application::GetInstance();
    auto before = app.RenderToImage(*renderer, scene->GetView(),
                                    scene->GetScene(), kWidth, kHeight);
    ASSERT_TRUE(before);

    auto positions = mesh.GetVertexPositions() +
                     core::Tensor::Init<float>({{0.5f, 0.f, 0.f},
                                                {0.5f, 0.f, 0.f},
                                                {0.5f, 0.f, 0.f},
                                                {0.5f, 0.f, 0.f}});
    mesh.SetVertexPositions(positions);
    mesh.SetVertexNormals(core::Tensor::Init<float>({{1.f, 0.f, 0.f},
                                                     {1.f, 0.f, 0.f},
                                                     {1.f, 0.f, 0.f},
                                                     {1.f, 0.f, 0.f}}));
    mesh.SetVertexColors(core::Tensor::Init<float>({{0.f, 1.f, 0.f},
                                                    {0.f, 1.f, 0.f},
                                                    {0.f, 1.f, 0.f},
                                                    {0.f, 1.f, 0.f}}));
    mesh.SetVertexAttr(
            "texture_uvs",
            core::Tensor::Init<float>(
                    {{1.f, 1.f}, {0.f, 1.f}, {0.f, 0.f}, {1.f, 0.f}}));
    const uint32_t flags = visualization::rendering::Scene::kUpdatePointsFlag |
                           visualization::rendering::Scene::kUpdateNormalsFlag |
                           visualization::rendering::Scene::kUpdateColorsFlag |
                           visualization::rendering::Scene::kUpdateUv0Flag;
    for (int i = 0; i < 8; ++i) {
        scene->UpdateGeometry("mesh", mesh, flags);
    }

    const auto& bounds = scene->GetBoundingBox();
    EXPECT_NEAR(bounds.min_bound_.x(), -0.5, 1e-6);
    EXPECT_NEAR(bounds.max_bound_.x(), 1.5, 1e-6);
    auto after = app.RenderToImage(*renderer, scene->GetView(),
                                   scene->GetScene(), kWidth, kHeight);
    ASSERT_TRUE(after);
    EXPECT_NE(before->data_, after->data_);

    auto changed_topology = mesh;
    changed_topology.SetTriangleIndices(
            core::Tensor::Init<int64_t>({{0, 1, 3}, {1, 2, 3}}));
    changed_topology.SetVertexPositions(
            mesh.GetVertexPositions() +
            core::Tensor::Ones({4, 3}, core::Float32, core::Device("CPU:0")));
    scene->UpdateGeometry("mesh", changed_topology, flags);
    const auto& unchanged_bounds = scene->GetBoundingBox();
    EXPECT_NEAR(unchanged_bounds.min_bound_.x(), -0.5, 1e-6);
    EXPECT_NEAR(unchanged_bounds.max_bound_.x(), 1.5, 1e-6);

    auto changed_layout = mesh;
    changed_layout.SetTriangleColors(
            core::Tensor::Init<float>({{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}}));
    changed_layout.SetVertexPositions(
            mesh.GetVertexPositions() +
            core::Tensor::Ones({4, 3}, core::Float32, core::Device("CPU:0")));
    scene->UpdateGeometry("mesh", changed_layout,
                          visualization::rendering::Scene::kUpdatePointsFlag);
    EXPECT_NEAR(scene->GetBoundingBox().min_bound_.x(), -0.5, 1e-6);
    EXPECT_NEAR(scene->GetBoundingBox().max_bound_.x(), 1.5, 1e-6);
}

TEST_F(TensorGeometryUpdateTest, TriangleAttributeLayoutUpdatesExpandedMesh) {
    auto& engine = visualization::rendering::EngineInstance::GetInstance();
    auto& resources =
            visualization::rendering::EngineInstance::GetResourceManager();
    auto renderer =
            std::make_unique<visualization::rendering::FilamentRenderer>(
                    engine, kWidth, kHeight, resources);
    auto scene =
            std::make_unique<visualization::rendering::Open3DScene>(*renderer);

    auto mesh = MakeQuad();
    mesh.SetTriangleColors(
            core::Tensor::Init<float>({{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}}));
    mesh.SetTriangleNormals(
            core::Tensor::Init<float>({{0.f, 0.f, 1.f}, {0.f, 0.f, 1.f}}));
    mesh.SetTriangleAttr(
            "texture_uvs",
            core::Tensor::Init<float>({{{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}},
                                       {{0.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}}}));
    visualization::rendering::MaterialRecord material;
    material.shader = "defaultUnlit";
    scene->AddGeometry("expanded", &mesh, material, false);

    mesh.SetVertexPositions(mesh.GetVertexPositions() * 0.5f);
    mesh.SetTriangleColors(
            core::Tensor::Init<float>({{0.f, 0.f, 1.f}, {1.f, 1.f, 0.f}}));
    mesh.SetTriangleNormals(
            core::Tensor::Init<float>({{1.f, 0.f, 0.f}, {1.f, 0.f, 0.f}}));
    mesh.SetTriangleAttr(
            "texture_uvs",
            core::Tensor::Init<float>({{{1.f, 1.f}, {0.f, 1.f}, {0.f, 0.f}},
                                       {{1.f, 1.f}, {0.f, 0.f}, {1.f, 0.f}}}));
    scene->UpdateGeometry(
            "expanded", mesh,
            visualization::rendering::Scene::kUpdatePointsFlag |
                    visualization::rendering::Scene::kUpdateNormalsFlag |
                    visualization::rendering::Scene::kUpdateColorsFlag |
                    visualization::rendering::Scene::kUpdateUv0Flag);
    const auto bounds = scene->GetScene()->GetGeometryBoundingBox("expanded");
    EXPECT_NEAR(bounds.min_bound_.x(), -0.5, 1e-6);
    EXPECT_NEAR(bounds.max_bound_.x(), 0.5, 1e-6);
}

}  // namespace
