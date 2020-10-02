// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/FilamentView.h"

// 4068: Filament has some clang-specific vectorizing pragma's that MSVC flags
// 4146: Filament's utils/algorithm.h utils::details::ctz() tries to negate
//       an unsigned int.
// 4293: Filament's utils/algorithm.h utils::details::clz() does strange
//       things with MSVC. Somehow sizeof(unsigned int) > 4, but its size is
//       32 so that x >> 32 gives a warning. (Or maybe the compiler can't
//       determine the if statement does not run.)
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068 4146 4293)
#endif  // _MSC_VER

#include <filament/Camera.h>
#include <filament/ColorGrading.h>
#include <filament/Engine.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/View.h>
#include <filament/Viewport.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/visualization/rendering/filament/FilamentCamera.h"
#include "open3d/visualization/rendering/filament/FilamentEntitiesMods.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

#define AUTO_CLEAR_COLOR 0

#if AUTO_CLEAR_COLOR
const filament::LinearColorA kDepthClearColor = {0.f, 0.f, 0.f, 0.f};
const filament::LinearColorA kNormalsClearColor = {0.5f, 0.5f, 0.5f, 1.f};
#endif

}  // namespace

FilamentView::FilamentView(filament::Engine& engine,
                           FilamentResourceManager& resource_mgr)
    : engine_(engine), resource_mgr_(resource_mgr) {
    view_ = engine_.createView();
    view_->setSampleCount(4);
    view_->setAntiAliasing(filament::View::AntiAliasing::FXAA);
    view_->setPostProcessingEnabled(true);
    view_->setAmbientOcclusion(filament::View::AmbientOcclusion::SSAO);
    view_->setVisibleLayers(kAllLayersMask, kMainLayer);
    color_grading_ =
            filament::ColorGrading::Builder()
                    .quality(filament::ColorGrading::QualityLevel::HIGH)
                    .toneMapping(filament::ColorGrading::ToneMapping::UCHIMURA)
                    .build(engine);
    view_->setColorGrading(color_grading_);

    camera_ = std::make_unique<FilamentCamera>(engine_);
    view_->setCamera(camera_->GetNativeCamera());

    camera_->SetProjection(90, 4.f / 3.f, 0.01, 1000,
                           Camera::FovType::Horizontal);
    // Default to MSAA 4x
    SetSampleCount(4);
    discard_buffers_ = View::TargetBuffers::All;
}

FilamentView::FilamentView(filament::Engine& engine,
                           FilamentScene& scene,
                           FilamentResourceManager& resource_mgr)
    : FilamentView(engine, resource_mgr) {
    scene_ = &scene;

    view_->setScene(scene_->GetNativeScene());
}

FilamentView::~FilamentView() {
    view_->setCamera(nullptr);
    view_->setScene(nullptr);

    camera_.reset();
    engine_.destroy(view_);
    engine_.destroy(color_grading_);
}

View::Mode FilamentView::GetMode() const { return mode_; }

void FilamentView::SetMode(Mode mode) {
// As color switching disabled, we don't need this code.
// Yet disabling this looks like a bad idea, so I leave code commented
#if AUTO_CLEAR_COLOR
    switch (mode) {
        case Mode::Color:
            view_->setVisibleLayers(kAllLayersMask, kMainLayer);
            view_->setClearColor(
                    {clearColor_.x(), clearColor_.y(), clearColor_.z(), 1.f});
            break;
        case Mode::Depth:
            view_->setVisibleLayers(kAllLayersMask, kMainLayer);
            view_->setClearColor(kDepthClearColor);
            break;
        case Mode::Normals:
            view_->setVisibleLayers(kAllLayersMask, kMainLayer);
            view_->setClearColor(kNormalsClearColor);
            break;
        case Mode::ColorMapX:
        case Mode::ColorMapY:
        case Mode::ColorMapZ:
            view_->setVisibleLayers(kAllLayersMask, kMainLayer);
            view_->setClearColor(kDepthClearColor);
            break;
    }
#endif

    mode_ = mode;
}

void FilamentView::SetDiscardBuffers(const TargetBuffers& buffers) {
    discard_buffers_ = buffers;
    view_->setRenderTarget(nullptr);
}

void FilamentView::SetSampleCount(int n) { view_->setSampleCount(n); }

int FilamentView::GetSampleCount() const { return view_->getSampleCount(); }

void FilamentView::SetViewport(std::int32_t x,
                               std::int32_t y,
                               std::uint32_t w,
                               std::uint32_t h) {
    view_->setViewport({x, y, w, h});
}

void FilamentView::SetSSAOEnabled(const bool enabled) {
    const auto option = enabled ? filament::View::AmbientOcclusion::SSAO
                                : filament::View::AmbientOcclusion::NONE;
    view_->setAmbientOcclusion(option);
}

Camera* FilamentView::GetCamera() const { return camera_.get(); }

void FilamentView::CopySettingsFrom(const FilamentView& other) {
    SetMode(other.mode_);
    view_->setRenderTarget(nullptr);

    auto vp = other.view_->getViewport();
    SetViewport(0, 0, vp.width, vp.height);

    // TODO: Consider moving this code to FilamentCamera
    auto& camera = view_->getCamera();
    auto& other_camera = other.GetNativeView()->getCamera();

    // TODO: Code below could introduce problems with culling,
    //        because Camera::setCustomProjection method
    //        assigns both culling projection and projection matrices
    //        to the same matrix. Which is good for ORTHO but
    //        makes culling matrix with infinite far plane for PERSPECTIVE
    //        See FCamera::setCustomProjection and FCamera::setProjection
    //        There is no easy way to fix it currently (Filament 1.4.3)
    camera.setCustomProjection(other_camera.getProjectionMatrix(),
                               other_camera.getNear(),
                               other_camera.getCullingFar());
    camera.setModelMatrix(other_camera.getModelMatrix());
}

void FilamentView::SetScene(FilamentScene& scene) {
    scene_ = &scene;
    view_->setScene(scene_->GetNativeScene());
}

void FilamentView::PreRender() {
    // auto& renderable_mgr = engine_.getRenderableManager();

    MaterialInstanceHandle material_handle;
    std::shared_ptr<filament::MaterialInstance> selected_material;
    if (mode_ == Mode::Depth) {
        material_handle = FilamentResourceManager::kDepthMaterial;
        // FIXME: Refresh parameters only then something ACTUALLY changed
        selected_material =
                resource_mgr_.GetMaterialInstance(material_handle).lock();
        if (selected_material) {
            const auto f = camera_->GetNativeCamera()->getCullingFar();
            const auto n = camera_->GetNativeCamera()->getNear();

            FilamentMaterialModifier(selected_material, material_handle)
                    .SetParameter("cameraNear", n)
                    .SetParameter("cameraFar", f)
                    .Finish();
        }
    } else if (mode_ == Mode::Normals) {
        material_handle = FilamentResourceManager::kNormalsMaterial;
        selected_material =
                resource_mgr_.GetMaterialInstance(material_handle).lock();
    } else if (mode_ >= Mode::ColorMapX) {
        material_handle = FilamentResourceManager::kColorMapMaterial;

        int coordinate_index = 0;
        switch (mode_) {
            case Mode::ColorMapX:
                coordinate_index = 0;
                break;
            case Mode::ColorMapY:
                coordinate_index = 1;
                break;
            case Mode::ColorMapZ:
                coordinate_index = 2;
                break;

            default:
                break;
        }

        selected_material =
                resource_mgr_.GetMaterialInstance(material_handle).lock();
        if (selected_material) {
            FilamentMaterialModifier(selected_material, material_handle)
                    .SetParameter("coordinateIndex", coordinate_index)
                    .Finish();
        }
    }

    // TODO: is any of this necessary?
    // if (scene_) {
    //     for (const auto& pair : scene_->entities_) {
    //         const auto& entity = pair.second;
    //         if (entity.info.type == EntityType::Geometry) {
    //             std::shared_ptr<filament::MaterialInstance> mat_inst;
    //             if (selected_material) {
    //                 mat_inst = selected_material;

    //                 if (mode_ >= Mode::ColorMapX) {
    //                     auto bbox = scene_->GetEntityBoundingBox(pair.first);
    //                     Eigen::Vector3f bbox_min =
    //                             bbox.GetMinBound().cast<float>();
    //                     Eigen::Vector3f bbox_max =
    //                             bbox.GetMaxBound().cast<float>();

    //                     FilamentMaterialModifier(selected_material,
    //                                              material_handle)
    //                             .SetParameter("bboxMin", bbox_min)
    //                             .SetParameter("bboxMax", bbox_max)
    //                             .Finish();
    //                 }
    //             } else {
    //                 mat_inst =
    //                         resource_mgr_.GetMaterialInstance(entity.material)
    //                                 .lock();
    //             }

    //             filament::RenderableManager::Instance inst =
    //                     renderable_mgr.getInstance(entity.info.self);
    //             renderable_mgr.setMaterialInstanceAt(inst, 0,
    //             mat_inst.get());
    //         }
    //     }
    // }
}

void FilamentView::PostRender() {}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
