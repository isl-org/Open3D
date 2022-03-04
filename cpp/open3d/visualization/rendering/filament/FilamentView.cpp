// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
#include "open3d/visualization/rendering/ColorGrading.h"
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
    SetAntiAliasing(true, false);
    SetPostProcessing(true);
    SetAmbientOcclusion(true, false);
    view_->setVisibleLayers(kAllLayersMask, kMainLayer);
    SetShadowing(true, ShadowType::kPCF);
    ColorGradingParams cp(ColorGradingParams::Quality::kHigh,
                          ColorGradingParams::ToneMapping::kUchimura);
    SetColorGrading(cp);

    camera_ = std::make_unique<FilamentCamera>(engine_);
    view_->setCamera(camera_->GetNativeCamera());

    camera_->SetProjection(90, 4.f / 3.f, 0.01, 1000,
                           Camera::FovType::Horizontal);

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

std::array<int, 4> FilamentView::GetViewport() const {
    auto vp = view_->getViewport();
    return {vp.left, vp.bottom, int(vp.width), int(vp.height)};
}

void FilamentView::SetPostProcessing(bool enabled) {
    view_->setPostProcessingEnabled(enabled);
}

void FilamentView::SetAmbientOcclusion(bool enabled,
                                       bool ssct_enabled /* = false */) {
    filament::View::AmbientOcclusionOptions options;
    options.enabled = enabled;
    options.ssct.enabled = ssct_enabled;
    view_->setAmbientOcclusionOptions(options);
}

void FilamentView::SetAntiAliasing(bool enabled, bool temporal /* = false */) {
    if (enabled) {
        filament::View::TemporalAntiAliasingOptions options;
        options.enabled = temporal;
        view_->setAntiAliasing(filament::View::AntiAliasing::FXAA);
        view_->setTemporalAntiAliasingOptions(options);
    } else {
        view_->setAntiAliasing(filament::View::AntiAliasing::NONE);
    }
}

void FilamentView::SetShadowing(bool enabled, ShadowType type) {
    if (enabled) {
        filament::View::ShadowType stype =
                (type == ShadowType::kPCF) ? filament::View::ShadowType::PCF
                                           : filament::View::ShadowType::VSM;
        view_->setShadowType(stype);
        view_->setShadowingEnabled(true);
    } else {
        view_->setShadowingEnabled(false);
    }
}

static inline filament::math::float3 eigen_to_float3(const Eigen::Vector3f& v) {
    return filament::math::float3(v.x(), v.y(), v.z());
}

static inline filament::math::float4 eigen_to_float4(const Eigen::Vector4f& v) {
    return filament::math::float4(v.x(), v.y(), v.z(), v.w());
}

void FilamentView::SetColorGrading(const ColorGradingParams& color_grading) {
    filament::ColorGrading::QualityLevel q =
            filament::ColorGrading::QualityLevel::LOW;
    switch (color_grading.GetQuality()) {
        case ColorGradingParams::Quality::kMedium:
            q = filament::ColorGrading::QualityLevel::MEDIUM;
            break;
        case ColorGradingParams::Quality::kHigh:
            q = filament::ColorGrading::QualityLevel::HIGH;
            break;
        case ColorGradingParams::Quality::kUltra:
            q = filament::ColorGrading::QualityLevel::ULTRA;
            break;
        default:
            break;
    }

    filament::ColorGrading::ToneMapping tm =
            filament::ColorGrading::ToneMapping::LINEAR;
    switch (color_grading.GetToneMapping()) {
        case ColorGradingParams::ToneMapping::kAcesLegacy:
            tm = filament::ColorGrading::ToneMapping::ACES_LEGACY;
            break;
        case ColorGradingParams::ToneMapping::kAces:
            tm = filament::ColorGrading::ToneMapping::ACES;
            break;
        case ColorGradingParams::ToneMapping::kFilmic:
            tm = filament::ColorGrading::ToneMapping::FILMIC;
            break;
        case ColorGradingParams::ToneMapping::kUchimura:
            tm = filament::ColorGrading::ToneMapping::UCHIMURA;
            break;
        case ColorGradingParams::ToneMapping::kReinhard:
            tm = filament::ColorGrading::ToneMapping::REINHARD;
            break;
        case ColorGradingParams::ToneMapping::kDisplayRange:
            tm = filament::ColorGrading::ToneMapping::DISPLAY_RANGE;
            break;
        default:
            break;
    }

    if (color_grading_) {
        engine_.destroy(color_grading_);
    }
    color_grading_ =
            filament::ColorGrading::Builder()
                    .quality(q)
                    .toneMapping(tm)
                    .whiteBalance(color_grading.GetTemperature(),
                                  color_grading.GetTint())
                    .channelMixer(
                            eigen_to_float3(color_grading.GetMixerRed()),
                            eigen_to_float3(color_grading.GetMixerGreen()),
                            eigen_to_float3(color_grading.GetMixerBlue()))
                    .shadowsMidtonesHighlights(
                            eigen_to_float4(color_grading.GetShadows()),
                            eigen_to_float4(color_grading.GetMidtones()),
                            eigen_to_float4(color_grading.GetHighlights()),
                            eigen_to_float4(color_grading.GetRanges()))
                    .slopeOffsetPower(
                            eigen_to_float3(color_grading.GetSlope()),
                            eigen_to_float3(color_grading.GetOffset()),
                            eigen_to_float3(color_grading.GetPower()))
                    .contrast(color_grading.GetContrast())
                    .vibrance(color_grading.GetVibrance())
                    .saturation(color_grading.GetSaturation())
                    .curves(eigen_to_float3(color_grading.GetShadowGamma()),
                            eigen_to_float3(color_grading.GetMidpoint()),
                            eigen_to_float3(color_grading.GetHighlightScale()))
                    .build(engine_);
    view_->setColorGrading(color_grading_);
}

void FilamentView::ConfigureForColorPicking() {
    view_->setSampleCount(1);
    SetPostProcessing(false);
    SetAmbientOcclusion(false, false);
    SetShadowing(false, ShadowType::kPCF);
    configured_for_picking_ = true;
}

void FilamentView::EnableViewCaching(bool enable) {
    caching_enabled_ = enable;

    if (caching_enabled_) {
        if (render_target_) {
            resource_mgr_.Destroy(render_target_);
            resource_mgr_.Destroy(color_buffer_);
            resource_mgr_.Destroy(depth_buffer_);
            render_target_ = RenderTargetHandle();
            color_buffer_ = TextureHandle();
            depth_buffer_ = TextureHandle();
        }

        // Create RenderTarget
        auto vp = view_->getViewport();
        color_buffer_ =
                resource_mgr_.CreateColorAttachmentTexture(vp.width, vp.height);
        depth_buffer_ =
                resource_mgr_.CreateDepthAttachmentTexture(vp.width, vp.height);
        render_target_ =
                resource_mgr_.CreateRenderTarget(color_buffer_, depth_buffer_);
        SetRenderTarget(render_target_);
    }

    if (!caching_enabled_) {
        view_->setRenderTarget(nullptr);
    }
}

bool FilamentView::IsCached() const { return caching_enabled_; }

TextureHandle FilamentView::GetColorBuffer() { return color_buffer_; }

void FilamentView::SetRenderTarget(const RenderTargetHandle render_target) {
    if (!render_target) {
        view_->setRenderTarget(nullptr);
    } else {
        auto rt_weak = resource_mgr_.GetRenderTarget(render_target);
        auto rt = rt_weak.lock();
        if (!rt) {
            utility::LogWarning(
                    "Invalid render target given to SetRenderTarget");
            view_->setRenderTarget(nullptr);
        } else {
            view_->setRenderTarget(rt.get());
        }
    }
}

Camera* FilamentView::GetCamera() const { return camera_.get(); }

void FilamentView::CopySettingsFrom(const FilamentView& other) {
    SetMode(other.mode_);
    view_->setRenderTarget(nullptr);
    auto vp = other.view_->getViewport();
    SetViewport(0, 0, vp.width, vp.height);
    camera_->CopyFrom(other.camera_.get());
    if (other.configured_for_picking_) {
        ConfigureForColorPicking();
    }
    if (other.color_grading_) {
        view_->setColorGrading(other.color_grading_);
    }
    auto ao_options = other.view_->getAmbientOcclusionOptions();
    view_->setAmbientOcclusionOptions(ao_options);
    auto aa_mode = other.view_->getAntiAliasing();
    auto temporal_options = other.view_->getTemporalAntiAliasingOptions();
    view_->setAntiAliasing(aa_mode);
    view_->setTemporalAntiAliasingOptions(temporal_options);
    view_->setShadowingEnabled(other.view_->isShadowingEnabled());
    view_->setPostProcessingEnabled(other.view_->isPostProcessingEnabled());
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
