// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
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

// ----------------------------------------------------------------------------
// ctor / dtor
// ----------------------------------------------------------------------------

FilamentView::FilamentView(filament::Engine& engine,
                           FilamentResourceManager& resource_mgr)
    : engine_(engine), resource_mgr_(resource_mgr) {
    view_ = engine_.createView();

    // Configure MSAA (4×) ---------------------------------------------------
    filament::View::MultiSampleAntiAliasingOptions msaa;
    msaa.enabled     = true;
    msaa.sampleCount = 4;
    msaa.customResolve = false;
    view_->setMultiSampleAntiAliasingOptions(msaa);

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

// ----------------------------------------------------------------------------
// view state ----------------------------------------------------------------
// ----------------------------------------------------------------------------

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

void FilamentView::SetWireframe(bool enable) {
    if (enable) {
        SetBloom(true, 0.5f, 8);
    } else {
        SetBloom(false);
    }
}

// ----------------------------------------------------------------------------
// MSAA helpers ---------------------------------------------------------------
// ----------------------------------------------------------------------------

void FilamentView::SetSampleCount(int n) {
    auto opts        = view_->getMultiSampleAntiAliasingOptions();
    opts.sampleCount = static_cast<uint8_t>(n);
    opts.enabled     = (n > 1);
    view_->setMultiSampleAntiAliasingOptions(opts);
}

int FilamentView::GetSampleCount() const {
    return view_->getMultiSampleAntiAliasingOptions().sampleCount;
}

// ----------------------------------------------------------------------------
// viewport / post fx ---------------------------------------------------------
// ----------------------------------------------------------------------------

void FilamentView::SetViewport(std::int32_t x,
                               std::int32_t y,
                               std::uint32_t w,
                               std::uint32_t h) {
    view_->setViewport({x, y, w, h});
}

std::array<int, 4> FilamentView::GetViewport() const {
    auto vp = view_->getViewport();
    return {vp.left, vp.bottom, static_cast<int>(vp.width),
            static_cast<int>(vp.height)};
}

void FilamentView::SetPostProcessing(bool enabled) {
    view_->setPostProcessingEnabled(enabled);
}

void FilamentView::SetAmbientOcclusion(bool enabled, bool ssct_enabled) {
    filament::View::AmbientOcclusionOptions options;
    options.enabled       = enabled;
    options.ssct.enabled  = ssct_enabled;
    view_->setAmbientOcclusionOptions(options);
}

void FilamentView::SetBloom(bool enabled, float strength, int spread) {
    filament::View::BloomOptions bloom;
    bloom.enabled   = enabled;
    bloom.strength  = strength;
    bloom.threshold = false;
    bloom.levels    = spread;
    view_->setBloomOptions(bloom);
}

void FilamentView::SetAntiAliasing(bool enabled, bool temporal) {
    // FXAA / TAA
    view_->setAntiAliasing(enabled ? filament::View::AntiAliasing::FXAA
                                   : filament::View::AntiAliasing::NONE);

    filament::View::TemporalAntiAliasingOptions taa =
            view_->getTemporalAntiAliasingOptions();
    taa.enabled = temporal;
    view_->setTemporalAntiAliasingOptions(taa);

    // MSAA flag (sample count se controla aparte)
    auto msaa = view_->getMultiSampleAntiAliasingOptions();
    msaa.enabled = enabled && msaa.sampleCount > 1;
    view_->setMultiSampleAntiAliasingOptions(msaa);
}

void FilamentView::SetShadowing(bool enabled, ShadowType type) {
    if (enabled) {
        auto stype = (type == ShadowType::kPCF) ? filament::View::ShadowType::PCF
                                                : filament::View::ShadowType::VSM;
        view_->setShadowType(stype);
        view_->setShadowingEnabled(true);
    } else {
        view_->setShadowingEnabled(false);
    }
}

// ----------------------------------------------------------------------------
// color grading --------------------------------------------------------------
// ----------------------------------------------------------------------------

static inline filament::math::float3 eigen_to_float3(const Eigen::Vector3f& v) {
    return {v.x(), v.y(), v.z()};
}
static inline filament::math::float4 eigen_to_float4(const Eigen::Vector4f& v) {
    return {v.x(), v.y(), v.z(), v.w()};
}

void FilamentView::SetColorGrading(const ColorGradingParams& params) {
    // quality ----------------------------------------------------------------
    using Q = ColorGradingParams::Quality;
    filament::ColorGrading::QualityLevel qlevel = filament::ColorGrading::QualityLevel::LOW;
    switch (params.GetQuality()) {
        case Q::kMedium: qlevel = filament::ColorGrading::QualityLevel::MEDIUM; break;
        case Q::kHigh:   qlevel = filament::ColorGrading::QualityLevel::HIGH;   break;
        case Q::kUltra:  qlevel = filament::ColorGrading::QualityLevel::ULTRA;  break;
        default: break;
    }

    if (color_grading_) {
        engine_.destroy(color_grading_);
    }

    color_grading_ = filament::ColorGrading::Builder()
            .quality(qlevel)
            .whiteBalance(params.GetTemperature(), params.GetTint())
            .channelMixer(eigen_to_float3(params.GetMixerRed()),
                          eigen_to_float3(params.GetMixerGreen()),
                          eigen_to_float3(params.GetMixerBlue()))
            .shadowsMidtonesHighlights(eigen_to_float4(params.GetShadows()),
                                       eigen_to_float4(params.GetMidtones()),
                                       eigen_to_float4(params.GetHighlights()),
                                       eigen_to_float4(params.GetRanges()))
            .slopeOffsetPower(eigen_to_float3(params.GetSlope()),
                              eigen_to_float3(params.GetOffset()),
                              eigen_to_float3(params.GetPower()))
            .contrast(params.GetContrast())
            .vibrance(params.GetVibrance())
            .saturation(params.GetSaturation())
            .curves(eigen_to_float3(params.GetShadowGamma()),
                    eigen_to_float3(params.GetMidpoint()),
                    eigen_to_float3(params.GetHighlightScale()))
            .build(engine_);

    view_->setColorGrading(color_grading_);
}

// ----------------------------------------------------------------------------
// picking / caching ----------------------------------------------------------
// ----------------------------------------------------------------------------

void FilamentView::ConfigureForColorPicking() {
    SetSampleCount(1);
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

// ----------------------------------------------------------------------------
// copy settings --------------------------------------------------------------
// ----------------------------------------------------------------------------

void FilamentView::CopySettingsFrom(const FilamentView& other) {
    SetMode(other.mode_);

    // viewport ---------------------------------------------------------------
    auto vp = other.view_->getViewport();
    SetViewport(0, 0, vp.width, vp.height);

    // camera -----------------------------------------------------------------
    camera_->CopyFrom(other.camera_.get());

    // color grading ----------------------------------------------------------
    if (other.color_grading_) {
        view_->setColorGrading(other.color_grading_);
    }

    // MSAA -------------------------------------------------------------------
    auto msaa = other.view_->getMultiSampleAntiAliasingOptions();
    view_->setMultiSampleAntiAliasingOptions(msaa);

    // AO, AA, TAA, shadows, post‑fx -----------------------------------------
    view_->setAmbientOcclusionOptions(other.view_->getAmbientOcclusionOptions());
    view_->setAntiAliasing(other.view_->getAntiAliasing());
    view_->setTemporalAntiAliasingOptions(other.view_->getTemporalAntiAliasingOptions());
    view_->setShadowingEnabled(other.view_->isShadowingEnabled());
    view_->setPostProcessingEnabled(other.view_->isPostProcessingEnabled());

    if (other.configured_for_picking_) {
        ConfigureForColorPicking();
    }
}

// ----------------------------------------------------------------------------
// scene / render hooks -------------------------------------------------------
// ----------------------------------------------------------------------------

void FilamentView::SetScene(FilamentScene& scene) {
    scene_ = &scene;
    view_->setScene(scene_->GetNativeScene());
}

void FilamentView::PreRender() {
    // ... (sin cambios) ------------------------------------------------------
}

void FilamentView::PostRender() {}

// ----------------------------------------------------------------------------
}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
