// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Metal compute backend for Gaussian splatting (macOS / Apple Silicon).
// MTLTexture creation and Filament import helpers are inlined here so they
// share the same translation unit as the backend that uses them.

#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatRenderer.h"

#if defined(__APPLE__)

#import <Metal/Metal.h>

#include <filament/Texture.h>
#include <filament/View.h>

#include <memory>
#include <unordered_map>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentNativeInterop.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"
#include "open3d/visualization/rendering/gaussian_splat/ComputeGPU.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatDataPacking.h"
#include "open3d/visualization/rendering/gaussian_splat/GaussianSplatPassRunner.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

// Forward declaration so PrepareAppleOutputTextures can call the release helper
// on partial-setup failure paths.
void ReleaseAppleOutputTextures(GaussianSplatRenderer::OutputTargets& targets);

/// Create the GS scene-depth and color MTLTextures and import them into
/// Filament as a shared render target.  Scene depth is always allocated to
/// keep render-target topology stable across scene-content changes.
bool PrepareAppleOutputTextures(FilamentView& view,
                                FilamentResourceManager& resource_mgr,
                                std::uint32_t width,
                                std::uint32_t height,
                                GaussianSplatRenderer::OutputTargets& targets) {
    FilamentMetalNativeHandles mh =
            GetFilamentMetalNativeHandles(EngineInstance::GetPlatform());
    if (!mh.valid) {
        return false;
    }

    id<MTLDevice> device =
            (__bridge id<MTLDevice>)reinterpret_cast<void*>(mh.device);
    if (!device) {
        return false;
    }

    const int w = static_cast<int>(width);
    const int h = static_cast<int>(height);

    MTLTextureDescriptor* depth_desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                         width:w
                                        height:h
                                     mipmapped:NO];
    depth_desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    depth_desc.storageMode = MTLStorageModePrivate;

    MTLTextureDescriptor* color_desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                         width:w
                                        height:h
                                     mipmapped:NO];
    // RenderTarget: needed for kCompositeSolid render-pass clear.
    // ShaderWrite: needed for kFull compute composite.
    // ShaderRead:  needed for ImGui overlay sampling.
    color_desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
                       MTLTextureUsageShaderWrite;
    color_desc.storageMode = MTLStorageModePrivate;

    // Always allocate scene_depth to ensure stable render-target topology.
    id<MTLTexture> scene_depth = [device newTextureWithDescriptor:depth_desc];
    if (!scene_depth) {
        return false;
    }
    scene_depth.label = @"gs.scene_depth";
    targets.scene_depth_mtl_texture = reinterpret_cast<std::uintptr_t>(
            (__bridge_retained void*)scene_depth);

    id<MTLTexture> gs_color = [device newTextureWithDescriptor:color_desc];
    if (!gs_color) {
        // Release already-retained depth before failing.
        id<MTLTexture> t =
                (__bridge_transfer id<MTLTexture>)reinterpret_cast<void*>(
                        targets.scene_depth_mtl_texture);
        (void)t;
        targets.scene_depth_mtl_texture = 0;
        return false;
    }
    gs_color.label = @"gs.color";
    targets.gs_color_mtl_texture =
            reinterpret_cast<std::uintptr_t>((__bridge_retained void*)gs_color);

    using Tex = filament::Texture;
    // Import scene depth (Filament writes, GS composite reads for occlusion).
    if (targets.scene_depth_mtl_texture != 0) {
        targets.depth = resource_mgr.CreateImportedMTLTexture(
                targets.scene_depth_mtl_texture, w, h,
                static_cast<int>(Tex::InternalFormat::DEPTH32F),
                static_cast<int>(Tex::Usage::DEPTH_ATTACHMENT |
                                 Tex::Usage::SAMPLEABLE));
        if (!targets.depth) {
            ReleaseAppleOutputTextures(targets);
            return false;
        }
    }
    // SAMPLEABLE:       ImGui reads the final GS color output.
    // COLOR_ATTACHMENT: required for the readback render target used by
    //                   readPixels in the offscreen (FilamentRenderToBuffer)
    //                   path.
    // BLIT_SRC:         required by Filament's readPixels precondition
    //                   (will be asserted in a later release of Filament).
    targets.color = resource_mgr.CreateImportedMTLTexture(
            targets.gs_color_mtl_texture, w, h,
            static_cast<int>(Tex::InternalFormat::RGBA16F),
            static_cast<int>(Tex::Usage::SAMPLEABLE |
                             Tex::Usage::COLOR_ATTACHMENT |
                             Tex::Usage::BLIT_SRC));

    if (!targets.color) {
        ReleaseAppleOutputTextures(targets);
        return false;
    }

    auto view_color = view.GetColorBuffer();
    if (!view_color) {
        ReleaseAppleOutputTextures(targets);
        return false;
    }

    if (targets.depth) {
        // Use shared depth: Filament writes into the depth texture that
        // the GS composite shader will later sample.
        targets.render_target =
                resource_mgr.CreateRenderTarget(view_color, targets.depth);
    } else {
        // No shared depth: create a Filament-owned depth attachment so
        // Filament renders normally (depth stays private to Filament).
        auto owned_depth = resource_mgr.CreateDepthAttachmentTexture(w, h);
        targets.depth = owned_depth;
        targets.render_target =
                resource_mgr.CreateRenderTarget(view_color, targets.depth);
    }
    view.SetRenderTarget(targets.render_target);

    auto* native = view.GetNativeView();
    auto msaa = native->getMultiSampleAntiAliasingOptions();
    msaa.enabled = false;
    native->setMultiSampleAntiAliasingOptions(msaa);
    view.SetPostProcessing(false);

    return static_cast<bool>(targets.render_target);
}

/// Release ARC-retained MTLTextures stored as uintptr_t in OutputTargets.
void ReleaseAppleOutputTextures(GaussianSplatRenderer::OutputTargets& targets) {
    if (targets.scene_depth_mtl_texture != 0) {
        id<MTLTexture> t =
                (__bridge_transfer id<MTLTexture>)reinterpret_cast<void*>(
                        targets.scene_depth_mtl_texture);
        (void)t;
        targets.scene_depth_mtl_texture = 0;
    }
    if (targets.gs_color_mtl_texture != 0) {
        id<MTLTexture> t =
                (__bridge_transfer id<MTLTexture>)reinterpret_cast<void*>(
                        targets.gs_color_mtl_texture);
        (void)t;
        targets.gs_color_mtl_texture = 0;
    }
}

}  // namespace

class GaussianSplatMetalBackend final : public GaussianSplatRenderer::Backend {
public:
    GaussianSplatMetalBackend(FilamentResourceManager& resource_mgr,
                              const GaussianSplatRenderer::RenderConfig& config)
        : config_(config) {
        (void)resource_mgr;
        FilamentMetalNativeHandles mh =
                GetFilamentMetalNativeHandles(EngineInstance::GetPlatform());
        if (mh.valid) {
            gpu_ = CreateComputeGpuContextMetal(mh.device, mh.command_queue);
        }
    }

    ~GaussianSplatMetalBackend() override { Cleanup(); }

    const char* GetName() const override { return "Metal"; }

    void BeginFrame(std::uint64_t) override {}

    void ForgetView(const FilamentView& view) override {
        // Release per-view GPU resources when the view is removed.
        auto it = view_states_.find(&view);
        if (it != view_states_.end()) {
            DestroyViewState(it->second);
            view_states_.erase(it);
        }
    }

    bool RenderGeometryStage(
            const FilamentView& view,
            const FilamentScene& scene,
            const GaussianSplatRenderer::ViewRenderData& render_data,
            GaussianSplatRenderer::OutputTargets& targets) override {
        // Pack view params and run all geometry compute passes on the Metal
        // command queue.
        if (!gpu_) {
            return false;
        }

        const GaussianSplatPackedAttrs* attrs =
                scene.GetGaussianSplatPackedAttrs();
        if (!attrs || attrs->splat_count == 0) {
            return false;
        }

        // Pack only the view-params UBO (288 bytes) — cheap every frame.
        PackedGaussianScene frame =
                PackGaussianViewParams(*attrs, render_data, config_);
        if (!frame.valid) {
            return false;
        }

        auto& vs = view_states_[&view];

        const std::uint64_t scene_id = scene.GetGeometryChangeId();
        const bool scene_changed =
                (scene_id != vs.cached_scene_id ||
                 attrs->splat_count != vs.cached_splat_count);

        return RunGaussianGeometryPasses(*gpu_, config_, frame, *attrs, vs,
                                         scene_id, scene_changed);
    }

    bool RenderCompositeStage(
            const FilamentView& view,
            const GaussianSplatRenderer::ViewRenderData&,
            GaussianSplatRenderer::OutputTargets& targets) override {
        // Dispatch the composite pass on the Metal command queue.
        if (!gpu_) {
            return false;
        }
        auto it = view_states_.find(&view);
        if (it == view_states_.end() || it->second.view_params_buf == 0) {
            return false;
        }
        return RunGaussianCompositePass(*gpu_, config_, it->second, targets);
    }

    bool PrepareOutputTextures(
            FilamentView& view,
            FilamentResourceManager& resource_mgr,
            std::uint32_t width,
            std::uint32_t height,
            GaussianSplatRenderer::OutputTargets& targets) override {
        // Create shared MTLTextures and import them into Filament for zero-copy
        // access.  Scene depth and GS color are always allocated.
        return PrepareAppleOutputTextures(view, resource_mgr, width, height,
                                          targets);
    }

    void ReleaseOutputTextures(
            FilamentResourceManager&,
            GaussianSplatRenderer::OutputTargets& targets) override {
        // Release ARC-retained MTLTexture handles.
        ReleaseAppleOutputTextures(targets);
    }

    bool ReadMergedDepthToUint16Cpu(const FilamentView& view,
                                    std::vector<std::uint16_t>& out,
                                    std::uint32_t width,
                                    std::uint32_t height) override {
        auto it = view_states_.find(&view);
        if (it == view_states_.end() || it->second.merged_depth_u16_tex == 0) {
            return false;
        }
        return gpu_->DownloadTextureR16UI(it->second.merged_depth_u16_tex,
                                          width, height, out);
    }

    bool ReadCompositeDepthToFloatCpu(const FilamentView& view,
                                      std::vector<float>& out,
                                      std::uint32_t width,
                                      std::uint32_t height) override {
        auto it = view_states_.find(&view);
        if (it == view_states_.end() || it->second.composite_depth_tex == 0) {
            return false;
        }
        return gpu_->DownloadTextureR32F(it->second.composite_depth_tex, width,
                                         height, out);
    }

private:
    void DestroyViewState(GaussianSplatViewGpuResources& vs) {
        // Free all per-view GPU buffers and textures tracked by this backend.
        if (!gpu_) {
            return;
        }
        auto destroy_buf = [&](std::uintptr_t& b) {
            if (b != 0) {
                gpu_->DestroyBuffer(b);
                b = 0;
            }
        };
        destroy_buf(vs.view_params_buf);
        destroy_buf(vs.positions_buf);
        destroy_buf(vs.scales_buf);
        destroy_buf(vs.rotations_buf);
        destroy_buf(vs.dc_opacity_buf);
        destroy_buf(vs.sh_buf);
        destroy_buf(vs.projected_buf);
        destroy_buf(vs.tile_counts_buf);
        destroy_buf(vs.tile_offsets_buf);
        destroy_buf(vs.tile_heads_buf);
        destroy_buf(vs.counters_buf);
        destroy_buf(vs.tile_entries_buf);
        destroy_buf(vs.dispatch_args_buf);
        destroy_buf(vs.sort_keys_buf[0]);
        destroy_buf(vs.sort_keys_buf[1]);
        destroy_buf(vs.sort_values_buf[0]);
        destroy_buf(vs.sort_values_buf[1]);
        destroy_buf(vs.histogram_buf);
        destroy_buf(vs.radix_params_buf);
        destroy_buf(vs.sorted_splat_indices_buf);
        destroy_buf(vs.mask_buf);
        if (vs.composite_depth_tex != 0) {
            gpu_->DestroyTexture(vs.composite_depth_tex);
            vs.composite_depth_tex = 0;
        }
        if (vs.merged_depth_u16_tex != 0) {
            gpu_->DestroyTexture(vs.merged_depth_u16_tex);
            vs.merged_depth_u16_tex = 0;
        }
    }

    void Cleanup() {
        for (auto& pair : view_states_) {
            DestroyViewState(pair.second);
        }
        view_states_.clear();
        gpu_.reset();
    }

    const GaussianSplatRenderer::RenderConfig& config_;
    std::unordered_map<const FilamentView*, GaussianSplatViewGpuResources>
            view_states_;
    std::unique_ptr<GaussianSplatGpuContext> gpu_;
};

std::unique_ptr<GaussianSplatRenderer::Backend> CreateGaussianSplatMetalBackend(
        FilamentResourceManager& resource_mgr,
        const GaussianSplatRenderer::RenderConfig& config) {
    return std::make_unique<GaussianSplatMetalBackend>(resource_mgr, config);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // defined(__APPLE__)
