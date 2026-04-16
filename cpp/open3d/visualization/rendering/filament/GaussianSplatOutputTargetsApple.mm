// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianSplatOutputTargetsApple.h"

#if defined(__APPLE__)

#import <Metal/Metal.h>

#include <filament/Texture.h>
#include <filament/View.h>

#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentNativeInterop.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"
#include "open3d/visualization/rendering/filament/GaussianSplatRenderer.h"

#include <cstdint>

namespace open3d {
namespace visualization {
namespace rendering {

bool PrepareGaussianImportedRenderTargetsApple(
        FilamentView& view,
        FilamentResourceManager& resource_mgr,
        std::uint32_t width,
        std::uint32_t height,
        bool needs_scene_depth,
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

    // Always allocate scene_depth for GS views to ensure stable render-target
    // topology and avoid Filament handle lifecycle hazards from topology
    // transitions. The composite shader's occlusion test is gated separately.
    id<MTLTexture> scene_depth = [device newTextureWithDescriptor:depth_desc];
    if (!scene_depth) {
        return false;
    }
    scene_depth.label = @"gs.scene_depth";
    targets.scene_depth_mtl_texture = reinterpret_cast<std::uintptr_t>(
            (__bridge_retained void*)scene_depth);

    id<MTLTexture> gs_color = [device newTextureWithDescriptor:color_desc];
    if (!gs_color) {
        ReleaseGaussianImportedMTLTexturesApple(targets);
        return false;
    }
    gs_color.label = @"gs.color";
    targets.gs_color_mtl_texture =
            reinterpret_cast<std::uintptr_t>((__bridge_retained void*)gs_color);

    using Tex = filament::Texture;
    // Import scene depth (always allocated; Filament writes, GS reads for
    // occlusion testing).
    if (targets.scene_depth_mtl_texture != 0) {
        targets.depth = resource_mgr.CreateImportedMTLTexture(
                targets.scene_depth_mtl_texture, w, h,
                static_cast<int>(Tex::InternalFormat::DEPTH32F),
                static_cast<int>(Tex::Usage::DEPTH_ATTACHMENT |
                                 Tex::Usage::SAMPLEABLE));
        if (!targets.depth) {
            ReleaseGaussianImportedMTLTexturesApple(targets);
            return false;
        }
    }
    // SAMPLEABLE:       ImGui reads the final GS color output.
    // COLOR_ATTACHMENT: required for the readback render target used by
    //                   readPixels in the offscreen (FilamentRenderToBuffer)
    //                   path.
    // BLIT_SRC:         required by Filament's readPixels precondition
    //                   (will be asserted in a later release of Filament).
    // The underlying Metal texture is created with MTLTextureUsageRenderTarget,
    // so the physical texture supports all three usages.
    targets.color = resource_mgr.CreateImportedMTLTexture(
            targets.gs_color_mtl_texture, w, h,
            static_cast<int>(Tex::InternalFormat::RGBA16F),
            static_cast<int>(Tex::Usage::SAMPLEABLE |
                             Tex::Usage::COLOR_ATTACHMENT |
                             Tex::Usage::BLIT_SRC));

    if (!targets.color) {
        ReleaseGaussianImportedMTLTexturesApple(targets);
        return false;
    }

    // A valid Filament color buffer is required to build the render target.
    // Mirror the GL backend: release partial allocations and fail if absent.
    auto view_color = view.GetColorBuffer();
    if (!view_color) {
        ReleaseGaussianImportedMTLTexturesApple(targets);
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

void ReleaseGaussianImportedMTLTexturesApple(
        GaussianSplatRenderer::OutputTargets& targets) {
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

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // defined(__APPLE__)
